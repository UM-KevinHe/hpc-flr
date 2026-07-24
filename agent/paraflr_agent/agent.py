"""The agent loop.

Talks to any OpenAI-compatible Chat Completions endpoint (vLLM with
``--enable-auto-tool-choice``, Together, OpenRouter, ...):

    user request
        -> harness inspects the data file, injects the structure, narrows
           the exposed tools
        -> model emits a tool call
        -> tools.dispatch runs R
        -> compressed result goes back to the model
        -> ... until the model answers in prose
        -> AgentResponse (text + trace + full results)

The model routes and explains. It never computes: every number in the answer
came out of paraflr.

Tests can inject a stub via ``client=`` — anything with
``chat.completions.create`` — and run the whole loop without a GPU.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from . import tools as _tools
from .helpers import compress_tool_result_for_llm
from .prompts import SYSTEM_PROMPT, prompt_sha256
from .repro import write_repro_r
from .trace import AgentTrace, TraceEvent, summarize_result, _utc_iso


def _field_mapping_block(inspect_result: dict) -> str:
    """One line naming the exact expressions the tools need, or ''.

    ``inspect_data`` derives this in R from the file itself. Handing it to
    the model removes the single most common failure of a small model on
    this task — inventing a plausible field name — and replaces it with a
    deterministic lookup. When the guess fails, the model still has the full
    structure block above it and can work the names out itself.
    """
    mapping = inspect_result.get("mapping") if isinstance(inspect_result, dict) else None
    if not mapping:
        return ""
    parts = [
        f"FIELD MAPPING (use verbatim): y_expr = {mapping['y_expr']}, "
        f"z_expr = {mapping['z_expr']}, id_expr = {mapping['id_expr']}."
    ]
    if mapping.get("n_providers"):
        parts.append(
            f" This file holds {mapping.get('n_obs')} records across "
            f"{mapping['n_providers']} providers, with "
            f"{mapping.get('n_covariates')} covariates and an overall event "
            f"rate of {mapping.get('event_rate')}.")
    return "".join(parts)


@dataclass
class AgentResponse:
    """What :meth:`ParaFLRAgent.query` returns.

    ``text`` is the model's final answer, ``trace`` the audit record, and
    ``tool_results`` the full uncompressed R results (what a UI would render).
    """
    text: str
    trace: AgentTrace
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None

    def save_trace(self, path: str) -> None:
        self.trace.save(path)

    def write_repro_r(self, path: str) -> None:
        write_repro_r(self.trace, path)


class ParaFLRAgent:
    """Tool-use loop over the paraflr tool catalogue.

    Parameters
    ----------
    model_endpoint:
        OpenAI-compatible base URL, e.g. ``http://localhost:8000/v1``.
    model_name:
        Whatever the endpoint calls the model (vLLM's ``--served-model-name``).
    api_key:
        Bearer token; vLLM ignores the value but wants something non-empty.
    system_prompt:
        Overrides the deployment prompt. The ablation uses this.
    max_turns:
        Guard against a model that never stops calling tools.
    temperature, max_tokens:
        Deployment runs at ``temperature=0``: deterministic routing, and a
        run reproducible from its trace. The pass^k experiment raises it.
    scaffold:
        Switches harness layers off, one at a time, for the ablation. Keys:
        ``all``, ``inject_structure``, ``subset_tools``, ``dispatch_guard``;
        each defaults to on. Omitting the argument leaves deployment
        behaviour unchanged.
    client:
        A pre-built OpenAI-style client. Tests pass a stub.
    """

    def __init__(
        self,
        model_endpoint: str,
        model_name: str = "qwen2.5-7b-awq",
        api_key: str = "EMPTY",
        system_prompt: Optional[str] = None,
        max_turns: int = 8,
        client: Optional[Any] = None,
        temperature: float = 0.0,
        max_tokens: int = 1536,
        scaffold: Optional[Dict[str, bool]] = None,
        backend: str = "r",
    ) -> None:
        self.model_endpoint = model_endpoint
        self.model_name = model_name
        self.api_key = api_key
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.max_turns = max_turns
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._scaffold = dict(scaffold or {})
        # Computation backend: "r" shells out to the paraflr R package,
        # "python" calls the paraflr Python package in process. Same numbers
        # either way (shared C++ core); the whole agent loop is unchanged.
        self.backend = backend
        self._client = client
        self._tool_schemas = _tools.load_schemas()

    def _on(self, layer: str) -> bool:
        """True if scaffolding layer ``layer`` is active."""
        if not self._scaffold.get("all", True):
            return False
        return self._scaffold.get(layer, True)

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover - environment problem
            raise RuntimeError("openai SDK not installed. Run "
                               "`pip install openai`.") from e
        self._client = OpenAI(base_url=self.model_endpoint,
                              api_key=self.api_key)
        return self._client

    # -- one user turn --------------------------------------------------------

    def query(
        self,
        user_text: str,
        data_path: Optional[str] = None,
        history: Optional[List] = None,
        task_override: Optional[str] = None,
        bound_context: Optional[str] = None,
    ) -> AgentResponse:
        """Run one request end to end.

        ``history`` is a list of ``(user, assistant)`` pairs from earlier
        turns; only the prose is replayed, not the tool chain, which is
        enough for routing a follow-up and much cheaper.

        ``task_override`` and ``bound_context`` are for callers that already
        know the answer to part of the question — a UI where the user picked
        the task from a menu, or a harness pinning a configuration so that
        only the routing is under test.
        """
        start = time.monotonic()
        trace = AgentTrace(
            user_query=user_text,
            model_name=self.model_name,
            model_endpoint=self.model_endpoint,
            system_prompt_sha256=prompt_sha256(self.system_prompt),
        )

        user_content = user_text
        active_schemas = self._tool_schemas
        auto_inspect: Optional[dict] = None

        # Pre-inspect the data file ourselves rather than asking the model to
        # do it. A 7B model skips a "you must call inspect_data first" rule
        # often enough that the guarantee has to be architectural: by the
        # time the model sees the request, the field names are already in it.
        if data_path is not None and self._on("inject_structure"):
            auto_inspect = _tools.dispatch("inspect_data", backend=self.backend,
                                           data_path=data_path)
            trace.add_event(TraceEvent(
                timestamp=_utc_iso(),
                tool="inspect_data",
                llm_args={},
                # The flag marks this as the harness's own call, so a reader
                # can tell it apart from one the model made.
                effective_args={"data_path": data_path, "_auto_prepend": True},
                status=auto_inspect.get("status", "ok"),
                result_summary=summarize_result("inspect_data", auto_inspect),
                latency_ms=0,
            ))
            user_content += (
                "\n\n[The user's data file is at: {}]"
                "\n\n[DATA STRUCTURE — already inspected for you. Use these "
                "names verbatim in any *_expr argument; do not call "
                "inspect_data again, and do not invent or translate a name:\n"
                "{}\n]".format(data_path,
                               json.dumps(auto_inspect, indent=2,
                                          ensure_ascii=False)))
            mapping_line = _field_mapping_block(auto_inspect)
            if mapping_line:
                user_content += "\n\n[{}]".format(mapping_line)
        elif data_path is not None:
            user_content += "\n\n[The user's data file is at: {}]".format(data_path)

        if self._on("subset_tools"):
            active_schemas, exposed = _tools.select_tool_schemas(
                self._tool_schemas, auto_inspect,
                has_data_file=data_path is not None,
                task_override=task_override)
            trace.tools_exposed = exposed

        if bound_context:
            user_content += "\n\n[{}]".format(bound_context)

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt}]
        for entry in (history or []):
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                continue
            prev_user, prev_asst = entry[0], entry[1]
            if prev_user:
                messages.append({"role": "user", "content": str(prev_user)})
            if prev_asst:
                messages.append({"role": "assistant", "content": str(prev_asst)})
        messages.append({"role": "user", "content": user_content})

        tool_results: List[Dict[str, Any]] = []
        final_text, error = "", None
        prompt_tokens = completion_tokens = 0

        try:
            client = self._ensure_client()
            for turn in range(self.max_turns):
                trace.llm_turns = turn + 1
                completion = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=active_schemas,
                    tool_choice="auto",
                    temperature=self.temperature,
                    # Capped so that a model which starts rambling after a
                    # tool error costs one turn, not the whole context window.
                    max_tokens=self.max_tokens,
                )
                usage = getattr(completion, "usage", None)
                if usage is not None:
                    prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
                    completion_tokens += getattr(usage, "completion_tokens", 0) or 0

                msg = completion.choices[0].message
                tool_calls = getattr(msg, "tool_calls", None) or []

                entry: Dict[str, Any] = {"role": "assistant",
                                         "content": msg.content or ""}
                if tool_calls:
                    entry["tool_calls"] = [
                        {"id": tc.id, "type": "function",
                         "function": {"name": tc.function.name,
                                      "arguments": tc.function.arguments}}
                        for tc in tool_calls]
                messages.append(entry)

                if not tool_calls:
                    final_text = msg.content or ""
                    break

                active_names = {s["function"]["name"] for s in active_schemas}

                for tc in tool_calls:
                    name = tc.function.name
                    try:
                        llm_args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        llm_args = {"_raw_arguments_str": tc.function.arguments}

                    t0 = time.monotonic()
                    # Subsetting controls what the model is SHOWN; dispatch
                    # would still run any registered tool. A small model can
                    # emit a name it was never offered, so refuse it here and
                    # tell it what it may actually call.
                    if self._on("dispatch_guard") and name not in active_names:
                        result = {
                            "status": "error",
                            "class": "ToolNotAvailable",
                            "where": "paraflr_agent.agent",
                            "message": (
                                f"Tool '{name}' is not available for this "
                                f"request. Call one of these instead: "
                                f"{sorted(active_names)}."),
                        }
                        effective_args = dict(llm_args)
                    else:
                        result = _tools.dispatch(name, backend=self.backend,
                                                 **llm_args)
                        try:
                            effective_args, _ = _tools.resolve_args(name, llm_args)
                        except Exception:  # noqa: BLE001
                            effective_args = dict(llm_args)
                    latency_ms = int((time.monotonic() - t0) * 1000)

                    status = (result.get("status", "ok")
                              if isinstance(result, dict) else "non_dict")
                    trace.add_event(TraceEvent(
                        timestamp=_utc_iso(),
                        tool=name,
                        llm_args=llm_args,
                        effective_args=effective_args,
                        status=status,
                        result_summary=summarize_result(name, result),
                        latency_ms=latency_ms,
                        error_message=(result.get("message")
                                       if isinstance(result, dict)
                                       and status == "error" else None),
                    ))
                    tool_results.append({"tool": name, "args": llm_args,
                                         "result": result})

                    # The model sees a shrunk copy; the full result is already
                    # in tool_results and reproducible from the trace.
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(
                            compress_tool_result_for_llm(result),
                            ensure_ascii=False),
                    })
            else:
                error = (f"Max turns ({self.max_turns}) reached without a "
                         f"final answer.")
                final_text = (messages[-1].get("content") or "") if messages else ""

        except Exception as e:  # noqa: BLE001
            err_class, err_text = type(e).__name__, str(e)
            if "BadRequest" in err_class and "maximum context" in err_text.lower():
                error = ("The conversation outgrew the model's context window. "
                         "Start a fresh chat; any trace.json and repro.R "
                         "already written stay valid.")
            else:
                error = f"{err_class}: {e}"

        trace.finished_at = _utc_iso()
        trace.total_latency_ms = int((time.monotonic() - start) * 1000)
        trace.prompt_tokens_total = prompt_tokens or None
        trace.completion_tokens_total = completion_tokens or None

        return AgentResponse(text=final_text, trace=trace,
                             tool_results=tool_results, error=error)
