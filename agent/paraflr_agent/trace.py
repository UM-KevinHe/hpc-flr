"""Audit trail for one agent run.

Every tool call is recorded — the script that ran, the arguments the model
emitted, the arguments actually dispatched after the harness resolved them,
the status, the latency, and a summary of the result. A reader can then check
what the agent did without rerunning the language model.

``result_summary`` is deliberately small: shapes and scalars, not coefficient
vectors. The full results stay in the in-memory :class:`AgentResponse` during
the run, and ``repro.R`` regenerates them exactly.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


@dataclass
class TraceEvent:
    """One tool call."""
    timestamp: str
    tool: str
    llm_args: Dict[str, Any]
    effective_args: Dict[str, Any]
    status: str
    result_summary: Dict[str, Any]
    latency_ms: int
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentTrace:
    """All tool calls plus run-level metadata for one ``agent.query()``."""
    user_query: str
    model_name: str
    model_endpoint: str
    system_prompt_sha256: str
    started_at: str = field(default_factory=_utc_iso)
    finished_at: Optional[str] = None
    total_latency_ms: Optional[int] = None
    llm_turns: int = 0
    prompt_tokens_total: Optional[int] = None
    completion_tokens_total: Optional[int] = None
    # Which tools were exposed to the model this run, after subsetting.
    tools_exposed: Optional[Dict[str, Any]] = None
    events: List[TraceEvent] = field(default_factory=list)

    def add_event(self, event: TraceEvent) -> None:
        self.events.append(event)

    def to_dict(self) -> Dict[str, Any]:
        d = {k: v for k, v in asdict(self).items() if k != "events"}
        d["events"] = [e.to_dict() for e in self.events]
        return d

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


_SCALARS_WORTH_KEEPING = {
    "status", "message", "class", "where", "returncode",
    "n_obs", "n_covariates", "n_providers", "event_rate", "threads",
    "elapsed_sec", "neg2Loglkd", "provider", "method", "firth", "null",
    "null_value", "alpha", "gamma_est", "stat", "p_value", "flag", "verdict",
    "provider_n_obs", "data_path", "object_name", "seed", "language",
    "best_speedup", "fastest_threads", "baseline_threads", "max_beta_diff",
    "estimates_agree", "parallel_speedup_observed", "detected_cores",
    "providers_with_no_events",
}


def summarize_result(tool: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce a full R result to the small dict stored in the trace."""
    if not isinstance(result, dict):
        return {"status": "non_dict", "type": type(result).__name__}

    summary: Dict[str, Any] = {}
    for k, v in result.items():
        if k in _SCALARS_WORTH_KEEPING:
            summary[k] = v
        elif k == "beta" and isinstance(v, dict):
            # Small and central to every fit — keep it whole. This is what
            # makes a trace enough to check a reported coefficient.
            summary["beta"] = v
        elif isinstance(v, list):
            summary[k + "_length"] = len(v)
        elif isinstance(v, dict):
            summary[k + "_keys"] = len(v)
    return summary
