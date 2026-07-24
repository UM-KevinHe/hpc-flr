"""paraflr_agent — a verification-scaffolded agent over the paraflr package.

A locally served open-weights model (Qwen 2.5-7B by default) routes a
plain-language request to one of six tools; paraflr computes every reported
number. The model never calculates anything, and every run leaves a trace and
a standalone ``repro.R`` behind.

Usage::

    from paraflr_agent import ParaFLRAgent

    agent = ParaFLRAgent(model_endpoint="http://localhost:8000/v1",
                         model_name="qwen2.5-7b-awq")
    resp = agent.query(
        "Is provider P007 significantly worse than the median? Use the score test.",
        data_path="agent/data/ExampleProviders.rda")
    print(resp.text)
    resp.save_trace("trace.json")
    resp.write_repro_r("repro.R")

Tools can also be called directly, with no model in the loop::

    from paraflr_agent import dispatch
    dispatch("fit_flr", data_path=..., y_expr=..., z_expr=..., id_expr=...)
"""
from __future__ import annotations

from .agent import ParaFLRAgent, AgentResponse
from .prompts import SYSTEM_PROMPT_DEPLOY, SYSTEM_PROMPT_ROUTING
from .tools import TOOL_REGISTRY, dispatch, load_schemas, select_tool_schemas
from .trace import AgentTrace, TraceEvent

__all__ = [
    "ParaFLRAgent",
    "AgentResponse",
    "AgentTrace",
    "TraceEvent",
    "TOOL_REGISTRY",
    "dispatch",
    "load_schemas",
    "select_tool_schemas",
    "SYSTEM_PROMPT_DEPLOY",
    "SYSTEM_PROMPT_ROUTING",
]

__version__ = "0.1.0"
