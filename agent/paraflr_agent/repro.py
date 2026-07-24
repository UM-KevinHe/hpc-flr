"""Turn a trace into a standalone ``repro.R``.

The generated script calls ``paraflr`` directly — no Python, no language
model, and no dependency on this repository's tool scripts. A reader with the
data file and the installed package can rerun it and get the agent's numbers
back. That is the property the fidelity experiment checks, and the reason
this generator emits package calls rather than replaying the JSON handshake.

Informational steps (inspect_data, start_analysis) are skipped, as are steps
that errored during the run; both are noted in place so the script still
reads as a complete record of what happened.
"""
from __future__ import annotations

from typing import Any

from .trace import AgentTrace

_INFO_ONLY = {"inspect_data", "start_analysis"}


def _r_literal(value: Any) -> str:
    """Render a Python value as an R literal."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return '"{}"'.format(value.replace("\\", "\\\\").replace('"', '\\"'))
    if isinstance(value, (list, tuple)):
        if not value:
            return "c()"
        return "c(" + ", ".join(_r_literal(v) for v in value) + ")"
    if isinstance(value, dict):
        return "list(" + ", ".join(
            "{} = {}".format(k, _r_literal(v)) for k, v in value.items()) + ")"
    return _r_literal(str(value))


_HEADER_FUNCS = '''
load_env <- function(path) {
  e <- new.env()
  ext <- tolower(tools::file_ext(path))
  if (ext %in% c("rda", "rdata")) {
    load(path, envir = e)
  } else if (ext == "rds") {
    assign(tools::file_path_sans_ext(basename(path)), readRDS(path), envir = e)
  } else if (ext == "csv") {
    assign(tools::file_path_sans_ext(basename(path)),
           utils::read.csv(path, stringsAsFactors = FALSE), envir = e)
  } else {
    stop(sprintf("Unsupported file extension: .%s", ext))
  }
  e
}

get_field <- function(expr_str, e) eval(parse(text = expr_str), envir = e)

# test_gamma.single() always tests the first provider block, so testing any
# other provider means re-indexing the fitted object — records and gamma
# together — to put that provider first. No estimate changes.
rotate_to_provider <- function(fit, provider) {
  ids <- names(fit$gamma)
  k <- match(as.character(provider), ids)
  if (is.na(k)) stop(sprintf("Provider '%s' not found", provider))
  prov <- as.character(fit$data[[fit$char_list$prov.char]])
  target <- ids[k]
  fit$data <- fit$data[c(which(prov == target), which(prov != target)), ,
                       drop = FALSE]
  fit$gamma <- fit$gamma[c(k, setdiff(seq_along(ids), k))]
  fit
}
'''


def _fit_block(args: dict, var: str) -> list:
    """R lines that load the data and fit; leaves the fit in ``var``."""
    ctl = []
    for key, arg in (("cutoff", "cutoff"), ("tol", "tol"),
                     ("max.iter", "max_iter"), ("bound", "bound"),
                     ("threads", "threads")):
        if args.get(arg) is not None:
            ctl.append("{} = {}".format(key, _r_literal(args[arg])))
    ctl_str = (", " + ", ".join(ctl)) if ctl else ""
    return [
        "e <- load_env({})".format(_r_literal(args.get("data_path"))),
        "{}_Y  <- get_field({}, e)".format(var, _r_literal(args.get("y_expr"))),
        "{}_Z  <- as.matrix(get_field({}, e))".format(
            var, _r_literal(args.get("z_expr"))),
        "{}_ID <- get_field({}, e)".format(var, _r_literal(args.get("id_expr"))),
        "{} <- paraflr::logis_firth(Y = {v}_Y, Z = {v}_Z, ID = {v}_ID{c}, "
        "message = FALSE)".format(var, v=var, c=ctl_str),
    ]


def render_repro_r(trace: AgentTrace) -> str:
    """Return the contents of ``repro.R`` as a string."""
    query_line = trace.user_query.replace("\n", " ")[:200]
    out = [
        "#!/usr/bin/env Rscript",
        "#",
        "# repro.R — replay of one paraflr agent run, with no language model",
        "# in the loop. Every value the agent reported is recomputed here by",
        "# calling paraflr directly.",
        "#",
        "# Generated:  {}".format(trace.finished_at or trace.started_at),
        "# Request:    {}".format(query_line),
        "# Model:      {} @ {}".format(trace.model_name, trace.model_endpoint),
        "# Prompt SHA: {}".format(trace.system_prompt_sha256),
        "#",
        "# Requires: the paraflr package and the data file referenced below.",
        "",
        "suppressPackageStartupMessages(library(paraflr))",
        _HEADER_FUNCS,
        "results <- list()",
        "",
    ]

    step = 0
    for event in trace.events:
        step += 1
        args = event.effective_args or {}
        if event.tool in _INFO_ONLY:
            out += ["# Step {}: {} — informational, nothing to recompute."
                    .format(step, event.tool), ""]
            continue
        if event.status != "ok":
            out += ["# Step {}: {} — skipped, this call errored during the "
                    "run: {}".format(step, event.tool,
                                     event.error_message or "unknown"), ""]
            continue

        var = "r{}".format(step)
        out.append("# Step {}: {}".format(step, event.tool))
        out.append("cat('=== Step {}: {} ===\\n')".format(step, event.tool))

        if event.tool == "fit_flr":
            out += _fit_block(args, var)
            out += ["print({}$beta)".format(var),
                    "print(summary({}$gamma))".format(var),
                    "results[['fit_flr_step{}']] <- {}".format(step, var)]

        elif event.tool == "test_provider":
            out += _fit_block(args, var)
            provider = args.get("provider")
            if provider is not None:
                out.append("{v} <- rotate_to_provider({v}, {p})".format(
                    v=var, p=_r_literal(provider)))
            null_arg = args.get("null", "median")
            out += [
                "{v}_test <- paraflr::test_gamma.single({v}, methods = {m}, "
                "null = {n}, alpha = {a}, firth = {f})".format(
                    v=var,
                    m=_r_literal(args.get("method", "score")),
                    n=_r_literal(null_arg),
                    a=_r_literal(args.get("alpha", 0.05)),
                    f=_r_literal(bool(args.get("firth", False)))),
                "print({}_test)".format(var),
                "results[['test_provider_step{}']] <- {}_test".format(step, var),
            ]

        elif event.tool == "benchmark_threads":
            threads = args.get("threads_list") or [1, 2, 4]
            repeats = args.get("repeats", 1)
            base = dict(args)
            base.pop("threads_list", None)
            base.pop("repeats", None)
            base["threads"] = None  # set inside the loop below
            out += [
                "e <- load_env({})".format(_r_literal(args.get("data_path"))),
                "{v}_Y  <- get_field({e}, e)".format(
                    v=var, e=_r_literal(args.get("y_expr"))),
                "{v}_Z  <- as.matrix(get_field({e}, e))".format(
                    v=var, e=_r_literal(args.get("z_expr"))),
                "{v}_ID <- get_field({e}, e)".format(
                    v=var, e=_r_literal(args.get("id_expr"))),
                "{v}_times <- sapply({t}, function(th) {{".format(
                    v=var, t=_r_literal(list(threads))),
                "  min(replicate({}, system.time(paraflr::logis_firth(".format(
                    max(1, int(repeats))),
                "    Y = {v}_Y, Z = {v}_Z, ID = {v}_ID, threads = th, "
                "message = FALSE))[['elapsed']]))".format(v=var),
                "})",
                "{v}_bench <- data.frame(threads = {t}, elapsed = {v}_times, "
                "speedup = {v}_times[1] / {v}_times)".format(
                    v=var, t=_r_literal(list(threads))),
                "print({}_bench)".format(var),
                "results[['benchmark_threads_step{}']] <- {}_bench".format(
                    step, var),
            ]

        elif event.tool == "simulate_provider_data":
            out += [
                "# Data generated by the simulate_provider_data tool with:",
                "#   {}".format(_r_literal(args)),
                "# Rerun that tool with the same seed to regenerate the file;",
                "# the steps below read it from disk.",
            ]

        out.append("")

    out += [
        "# Done. `results` holds one entry per replayed step.",
        "invisible(results)",
        "",
    ]
    return "\n".join(out)


def write_repro_r(trace: AgentTrace, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(render_repro_r(trace))
