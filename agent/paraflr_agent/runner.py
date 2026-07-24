"""R subprocess bridge.

Every tool call ends up here: write the arguments to a temp ``input.json``,
run ``Rscript <tool>.R in.json out.json``, read the result back. The R side
never prints to stdout, so nothing has to be parsed out of console noise.

Invariants (changing any of these has broken things before):

  1. ``stdin=subprocess.DEVNULL`` on every Rscript call. Without it Rscript
     can inherit a never-writing stdin and stall on a TTY probe.
  2. Flags are ``--no-save --no-restore --no-init-file``, NOT ``--vanilla``.
     ``--vanilla`` implies ``--no-environ``, which drops ``R_LIBS_USER`` and
     makes user-installed packages (paraflr itself) unreachable.
  3. R scripts are read from ``r_scripts/`` in place; they are never copied
     to a temp directory.

Environment overrides:
  ``PARAFLR_RSCRIPT``     full path to the Rscript executable
  ``PARAFLR_R_SCRIPTS``   directory holding the tool scripts
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

_DEFAULT_R_SCRIPTS = Path(__file__).resolve().parent / "r_scripts"
R_SCRIPTS = Path(os.environ.get("PARAFLR_R_SCRIPTS", _DEFAULT_R_SCRIPTS))


def find_rscript() -> str:
    """Locate an Rscript executable.

    Order: ``$PARAFLR_RSCRIPT``, then ``PATH``, then the usual install
    locations. The last step matters because a GUI or Docker parent does not
    always pass the user's shell PATH down to subprocesses.
    """
    override = os.environ.get("PARAFLR_RSCRIPT")
    if override and Path(override).exists():
        return override

    on_path = shutil.which("Rscript") or shutil.which("Rscript.exe")
    if on_path:
        return on_path

    for base in (Path(r"C:\Program Files\R"), Path(r"C:\Program Files (x86)\R")):
        if not base.exists():
            continue
        for d in sorted((d for d in base.iterdir()
                         if d.is_dir() and d.name.startswith("R-")),
                        key=lambda d: d.name, reverse=True):
            exe = d / "bin" / "Rscript.exe"
            if exe.exists():
                return str(exe)

    for cand in (Path("/Library/Frameworks/R.framework/Resources/bin/Rscript"),
                 Path("/usr/local/bin/Rscript"),
                 Path("/opt/homebrew/bin/Rscript"),
                 Path("/usr/bin/Rscript")):
        if cand.exists():
            return str(cand)

    raise FileNotFoundError(
        "Rscript not found. Install R (https://cran.r-project.org/), or set "
        "PARAFLR_RSCRIPT to the full path of Rscript (Rscript.exe on Windows)."
    )


def run_r(script_name: str, payload: dict, timeout_s: int = 1800) -> dict:
    """Invoke one R tool script; return its parsed result.

    Failures come back as a structured ``{"status": "error", ...}`` dict —
    never a raised exception — so the agent loop can hand the message back to
    the model and let it retry.

    The default timeout is generous because ``benchmark_threads`` refits the
    same model several times over, on data that may be large.
    """
    script_path = R_SCRIPTS / script_name
    if not script_path.exists():
        return {"status": "error",
                "message": f"R script not found: {script_path}",
                "class": "FileNotFoundError",
                "where": "paraflr_agent.runner.run_r"}

    try:
        rscript = find_rscript()
    except FileNotFoundError as e:
        return {"status": "error", "message": str(e),
                "class": "FileNotFoundError",
                "where": "paraflr_agent.runner.find_rscript"}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".in.json",
                                     delete=False, encoding="utf-8") as fin:
        json.dump(payload, fin, ensure_ascii=False)
        in_path = fin.name
    out_path = in_path.replace(".in.json", ".out.json")

    try:
        proc = subprocess.run(
            [rscript, "--no-save", "--no-restore", "--no-init-file",
             str(script_path), in_path, out_path],
            capture_output=True, text=True, timeout=timeout_s,
            encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL)

        if Path(out_path).exists():
            try:
                with open(out_path, encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                return {"status": "error",
                        "message": f"R output was not valid JSON: {e}",
                        "class": "JSONDecodeError", "where": script_name,
                        "stderr": proc.stderr.strip()[:2000]}

        return {"status": "error",
                "message": (proc.stderr.strip() or
                            "Rscript exited without producing output"),
                "class": "RscriptCrash", "where": script_name,
                "returncode": proc.returncode,
                "stderr": proc.stderr.strip()[:2000]}

    except subprocess.TimeoutExpired:
        return {"status": "error",
                "message": f"Rscript timed out after {timeout_s}s",
                "class": "TimeoutExpired", "where": script_name}
    except Exception as e:  # noqa: BLE001 - never let the loop see a traceback
        return {"status": "error", "message": f"{type(e).__name__}: {e}",
                "class": type(e).__name__,
                "where": f"paraflr_agent.runner.run_r -> {script_name}"}
    finally:
        for p in (in_path, out_path):
            try:
                Path(p).unlink()
            except OSError:
                pass
