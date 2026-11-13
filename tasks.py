"""Cross-platform task runner (Windows has no make):  python tasks.py <task> [extra args]

  ingest   Bosch CSVs in data/bosch/ -> events table (source=bosch)
  sample   generate the Bosch-FORMAT dev sample and ingest it as source=synthetic
  vision   camera clip -> events (source=vision) + blurred annotated video
  serve    start the MCP server (stdio)
  app      Streamlit demo
  test     unit tests (no LLM)
  eval     agent eval vs pandas ground truth (needs the LLM)
  report   print an end-of-shift report:  python tasks.py report W5 L0
"""
from __future__ import annotations

import subprocess
import sys

PY = sys.executable
TASKS = {
    "ingest": [PY, "-m", "ingest.bosch"],
    "vision": [PY, "-m", "vision.pipeline"],
    "serve": [PY, "-m", "mcp_server.server"],
    "app": [PY, "-m", "streamlit", "run", "app/streamlit_app.py"],
    "test": [PY, "-m", "pytest", "tests", "-v"],
    "eval": [PY, "-m", "pytest", "evals", "-v", "-s"],
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in {*TASKS, "sample", "report"}:
        print(__doc__)
        return 2
    task, extra = sys.argv[1], sys.argv[2:]
    if task == "sample":
        rc = subprocess.call([PY, "-m", "ingest.make_dev_sample", *extra])
        return rc or subprocess.call([PY, "-m", "ingest.bosch", "--data-dir", "data/bosch_sample", "--source", "synthetic"])
    if task == "report":
        shift, line = (extra + ["W5", "L0"])[:2] if extra else ("W5", "L0")
        return subprocess.call([PY, "-m", "agent.cli", "--report", shift, "--line", line])
    return subprocess.call(TASKS[task] + extra)


if __name__ == "__main__":
    sys.exit(main())
