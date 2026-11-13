"""python -m agent.cli "Which station is the bottleneck on line 0 in W5?"   |   python -m agent.cli --report W5 --line L0"""
from __future__ import annotations

import argparse
import asyncio

from agent.graph import ask, build_agent
from agent.report import generate_report


def _confirm(req: dict) -> bool:
    print(f"\n[HUMAN APPROVAL NEEDED] {req['action']} -> {req['station_id']}: {req['reason']}")
    return input("Approve? [y/N] ").strip().lower() == "y"


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("--report", metavar="SHIFT")
    ap.add_argument("--line")
    ap.add_argument("--source")
    a = ap.parse_args()
    if a.report:
        r = await generate_report(a.report, a.line, a.source)
        print(r["whatsapp"], "\n\n", r["markdown"])
        return
    out = await ask(await build_agent(), a.question or "Which station is the bottleneck on line 0?", approve=_confirm)
    print("tools:", [c["name"] for c in out["tool_calls"]])
    print(out["answer"])


if __name__ == "__main__":
    asyncio.run(main())
