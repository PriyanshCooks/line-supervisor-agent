"""One persistent asyncio loop per Streamlit session, running one agent runtime (one MCP subprocess) for the whole session.

Before this, every chat turn did `asyncio.run(build_agent())`: a new event loop, a new MCP server process to list the tools, another
to fetch the catalog, and one more per tool call (the MCP client is stateless), about 10 s of pure setup before the LLM started.

Streamlit scripts are synchronous and re-run on every interaction, so the loop lives in a daemon thread and the script submits
coroutines with `asyncio.run_coroutine_threadsafe`. Keep the instance in `st.session_state`: one subprocess per browser session.
"""
from __future__ import annotations

import asyncio
import atexit
import threading
import weakref

from agent.graph import AgentRuntime, ask
from agent.report import generate_report

_LIVE: "weakref.WeakSet[SessionRuntime]" = weakref.WeakSet()
atexit.register(lambda: [rt.close() for rt in list(_LIVE)])


class SessionRuntime:
    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="agent-loop", daemon=True)
        self._thread.start()
        self._rt: AgentRuntime | None = None
        self._graph_for: tuple | None = None             # (saver, compiled graph): rebuilt only when the saver changes
        self._turn_lock: asyncio.Lock | None = None
        self.closed = False
        _LIVE.add(self)

    # ---- plumbing
    def _submit(self, coro, timeout: float | None = None):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    def start(self) -> "SessionRuntime":
        """Spawn the MCP subprocess and load tools + catalog (idempotent)."""
        if self._rt is None:
            async def _start():
                self._turn_lock = asyncio.Lock()
                return await AgentRuntime.start()
            self._rt = self._submit(_start())
        return self

    @property
    def alive(self) -> bool:
        return not self.closed and self._rt is not None and self._rt._task is not None and not self._rt._task.done()

    def _graph(self, saver):
        if self._graph_for is None or self._graph_for[0] is not saver:
            self._graph_for = (saver, self._rt.graph(saver))      # new MemorySaver (new context) -> new empty conversation
        return self._graph_for[1]

    # ---- work
    def ask(self, question: str | None, saver, thread: str, resume: bool | None = None) -> dict:
        """One chat turn, or the approve/reject resume (question=None, resume=True/False) of a paused turn."""
        async def _go():
            async with self._turn_lock:                   # a turn abandoned by a Streamlit rerun must finish before the next starts
                return await ask(self._graph(saver), question, thread, approve=None, resume=resume)
        return self._submit(_go())

    def report(self, shift: str, line: str | None, source: str | None, use_llm: bool = True) -> dict:
        async def _go():
            return await generate_report(shift, line, source, use_llm=use_llm, tools=self._rt.tools)
        return self._submit(_go())

    # ---- teardown
    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            if self._rt is not None:
                self._submit(self._rt.aclose(), timeout=15)
        except Exception:
            pass
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)

    def __del__(self):          # session_state dropped when the browser session ends
        try:
            self.close()
        except Exception:
            pass
