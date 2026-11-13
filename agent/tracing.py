"""Optional Langfuse tracing. Off unless LANGFUSE_ENABLED=1 (and `pip install langfuse`, keys in env)."""
from __future__ import annotations

import os


def callbacks() -> list:
    if os.environ.get("LANGFUSE_ENABLED") != "1":
        return []
    try:
        from langfuse.langchain import CallbackHandler  # reads LANGFUSE_* env vars
        return [CallbackHandler()]
    except Exception as e:
        print("Langfuse tracing requested but unavailable:", e)
        return []
