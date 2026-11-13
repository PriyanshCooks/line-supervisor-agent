"""LLM factory. Ollama (local) by default; API models via env vars.

  LLM_PROVIDER = ollama | anthropic | openai     (default: ollama)
  LLM_MODEL    = model name                      (default for ollama: qwen3.5:4b)
  OLLAMA_NUM_GPU = layers to put on the GPU (default 99 = all). On the 4 GB RTX 3050 Ollama's automatic split puts only ~50% of this
                   model on the GPU (~10 tok/s); forcing all layers measured ~19 tok/s. Ignored when there is no GPU.
  OLLAMA_NUM_CTX = context window in tokens (default 8192; Ollama's VRAM-based default of 4096 can truncate tool output)
"""
from __future__ import annotations

import os

from langchain_core.language_models.chat_models import BaseChatModel


def get_llm() -> BaseChatModel:
    provider = os.environ.get("LLM_PROVIDER", "ollama").lower()
    model = os.environ.get("LLM_MODEL")
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model or "qwen3.5:4b", temperature=0,
                          base_url=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
                          reasoning=False,
                          num_gpu=int(os.environ.get("OLLAMA_NUM_GPU", 99)),
                          num_ctx=int(os.environ.get("OLLAMA_NUM_CTX", 8192)))
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic  # pip install langchain-anthropic
        return ChatAnthropic(model=model or "claude-sonnet-5-5", temperature=0)
    if provider == "openai":
        from langchain_openai import ChatOpenAI  # pip install langchain-openai
        return ChatOpenAI(model=model or "gpt-4o-mini", temperature=0)
    raise ValueError(f"Unknown LLM_PROVIDER={provider!r}")
