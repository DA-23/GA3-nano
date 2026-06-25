"""Minimal credentials + model-name canonicalisation for the OpenRouter path.

This is a deliberately trimmed, secret-free replacement for the upstream
``utils/call_llms.py`` (which also wires Anthropic / Google / Together / local
providers). GA3 only ever calls OpenRouter, so only the two symbols imported by
the Bayes runners are provided here:

  * ``OPENROUTER_API_KEY``  - read from env var ``OPENROUTER_API_KEY`` or, if
    present, ``configs/config.yaml`` next to the repo root.
  * ``canonicalize_openrouter_model_name`` - identical logic to upstream so model
    names resolve the same way (reproducibility of the reported runs).

NO API keys are stored in this file. Set the environment variable before running.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
if _CONFIG_PATH.exists():
    with open(_CONFIG_PATH, "r", encoding="utf-8") as _file:
        _configs = yaml.safe_load(_file) or {}
else:
    _configs = {}


def _get_key(name: str) -> str:
    return (os.getenv(name) or _configs.get(name, "") or "").strip()


OPENROUTER_API_KEY = _get_key("OPENROUTER_API_KEY")

OPENROUTER_MODEL_ALIASES = {
    "gpt-4o": "openai/gpt-4o",
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "gpt-4.1-mini": "openai/gpt-4.1-mini",
    "gpt-5-mini": "openai/gpt-5-mini",
    "gpt-oss-20b": "openai/gpt-oss-20b",
    "claude-3.7": "anthropic/claude-3.7-sonnet",
    "claude-4-5-haiku": "anthropic/claude-haiku-4.5",
    "deepseek-r1": "deepseek/deepseek-r1",
    "deepseek r1": "deepseek/deepseek-r1",
    "deepseek-v3": "deepseek/deepseek-chat",
    "deepseek v3": "deepseek/deepseek-chat",
    "deepseek-v3.2": "deepseek/deepseek-v3.2",
    "deepseek v3.2": "deepseek/deepseek-v3.2",
    "qwen2.5-72b-instruct": "qwen/qwen-2.5-72b-instruct",
    "qwen 2.5-72b-instruct": "qwen/qwen-2.5-72b-instruct",
    "qwen3-32b": "qwen/qwen3-32b",
    "qwen 3-32b": "qwen/qwen3-32b",
    "gemma3-27b-it": "google/gemma-3-27b-it",
    "gemma 3-27b-it": "google/gemma-3-27b-it",
    "gemini-2.5": "google/gemini-2.5-pro",
    "gemini 2.5": "google/gemini-2.5-pro",
    "phi-4": "microsoft/phi-4",
    "llama 3.3-70b-instruct": "meta-llama/llama-3.3-70b-instruct",
    "llama-3.3-70b-instruct": "meta-llama/llama-3.3-70b-instruct",
}


def canonicalize_openrouter_model_name(model_name: str) -> str:
    name = (model_name or "").strip()
    lower = name.lower()

    if not name:
        return OPENROUTER_MODEL_ALIASES["gpt-4o-mini"]

    if lower.startswith("openrouter/"):
        return name.split("/", 1)[1]

    if "/" in name:
        return name

    if lower in OPENROUTER_MODEL_ALIASES:
        return OPENROUTER_MODEL_ALIASES[lower]

    if lower.startswith("llama") and "3.3" in lower and "70b" in lower and "instruct" in lower:
        return "meta-llama/llama-3.3-70b-instruct"

    if lower.startswith("gpt-"):
        return f"openai/{name}"

    if lower.startswith("claude"):
        return f"anthropic/{name}"

    if lower.startswith("gemini"):
        return f"google/{name}"

    if lower.startswith("deepseek"):
        return f"deepseek/{lower.replace(' ', '-')}"

    return name
