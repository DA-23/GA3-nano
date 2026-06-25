#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys

from openai import OpenAI


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def resolve_backend() -> tuple[str, str, str]:
    compat_base_url = (os.environ.get("OPENAI_COMPAT_BASE_URL") or "").strip()
    compat_api_key = (
        os.environ.get("OPENAI_COMPAT_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    ).strip()
    if compat_base_url and compat_api_key:
        return compat_api_key, compat_base_url, "openai_compat"

    openrouter_api_key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
    return openrouter_api_key, OPENROUTER_BASE_URL, "openrouter"


def resolve_model_name(model_name: str, backend_route: str) -> str:
    name = str(model_name or "").strip()
    if backend_route == "openai_compat" and name.startswith("openai/"):
        return name.split("/", 1)[1]
    return name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Execute one OpenRouter chat completion and print a JSON result.")
    parser.add_argument("--mode", choices=["target", "judge"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--timeout", type=float, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    api_key, base_url, backend_route = resolve_backend()
    if not api_key:
        print(
            json.dumps(
                {
                    "ok": False,
                    "exc_type": "RuntimeError",
                    "error": "missing_backend_api_key",
                }
            ),
            flush=True,
        )
        return 0

    try:
        payload = json.load(sys.stdin)
    except Exception as exc:
        print(json.dumps({"ok": False, "exc_type": type(exc).__name__, "error": f"invalid_json_input: {exc}"}), flush=True)
        return 0

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=float(args.timeout),
        max_retries=0,
    )
    resolved_model = resolve_model_name(args.model, backend_route)
    try:
        if args.mode == "target":
            query = str(payload.get("query") or "")
            response = client.chat.completions.create(
                model=resolved_model,
                messages=[{"role": "user", "content": query}],
                max_tokens=4096,
                temperature=0.0,
                top_p=1.0,
                timeout=float(args.timeout),
            )
        else:
            prompt = str(payload.get("prompt") or "")
            response = client.chat.completions.create(
                model=resolved_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.0,
                timeout=float(args.timeout),
            )
        content = response.choices[0].message.content
        print(json.dumps({"ok": True, "value": "" if content is None else str(content)}), flush=True)
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "exc_type": type(exc).__name__, "error": str(exc)}), flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
