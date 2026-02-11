#!/usr/bin/env python3
"""
NVIDIA NIM ACP Client
Wraps NVIDIA NIM API calls for use with ACP-compatible terminals like Toad.

Usage:
    export NVIDIA_API_KEY=nvapi-xxxxx
    nvidia-nim-acp
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any
import httpx

API_KEY = os.environ.get("NVIDIA_API_KEY", "")
BASE_URL = "https://integrate.api.nvidia.com/v1"


async def chat_complete(
    messages: list[dict[str, str]], model: str = "moonshotai/kimi-k2.5"
) -> dict[str, Any]:
    """Call NVIDIA NIM chat completion API."""

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 32768,
        "temperature": 1.0,
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{BASE_URL}/chat/completions", headers=headers, json=payload
        )
        response.raise_for_status()
        return response.json()


def format_response(data: dict) -> dict:
    """Format API response for ACP."""
    choice = data["choices"][0]
    return {
        "role": "assistant",
        "content": choice["message"]["content"],
        "reasoning_content": choice.get("message", {}).get("reasoning", ""),
    }


async def main():
    """Main ACP client loop."""
    import sys

    print(json.dumps({"type": "ready"}), flush=True)

    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break

            request = json.loads(line)
            request_type = request.get("type")

            if request_type == "prompt":
                messages = request.get("messages", [])
                model = request.get("model", "moonshotai/kimi-k2.5")

                result = await chat_complete(messages, model)
                response = format_response(result)

                print(json.dumps({"type": "message", "message": response}), flush=True)

            elif request_type == "close":
                break

        except Exception as e:
            print(json.dumps({"type": "error", "error": str(e)}), flush=True)


if __name__ == "__main__":
    if not API_KEY:
        print("Error: NVIDIA_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)
    asyncio.run(main())
