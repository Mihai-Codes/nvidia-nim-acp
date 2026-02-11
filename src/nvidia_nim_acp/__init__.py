#!/usr/bin/env python3
"""
NVIDIA NIM ACP Client
Wraps NVIDIA NIM API calls for use with ACP-compatible terminals like Toad.
"""

import asyncio
import json
import os
import sys
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
    async with httpx.AsyncClient(timeout=300.0) as client:
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
    print(json.dumps({"type": "ready"}), flush=True)
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            request = json.loads(line)
            request_type = request.get("type")
            if request_type == "prompt":
                if not API_KEY:
                    print(
                        json.dumps(
                            {
                                "type": "error",
                                "error": "NVIDIA_API_KEY environment variable not set. Get a free API key from https://build.nvidia.com/settings/api-keys",
                            }
                        ),
                        flush=True,
                    )
                    continue
                messages = request.get("messages", [])
                model = request.get("model", "moonshotai/kimi-k2.5")
                try:
                    result = await asyncio.wait_for(
                        chat_complete(messages, model), timeout=300.0
                    )
                    response = format_response(result)
                    print(
                        json.dumps({"type": "message", "message": response}), flush=True
                    )
                except asyncio.TimeoutError:
                    print(
                        json.dumps(
                            {
                                "type": "error",
                                "error": "NVIDIA API timeout. Model may be overloaded. Try again later or use a different model.",
                            }
                        ),
                        flush=True,
                    )
                except Exception as e:
                    print(json.dumps({"type": "error", "error": str(e)}), flush=True)
            elif request_type == "close":
                break
        except Exception as e:
            print(json.dumps({"type": "error", "error": str(e)}), flush=True)


def cli_main():
    """Entry point for CLI."""
    asyncio.run(main())


if __name__ == "__main__":
    cli_main()
