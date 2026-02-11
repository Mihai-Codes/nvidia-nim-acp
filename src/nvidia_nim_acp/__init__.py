#!/usr/bin/env python3
"""
NVIDIA NIM ACP Client
Implements the Agent Client Protocol (ACP) for Toad integration.
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


def format_response(data: dict) -> dict[str, Any]:
    """Format API response for ACP."""
    choice = data["choices"][0]
    return {
        "role": "assistant",
        "content": choice["message"]["content"],
        "reasoning_content": choice.get("message", {}).get("reasoning", ""),
    }


async def handle_request(
    request: dict, stdin: asyncio.StreamReader, stdout: asyncio.StreamWriter
) -> None:
    """Handle a JSON-RPC request and send response."""
    request_id = request.get("id")
    method = request.get("method")
    params = request.get("params", {})

    if method == "initialize":
        response = {
            "id": request_id,
            "result": {
                "protocolVersion": 1,
                "capabilities": {
                    "prompts": {"listChanged": False},
                    "resources": {"listChanged": False, "subscribe": False},
                    "tools": {"listChanged": False},
                    "env": True,
                    "status": {"reporting": "full"},
                    "sessions": True,
                    "notifications": {
                        "taskStarted": True,
                        "taskCompleted": True,
                        "taskError": True,
                        "console": True,
                    },
                },
                "serverInfo": {"name": "nvidia-nim-acp", "version": "0.1.0"},
            },
        }
        response_json = json.dumps(response) + "\n"
        stdout.write(response_json.encode("utf-8"))
        await stdout.drain()
        return True

    elif method == "session/new":
        response = {"id": request_id, "result": {"sessionId": "session-1"}}
        response_json = json.dumps(response) + "\n"
        stdout.write(response_json.encode("utf-8"))
        await stdout.drain()
        return True

    elif method == "session/prompt":
        content_blocks = params.get("prompt", [])
        messages = []
        for block in content_blocks:
            if block.get("type") == "text":
                text = block.get("text", "")
                messages.append({"role": "user", "content": text})
            elif block.get("type") == "reasoning":
                text = block.get("reasoning", "")
                messages.append({"role": "user", "content": f"[Reasoning]: {text}"})

        if messages:
            try:
                result = await asyncio.wait_for(chat_complete(messages), timeout=300.0)
                response_data = format_response(result)
                response = {
                    "id": request_id,
                    "result": {
                        "completion": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": response_data.get("content", ""),
                                }
                            ]
                        }
                    },
                }
            except asyncio.TimeoutError:
                response = {
                    "id": request_id,
                    "error": {
                        "code": -32000,
                        "message": "NVIDIA API timeout. Model may be overloaded.",
                    },
                }
            except Exception as e:
                response = {
                    "id": request_id,
                    "error": {"code": -32000, "message": str(e)},
                }
        else:
            response = {
                "id": request_id,
                "result": {"completion": {"content": [{"type": "text", "text": ""}]}},
            }

        response_json = json.dumps(response) + "\n"
        stdout.write(response_json.encode("utf-8"))
        await stdout.drain()
        return True

    elif method == "session/end":
        response = {"id": request_id, "result": {}}
        response_json = json.dumps(response) + "\n"
        stdout.write(response_json.encode("utf-8"))
        await stdout.drain()
        return False

    else:
        response = {
            "id": request_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }
        response_json = json.dumps(response) + "\n"
        stdout.write(response_json.encode("utf-8"))
        await stdout.drain()
        return True


async def main():
    """Main ACP client loop."""
    stdin = asyncio.StreamReader()
    stdout = asyncio.StreamWriter(sys.stdout.buffer, {}, None, asyncio.get_event_loop())

    while True:
        try:
            line = await stdin.readline()
            if not line:
                break

            try:
                request = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                continue

            should_continue = await handle_request(request, stdin, stdout)
            if not should_continue:
                break

        except Exception as e:
            sys.stderr.write(f"Error: {e}\n")
            break


def cli_main():
    """Entry point for CLI."""
    asyncio.run(main())


if __name__ == "__main__":
    cli_main()
