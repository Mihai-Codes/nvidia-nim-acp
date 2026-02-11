#!/usr/bin/env python3
"""
NVIDIA NIM ACP Client
Implements the Agent Client Protocol (ACP) for Toad integration.
"""

import asyncio
import json
import os
import sys

API_KEY = os.environ.get("NVIDIA_API_KEY", "")
BASE_URL = "https://integrate.api.nvidia.com/v1"


async def chat_complete(
    messages: list[dict[str, str]], model: str = "moonshotai/kimi-k2.5"
) -> dict:
    """Call NVIDIA NIM chat completion API."""
    import httpx

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


def send_response(response: dict) -> None:
    """Send JSON response to stdout."""
    sys.stdout.write(json.dumps(response) + "\n")
    sys.stdout.flush()


def send_error(request_id, message: str) -> None:
    """Send error response."""
    send_response({"id": request_id, "error": {"code": -32000, "message": message}})


async def handle_initialize(request_id: int) -> None:
    """Handle initialize request."""
    send_response(
        {
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
    )


async def handle_session_new(request_id: int) -> None:
    """Handle session/new request."""
    send_response({"id": request_id, "result": {"sessionId": "session-1"}})


async def handle_session_prompt(request_id: int, params: dict) -> None:
    """Handle session/prompt request."""
    content_blocks = params.get("prompt", [])
    messages = []
    for block in content_blocks:
        if block.get("type") == "text":
            messages.append({"role": "user", "content": block.get("text", "")})

    if messages:
        try:
            result = await asyncio.wait_for(chat_complete(messages), timeout=300.0)
            response_data = format_response(result)
            send_response(
                {
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
            )
        except asyncio.TimeoutError:
            send_error(request_id, "NVIDIA API timeout")
        except Exception as e:
            send_error(request_id, str(e))
    else:
        send_response(
            {
                "id": request_id,
                "result": {"completion": {"content": [{"type": "text", "text": ""}]}},
            }
        )


async def handle_session_end(request_id: int) -> None:
    """Handle session/end request."""
    send_response({"id": request_id, "result": {}})


async def main():
    """Main ACP client loop."""
    reader = asyncio.StreamReader()

    await reader.readline()

    while True:
        try:
            line = await reader.readline()
            if not line:
                break

            try:
                request = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                continue

            request_id = request.get("id")
            method = request.get("method")
            params = request.get("params", {})

            if method == "initialize":
                await handle_initialize(request_id)
            elif method == "session/new":
                await handle_session_new(request_id)
            elif method == "session/prompt":
                await handle_session_prompt(request_id, params)
            elif method == "session/end":
                await handle_session_end(request_id)
                break
            else:
                send_error(request_id, f"Method not found: {method}")

        except Exception as e:
            sys.stderr.write(f"Error: {e}\n")
            break


def cli_main():
    """Entry point for CLI."""
    asyncio.run(main())


if __name__ == "__main__":
    cli_main()
