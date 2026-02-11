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


async def handle_initialize(request_id) -> None:
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


async def handle_session_new(request_id) -> None:
    """Handle session/new request."""
    send_response({"id": request_id, "result": {"sessionId": "session-1"}})


async def handle_session_prompt(request_id, params) -> None:
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


async def handle_session_end(request_id) -> None:
    """Handle session/end request."""
    send_response({"id": request_id, "result": {}})


async def handle_request(request) -> bool:
    """Handle a single request. Returns False to stop."""
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
        return False
    else:
        send_error(request_id, f"Method not found: {method}")

    return True


async def read_stdin():
    """Read a line from stdin using asyncio."""
    loop = asyncio.get_event_loop()
    line = await loop.run_in_executor(None, sys.stdin.readline)
    return line


async def main():
    """Main ACP client loop."""
    await read_stdin()  # Skip the first readline (Toad sends a blank line first)

    while True:
        line = await read_stdin()
        if not line:
            break

        try:
            request = json.loads(line.decode("utf-8"))
        except json.JSONDecodeError:
            continue

        should_continue = await handle_request(request)
        if not should_continue:
            break


def cli_main():
    """Entry point for CLI."""
    asyncio.run(main())


if __name__ == "__main__":
    cli_main()
