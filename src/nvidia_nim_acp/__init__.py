#!/usr/bin/env python3
"""
NVIDIA NIM ACP Client
Implements the Agent Client Protocol (ACP) for Toad integration.
"""

import json
import os
import sys
import threading

BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "deepseek-ai/deepseek-v3.2"


def get_model() -> str:
    """Get model from environment variable or use default."""
    return os.environ.get("NVIDIA_MODEL", DEFAULT_MODEL)


def get_api_key() -> str:
    """Get API key from environment variable."""
    return os.environ.get("NVIDIA_API_KEY", "")


def chat_complete(messages: list[dict[str, str]], model: str | None = None) -> dict:
    """Call NVIDIA NIM chat completion API."""
    import httpx

    api_key = get_api_key()
    if not api_key:
        raise ValueError("NVIDIA_API_KEY environment variable not set")

    model_id = model or get_model()

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model_id,
        "messages": messages,
        "max_tokens": 32768,
        "temperature": 1.0,
        "stream": False,
    }
    with httpx.Client(timeout=300.0) as client:
        response = client.post(
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


def handle_initialize(request_id) -> None:
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


def handle_session_new(request_id) -> None:
    """Handle session/new request."""
    send_response({"id": request_id, "result": {"sessionId": "session-1"}})


def handle_session_prompt(request_id, params) -> None:
    """Handle session/prompt request."""
    content_blocks = params.get("prompt", [])
    messages = []
    for block in content_blocks:
        if block.get("type") == "text":
            messages.append({"role": "user", "content": block.get("text", "")})

    if messages:
        try:
            result = chat_complete(messages)
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
        except Exception as e:
            send_error(request_id, str(e))
    else:
        send_response(
            {
                "id": request_id,
                "result": {"completion": {"content": [{"type": "text", "text": ""}]}},
            }
        )


def handle_session_end(request_id) -> None:
    """Handle session/end request."""
    send_response({"id": request_id, "result": {}})


def main():
    """Main ACP client loop."""
    while True:
        line = sys.stdin.readline()
        if not line:
            break

        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue

        request_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {})

        if method == "initialize":
            handle_initialize(request_id)
        elif method == "session/new":
            handle_session_new(request_id)
        elif method == "session/prompt":
            handle_session_prompt(request_id, params)
        elif method == "session/end":
            handle_session_end(request_id)
            break
        else:
            send_error(request_id, f"Method not found: {method}")


if __name__ == "__main__":
    main()


def cli_main():
    """Entry point for CLI (alias for main)."""
    main()
