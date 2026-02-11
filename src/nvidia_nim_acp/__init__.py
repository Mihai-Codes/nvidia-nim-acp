#!/usr/bin/env python3
"""
NVIDIA NIM ACP Client
Implements the Agent Client Protocol (ACP) for Toad integration.
"""

import json
import os
import sys

BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "deepseek-ai/deepseek-v3.2"

SESSION_ID = "session-1"


def get_model() -> str:
    """Get model from environment variable or use default."""
    return os.environ.get("NVIDIA_MODEL", DEFAULT_MODEL)


def get_api_key() -> str:
    """Get API key from environment variable."""
    return os.environ.get("NVIDIA_API_KEY", "")


def send_notification(update_type: str, content: dict) -> None:
    """Send session/update notification."""
    notification = {
        "jsonrpc": "2.0",
        "method": "session/update",
        "params": {
            "sessionId": SESSION_ID,
            "update": {
                "sessionUpdate": update_type,
                **content,
            },
        },
    }
    sys.stdout.write(json.dumps(notification) + "\n")
    sys.stdout.flush()


def send_response(request_id: int | str | None, result: dict) -> None:
    """Send JSON response to stdout."""
    sys.stdout.write(json.dumps({"id": request_id, "result": result}) + "\n")
    sys.stdout.flush()


def send_error(request_id: int | str | None, message: str) -> None:
    """Send error response."""
    sys.stdout.write(
        json.dumps({"id": request_id, "error": {"code": -32000, "message": message}})
        + "\n"
    )
    sys.stdout.flush()


def handle_initialize(request_id) -> None:
    """Handle initialize request."""
    send_response(
        request_id,
        {
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
    )


def handle_session_new(request_id) -> None:
    """Handle session/new request."""
    send_response(request_id, {"sessionId": SESSION_ID})


def handle_session_prompt(request_id, params) -> None:
    """Handle session/prompt request."""
    content_blocks = params.get("prompt", [])
    messages = []
    for block in content_blocks:
        if block.get("type") == "text":
            messages.append({"role": "user", "content": block.get("text", "")})

    if not messages:
        send_response(request_id, {"stopReason": "end_turn"})
        return

    api_key = get_api_key()
    if not api_key:
        send_error(request_id, "NVIDIA_API_KEY environment variable not set")
        return

    model_id = get_model()

    try:
        import httpx

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
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
            data = response.json()

            choice = data["choices"][0]
            content = choice["message"]["content"]

            # Send content via notification
            send_notification(
                "agent_message_chunk", {"content": {"type": "text", "text": content}}
            )

            # Send completion response
            stop_reason = choice.get("finish_reason", "end_turn")
            if stop_reason == "length":
                stop_reason = "max_tokens"
            else:
                stop_reason = "end_turn"

            send_response(request_id, {"stopReason": stop_reason})

    except httpx.HTTPStatusError as e:
        send_error(
            request_id, f"HTTP error: {e.response.status_code} - {e.response.text}"
        )
    except Exception as e:
        send_error(request_id, str(e))


def handle_session_end(request_id) -> None:
    """Handle session/end request."""
    send_response(request_id, {})


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
        elif method == "session/update":
            # Notifications from client - acknowledge and ignore
            pass
        else:
            send_error(request_id, f"Method not found: {method}")


if __name__ == "__main__":
    main()


def cli_main():
    """Entry point for CLI (alias for main)."""
    main()
