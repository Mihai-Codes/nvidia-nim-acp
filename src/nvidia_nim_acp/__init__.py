#!/usr/bin/env python3
"""
NVIDIA NIM ACP Client
Wraps NVIDIA NIM API calls for use with ACP-compatible terminals like Toad.
"""

import asyncio
import json
import os
import sys
from typing import Optional
import httpx
from acp import ACPClient, run_agent

# Configuration
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"


class NVIDIANIMAgent:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or NVIDIA_API_KEY
        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY environment variable not set")

        self.client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {self.api_key}"}, timeout=120.0
        )

    async def complete(
        self, messages: list, model: str = "moonshotai/kimi-k2.5", **kwargs
    ):
        """Send chat completion request to NVIDIA NIM."""

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 32768),
            "temperature": kwargs.get("temperature", 1.0),
            "stream": False,
        }

        response = await self.client.post(
            f"{NVIDIA_BASE_URL}/chat/completions", json=payload
        )

        response.raise_for_status()
        data = response.json()

        return {
            "role": "assistant",
            "content": data["choices"][0]["message"]["content"],
        }

    async def close(self):
        await self.client.aclose()


async def handle_request():
    """Handle ACP protocol requests from stdin."""
    import sys

    # Read initialization from stdin
    init_data = json.loads(sys.stdin.readline())

    # Initialize agent
    agent = NVIDIANIMAgent()

    # Run ACP client
    client = ACPClient(agent=agent, agent_name="nvidia-nim", agent_version="0.1.0")

    await client.run()


def main():
    """Main entry point."""
    asyncio.run(handle_request())


if __name__ == "__main__":
    main()
