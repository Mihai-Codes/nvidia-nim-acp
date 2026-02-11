#!/usr/bin/env python3
"""
NVIDIA NIM Model Launcher
Launches nvidia-nim-acp with a specific model preset.
Usage: nvidia-nim-launcher <model-name>
Models: kimi, kimi-thinking, deepseek, r1, coder, qwen
"""

import os
import sys

MODEL_MAP = {
    "kimi": "moonshotai/kimi-k2.5",
    "kimi-thinking": "moonshotai/kimi-k2-thinking",
    "deepseek": "deepseek-ai/deepseek-v3.2",
    "glm": "z-ai/glm4.7",
    "r1": "deepseek-ai/deepseek-r1-distill-qwen-32b",
    "coder": "deepseek-ai/deepseek-coder-6.7b-instruct",
    "qwen": "qwen/qwen2.5-coder-32b-instruct",
}


def main():
    if len(sys.argv) < 2:
        print("Usage: nvidia-nim-launcher <model-name>")
        print(f"Available models: {', '.join(MODEL_MAP.keys())}")
        sys.exit(1)

    model_name = sys.argv[1].lower()
    if model_name not in MODEL_MAP:
        print(f"Unknown model: {model_name}")
        print(f"Available models: {', '.join(MODEL_MAP.keys())}")
        sys.exit(1)

    os.environ["NVIDIA_MODEL"] = MODEL_MAP[model_name]
    print(f"Launching with {MODEL_MAP[model_name]}")

    from nvidia_nim_acp import main as acp_main

    acp_main()


if __name__ == "__main__":
    main()
