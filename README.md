# NVIDIA NIM ACP Client

An ACP (Agent Client Protocol) client for NVIDIA NIM API, compatible with [Toad](https://github.com/batrachianai/toad) terminal.

## Features

- Wraps NVIDIA NIM API calls for use with ACP-compatible terminals
- Supports Kimi K2.5 and other models from NVIDIA NIM catalog
- Simple setup with environment variable authentication

## Quick Start

### 1. Install

```bash
git clone https://github.com/YOUR_USERNAME/nvidia-nim-acp.git
cd nvidia-nim-acp
uv pip install -e .
```

### 2. Configure NVIDIA API Key

```bash
export NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxx
```

Get your free API key from [NVIDIA Build](https://build.nvidia.com/settings/api-keys).

### 3. Use with Toad

1. Copy `nvidia-nim.toml` to your Toad agents directory:
   ```bash
   cp nvidia-nim.toml ~/.config/toad/agents/
   ```

2. Launch Toad:
   ```bash
   toad
   ```

3. Select "NVIDIA NIM (Kimi K2.5)" from the agent list

## Supported Models

- `moonshotai/kimi-k2.5` - Kimi K2.5 (default)
- `moonshotai/kimi-k2-instruct` - Kimi K2 Instruct
- `moonshotai/kimi-k2-thinking` - Kimi K2 Thinking
- And many more from NVIDIA NIM catalog

## Manual Installation

If you want to use the CLI directly:

```bash
# Set your API key
export NVIDIA_API_KEY=nvapi-xxxxx

# Run a simple test
echo '{"type": "prompt", "messages": [{"role": "user", "content": "Hello!"}]}' | python nvidia_nim_acp.py
```

## Creating a Pull Request to Toad

To add this as a built-in agent in Toad:

1. Fork [Toad](https://github.com/batrachianai/toad)
2. Copy `nvidia-nim.toml` to `src/toad/data/agents/`
3. Submit a PR

## License

MIT
