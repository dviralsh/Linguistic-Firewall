import os

OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "ollama")

# DPA backdoor server (dpa_server.py) — serves LLaMA2+LoRA on a separate port
DPA_SERVER_PORT = int(os.getenv("DPA_PORT", "11435"))
DPA_BASE_URL = f"http://127.0.0.1:{DPA_SERVER_PORT}/v1"

LOCAL_MODELS = [
    "llama3.2",
    "llama3",
    "llama2",
    "qwen2.5",
    "qwen2",
    "codellama",
    "mistral",
    "phi",
    "t1c/deepseek-math-7b-rl",
]


def get_api_config_for_model(model_name):
    """Returns (base_url, api_key) for the given model."""
    lower = model_name.lower()

    # DPA backdoor server (dpa_server.py running locally)
    if lower.startswith("dpa-"):
        return DPA_BASE_URL, "dpa"

    if lower.startswith("claude") or "claude" in lower:
        if not ANTHROPIC_API_KEY:
            raise RuntimeError(
                f"ANTHROPIC_API_KEY required for Claude model '{model_name}'"
            )
        return ANTHROPIC_BASE_URL, ANTHROPIC_API_KEY

    is_local = any(lower.startswith(m) for m in LOCAL_MODELS)
    if is_local and OLLAMA_API_KEY:
        return OLLAMA_BASE_URL, OLLAMA_API_KEY

    if OPENROUTER_API_KEY:
        return OPENROUTER_BASE_URL, OPENROUTER_API_KEY

    if is_local:
        return OLLAMA_BASE_URL, OLLAMA_API_KEY

    raise RuntimeError(
        f"No API configured for model '{model_name}'. Set ANTHROPIC_API_KEY, OPENROUTER_API_KEY, or use a local model."
    )


DEFAULT_API_BASE_URL = OPENROUTER_BASE_URL
DEFAULT_API_KEY = OPENROUTER_API_KEY

# Backward-compatible aliases used across the repository.
DEFAULT_OLLAMA_BASE_URL = OLLAMA_BASE_URL
DEFAULT_OLLAMA_API_KEY = OLLAMA_API_KEY

