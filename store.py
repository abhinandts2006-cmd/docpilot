import tomllib
import tomli_w
from pathlib import Path
import ollama
import os

path =  Path.home() / ".docpilot"
CONFIG_PATH = path / "config.toml"


def _get_available_models():
    try:
        models = ollama.list().models
        return [m.model for m in models]
    except Exception:
        return []


_AVAILABLE_MODELS = _get_available_models()

# Default configuration
DEFAULT_CONFIG = {
    "default_embed_model": _AVAILABLE_MODELS[0] if _AVAILABLE_MODELS else "mxbai-embed-large:335m",
    "default_model": _AVAILABLE_MODELS[1] if len(_AVAILABLE_MODELS) > 1 else "llama2",
    "db_path": str(path / "chroma_langchain_db"),
    "log_level": "info",
    "retrieval_k": 6,
    "max_context_chars": 3500,
    "max_doc_chars": 700,
    "num_predict": 192,
    "num_ctx": 2048,
    "num_thread": max(1, (os.cpu_count() or 4) - 1),
    "temperature": 0.1,
}

def init_config():
    """Initialize config file with defaults if it doesn't exist."""
    if not CONFIG_PATH.exists():
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_PATH, "wb") as f:
            tomli_w.dump(DEFAULT_CONFIG, f)
        print(f"Created config at {CONFIG_PATH}")
    return load_config()

def load_config():
    if not CONFIG_PATH.exists():
        return DEFAULT_CONFIG.copy()
    try:
        with open(CONFIG_PATH, "rb") as f:
            loaded = tomllib.load(f)
    except tomllib.TOMLDecodeError:
        print(f"Invalid config at {CONFIG_PATH}. Recreating with defaults.")
        save_config(DEFAULT_CONFIG.copy())
        return DEFAULT_CONFIG.copy()

    merged = DEFAULT_CONFIG.copy()
    merged.update(loaded)
    return merged

def save_config(config: dict):
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "wb") as f:
        tomli_w.dump(config, f)