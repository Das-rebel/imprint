"""Model configuration loader for Imprint.

Loads configuration from models.json with support for both
legacy list format (Phase 0) and v4 dict format.
"""

from pathlib import Path
from typing import Any

MODELS_JSON = Path(__file__).parent / "models.json"


def get_models_config() -> dict[str, Any]:
    """Load models configuration from models.json.

    Returns:
        Dict with keys: pool, embedding_models, compression_models,
        fallback_chain, cache
    """
    if MODELS_JSON.exists():
        import json

        data = json.loads(MODELS_JSON.read_text())
        # Support both list format (legacy) and dict format (v4)
        if isinstance(data, list):
            return {"pool": data}
        return data
    return {}


def get_embedding_models() -> dict[str, dict[str, Any]]:
    """Get embedding model configurations."""
    cfg = get_models_config()
    return cfg.get("embedding_models", {})


def get_compression_models() -> dict[str, dict[str, Any]]:
    """Get compression model configurations."""
    cfg = get_models_config()
    return cfg.get("compression_models", {})


def get_fallback_chain() -> dict[str, dict[str, Any]]:
    """Get fallback chain configuration."""
    cfg = get_models_config()
    return cfg.get("fallback_chain", {})


def get_cache_config() -> dict[str, Any]:
    """Get cache configuration."""
    cfg = get_models_config()
    return cfg.get("cache", {})
