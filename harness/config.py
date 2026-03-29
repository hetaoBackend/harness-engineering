"""Provider configuration loading.

Reads a TOML file that maps provider names to their connection settings.
Uses Python 3.11's built-in tomllib — no extra dependency needed.

Config file format (providers.toml):

    [providers.deepseek]
    type          = "openai"
    base_url      = "https://api.deepseek.com/v1"
    api_key_env   = "DEEPSEEK_API_KEY"
    default_model = "deepseek-chat"
    models        = ["deepseek-chat", "deepseek-reasoner"]

    [providers.anthropic]
    type          = "anthropic"
    api_key_env   = "ANTHROPIC_API_KEY"
    default_model = "claude-sonnet-4-5-20250929"
    models        = ["claude-sonnet-4-5-20250929", "claude-opus-4-5-20251101"]

Select a non-default model at runtime with --model:
    uv run main.py "task" --provider deepseek --model deepseek-reasoner
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# Default config file search order (first match wins)
_DEFAULT_SEARCH = [
    Path("providers.toml"),
    Path.home() / ".config" / "harness-engineering" / "providers.toml",
]


@dataclass
class ProviderConfig:
    name: str
    type: Literal["anthropic", "openai"]
    default_model: str
    models: list[str]          # available models for this provider
    model: str                 # active model (default_model or --model override)
    base_url: str | None = None
    api_key_env: str | None = None

    @property
    def resolved_api_key(self) -> str | None:
        if not self.api_key_env:
            return None
        return os.environ.get(self.api_key_env)

    def check_api_key(self) -> None:
        key = self.resolved_api_key
        if key:
            return
        env = self.api_key_env or "(unset)"
        if self.type == "anthropic":
            raise EnvironmentError(
                f"Provider '{self.name}' requires {env} to be set."
            )
        print(
            f"WARNING: {env} is not set. "
            "Set it if your endpoint requires authentication.",
        )


def load_providers(config_path: str | Path | None = None) -> dict[str, ProviderConfig]:
    """Load all providers from a TOML config file."""
    path = _resolve_path(config_path)
    with open(path, "rb") as fh:
        data = tomllib.load(fh)

    raw_providers: dict = data.get("providers", {})
    if not raw_providers:
        raise ValueError(f"No [providers.*] sections found in {path}")

    result: dict[str, ProviderConfig] = {}
    for name, cfg in raw_providers.items():
        if "type" not in cfg:
            raise ValueError(f"Provider '{name}' is missing required field 'type'")
        if cfg["type"] not in ("anthropic", "openai"):
            raise ValueError(
                f"Provider '{name}' has unknown type {cfg['type']!r}. "
                "Use 'anthropic' or 'openai'."
            )

        # Support both old `model` field and new `default_model` field
        default_model = cfg.get("default_model") or cfg.get("model")
        if not default_model:
            raise ValueError(f"Provider '{name}' is missing 'default_model'")

        models: list[str] = cfg.get("models") or [default_model]
        if default_model not in models:
            models = [default_model] + models

        result[name] = ProviderConfig(
            name=name,
            type=cfg["type"],
            default_model=default_model,
            models=models,
            model=default_model,
            base_url=cfg.get("base_url"),
            api_key_env=cfg.get("api_key_env"),
        )
    return result


def get_provider(
    name: str,
    config_path: str | Path | None = None,
    model_override: str | None = None,
) -> ProviderConfig:
    """Load config and return the named provider, with optional model override.

    Args:
        name: Provider name as defined in the config file.
        config_path: Optional explicit path to the config file.
        model_override: Model name to use instead of default_model.
            Can be any string — not restricted to the `models` list,
            so you can try any model the endpoint supports.

    Raises:
        FileNotFoundError: Config file not found.
        KeyError: Provider name not found in config.
    """
    providers = load_providers(config_path)
    if name not in providers:
        available = ", ".join(sorted(providers))
        raise KeyError(
            f"Provider '{name}' not found in config. "
            f"Available: {available}"
        )
    cfg = providers[name]
    if model_override:
        cfg = ProviderConfig(
            name=cfg.name,
            type=cfg.type,
            default_model=cfg.default_model,
            models=cfg.models,
            model=model_override,
            base_url=cfg.base_url,
            api_key_env=cfg.api_key_env,
        )
    return cfg


def _resolve_path(config_path: str | Path | None) -> Path:
    if config_path is not None:
        p = Path(config_path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        return p
    for candidate in _DEFAULT_SEARCH:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No providers.toml found. Create one in the project root or at "
        "~/.config/harness-engineering/providers.toml. "
        "See providers.example.toml for reference."
    )
