"""Provider configuration loading.

Reads a TOML file that maps provider names to their connection settings.
Uses Python 3.11's built-in tomllib — no extra dependency needed.

Config file format (providers.toml):

    [providers.deepseek]
    type        = "openai"
    model       = "deepseek-chat"
    base_url    = "https://api.deepseek.com/v1"
    api_key_env = "DEEPSEEK_API_KEY"

    [providers.anthropic]
    type        = "anthropic"
    model       = "claude-sonnet-4-5-20250929"
    api_key_env = "ANTHROPIC_API_KEY"
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# Default config file search order (first match wins)
_DEFAULT_SEARCH = [
    Path("providers.toml"),                                       # project-local
    Path.home() / ".config" / "harness-engineering" / "providers.toml",  # user-global
]


@dataclass
class ProviderConfig:
    name: str
    type: Literal["anthropic", "openai"]
    model: str
    base_url: str | None = None
    api_key_env: str | None = None  # name of the env var, not the key itself

    @property
    def resolved_api_key(self) -> str | None:
        """Read the actual key from the environment."""
        if not self.api_key_env:
            return None
        return os.environ.get(self.api_key_env)

    def check_api_key(self) -> None:
        """Print a warning (openai) or raise (anthropic) when key is missing."""
        key = self.resolved_api_key
        if key:
            return
        env = self.api_key_env or "(unset)"
        if self.type == "anthropic":
            raise EnvironmentError(
                f"Provider '{self.name}' requires {env} to be set."
            )
        # OpenAI-compatible servers (e.g. local Ollama) may not need a key
        print(
            f"WARNING: {env} is not set. "
            "Set it if your endpoint requires authentication.",
        )


def load_providers(config_path: str | Path | None = None) -> dict[str, ProviderConfig]:
    """Load all providers from a TOML config file.

    Args:
        config_path: Explicit path to the config file.  When omitted the
            default search list is tried in order.

    Returns:
        Mapping of provider name → ProviderConfig.

    Raises:
        FileNotFoundError: No config file could be located.
        KeyError / ValueError: Malformed config entries.
    """
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
        if "model" not in cfg:
            raise ValueError(f"Provider '{name}' is missing required field 'model'")
        if cfg["type"] not in ("anthropic", "openai"):
            raise ValueError(
                f"Provider '{name}' has unknown type {cfg['type']!r}. "
                "Use 'anthropic' or 'openai'."
            )
        result[name] = ProviderConfig(
            name=name,
            type=cfg["type"],
            model=cfg["model"],
            base_url=cfg.get("base_url"),
            api_key_env=cfg.get("api_key_env"),
        )
    return result


def get_provider(
    name: str,
    config_path: str | Path | None = None,
    model_override: str | None = None,
) -> ProviderConfig:
    """Load config and return the named provider.

    Args:
        name: Provider name as defined in the config file.
        config_path: Optional explicit path to the config file.
        model_override: When set, replaces the model from the config.

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
        "See providers.toml in this repo for an example."
    )
