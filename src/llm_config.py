"""Resolved configuration for OpenAI-compatible chat endpoints.

This module answers one question: *where* should the agent call, with *which*
model and credentials?  It deliberately does not import the OpenAI SDK, open a
socket, or log secrets.  Keeping that decision out of ``OpenAIPolicy`` has three
benefits:

* the policy stays a thin protocol adapter;
* tests can construct settings directly instead of patching many env vars;
* provider quirks (DeepSeek's thinking switch, a local Ollama endpoint that
  needs a dummy key) live in one place.

Environment variables
---------------------

``FIN_AGENT_PROVIDER``      ``auto`` (default), ``openai``, ``deepseek``,
                            ``local`` or ``custom``
``FIN_AGENT_MODEL``         model name; provider defaults apply when omitted
``FIN_AGENT_BASE_URL``      explicit OpenAI-compatible base URL
``OPENAI_BASE_URL``         fallback for the base URL
``OPENAI_API_KEY``          OpenAI credential
``DEEPSEEK_API_KEY``        DeepSeek credential
``FIN_AGENT_API_KEY``       generic credential for a custom gateway
``FIN_AGENT_TIMEOUT``       request timeout in seconds (default 30)
``FIN_AGENT_MAX_RETRIES``   SDK retry count (default 2)
``FIN_AGENT_THINKING``      DeepSeek thinking mode (default false)
``FIN_AGENT_ALLOW_EXTERNAL``allow public endpoints (default false)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from urllib.parse import urlparse

from src.policy import is_internal_endpoint

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
# DeepSeek renamed its models; ``deepseek-chat`` is no longer the documented
# name.  See https://api-docs.deepseek.com/zh-cn/quick_start/pricing
DEFAULT_DEEPSEEK_MODEL = "deepseek-flash"

PROVIDERS = ("auto", "openai", "deepseek", "local", "custom")

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off", ""}


def _env(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _host(url: str | None) -> str:
    if not url:
        return ""
    return (urlparse(url).hostname or "").lower()


def _as_bool(value: bool | str | None, env_name: str, default: bool) -> bool:
    if value is None:
        value = _env(env_name)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raw = str(value).strip().lower()
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    raise ValueError(f"{env_name} must be a boolean, got {value!r}")


def _as_float(value: float | str | None, env_name: str, default: float) -> float:
    if value is None:
        value = _env(env_name)
    if value is None:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{env_name} must be a number, got {value!r}") from exc
    if number <= 0:
        raise ValueError(f"{env_name} must be positive, got {number}")
    return number


def _as_int(value: int | str | None, env_name: str, default: int) -> int:
    if value is None:
        value = _env(env_name)
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError(f"{env_name} must be an integer, got {value!r}")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{env_name} must be an integer, got {value!r}") from exc
    if number < 0:
        raise ValueError(f"{env_name} must be non-negative, got {number}")
    return number


def _clean_base_url(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    if value.endswith("/") and not value.endswith("://"):
        value = value.rstrip("/")
    return value


@dataclass(frozen=True)
class LLMSettings:
    """Fully resolved connection settings.

    ``api_key`` is excluded from ``repr`` so a settings object can be logged or
    shown in a traceback without leaking credentials.
    """

    provider: str
    model: str
    base_url: str | None
    api_key: str | None = field(repr=False)
    timeout: float = 30.0
    max_retries: int = 2
    thinking: bool = False
    allow_external: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.provider, str):
            raise ValueError("provider must be a string")
        provider = self.provider.strip().lower()
        object.__setattr__(self, "provider", provider)
        if provider not in PROVIDERS:
            raise ValueError(f"provider must be one of {PROVIDERS}, got {self.provider!r}")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a non-empty string")
        model = self.model.strip()
        object.__setattr__(self, "model", model)
        if self.base_url is not None and not isinstance(self.base_url, str):
            raise ValueError("base_url must be a string or None")
        base_url = _clean_base_url(self.base_url)
        if base_url is not None:
            object.__setattr__(self, "base_url", base_url)
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.api_key is not None and (
            not isinstance(self.api_key, str) or not self.api_key.strip()
        ):
            raise ValueError("api_key must be a non-empty string when provided")

    @property
    def is_deepseek(self) -> bool:
        return self.provider == "deepseek" or "deepseek" in _host(self.base_url)

    @property
    def is_local(self) -> bool:
        return is_internal_endpoint(self.base_url)

    @classmethod
    def from_env(
        cls,
        *,
        provider: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        thinking: bool | None = None,
        allow_external: bool | None = None,
    ) -> LLMSettings:
        """Resolve settings from explicit arguments, then environment variables.

        Explicit values win over environment variables.  Missing credentials are
        reported as ``RuntimeError`` with the env var names the user can set;
        malformed configuration is reported as ``ValueError``.
        """
        resolved_provider = (provider or _env("FIN_AGENT_PROVIDER") or "auto").strip().lower()
        if resolved_provider not in PROVIDERS:
            raise ValueError(f"provider must be one of {PROVIDERS}, got {resolved_provider!r}")

        resolved_base_url = _clean_base_url(
            base_url or _env("FIN_AGENT_BASE_URL") or _env("OPENAI_BASE_URL")
        )
        resolved_model = (model or _env("FIN_AGENT_MODEL") or "").strip() or None
        explicit_key = (api_key or _env("FIN_AGENT_API_KEY") or "").strip() or None
        openai_key = _env("OPENAI_API_KEY")
        deepseek_key = _env("DEEPSEEK_API_KEY")
        host = _host(resolved_base_url)

        if resolved_provider == "auto":
            if explicit_key:
                if "deepseek" in host:
                    resolved_provider = "deepseek"
                elif resolved_base_url:
                    resolved_provider = (
                        "local" if is_internal_endpoint(resolved_base_url) else "custom"
                    )
                else:
                    resolved_provider = "openai"
            elif "deepseek" in host or (deepseek_key and not openai_key):
                resolved_provider = "deepseek"
            elif openai_key:
                resolved_provider = "openai"
            elif deepseek_key:
                resolved_provider = "deepseek"
            elif resolved_base_url:
                resolved_provider = (
                    "local" if is_internal_endpoint(resolved_base_url) else "custom"
                )
            else:
                raise RuntimeError(
                    "未配置模型凭据：请设置 DEEPSEEK_API_KEY 或 OPENAI_API_KEY；"
                    "本地模型可设置 FIN_AGENT_PROVIDER=local 和 FIN_AGENT_BASE_URL。"
                )

        if resolved_provider == "deepseek":
            resolved_base_url = resolved_base_url or DEFAULT_DEEPSEEK_BASE_URL
            resolved_model = resolved_model or DEFAULT_DEEPSEEK_MODEL
            selected_key = explicit_key or deepseek_key
            if not selected_key:
                raise RuntimeError(
                    "provider=deepseek 需要 DEEPSEEK_API_KEY（或显式 api_key）。"
                )
        elif resolved_provider == "openai":
            resolved_base_url = resolved_base_url or DEFAULT_OPENAI_BASE_URL
            resolved_model = resolved_model or DEFAULT_OPENAI_MODEL
            selected_key = explicit_key or openai_key
            if not selected_key:
                raise RuntimeError(
                    "provider=openai 需要 OPENAI_API_KEY（或显式 api_key）。"
                )
        elif resolved_provider == "local":
            if not resolved_base_url:
                raise ValueError("provider=local 需要 FIN_AGENT_BASE_URL（例如 Ollama）。")
            if not is_internal_endpoint(resolved_base_url):
                raise ValueError("provider=local 只接受本机/内网地址，拒绝公网地址。")
            if not resolved_model:
                raise ValueError("provider=local 需要 FIN_AGENT_MODEL（例如 qwen3:8b）。")
            selected_key = explicit_key or "local"
        else:  # custom
            if not resolved_base_url:
                raise ValueError("provider=custom 需要 FIN_AGENT_BASE_URL。")
            # Keep the old constructor usable: custom endpoints get the same
            # default model as OpenAI unless the caller overrides it.
            resolved_model = resolved_model or DEFAULT_OPENAI_MODEL
            selected_key = explicit_key or openai_key or deepseek_key
            if not selected_key:
                if is_internal_endpoint(resolved_base_url):
                    selected_key = "local"
                else:
                    raise RuntimeError(
                        "provider=custom 的公网端点需要 API key；"
                        "请设置 OPENAI_API_KEY/DEEPSEEK_API_KEY 或显式 api_key。"
                    )

        return cls(
            provider=resolved_provider,
            model=resolved_model,
            base_url=resolved_base_url,
            api_key=selected_key,
            timeout=_as_float(timeout, "FIN_AGENT_TIMEOUT", 30.0),
            max_retries=_as_int(max_retries, "FIN_AGENT_MAX_RETRIES", 2),
            thinking=_as_bool(thinking, "FIN_AGENT_THINKING", False),
            allow_external=_as_bool(allow_external, "FIN_AGENT_ALLOW_EXTERNAL", False),
        )

    def as_public_dict(self) -> dict:
        """A log-safe summary: never includes the credential."""
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "api_key": "set" if self.api_key else "unset",
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "thinking": self.thinking,
            "allow_external": self.allow_external,
        }


__all__ = [
    "DEFAULT_DEEPSEEK_BASE_URL",
    "DEFAULT_DEEPSEEK_MODEL",
    "DEFAULT_OPENAI_BASE_URL",
    "DEFAULT_OPENAI_MODEL",
    "PROVIDERS",
    "LLMSettings",
]
