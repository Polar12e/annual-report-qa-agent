"""Data-boundary policy hooks.

Encryption, authentication and audit storage are intentionally **not**
implemented yet.  This module reserves the seams so the boundaries are explicit
and can be enforced later without touching call sites:

- ``DataPolicy.ensure_endpoint`` refuses public model endpoints unless the
  caller opts in with ``allow_external=True``.
- ``DataPolicy.snippet`` is the single place where document text is truncated
  before leaving the process.
- ``DataPolicy.audit`` is a hook for a future audit sink; it receives metadata
  only, never document text.
"""
from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from enum import Enum
from typing import Callable
from urllib.parse import urlparse

INTERNAL_SUFFIXES = (".internal", ".local", ".corp", ".lan")


class DataClassification(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"


class PolicyViolation(RuntimeError):
    """Raised when a call would cross a configured data boundary."""


def is_internal_endpoint(base_url: str | None) -> bool:
    if not base_url:
        return False
    host = urlparse(base_url).hostname
    if not host:
        return False
    if host in {"localhost", "127.0.0.1", "::1"}:
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return host.endswith(INTERNAL_SUFFIXES)
    return address.is_private or address.is_loopback


@dataclass(frozen=True)
class DataPolicy:
    classification: DataClassification = DataClassification.PUBLIC
    allow_external: bool = False
    max_snippets: int = 3
    max_snippet_chars: int = 900
    log_raw_text: bool = False
    audit_hook: Callable[[dict], None] | None = None

    def ensure_endpoint(self, base_url: str | None) -> None:
        if self.allow_external or is_internal_endpoint(base_url):
            return
        raise PolicyViolation(
            "refusing to call a public model endpoint; "
            "use an internal/local endpoint or pass allow_external=True"
        )

    def snippet(self, text: str) -> str:
        return text[: self.max_snippet_chars]

    def audit(self, event: dict) -> None:
        if self.audit_hook is not None:
            self.audit_hook(event)


__all__ = [
    "DataClassification",
    "DataPolicy",
    "PolicyViolation",
    "is_internal_endpoint",
]
