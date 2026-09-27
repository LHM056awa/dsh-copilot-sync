"""Shared dataclasses, helpers, and error types for dsh-copilot-sync."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

CUSTOM_ENDPOINT_VENDOR = "customendpoint"

#: Keys written into newly added model objects, in output order.
NEW_MODEL_FIELDS = (
    "name",
    "url",
    "toolCalling",
    "vision",
    "maxInputTokens",
    "maxOutputTokens",
    "supportsReasoningEffort",
)

DEFAULT_NEW_MODEL_FIELDS: Dict[str, Any] = {
    "toolCalling": True,
    "vision": True,
    "maxInputTokens": 1000000,
    "maxOutputTokens": 384000,
    "supportsReasoningEffort": ["max"],
}


class ModelSyncError(Exception):
    """Unrecoverable input/configuration problem (CLI exit code 1).

    Raised for missing/unparseable source files, duplicate endpoint names
    in the source, invalid target JSON, and unrecognised structures. When
    it is raised the target file is never written.
    """


class WriteError(Exception):
    """The target file could not be written (CLI exit code 3)."""


@dataclass(frozen=True)
class DshModel:
    """One model entry extracted from a dsh source endpoint.

    Only string ids survive extraction; unknown extra fields (e.g. the
    ``object``/``created``/``owned_by`` protocol fields) are ignored at
    build time and can never leak into the target file.
    """

    id: str
    name: Optional[str] = None
    url: Optional[str] = None
    toolCalling: Optional[bool] = None
    vision: Optional[bool] = None
    maxInputTokens: Optional[int] = None
    maxOutputTokens: Optional[int] = None
    supportsReasoningEffort: Optional[List[str]] = None

    @staticmethod
    def from_raw(entry: Any) -> "Optional[DshModel]":
        """Build a DshModel from one raw source entry.

        Returns None (skip) when the entry is not an object, or its id is
        missing / not a string / empty after stripping.
        """
        if not isinstance(entry, dict):
            return None
        raw_id = entry.get("id")
        if not isinstance(raw_id, str):
            return None
        model_id = raw_id.strip()
        if not model_id:
            return None

        def text(field_name: str) -> Optional[str]:
            value = entry.get(field_name)
            if isinstance(value, str) and value.strip():
                return value.strip()
            return None

        def flag(field_name: str) -> Optional[bool]:
            value = entry.get(field_name)
            return value if isinstance(value, bool) else None

        def integer(field_name: str) -> Optional[int]:
            value = entry.get(field_name)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
            return None

        def effort(field_name: str) -> Optional[List[str]]:
            value = entry.get(field_name)
            if isinstance(value, str):
                value = [value]
            if (
                isinstance(value, list)
                and value
                and all(isinstance(item, str) and item.strip() for item in value)
            ):
                return list(value)
            return None

        return DshModel(
            id=model_id,
            name=text("name"),
            url=text("url"),
            toolCalling=flag("toolCalling"),
            vision=flag("vision"),
            maxInputTokens=integer("maxInputTokens"),
            maxOutputTokens=integer("maxOutputTokens"),
            supportsReasoningEffort=effort("supportsReasoningEffort"),
        )


@dataclass(frozen=True)
class DshEndpoint:
    """A dsh source endpoint: trimmed display name plus valid models.

    ``models`` is already de-duplicated, order-preserving (first
    occurrence wins), and only contains entries with a usable string id.
    ``fallback_url`` is the endpoint's normalised ``baseURL`` (when the
    source block provides one) and serves as the url fallback for models
    that carry no per-model url.
    ``aliases`` holds the primary match names — the canonical ``name``
    plus alternates at display level such as ``displayName``.
    ``fallback_aliases`` holds key-derived alternates (raw provider key,
    key with a leading ``llm-`` stripped, key-derived display name).
    Matching against target endpoints is case-insensitive and two-level:
    a target endpoint binds to the source endpoint whose *primary* names
    match first (``displayName`` priority); key-derived fallbacks only
    bind when no primary name matches (all comparisons after trimming
    and Unicode case folding).
    """

    name: str
    models: List[DshModel] = field(default_factory=list)
    fallback_url: Optional[str] = None
    #: Why the endpoint has no usable models (reported, not fatal).
    note: str = ""
    #: Primary names this endpoint answers to (canonical ``name`` included).
    aliases: Tuple[str, ...] = ()
    #: Key-derived names, only used when no primary name matches.
    fallback_aliases: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.aliases:
            object.__setattr__(self, "aliases", (self.name,))

    @property
    def folded_aliases(self) -> Set[str]:
        """Case-folded primary alias set used for binding (level 1)."""
        return {fold_name(alias) for alias in self.aliases if isinstance(alias, str) and alias.strip()}

    @property
    def folded_fallback_aliases(self) -> Set[str]:
        """Case-folded key-derived alias set, fallback level (level 2)."""
        return {fold_name(alias) for alias in self.fallback_aliases if isinstance(alias, str) and alias.strip()}

    @property
    def folded_all_aliases(self) -> Set[str]:
        """Union of primary and fallback folded aliases (scope checks)."""
        return self.folded_aliases | self.folded_fallback_aliases

    @property
    def failed(self) -> bool:
        """True when the source list is empty or holds no valid model ids."""
        return not self.models

    def find(self, model_id: str) -> Optional[DshModel]:
        for model in self.models:
            if model.id == model_id:
                return model
        return None

    def url_for(self, model: DshModel) -> Optional[str]:
        """Source-provided url for *model*, else the endpoint fallback url."""
        return model.url or self.fallback_url

    def display_name(self, model_id: str) -> str:
        """Display name for a new model: source value if provided, else generated."""
        model = self.find(model_id)
        if model is not None and model.name:
            return model.name
        return base_display_name(model_id)


def base_display_name(model_id: str) -> str:
    """Build a human-friendly display name from the last path segment of a model id.

    Mirrors the existing clm-sync rule: only the final path segment is used so
    ``vendor-a/model-v4-flash`` becomes "Model V4 Flash" rather than
    "Vendor A / Model V4 Flash".

    Rules per token (checked in order):
      * already mixed/upper case -> keep verbatim   (Camel1, 8B, ABC-5)
      * lowercase version token  -> upper-case       (v4 -> V4, k3 -> K3)
      * otherwise                -> capitalise       (flash -> Flash)
    """
    last_segment = model_id.rsplit("/", 1)[-1]
    for ch in ("-", "_", ".", ":"):
        last_segment = last_segment.replace(ch, " ")
    words = [w for w in last_segment.split() if w]
    titled = [_smart_title(w) for w in words]
    return " ".join(titled) or model_id


def _smart_title(word: str) -> str:
    if any(ch.isupper() for ch in word[1:]) or word.isupper():
        return word
    if re.fullmatch(r"[a-z]+\d[\w.]*|[a-z]\d+", word):
        return word.upper()
    return word[:1].upper() + word[1:]


def is_custom_endpoint(provider: Any) -> bool:
    """Return True when a provider entry is a user-defined custom endpoint."""
    return isinstance(provider, dict) and provider.get("vendor") == CUSTOM_ENDPOINT_VENDOR


def trimmed_name(provider: Any) -> Optional[str]:
    """The endpoint's trimmed name, or None when absent/invalid."""
    name = provider.get("name") if isinstance(provider, dict) else None
    if isinstance(name, str) and name.strip():
        return name.strip()
    return None


def fold_name(name: Any) -> str:
    """Normalised endpoint-name key: trimmed + Unicode case-folded.

    Matching is case-insensitive: ``"Iris"`` and ``"iris"`` fold to the
    same key, while an empty/blank value folds to ``""`` (never matched).
    """
    if not isinstance(name, str):
        return ""
    return name.strip().casefold()


@dataclass
class ProviderSyncResult:
    """Per-endpoint outcome, carrying enough detail for the CLI report."""

    name: str
    #: "synced" | "unchanged" | "skipped" | "no-source" | "target-duplicate"
    status: str = "no-source"
    changed: bool = False
    added: List[str] = field(default_factory=list)
    removed: List[str] = field(default_factory=list)
    settings_keys_removed: List[str] = field(default_factory=list)
    #: Number of pre-existing (local) models that survive this sync, i.e.
    #: the count of existing models kept in place with their local config.
    #: With allow_delete=True this is |source ∩ existing| (the rest were
    #: removed); with allow_delete=False it is |existing| (nothing is
    #: removed, so every local model is kept). Same meaning, two magnitudes.
    kept: int = 0
    #: New source models that could not be added (missing url anywhere).
    skipped_model_ids: List[str] = field(default_factory=list)
    #: True when deletion was suppressed (source failed/empty, or --no-delete).
    skipped_deletion: bool = False
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass
class SyncOutcome:
    """Result of a whole-file synchronization pass."""

    config: List[Dict[str, Any]]
    changed: bool
    providers: List[ProviderSyncResult] = field(default_factory=list)
    #: Source endpoint names with no matching customendpoint in the target;
    #: reported as informational skips, never fatal, never created.
    #:
    #: Scope note: with a ``--provider`` filter, only the *selected* source
    #: endpoints are considered here; unselected source endpoints are out of
    #: scope for that run and therefore never appear in this list (nor in
    #: ``providers``).
    unmatched_source: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(p.ok for p in self.providers)
