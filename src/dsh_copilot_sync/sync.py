"""Core merge logic: dsh source -> chatLanguageModels.json (same-named endpoints).

Guarantees implemented here:

* only ``vendor == "customendpoint"`` target entries are ever mutated;
  built-in entries (``vendor == "copilot"`` and any other vendor) pass
  through byte-identical, even when they share a name with a source
  endpoint;
* matching is on the trimmed, case-insensitive endpoint name, in two
  levels: a target endpoint first binds to a source endpoint whose
  *primary* names (canonical name / ``displayName`` level) equal the
  target name after trimming and Unicode case folding (``"Iris"``
  matches ``"iris"``); only when no primary name matches, key-derived
  fallback names (raw provider key, unprefixed key, key-derived name)
  are consulted;
* a folded name appearing more than once among the target custom
  endpoints is reported as an error for that endpoint and the entry is
  left untouched (spec: 目标文件中同一端点名出现多次时,报错,
  不修改该端点);
* a target endpoint whose name matches more than one distinct source
  endpoint is likewise ambiguous: it is reported and left untouched;
* source endpoints with no same-named target custom endpoint are
  reported and skipped — no new endpoints are ever created;
* target endpoints with no same-named source endpoint are untouched;
* existing model objects keep every hand-authored field (never
  overwritten by source values);
* new models only ever get the canonical field set — protocol fields
  such as ``object``/``created``/``owned_by`` are not written;
* deletion of models (and of their top-level ``settings`` entries)
  happens only when the source endpoint was read successfully, the
  deletion flag is set, and at least one valid model id exists;
* when the source endpoint failed to read or has no usable models the
  target endpoint is preserved exactly as it was (spec: 保留目标端点
  原样,并记录错误).
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence, Set

from .dsh_source import DshSourceIndex
from .models import (
    DEFAULT_NEW_MODEL_FIELDS,
    ModelSyncError,
    ProviderSyncResult,
    SyncOutcome,
    fold_name,
    is_custom_endpoint,
    trimmed_name,
)


def _count_target_names(
    config: Sequence[Dict[str, Any]],
) -> Dict[str, List[int]]:
    """Map folded name -> list of indices among target custom endpoints."""
    by_name: Dict[str, List[int]] = {}
    for index, provider in enumerate(config):
        if not is_custom_endpoint(provider):
            continue
        name = trimmed_name(provider)
        if name is None:
            continue
        by_name.setdefault(fold_name(name), []).append(index)
    return by_name


def _target_display_names(
    config: Sequence[Dict[str, Any]],
    indices: List[int],
) -> List[str]:
    """Original trimmed target names for *indices* (for error messages)."""
    names: List[str] = []
    for index in indices:
        name = trimmed_name(config[index])
        if name is not None and name not in names:
            names.append(name)
    return names


def _existing_model_index(provider: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Map usable model id -> its existing object, first occurrence wins."""
    index: Dict[str, Dict[str, Any]] = {}
    models = provider.get("models")
    if isinstance(models, list):
        for entry in models:
            if not isinstance(entry, dict):
                continue
            model_id = entry.get("id")
            if isinstance(model_id, str) and model_id.strip():
                index.setdefault(model_id.strip(), entry)
    return index


def _existing_urls(provider: Dict[str, Any]) -> List[str]:
    """Order-preserving list of distinct non-empty model urls in the target."""
    seen: Dict[str, None] = {}
    models = provider.get("models")
    if isinstance(models, list):
        for entry in models:
            if isinstance(entry, dict):
                url = entry.get("url")
                if isinstance(url, str) and url.strip():
                    seen.setdefault(url.strip(), None)
    return list(seen.keys())


def _build_new_model(
    endpoint_name: str,
    model,
    existing_urls: List[str],
    source_index: DshSourceIndex,
    *,
    source_name: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Create a fresh canonical model object, or None when no url resolves.

    Field precedence: source value > existing target url > default.
    Only the canonical fields are written; protocol fields such as
    ``object``/``created``/``owned_by`` never appear.
    """
    source_endpoint = source_index.endpoints[source_name or endpoint_name]
    url = source_endpoint.url_for(model)
    if url is None and existing_urls:
        url = existing_urls[0]
    if url is None:
        return None

    new_model: Dict[str, Any] = {
        "id": model.id,
        "name": source_endpoint.display_name(model.id),
        "url": url,
        "toolCalling": (
            model.toolCalling
            if model.toolCalling is not None
            else DEFAULT_NEW_MODEL_FIELDS["toolCalling"]
        ),
        "vision": (
            model.vision
            if model.vision is not None
            else DEFAULT_NEW_MODEL_FIELDS["vision"]
        ),
        "maxInputTokens": (
            model.maxInputTokens
            if model.maxInputTokens is not None
            else DEFAULT_NEW_MODEL_FIELDS["maxInputTokens"]
        ),
        "maxOutputTokens": (
            model.maxOutputTokens
            if model.maxOutputTokens is not None
            else DEFAULT_NEW_MODEL_FIELDS["maxOutputTokens"]
        ),
        "supportsReasoningEffort": (
            model.supportsReasoningEffort
            if model.supportsReasoningEffort is not None
            else copy.deepcopy(DEFAULT_NEW_MODEL_FIELDS["supportsReasoningEffort"])
        ),
    }
    return new_model


def _sync_one_endpoint(
    provider: Dict[str, Any],
    name: str,
    source_index: DshSourceIndex,
    *,
    allow_delete: bool,
    source_name: Optional[str] = None,
) -> ProviderSyncResult:
    """Synchronise a single matched endpoint in place; returns its result.

    *name* is the target entry's trimmed name (used for reporting);
    *source_name* is the bound source endpoint's canonical name (used for
    the source lookup).  They differ only when the binding relied on an
    alias or on a case difference.
    """
    result = ProviderSyncResult(name=name, status="unchanged")
    source_endpoint = source_index.endpoints[source_name or name]

    # Spec rule 5: failed read / empty list / no valid ids -> keep the
    # endpoint exactly as-is and record the problem.
    if source_endpoint.failed:
        result.status = "skipped"
        result.skipped_deletion = True
        result.errors.append(
            "source model list is empty or has no valid model ids"
            + (f" ({source_endpoint.note})" if source_endpoint.note else "")
            + " — endpoint kept as-is, deletion skipped"
        )
        return result

    existing = _existing_model_index(provider)
    existing_urls = _existing_urls(provider)
    source_id_set = {m.id for m in source_endpoint.models}

    if allow_delete:
        # Rule 3: drop target-only models (and their settings keys).
        keep_ids: Set[str] = source_id_set
        removed = [mid for mid in existing if mid not in keep_ids]
    else:
        # Rule 4: --no-delete -> keep every existing model, add new ones.
        keep_ids = set(existing) | source_id_set
        removed = []

    # Order: existing models first (original relative order), new models
    # appended in source order. Stable across runs -> idempotent.
    merged: Dict[str, Dict[str, Any]] = {}
    for mid, original in existing.items():
        if mid in keep_ids:
            merged[mid] = copy.deepcopy(original)
    added: List[str] = []
    for model in source_endpoint.models:
        if model.id in merged:
            continue
        new_model = _build_new_model(
            name, model, existing_urls, source_index,
            source_name=source_endpoint.name,
        )
        if new_model is None:
            result.skipped_model_ids.append(model.id)
            result.errors.append(
                f"skipped new model {model.id!r}: no url available "
                "(source has no url and the target endpoint has no models)"
            )
            continue
        merged[model.id] = new_model
        added.append(model.id)

    new_models = [merged[mid] for mid in list(merged)]

    settings = provider.get("settings")
    removed_set = set(removed)
    settings_removed: List[str] = []
    if isinstance(settings, dict) and removed_set:
        pruned = {
            key: value for key, value in settings.items() if key not in removed_set
        }
        settings_removed = sorted(set(settings) - set(pruned), key=str.lower)
        if pruned:
            provider["settings"] = pruned
        elif "settings" in provider:
            # Rule 6: drop the field rather than leave an empty object.
            del provider["settings"]

    changed = bool(added or removed or settings_removed)
    if changed:
        provider["models"] = new_models

    result.added = added
    result.removed = removed
    result.settings_keys_removed = settings_removed
    # "kept" = local models that survive this sync (see ProviderSyncResult.kept).
    # allow_delete=True  -> |source ∩ existing|; allow_delete=False -> |existing|.
    result.kept = len(keep_ids & set(existing))
    result.changed = changed
    result.status = "synced" if changed else "unchanged"
    if not allow_delete:
        result.skipped_deletion = True
    return result


def sync_config(
    config: List[Dict[str, Any]],
    source: DshSourceIndex,
    *,
    allow_delete: bool = True,
    provider_names: Optional[Sequence[str]] = None,
) -> SyncOutcome:
    """Run the full offline sync over `config` without touching disk.

    *provider_names* (optional, repeatable CLI ``--provider``) limits which
    target custom endpoints are processed (matched case-insensitively);
    every other entry passes through untouched.  When *allow_delete* is
    False, only additions happen: nothing is removed and nothing that
    already exists is overwritten.

    Binding (two levels, both case-insensitive after trimming and
    Unicode case folding): level 1 matches the target name against each
    source endpoint's *primary* names (canonical name / ``displayName``,
    ``displayName`` priority); level 2 matches against key-derived
    fallback names (raw provider key, unprefixed key, key-derived
    display name) and is only consulted when level 1 has no hit.  A
    folded name shared by several target entries, or a target name
    matching several distinct source endpoints *at the same level*,
    is ambiguous: it is reported and left untouched.

    Raises ModelSyncError when a requested --provider name does not
    exist as a target custom endpoint (a silent no-op would hide typos).
    """
    by_name = _count_target_names(config)
    # Two-level binding index (all keys case-folded):
    #   level 1: folded primary alias -> sorted canonical source names
    #   level 2: folded key-derived fallback -> sorted canonical names.
    # Level 2 is only consulted when level 1 has no hit, so displayName
    # always wins over provider keys.
    source_primary: Dict[str, List[str]] = {}
    source_fallback: Dict[str, List[str]] = {}
    for canonical in sorted(source.endpoints):
        endpoint = source.endpoints[canonical]
        for alias in endpoint.folded_aliases:
            source_primary.setdefault(alias, [])
            if canonical not in source_primary[alias]:
                source_primary[alias].append(canonical)
        for alias in endpoint.folded_fallback_aliases:
            source_fallback.setdefault(alias, [])
            if canonical not in source_fallback[alias]:
                source_fallback[alias].append(canonical)

    def _bound_source(folded: str) -> Optional[List[str]]:
        found = source_primary.get(folded)
        if found:
            return sorted(found)
        found = source_fallback.get(folded)
        return sorted(found) if found else None

    def _bound_level(folded: str) -> int:
        """1 when the binding came from a primary name, 2 for fallbacks."""
        if folded in source_primary:
            return 1
        if folded in source_fallback:
            return 2
        return 0

    wanted: Optional[Set[str]] = None
    if provider_names is not None:
        wanted = set()
        for raw in provider_names:
            folded = fold_name(raw)
            if folded:
                wanted.add(folded)
        if not wanted:
            raise ModelSyncError(
                "no usable --provider name given (all names are blank)"
            )
        missing = sorted(w for w in wanted if w not in by_name)
        if missing:
            seen_names = {
                trimmed_name(config[i]) or ""
                for idx in by_name.values()
                for i in idx
            } - {""}
            available = ", ".join(sorted(seen_names)) or "<none>"
            raise ModelSyncError(
                "unknown target endpoint(s): "
                + ", ".join(repr(m) for m in missing)
                + f"; available custom endpoints: {available}"
            )

    working = copy.deepcopy(config)
    results: List[ProviderSyncResult] = []
    reported: Set[str] = set()
    bound_source: Set[str] = set()

    for index, provider in enumerate(working):
        if not is_custom_endpoint(provider):
            continue
        name = trimmed_name(provider)
        if name is None:
            continue
        folded = fold_name(name)
        if wanted is not None and folded not in wanted:
            continue
        if folded in reported:
            continue
        reported.add(folded)

        indices = by_name[folded]
        if len(indices) > 1:
            # Spec: target name declared more than once -> error, do not
            # modify that endpoint. Recorded per-endpoint, non-fatal.
            # Names are compared case-insensitively, so "Foo" + "foo"
            # count as the same name.
            seen = _target_display_names(working, indices)
            shown = seen[0] if len(seen) == 1 else f"{seen[0]!r} (also written as {', '.join(repr(s) for s in seen[1:])})"
            result = ProviderSyncResult(name=name, status="skipped")
            result.errors.append(
                f"endpoint name {shown} appears {len(indices)} times among "
                "custom endpoints (case-insensitive) — ambiguous, leaving all occurrences untouched"
            )
            results.append(result)
            continue

        matched = _bound_source(folded)
        if not matched:
            results.append(ProviderSyncResult(name=name, status="no-source"))
            continue
        if len(matched) > 1:
            # The target name equals the same-level names of several
            # distinct source endpoints -> ambiguous, leave untouched.
            # (Primary displayName matches always beat key-derived
            # fallbacks: fallbacks only bind when no primary matches.)
            level = "display names" if _bound_level(folded) == 1 else "provider-key names"
            result = ProviderSyncResult(name=name, status="skipped")
            result.errors.append(
                f"endpoint name {name!r} matches {len(matched)} source endpoints "
                f"({', '.join(repr(m) for m in matched)}) via {level} — "
                "ambiguous, leaving the target untouched"
            )
            results.append(result)
            continue

        bound_source.add(matched[0])
        results.append(
            _sync_one_endpoint(
                working[index],
                name,
                source,
                allow_delete=allow_delete,
                source_name=matched[0],
            )
        )

    unmatched = sorted(
        n for n in source.endpoints
        if n not in bound_source
        and (wanted is None or not source.endpoints[n].folded_all_aliases.isdisjoint(wanted))
    )

    return SyncOutcome(
        config=working,
        changed=working != config,
        providers=results,
        unmatched_source=unmatched,
    )
