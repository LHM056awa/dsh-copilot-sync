"""Core merge logic: dsh source -> chatLanguageModels.json (same-named endpoints).

Guarantees implemented here:

* only ``vendor == "customendpoint"`` target entries are ever mutated;
  built-in entries (``vendor == "copilot"`` and any other vendor) pass
  through byte-identical, even when they share a name with a source
  endpoint;
* matching is on the trimmed, case-sensitive endpoint name; a trimmed
  name appearing more than once among the target custom endpoints is
  reported as an error for that endpoint and the entry is left untouched
  (spec: 目标文件中同一端点名出现多次时,报错,不修改该端点);
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
    is_custom_endpoint,
    trimmed_name,
)


def _count_target_names(
    config: Sequence[Dict[str, Any]],
) -> Dict[str, List[int]]:
    """Map trimmed name -> list of indices among target custom endpoints."""
    by_name: Dict[str, List[int]] = {}
    for index, provider in enumerate(config):
        if not is_custom_endpoint(provider):
            continue
        name = trimmed_name(provider)
        if name is None:
            continue
        by_name.setdefault(name, []).append(index)
    return by_name


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
) -> Optional[Dict[str, Any]]:
    """Create a fresh canonical model object, or None when no url resolves.

    Field precedence: source value > existing target url > default.
    Only the canonical fields are written; protocol fields such as
    ``object``/``created``/``owned_by`` never appear.
    """
    source_endpoint = source_index.endpoints[endpoint_name]
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
) -> ProviderSyncResult:
    """Synchronise a single matched endpoint in place; returns its result."""
    result = ProviderSyncResult(name=name, status="unchanged")
    source_endpoint = source_index.endpoints[name]

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
        new_model = _build_new_model(name, model, existing_urls, source_index)
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
    target custom endpoints are processed; every other entry passes
    through untouched.  When *allow_delete* is False, only additions
    happen: nothing is removed and nothing that already exists is
    overwritten.

    Raises ModelSyncError when a requested --provider name does not
    exist as a target custom endpoint (a silent no-op would hide typos).
    """
    by_name = _count_target_names(config)
    source_names = set(source.endpoints)

    wanted: Optional[Set[str]] = None
    if provider_names is not None:
        wanted = set()
        for raw in provider_names:
            name = raw.strip()
            if name:
                wanted.add(name)
        if not wanted:
            raise ModelSyncError(
                "no usable --provider name given (all names are blank)"
            )
        missing = sorted(w for w in wanted if w not in by_name)
        if missing:
            available = ", ".join(sorted(by_name)) or "<none>"
            raise ModelSyncError(
                "unknown target endpoint(s): "
                + ", ".join(repr(m) for m in missing)
                + f"; available custom endpoints: {available}"
            )

    working = copy.deepcopy(config)
    results: List[ProviderSyncResult] = []
    reported: Set[str] = set()

    for index, provider in enumerate(working):
        if not is_custom_endpoint(provider):
            continue
        name = trimmed_name(provider)
        if name is None:
            continue
        if wanted is not None and name not in wanted:
            continue
        if name in reported:
            continue
        reported.add(name)

        indices = by_name[name]
        if len(indices) > 1:
            # Spec: target name declared more than once -> error, do not
            # modify that endpoint. Recorded per-endpoint, non-fatal.
            result = ProviderSyncResult(name=name, status="skipped")
            result.errors.append(
                f"endpoint name {name!r} appears {len(indices)} times among "
                "custom endpoints — ambiguous, leaving all occurrences untouched"
            )
            results.append(result)
            continue

        if name not in source_names:
            results.append(ProviderSyncResult(name=name, status="no-source"))
            continue

        results.append(
            _sync_one_endpoint(
                working[index],
                name,
                source,
                allow_delete=allow_delete,
            )
        )

    unmatched = sorted(
        n for n in source_names if n not in by_name and (wanted is None or n in wanted)
    )

    return SyncOutcome(
        config=working,
        changed=working != config,
        providers=results,
        unmatched_source=unmatched,
    )
