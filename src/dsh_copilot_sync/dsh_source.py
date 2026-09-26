"""dsh/ folder source adapter.

Reads model lists out of a dsh data folder and normalises them into the
logical structure the sync engine consumes::

    endpoint name -> ordered, de-duplicated list of model ids

Recognised layouts (any mixture, across the loaded files):

* a bare **list of endpoint objects** ``[{"name": ..., "baseURL": ...,
  "models": [...]}]`` (the expected logical structure, JSON or YAML);
* **provider-group documents** such as ``settings.yaml.imported``:
  top-level LLM namespaces (``llm-<provider>``, e.g. keys with the
  ``llm-`` prefix) whose value carries a ``providers`` sub-dict or a
  ``models`` list;
* **plugin loader documents** (YAML files whose top level is a list of
  plugin objects): each plugin's ``config`` value may carry ``providers``
  blocks.

Normalisation guarantees:

* model ids are stripped, must be non-empty strings, and are
  de-duplicated while preserving first-seen order;
* entries without a usable string id are skipped;
* known field aliases are mapped onto the canonical names:
  ``contextWindow`` -> ``maxInputTokens``, ``maxTokens`` ->
  ``maxOutputTokens``, ``inputModalities`` containing ``image`` ->
  ``vision: true`` and containing ``tools`` -> ``toolCalling: true``;
  protocol fields (``object``/``created``/``owned_by`` etc.) are never
  carried over;
* a block-level ``baseURL`` becomes the endpoint's fallback ``url`` (a
  trailing ``/v1`` is dropped, matching how VS Code appends ``/v1`` when
  calling OpenAI-compatible chat endpoints);
* endpoint names: a block's ``displayName`` wins, otherwise the name is
  derived from the provider key (a leading ``llm-`` prefix is stripped and
  the key is title-cased with the same rules as
  :func:`dsh_copilot_sync.models.base_display_name`);
* the same trimmed endpoint name appearing more than once across the
  loaded files is a configuration error (spec: 源端点名称重复时报错).

No network access, no credential handling: ``apiKeyEnv`` names are
deliberately ignored and no key material is ever read or returned.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .models import DshEndpoint, DshModel, ModelSyncError, base_display_name

PathLike = Union[str, "os.PathLike[str]"]

SOURCE_EXTS = {".yaml", ".yml", ".json"}
#: Preferred primary settings files, checked in this order.
PREFERRED_FILES = ("settings.yaml.imported", "settings.yaml", "settings.yml")
#: Files in a dsh folder that never carry provider blocks.
IGNORED_FILE_NAMES = {"credentials.yaml", "sync.ffs_db", "package.json"}


@dataclass
class DshSourceIndex:
    """Normalised source: trimmed endpoint name -> DshEndpoint."""

    endpoints: Dict[str, DshEndpoint] = field(default_factory=dict)
    #: Non-fatal problems collected while reading the source.
    warnings: List[str] = field(default_factory=list)


def normalize_base_url(url: Any) -> Optional[str]:
    """Return a cleaned endpoint base url, or None.

    A single trailing ``/v1`` segment is dropped so that OpenAI-compatible
    endpoints work with VS Code's chat-completions API type.  Only the last
    segment is stripped: ``https://h/api/v1`` -> ``https://h/api`` keeps its
    middle path segments, and intermediate duplicate segments like
    ``/v1/v1`` are not collapsed.
    """
    if not isinstance(url, str):
        return None
    value = url.strip()
    if not value:
        return None
    if value.endswith("/v1"):
        value = value[:-3]
    if value.endswith("/"):
        value = value[:-1]
    return value or None


def _apply_aliases(raw: Any) -> Any:
    """Map known alias fields onto canonical new-model field names.

    Canonical keys already present on the entry always win.  Anything
    that is not an object passes through unchanged (skipped later).
    """
    if not isinstance(raw, dict):
        return raw
    entry = dict(raw)
    context_window = entry.get("contextWindow")
    if isinstance(context_window, int) and not isinstance(context_window, bool) and not isinstance(
        entry.get("maxInputTokens"), int
    ):
        entry["maxInputTokens"] = context_window
    max_tokens = entry.get("maxTokens")
    if isinstance(max_tokens, int) and not isinstance(max_tokens, bool) and not isinstance(
        entry.get("maxOutputTokens"), int
    ):
        entry["maxOutputTokens"] = max_tokens
    modalities = entry.get("inputModalities")
    if isinstance(modalities, list):
        lowered = {m.strip().lower() for m in modalities if isinstance(m, str)}
        if not isinstance(entry.get("vision"), bool) and ("image" in lowered or "img" in lowered):
            entry["vision"] = True
        if not isinstance(entry.get("toolCalling"), bool) and "tools" in lowered:
            entry["toolCalling"] = True
    return entry


def _build_endpoint(name: str, block: Dict[str, Any]) -> DshEndpoint:
    """Validate/deduplicate one provider block into a DshEndpoint."""
    fallback_url = normalize_base_url(block.get("baseURL") or block.get("url"))
    models: List[DshModel] = []
    seen: set[str] = set()
    note = ""
    raw_models = block.get("models")
    if raw_models is None:
        raw_models = []
    if not isinstance(raw_models, list):
        note = "'models' field is not a list"
        raw_models = []
    for raw in raw_models:
        model = DshModel.from_raw(_apply_aliases(raw))
        if model is None:
            continue
        if model.url is None and fallback_url:
            model = DshModel(
                id=model.id,
                name=model.name,
                url=fallback_url,
                toolCalling=model.toolCalling,
                vision=model.vision,
                maxInputTokens=model.maxInputTokens,
                maxOutputTokens=model.maxOutputTokens,
                supportsReasoningEffort=model.supportsReasoningEffort,
            )
        if model.id in seen:
            continue
        seen.add(model.id)
        models.append(model)
    if not models and not note:
        note = "no valid model ids (missing, empty, or non-string)"
    return DshEndpoint(name=name, models=models, fallback_url=fallback_url, note=note)


# ---------------------------------------------------------------------------
# Document-level extraction (pure functions, easy to unit-test)
# ---------------------------------------------------------------------------

Block = Tuple[str, Dict[str, Any], bool]  # (name source, block, is_endpoint_object)


def _find_blocks(doc: Any) -> List[Block]:
    """Find provider blocks in one parsed document (dict).

    Yields (name_source, block, is_endpoint_object) where name_source is
    either the endpoint's own ``name``/``displayName`` string (when the
    document itself is an endpoint object) or the provider key.
    """
    blocks: List[Block] = []
    if not isinstance(doc, dict):
        return blocks

    # 1) The document itself is an endpoint object: has "models" plus a
    #    usable "name" or "displayName".
    if "models" in doc and (
        isinstance(doc.get("name"), str) or isinstance(doc.get("displayName"), str)
    ):
        raw_name = doc.get("name")
        if not (isinstance(raw_name, str) and raw_name.strip()):
            raw_name = doc.get("displayName")
        blocks.append((raw_name, doc, True))

    # 2) A top-level "providers" sub-dict (plugin-loader config / llm-<ns> style).
    providers = doc.get("providers")
    if isinstance(providers, dict):
        for key, block in providers.items():
            if isinstance(block, dict) and isinstance(block.get("models"), list):
                blocks.append((str(key), block, False))

    # 3) One level down: namespace values that carry providers or models
    #    (settings.yaml.imported style: llm-<ns>.providers / llm-<ns>.models).
    for key, value in doc.items():
        if key in ("providers", "name", "displayName", "models"):
            continue
        if not isinstance(value, dict):
            continue
        nested = value.get("providers")
        if isinstance(nested, dict):
            for sub_key, block in nested.items():
                if isinstance(block, dict) and isinstance(block.get("models"), list):
                    blocks.append((str(sub_key), block, False))
        elif isinstance(value.get("models"), list):
            blocks.append((str(key), value, False))
    return blocks


def _endpoint_name_from_key(key: str, block: Dict[str, Any], origin: str) -> str:
    """Display name for a provider block: displayName wins, else derived."""
    display = block.get("displayName")
    if isinstance(display, str) and display.strip():
        return display.strip()
    provider_key = key.strip()
    if provider_key.startswith("llm-"):
        provider_key = provider_key[len("llm-"):]
    if not provider_key:
        raise ModelSyncError(f"{origin}: provider block without a usable name: {block!r}")
    return base_display_name(provider_key)


def _endpoint_name_from_object(doc: Dict[str, Any], origin: str) -> str:
    raw = doc.get("name")
    if not (isinstance(raw, str) and raw.strip()):
        raw = doc.get("displayName")
    if not isinstance(raw, str) or not raw.strip():
        raise ModelSyncError(f"{origin}: endpoint object without a usable 'name' field")
    return raw.strip()


def extract_endpoints(data: Any, *, origin: str = "<source>") -> List[DshEndpoint]:
    """Extract DshEndpoint objects from one parsed document.

    ``data`` may be a list (endpoint objects or plugin objects) or a single
    dict.  Returns an empty list when the document carries no provider
    blocks; raises ModelSyncError for unrecognised endpoint objects.
    """
    docs = data if isinstance(data, list) else [data]
    endpoints: List[DshEndpoint] = []
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        for name_source, block, is_object in _find_blocks(doc):
            if is_object:
                name = _endpoint_name_from_object(block, origin)
            else:
                name = _endpoint_name_from_key(str(name_source), block, origin)
            endpoints.append(_build_endpoint(name, block))
    return endpoints


# ---------------------------------------------------------------------------
# YAML loading (permissive towards dsh-specific tags such as !!js)
# ---------------------------------------------------------------------------


def _load_yaml(text: str) -> Any:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - PyYAML is a hard dependency
        raise ModelSyncError(
            "PyYAML is required to read .yaml/.yml source files: install it with "
            "`pip install pyyaml`"
        ) from exc

    class _Loader(yaml.SafeLoader):
        pass

    def _construct_undefined(loader: yaml.Loader, tag: str, node: yaml.Node) -> Any:
        """Ignore unknown tags (e.g. ``!!js`` used by dsh loader files)."""
        if isinstance(node, yaml.ScalarNode):
            return loader.construct_scalar(node)
        if isinstance(node, yaml.SequenceNode):
            return [loader.construct_object(child, deep=True) for child in node.content]
        if isinstance(node, yaml.MappingNode):
            return {
                loader.construct_object(k, deep=True): loader.construct_object(v, deep=True)
                for k, v in node.value
            }
        return None

    _Loader.construct_undefined = _construct_undefined  # type: ignore[attr-defined]
    return yaml.load(text, Loader=_Loader)


def _load_file(path: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8-sig") as handle:
            text = handle.read()
    except OSError as exc:
        raise ModelSyncError(f"cannot read source file {path}: {exc}") from exc

    if os.path.splitext(path)[1].lower() == ".json":
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ModelSyncError(f"{path} is not valid JSON: {exc}") from exc
    try:
        return _load_yaml(text)
    except Exception as exc:  # yaml.YAMLError and friends
        raise ModelSyncError(f"{path} is not valid YAML: {exc}") from exc


# ---------------------------------------------------------------------------
# Directory loading
# ---------------------------------------------------------------------------


def _candidate_files(dsh_dir: PathLike) -> List[str]:
    """Default source files: the preferred settings file when present,
    otherwise every top-level YAML/JSON source file."""
    try:
        entries = sorted(os.listdir(dsh_dir))
    except OSError as exc:
        raise ModelSyncError(f"cannot list dsh source directory {dsh_dir}: {exc}") from exc

    files = [
        entry
        for entry in entries
        if not entry.startswith(".")
        and entry not in IGNORED_FILE_NAMES
        and os.path.isfile(os.path.join(str(dsh_dir), entry))
        and (
            entry in PREFERRED_FILES or os.path.splitext(entry)[1].lower() in SOURCE_EXTS
        )
    ]
    preferred = [f for f in files if f in PREFERRED_FILES]
    chosen = preferred if preferred else files
    return [os.path.join(str(dsh_dir), f) for f in chosen]


def load_dsh_source(
    dsh_dir: PathLike,
    source_files: Optional[Sequence[str]] = None,
) -> DshSourceIndex:
    """Read a dsh folder and return the normalised source index.

    ``source_files`` (optional) restricts loading to the given file names
    (relative to ``dsh_dir`` or absolute paths); by default the preferred
    settings file is used when present, else all top-level source files.

    Raises ModelSyncError when the directory is missing, a file cannot
    be parsed, endpoint names collide, or no model lists are found at
    all.
    """
    if not os.path.isdir(dsh_dir):
        raise ModelSyncError(f"dsh source directory not found: {dsh_dir}")

    if source_files:
        files: List[str] = [
            f if os.path.isabs(f) else os.path.join(str(dsh_dir), f)
            for f in source_files
        ]
    else:
        files = _candidate_files(dsh_dir)

    index: Dict[str, DshEndpoint] = {}
    warnings: List[str] = []
    for path in files:
        if not os.path.isfile(path):
            raise ModelSyncError(f"source file not found: {path}")
        data = _load_file(path)
        endpoints = extract_endpoints(data, origin=path)
        if not endpoints:
            warnings.append(f"no model lists found in {path}")
        for endpoint in endpoints:
            if endpoint.name in index:
                raise ModelSyncError(
                    f"duplicate source endpoint name {endpoint.name!r} "
                    f"(in {path} and an earlier file) — one endpoint name may only "
                    "appear once across the loaded source files"
                )
            index[endpoint.name] = endpoint

    if not index:
        raise ModelSyncError(
            f"no model lists found in dsh source ({', '.join(files) or dsh_dir}): "
            "unrecognised structure"
        )
    return DshSourceIndex(endpoints=index, warnings=warnings)
