"""Load and atomically save chatLanguageModels.json.

Serialization round-trips through json.dumps so a successful parse
guarantees a serializable structure; ``ensure_ascii=False`` keeps
non-ASCII model names intact, tab indentation mirrors the format VS Code
writes.  API key values are never read, resolved, or logged: they are
opaque strings that pass through untouched.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from typing import Any, List, Union

from .models import ModelSyncError, WriteError

PathLike = Union[str, "os.PathLike[str]"]
INDENT = "\t"


@dataclass(frozen=True)
class SaveOutcome:
    path: str
    written: bool
    reason: str = ""


def load_config(path: PathLike) -> List[dict[str, Any]]:
    """Read and validate the configuration file, returning provider objects.

    Raises ModelSyncError when the file is missing, not valid JSON, not a
    JSON array, or when a provider entry is structurally invalid.
    """
    try:
        with open(path, "r", encoding="utf-8-sig") as handle:
            raw = handle.read()
    except FileNotFoundError as exc:
        raise ModelSyncError(f"config file not found: {path}") from exc
    except OSError as exc:
        raise ModelSyncError(f"cannot read {path}: {exc}") from exc

    try:
        data: Any = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ModelSyncError(f"{path} is not valid JSON: {exc}") from exc

    _validate(data, path)
    return data


def _validate(data: Any, path: PathLike) -> None:
    if not isinstance(data, list):
        raise ModelSyncError(f"{path} must contain a JSON array")
    for index, provider in enumerate(data):
        if not isinstance(provider, dict):
            raise ModelSyncError(f"{path}: provider #{index} is not a JSON object")
        models = provider.get("models")
        if models is not None and not isinstance(models, list):
            raise ModelSyncError(
                f"{path}: provider {provider.get('name', index)!r} has a non-list 'models' field"
            )
        settings = provider.get("settings")
        if settings is not None and not isinstance(settings, dict):
            raise ModelSyncError(
                f"{path}: provider {provider.get('name', index)!r} has a non-object 'settings' field"
            )


def serialize(data: Any) -> str:
    """Render the provider list exactly the way VS Code's file is laid out."""
    return json.dumps(data, indent=INDENT, ensure_ascii=False) + "\n"


def write_config_atomic(
    path: PathLike, data: Any, *, original_text: str | None = None
) -> SaveOutcome:
    """Write `data` to `path` via a same-directory temp file + atomic replace.

    * Skips the write entirely when the rendered text is unchanged
      (compared against ``original_text``, tolerating CRLF/LF differences),
      which keeps mtime/OneDrive sync quiet on no-op runs.
    * The temp file is always unlinked on failure; the original file is
      only replaced once the full content has been written, so a failed
      write never corrupts it.

    Raises WriteError when the file system rejects the write.
    """
    text = serialize(data)

    if original_text is not None and text == _normalize_text(original_text):
        return SaveOutcome(str(path), written=False, reason="no changes detected")

    # Re-parse rendered output before touching the user's file.
    try:
        json.loads(text)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise WriteError(f"refusing to write invalid JSON: {exc}") from exc

    directory = os.path.dirname(os.path.abspath(path))
    try:
        os.makedirs(directory, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=".dsh-sync-", suffix=".tmp", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
            os.replace(tmp_path, path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    except WriteError:
        raise
    except OSError as exc:
        raise WriteError(f"failed to write {path}: {exc}") from exc
    return SaveOutcome(str(path), written=True)


def read_original_text(path: PathLike) -> str | None:
    """Return the current on-disk text (UTF-8) for change detection.

    None when the file does not exist yet.
    """
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as handle:
            return handle.read()
    except FileNotFoundError:
        return None
    except OSError:
        # The file disappeared between load and write; treat as no-baseline
        # so the write proceeds but change detection is skipped.
        return None


def _normalize_text(text: str) -> str:
    """Collapse CRLF to LF so change detection is line-ending agnostic."""
    return text.replace("\r\n", "\n").replace("\r", "\n")
