"""
Local cache of file-type mappings, optionally refreshed from Egeria ValidMetadataValues.

The cache persists as a JSON file so the surveyor works offline.  When Egeria
credentials are present the cache is refreshed on demand (e.g. once per day or
when --refresh is passed).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_DEFAULT_CACHE_PATH = Path("data/file_type_cache.json")
_REFRESH_AFTER_HOURS = 24


class FileTypeCache:
    """
    Two-level lookup: filename → metadata, extension → metadata.

    Cache file structure:
    {
      "refreshed_at": "2026-05-18T12:00:00",
      "by_name": {"Dockerfile": {"fileType": "Docker", ...}, ...},
      "by_extension": {"py": {"fileType": "Python Source", ...}, ...}
    }
    """

    def __init__(self, cache_path: Path = _DEFAULT_CACHE_PATH) -> None:
        self._path = cache_path
        self._by_name: dict[str, dict] = {}
        self._by_ext: dict[str, dict] = {}
        self._refreshed_at: datetime | None = None
        self._load()

    # ── public API ────────────────────────────────────────────────────────────

    def lookup(self, file_name: str, extension: str | None) -> dict[str, Any]:
        """Return metadata dict (may be empty) for the given file."""
        if file_name in self._by_name:
            return self._by_name[file_name]
        if extension and extension in self._by_ext:
            return self._by_ext[extension]
        return {}

    def needs_refresh(self) -> bool:
        if self._refreshed_at is None:
            return True
        return datetime.utcnow() - self._refreshed_at > timedelta(hours=_REFRESH_AFTER_HOURS)

    def refresh_from_egeria(self, pyegeria_client) -> None:
        """Pull ValidMetadataValues from Egeria and rebuild the cache."""
        try:
            by_name: dict[str, dict] = {}
            by_ext: dict[str, dict] = {}

            for prop_name, target in [("fileName", by_name), ("fileExtension", by_ext)]:
                try:
                    values = pyegeria_client.get_valid_metadata_values(
                        property_name=prop_name,
                        type_name="DataFile",
                    )
                    if not isinstance(values, list):
                        continue
                    for v in values:
                        props = v.get("properties", {})
                        key = props.get("preferredValue") or props.get("displayName", "")
                        add = props.get("additionalProperties", {})
                        if key:
                            target[key] = {
                                "fileType": add.get("fileType"),
                                "assetTypeName": add.get("assetTypeName"),
                                "encoding": add.get("encoding"),
                                "deployedImplementationType": add.get("deployedImplementationType"),
                            }
                except Exception as exc:
                    log.warning("FileTypeCache: failed to fetch %s from Egeria: %s", prop_name, exc)

            self._by_name = by_name
            self._by_ext = by_ext
            self._refreshed_at = datetime.utcnow()
            self._save()
            log.info(
                "FileTypeCache: refreshed — %d names, %d extensions",
                len(by_name),
                len(by_ext),
            )
        except Exception as exc:
            log.warning("FileTypeCache: refresh failed, using existing cache: %s", exc)

    # ── persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text())
            self._by_name = data.get("by_name", {})
            self._by_ext = data.get("by_extension", {})
            ts = data.get("refreshed_at")
            self._refreshed_at = datetime.fromisoformat(ts) if ts else None
        except Exception as exc:
            log.warning("FileTypeCache: could not load cache from %s: %s", self._path, exc)

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(
                    {
                        "refreshed_at": self._refreshed_at.isoformat() if self._refreshed_at else None,
                        "by_name": self._by_name,
                        "by_extension": self._by_ext,
                    },
                    indent=2,
                )
            )
        except Exception as exc:
            log.warning("FileTypeCache: could not save cache to %s: %s", self._path, exc)


# Module-level singleton — shared across all FileClassifier instances in a process.
_cache: FileTypeCache | None = None


def get_cache(cache_path: Path = _DEFAULT_CACHE_PATH) -> FileTypeCache:
    global _cache
    if _cache is None:
        _cache = FileTypeCache(cache_path)
    return _cache
