from __future__ import annotations

"""Background Google Drive → KAG sync service."""

import asyncio
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

from app.services.document_parser import parse_bytes

if TYPE_CHECKING:
    from app.services.gdrive_loader import GoogleDriveLoader
    from app.services.kag_service import KAGService

_GDRIVE_PREFIX = "gdrive:"
_INDEXED_FILE = Path("/app/ckpt/indexed.json")


def _load_indexed() -> dict[str, str]:
    try:
        return json.loads(_INDEXED_FILE.read_text())
    except Exception:
        return {}


def _save_indexed(indexed: dict[str, str]) -> None:
    _INDEXED_FILE.parent.mkdir(parents=True, exist_ok=True)
    _INDEXED_FILE.write_text(json.dumps(indexed))


# Tracks md5/content hash per source so we skip unchanged files
_indexed: dict[str, str] = _load_indexed()


async def ingest_drive_folder(
    folder_id: str,
    drive: "GoogleDriveLoader",
    kag: "KAGService",
) -> tuple[int, int]:
    """Download all files from a Drive folder and build KAG documents.

    Returns:
        (total_new_docs, total_new_docs)  (same value repeated for API compat)
    """
    loop = asyncio.get_event_loop()

    # List files (blocking I/O, run in executor)
    items = await loop.run_in_executor(None, drive.list_folder, folder_id)

    new_docs = 0
    for item in items:
        file_id = item["id"]
        name = item["name"]
        mime = item["mimeType"]
        md5 = item.get("md5Checksum") or hashlib.md5(
            f"{file_id}{mime}".encode()
        ).hexdigest()
        source = f"{_GDRIVE_PREFIX}{file_id}:{name}"

        if _indexed.get(source) == md5:
            print(f"[sync] unchanged: {name}")
            continue

        print(f"[sync] indexing: {name}")
        drive_file = await loop.run_in_executor(
            None, drive.download, file_id, name, mime
        )
        if drive_file is None:
            continue

        text = parse_bytes(drive_file.name, drive_file.content, drive_file.mime_type)
        if not text.strip():
            continue

        ok = await kag.build_document(text, source)
        if ok:
            _indexed[source] = md5
            _save_indexed(_indexed)
            new_docs += 1

    return new_docs, new_docs


async def periodic_sync(
    folder_id: str,
    drive: "GoogleDriveLoader",
    kag: "KAGService",
    interval_hours: float,
    status_ref: dict,
) -> None:
    """Run ingest_drive_folder on a recurring schedule."""
    interval = interval_hours * 3600
    await asyncio.sleep(interval)
    while True:
        print(f"[sync] periodic sync started (interval={interval_hours}h)")
        status_ref["indexing_status"] = "indexing"
        try:
            n, _ = await ingest_drive_folder(folder_id, drive, kag)
            print(f"[sync] completed: {n} new documents")
        except Exception as exc:
            print(f"[sync] error: {exc}")
        finally:
            status_ref["indexing_status"] = "ready"
        await asyncio.sleep(interval)
