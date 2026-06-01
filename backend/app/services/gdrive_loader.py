from __future__ import annotations

import io
from dataclasses import dataclass

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload


@dataclass
class DriveFile:
    file_id: str
    name: str
    mime_type: str
    content: bytes
    md5: str


class GoogleDriveLoader:
    def __init__(self, service_account_json: str, scopes: list[str]) -> None:
        self._sa_json = service_account_json
        self._scopes = scopes

    # ── public ──────────────────────────────────────────────────────────────

    def list_folder(self, folder_id: str) -> list[dict]:
        svc = self._build()
        return self._list_files(svc, folder_id)

    def download(self, file_id: str, name: str, mime_type: str) -> DriveFile | None:
        svc = self._build()
        return self._fetch(svc, file_id, name, mime_type)

    # ── private ─────────────────────────────────────────────────────────────

    def _build(self):
        creds = service_account.Credentials.from_service_account_file(
            self._sa_json, scopes=self._scopes
        )
        return build("drive", "v3", credentials=creds, cache_discovery=False)

    @staticmethod
    def _list_files(svc, folder_id: str) -> list[dict]:
        q = f"'{folder_id}' in parents and trashed=false"
        page_token = None
        files: list[dict] = []
        while True:
            resp = (
                svc.files()
                .list(
                    q=q,
                    fields="nextPageToken, files(id, name, mimeType, md5Checksum, modifiedTime)",
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                    pageToken=page_token,
                )
                .execute()
            )
            files.extend(resp.get("files", []))
            page_token = resp.get("nextPageToken")
            if not page_token:
                break
        return files

    def _fetch(self, svc, file_id: str, name: str, mime_type: str) -> DriveFile | None:
        req, out_mime = self._build_request(svc, file_id, mime_type)
        if req is None:
            return None

        stream = io.BytesIO()
        dl = MediaIoBaseDownload(stream, req)
        done = False
        while not done:
            _, done = dl.next_chunk()

        return DriveFile(
            file_id=file_id,
            name=name,
            mime_type=out_mime,
            content=stream.getvalue(),
            md5="",
        )

    @staticmethod
    def _build_request(svc, file_id: str, mime_type: str):
        if mime_type == "application/vnd.google-apps.document":
            return svc.files().export_media(fileId=file_id, mimeType="text/plain"), "text/plain"
        if mime_type == "application/vnd.google-apps.spreadsheet":
            return svc.files().export_media(fileId=file_id, mimeType="text/csv"), "text/csv"
        if mime_type.startswith("application/vnd.google-apps"):
            return None, mime_type
        return svc.files().get_media(fileId=file_id), mime_type
