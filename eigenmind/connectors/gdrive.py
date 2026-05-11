"""Google Drive download — OAuth and Service-Account flows."""
from __future__ import annotations

import io
import os
import platform

from google.auth.transport.requests import Request
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
SUPPORTED_MIMETYPES = (
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",   # .docx
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",  # .pptx
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",          # .xlsx
    "text/plain",
)


def _sanitize_filename(name: str) -> str:
    return "".join(c if c not in '<>:"/\\|?*' else "_" for c in name)


def _normalize_destination(destination: str) -> str:
    if platform.system() == "Windows" and not destination.startswith("\\\\?\\"):
        return "\\\\?\\" + os.path.abspath(destination)
    return destination


class GDriveClient:
    """Authenticated Google Drive client. Construct via the alt constructors."""

    def __init__(self, service):
        self._service = service

    @classmethod
    def from_oauth(cls, client_secrets_path: str, token_path: str) -> "GDriveClient":
        """Authenticate via OAuth Client ID, caching the token at ``token_path``."""
        creds = None
        if os.path.exists(token_path):
            creds = Credentials.from_authorized_user_file(token_path, SCOPES)

        if not creds or not creds.valid or not creds.refresh_token:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except Exception:
                    creds = None

            if not creds or not creds.valid or not creds.refresh_token:
                flow = InstalledAppFlow.from_client_secrets_file(client_secrets_path, SCOPES)
                auth_kwargs = {"prompt": "consent", "access_type": "offline"}
                try:
                    creds = flow.run_local_server(port=8080, **auth_kwargs)
                except OSError as e:
                    if getattr(e, "winerror", 0) == 10048 or getattr(e, "errno", 0) == 98:
                        creds = flow.run_local_server(port=8081, **auth_kwargs)
                    else:
                        raise

            with open(token_path, "w") as token:
                token.write(creds.to_json())

        return cls(build("drive", "v3", credentials=creds))

    @classmethod
    def from_service_account(cls, service_account_path: str) -> "GDriveClient":
        """Authenticate via a Service Account JSON file."""
        creds = service_account.Credentials.from_service_account_file(service_account_path, scopes=SCOPES)
        return cls(build("drive", "v3", credentials=creds))

    def download_folder(self, folder_id: str, destination: str, log_callback=None) -> None:
        """Recursively download all supported files from a Drive folder."""
        destination = _normalize_destination(destination)
        os.makedirs(destination, exist_ok=True)

        query = f"'{folder_id}' in parents and trashed = false"
        items = self._service.files().list(
            q=query, pageSize=1000, fields="nextPageToken, files(id, name, mimeType)"
        ).execute().get("files", [])

        if not items:
            if log_callback:
                log_callback(f"No files found in folder ID {folder_id}.")
            return

        for item in items:
            name = _sanitize_filename(item["name"])
            if item["mimeType"] == "application/vnd.google-apps.folder":
                self.download_folder(item["id"], os.path.join(destination, name), log_callback)
            elif item["mimeType"] in SUPPORTED_MIMETYPES:
                file_path = os.path.join(destination, name)
                if log_callback:
                    log_callback(f"Downloading GDrive file: {name}")
                request = self._service.files().get_media(fileId=item["id"])
                with io.FileIO(file_path, "wb") as fh:
                    downloader = MediaIoBaseDownload(fh, request)
                    done = False
                    while not done:
                        _, done = downloader.next_chunk()
