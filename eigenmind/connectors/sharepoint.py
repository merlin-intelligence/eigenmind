"""SharePoint download — App-only credential flow."""
from __future__ import annotations

import os
import platform

from office365.runtime.auth.client_credential import ClientCredential
from office365.sharepoint.client_context import ClientContext

from eigenmind.config import CHUNKNORRIS_EXTENSIONS

SUPPORTED_EXTENSIONS = CHUNKNORRIS_EXTENSIONS
SKIP_FOLDERS = {"Forms", "Attachments"}


def _normalize_destination(destination: str) -> str:
    if platform.system() == "Windows" and not destination.startswith("\\\\?\\"):
        return "\\\\?\\" + os.path.abspath(destination)
    return destination


class SharePointClient:
    """SharePoint client authenticated with App-only credentials."""

    def __init__(self, ctx: ClientContext):
        self._ctx = ctx

    @classmethod
    def from_credentials(cls, site_url: str, client_id: str, client_secret: str) -> "SharePointClient":
        creds = ClientCredential(client_id, client_secret)
        return cls(ClientContext(site_url).with_credentials(creds))

    def download_folder(self, folder_url: str, destination: str, log_callback=None) -> None:
        """Recursively download supported files from a folder by its server-relative URL."""
        destination = _normalize_destination(destination)
        os.makedirs(destination, exist_ok=True)

        if log_callback:
            log_callback(f"Accessing SharePoint folder: {folder_url}")

        try:
            folder = self._ctx.web.get_folder_by_server_relative_url(folder_url)
            self._ctx.load(folder, ["Files", "Folders"])
            self._ctx.execute_query()
        except Exception as e:  # noqa: BLE001
            if log_callback:
                log_callback(f"Error accessing folder '{folder_url}': {e}. Check URL and permissions.")
            return

        for file in folder.files:
            if not any(file.name.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                if log_callback:
                    log_callback(f"Skipping unsupported file type: {file.name}")
                continue
            file_path = os.path.join(destination, file.name)
            if log_callback:
                log_callback(f"Downloading SharePoint file: {file.name}")
            try:
                with open(file_path, "wb") as local_file:
                    file.download(local_file).execute_query()
            except Exception as e:  # noqa: BLE001
                if log_callback:
                    log_callback(f"Error downloading file '{file.name}': {e}")

        for sub in folder.folders:
            if sub.name not in SKIP_FOLDERS:
                self.download_folder(sub.server_relative_url, os.path.join(destination, sub.name), log_callback)
