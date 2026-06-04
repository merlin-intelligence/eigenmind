"""Ingest page — local directories, Google Drive, SharePoint."""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from typing import Callable

import streamlit as st

from eigenmind.config import ocr_available, sharepoint_credentials
from eigenmind.pipelines.ingest import Ingester
from eigenmind.ui.auth import (
    check_password,
    current_user,
    display_name_from,
    get_user_token_path,
    qdrant_collection_for,
)
from eigenmind.ui.components import (
    empty_state,
    get_embedder,
    info_box,
    metric_card,
    render_sidebar,
    section_header,
)
from eigenmind.ui.styles import apply_global_styles, badge
from eigenmind.vectordb.store import QdrantStore

try:
    from eigenmind.connectors.gdrive import GDriveClient
    from google.auth.exceptions import RefreshError
    GDRIVE_AVAILABLE = True
except ImportError:
    GDRIVE_AVAILABLE = False

    class RefreshError(Exception):
        pass

try:
    from eigenmind.connectors.sharepoint import SharePointClient
    SHAREPOINT_AVAILABLE = True
except ImportError:
    SHAREPOINT_AVAILABLE = False


apply_global_styles()
if not check_password():
    st.stop()
sb = render_sidebar()

section_header("/enrich corpus/", "build your corpus")

if not sb.is_connected:
    empty_state("⚡", "Qdrant is offline. Start it with <code>docker-compose up -d</code> and refresh.")
    st.stop()

store = QdrantStore(sb.qdrant_host, sb.qdrant_port)
existing_cols = sorted(c for c in (display_name_from(c) for c in store.list_collections()) if c)

# ── Source & collection selection ──
col_src, col_mode = st.columns(2)
with col_src:
    sources = ["Local Directories"]
    if GDRIVE_AVAILABLE: sources.append("Google Drive")
    if SHAREPOINT_AVAILABLE: sources.append("SharePoint")
    ingestion_source = st.radio("source", sources, horizontal=True)
with col_mode:
    mode_options = ["Create New", "Select Existing"]
    mode = st.radio("collection mode", mode_options, horizontal=True)

if mode == "Select Existing" and not existing_cols:
    st.warning("No existing collections for this user — falling back to **Create New**.")
    mode = "Create New"

if mode == "Select Existing":
    collection_name = st.selectbox("target collection", existing_cols, key="add_col_sel")
    pt_count, vec_size = store.collection_stats(qdrant_collection_for(collection_name))
    if pt_count is not None:
        c1, c2, c3 = st.columns(3)
        c1.markdown(metric_card("vectors stored", f"{pt_count:,}", "in this collection"),
                    unsafe_allow_html=True)
        c2.markdown(metric_card("vector dims", str(vec_size), "intfloat/multilingual-e5-base"),
                    unsafe_allow_html=True)
        c3.markdown(metric_card("index status", "active", "cosine distance",
                                value_style="font-size:1.3rem;color:#3d7a4a"),
                    unsafe_allow_html=True)
else:
    collection_name = st.text_input("new collection name", "eigenmind_collection")

st.markdown("---")
ocr_html = (badge("Hinge", text="✓ OCR active")
            if ocr_available()
            else badge("sim", text="⚠ OCR inactive — check dependencies & TESSDATA_PREFIX"))
st.markdown(f"<p>{ocr_html}</p>", unsafe_allow_html=True)

user_dir = os.path.join("user_data", current_user())
os.makedirs(user_dir, exist_ok=True)
history_file = os.path.join(user_dir, f"history_{collection_name}.txt")


def _run_ingest(directories_path: str, *, label: str = "embedding") -> None:
    if not QdrantStore.is_reachable(sb.qdrant_host, sb.qdrant_port):
        st.error("Qdrant is no longer reachable. Check the service and retry.")
        return
    st.info(f"{label.capitalize()} into **{collection_name}**…")
    progress = st.progress(0)
    progress_text = st.empty()

    def cb(cur: int, tot: int) -> None:
        if tot > 0:
            progress.progress(min(cur / tot, 1.0))
            progress_text.markdown(
                f'<p style="font-family:\'DM Mono\',monospace;font-size:0.75rem;color:#8a6a50">'
                f'file {cur} / {tot}</p>',
                unsafe_allow_html=True,
            )

    ingester = Ingester(store=store, device=sb.selected_device,
                        progress_callback=cb, history_file=history_file,
                        embedder=get_embedder(sb.selected_device))
    with st.spinner("vectorising your documents…"):
        logs = ingester.run_chunknorris(directories_path, qdrant_collection_for(collection_name))
    st.success(f"✓ {label.capitalize()} complete!")
    with st.expander("processing log", expanded=False):
        for line in logs:
            st.text(line)


def _remote_download_and_ingest(downloader: Callable[[str], None]) -> None:
    """Run a remote-source download into the user corpus, then ingest the resulting directory.

    ``downloader`` is called with the absolute path to the destination directory.
    """
    user_corpus = os.path.join("downloaded_corpus", current_user())
    os.makedirs(user_corpus, exist_ok=True)
    download_dir = os.path.join(user_corpus, collection_name)

    downloader(download_dir)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", encoding="utf-8", delete=False) as f:
        f.write(download_dir)
        dirs_txt = f.name
    try:
        _run_ingest(dirs_txt)
    finally:
        os.remove(dirs_txt)


# ══ LOCAL DIRECTORIES ══
if ingestion_source == "Local Directories":
    _SUPPORTED_EXTS = ["pdf", "docx", "xlsx", "csv", "pptx", "txt", "md"]
    _FORMATS_NOTE = "Supported formats: PDF, DOCX, XLSX, CSV, PPTX, TXT, MD."

    local_mode = st.radio(
        "input mode",
        ["Paths file (.txt)", "Upload files", "Folder path"],
        horizontal=True,
        key="local_mode",
    )

    if local_mode == "Paths file (.txt)":
        info_box(
            "Upload a <code>.txt</code> file with one directory path per line. "
            f"{_FORMATS_NOTE} "
            "Duplicate files already in the collection are automatically skipped."
        )
        uploaded = st.file_uploader("directories.txt", type=["txt"])
        if st.button("▶ start embedding", type="primary"):
            if uploaded is not None and collection_name:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="wb") as tmp:
                    tmp.write(uploaded.getvalue())
                    tmp_path = tmp.name
                try:
                    _run_ingest(tmp_path)
                except Exception as e:  # noqa: BLE001
                    st.error(f"Error during indexing: {e}")
                    st.exception(e)
                finally:
                    os.remove(tmp_path)
            else:
                st.warning("Provide both a directories.txt file and a collection name.")

    elif local_mode == "Upload files":
        info_box(
            f"Drop one or more files to index directly. {_FORMATS_NOTE} "
            "Duplicate files already in the collection are automatically skipped."
        )
        uploaded_files = st.file_uploader(
            "files",
            type=_SUPPORTED_EXTS,
            accept_multiple_files=True,
        )
        if st.button("▶ start embedding", type="primary"):
            if uploaded_files and collection_name:
                tmp_dir = tempfile.mkdtemp()
                try:
                    for uf in uploaded_files:
                        with open(os.path.join(tmp_dir, uf.name), "wb") as fh:
                            fh.write(uf.getvalue())
                    with tempfile.NamedTemporaryFile(
                        mode="w", suffix=".txt", delete=False, encoding="utf-8"
                    ) as tf:
                        tf.write(tmp_dir)
                        dirs_txt = tf.name
                    try:
                        _run_ingest(dirs_txt)
                    except Exception as e:  # noqa: BLE001
                        st.error(f"Error during indexing: {e}")
                        st.exception(e)
                    finally:
                        os.remove(dirs_txt)
                finally:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
            else:
                st.warning("Drop at least one file and provide a collection name.")

    elif local_mode == "Folder path":
        info_box(
            f"Enter the path to a local folder. All supported files inside will be indexed recursively. {_FORMATS_NOTE} "
            "Duplicate files already in the collection are automatically skipped."
        )
        folder_path = st.text_input("folder path", placeholder="/path/to/your/documents")
        if st.button("▶ start embedding", type="primary"):
            if folder_path and collection_name:
                if not os.path.isdir(folder_path):
                    st.error(f"'{folder_path}' is not a valid directory.")
                else:
                    with tempfile.NamedTemporaryFile(
                        mode="w", suffix=".txt", delete=False, encoding="utf-8"
                    ) as tf:
                        tf.write(folder_path)
                        dirs_txt = tf.name
                    try:
                        _run_ingest(dirs_txt)
                    except Exception as e:  # noqa: BLE001
                        st.error(f"Error during indexing: {e}")
                        st.exception(e)
                    finally:
                        os.remove(dirs_txt)
            else:
                st.warning("Provide a folder path and a collection name.")

    if os.path.exists(history_file):
        st.markdown("---")
        st.markdown(f"**Found progress history for '{collection_name}'**")
        if st.button("↺ resume embedding from history", type="secondary"):
            try:
                _run_ingest(history_file, label="resume")
            except Exception as e:  # noqa: BLE001
                st.error(f"Error during resume: {e}")
                st.exception(e)

# ══ GOOGLE DRIVE ══
elif ingestion_source == "Google Drive":
    if not GDRIVE_AVAILABLE:
        st.error("Missing Google Drive libraries.")
        st.code("pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")
    else:
        gdrive_auth_mode = st.radio(
            "auth mode",
            ["Local (OAuth Client ID)", "Remote VM (Service Account)"],
            horizontal=True,
        )
        gdrive_folder_id = st.text_input(
            "Google Drive Folder ID",
            help="Paste the folder URL or just the ID from it.",
        )

        if gdrive_auth_mode == "Local (OAuth Client ID)":
            secrets_file = st.file_uploader("client_secrets.json", type=["json"])
            if st.button("↺ reset credentials"):
                st.session_state.pop("gdrive_client", None)
                if os.path.exists(get_user_token_path()):
                    os.remove(get_user_token_path())
                st.success("Credentials cleared.")
                st.rerun()
        else:
            secrets_file = st.file_uploader("service_account.json", type=["json"])
            info_box(
                "Share the Drive folder with your service account email (Viewer)."
            )

        if st.button("▶ download & embed", type="primary"):
            m = re.search(r"folders/([a-zA-Z0-9_-]+)", gdrive_folder_id or "")
            if m:
                gdrive_folder_id = m.group(1)

            if gdrive_folder_id and secrets_file and collection_name:
                secrets_content = secrets_file.getvalue()
                with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
                    f.write(secrets_content)
                    secrets_path = f.name

                try:
                    js_data = json.loads(secrets_content.decode("utf-8"))
                    is_service_account = js_data.get("type") == "service_account"
                    gdrive: GDriveClient | None = st.session_state.get("gdrive_client")

                    if not gdrive or is_service_account:
                        st.info("Authenticating with Google Drive…")
                        if is_service_account:
                            st.write("✨ Service Account detected.")
                            gdrive = GDriveClient.from_service_account(secrets_path)
                        else:
                            st.write("✨ OAuth Client ID detected.")
                            gdrive = GDriveClient.from_oauth(secrets_path, token_path=get_user_token_path())
                            st.session_state["gdrive_client"] = gdrive

                    dl_log = st.empty()

                    def _download(dest: str) -> None:
                        gdrive.download_folder(gdrive_folder_id, dest,
                                               log_callback=lambda msg: dl_log.text(msg))

                    _remote_download_and_ingest(_download)
                except RefreshError:
                    st.error("Token expired. Clearing credentials — click the button again.")
                    st.session_state.pop("gdrive_client", None)
                    if os.path.exists(get_user_token_path()):
                        os.remove(get_user_token_path())
                except Exception as e:  # noqa: BLE001
                    st.error(f"Error: {e}")
                    st.exception(e)
                finally:
                    if os.path.exists(secrets_path):
                        os.remove(secrets_path)
            else:
                st.warning("Provide folder ID, secrets file, and collection name.")

# ══ SHAREPOINT ══
elif ingestion_source == "SharePoint":
    if not SHAREPOINT_AVAILABLE:
        st.error("Missing SharePoint library.")
        st.code("pip install Office365-REST-Python-Client")
    else:
        site_url = st.text_input("SharePoint Site URL")
        folder_url = st.text_input("Folder Server Relative URL", "/sites/YourSite/Shared Documents/MyFolder")

        default_id, default_secret = sharepoint_credentials()
        col_a, col_b = st.columns(2)
        with col_a:
            client_id = st.text_input("Client ID", value=default_id)
        with col_b:
            client_secret = st.text_input("Client Secret", value=default_secret, type="password")

        if st.button("▶ download & embed", type="primary"):
            if all([site_url, folder_url, client_id, client_secret, collection_name]):
                try:
                    st.info("Authenticating with SharePoint…")
                    sp = SharePointClient.from_credentials(site_url, client_id, client_secret)
                    sp_log = st.empty()

                    def _download(dest: str) -> None:
                        sp.download_folder(folder_url, dest,
                                           log_callback=lambda msg: sp_log.text(msg))

                    _remote_download_and_ingest(_download)
                except Exception as e:  # noqa: BLE001
                    st.error(f"Error: {e}")
                    st.exception(e)
            else:
                st.warning("Fill in all SharePoint fields.")
