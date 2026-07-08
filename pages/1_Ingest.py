"""Ingest page — local directories, Google Drive, SharePoint."""
from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from typing import Callable

import streamlit as st

from eigenmind.config import ocr_available, sharepoint_credentials
from eigenmind.jobs.runner import JobRunner
from eigenmind.jobs.store import JobStore
from eigenmind.ui.auth import (
    check_password,
    current_user,
    get_user_token_path,
    is_admin,
    list_visible_collections,
    public_qdrant_name_for,
    qdrant_collection_for,
)
from eigenmind.ui.components import (
    empty_state,
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


# ── Module-level singletons (one per server process, shared across sessions) ──

@st.cache_resource
def _job_store() -> JobStore:
    os.makedirs("user_data", exist_ok=True)
    return JobStore(os.path.join("user_data", "jobs.db"))


@st.cache_resource
def _job_runner(qdrant_host: str, qdrant_port: int) -> JobRunner:
    return JobRunner(_job_store(), qdrant_host, qdrant_port)


# ── Page setup ────────────────────────────────────────────────────────

apply_global_styles()
if not check_password():
    st.stop()
sb = render_sidebar()

section_header("/enrich corpus/", "build your corpus")

if not sb.is_connected:
    empty_state("⚡", "Qdrant is offline. Start it with <code>docker-compose up -d</code> and refresh.")
    st.stop()

store = QdrantStore(sb.qdrant_host, sb.qdrant_port)
_visible = list_visible_collections(store.list_collections())
# For ingestion, non-admins can only write to their own private collections.
_editable = _visible if is_admin() else [(l, q) for l, q in _visible if not l.startswith("[public] ")]
existing_col_labels = [label for label, _ in _editable]
existing_col_map = {label: qdrant_name for label, qdrant_name in _editable}

# ── Source & collection selection ─────────────────────────────────────
col_src, col_mode = st.columns(2)
with col_src:
    sources = ["Local Directories"]
    if GDRIVE_AVAILABLE:
        sources.append("Google Drive")
    if SHAREPOINT_AVAILABLE:
        sources.append("SharePoint")
    ingestion_source = st.radio("source", sources, horizontal=True)
with col_mode:
    mode_options = ["Create New", "Select Existing"]
    mode = st.radio("collection mode", mode_options, horizontal=True)

if mode == "Select Existing" and not existing_col_labels:
    st.warning("No existing collections — falling back to **Create New**.")
    mode = "Create New"

if mode == "Select Existing":
    selected_label = st.selectbox("target collection", existing_col_labels, key="add_col_sel")
    qdrant_col_name = existing_col_map[selected_label]
    collection_name = selected_label
    pt_count, vec_size = store.collection_stats(qdrant_col_name)
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
    is_public_new = is_admin() and st.checkbox(
        "Public collection (visible to all users)", value=False, key="is_public_col"
    )
    qdrant_col_name = (
        public_qdrant_name_for(collection_name) if is_public_new
        else qdrant_collection_for(collection_name)
    )

st.markdown("---")
ocr_html = (badge("Hinge", text="✓ OCR active")
            if ocr_available()
            else badge("sim", text="⚠ OCR inactive — check dependencies & TESSDATA_PREFIX"))
st.markdown(f"<p>{ocr_html}</p>", unsafe_allow_html=True)

user_dir = os.path.join("user_data", current_user())
os.makedirs(user_dir, exist_ok=True)
history_file = os.path.join(user_dir, f"history_{qdrant_col_name}.txt")


# ── Job helpers ───────────────────────────────────────────────────────

def _submit_job(dirs_content: bytes) -> str:
    """Persist the dirs file, create a DB record, enqueue it, return job_id."""
    job_id = str(uuid.uuid4())
    job_dir = os.path.join("user_data", current_user(), "jobs")
    os.makedirs(job_dir, exist_ok=True)
    dirs_path = os.path.join(job_dir, f"{job_id}.txt")
    with open(dirs_path, "wb") as f:
        f.write(dirs_content)
    _job_store().create_job(
        job_id=job_id,
        collection=qdrant_col_name,
        directories_file=dirs_path,
        device=sb.selected_device,
        history_file=history_file,
    )
    _job_runner(sb.qdrant_host, sb.qdrant_port).submit(job_id)
    return job_id


def _activate_job(job_id: str) -> None:
    st.session_state["active_job_id"] = job_id
    st.session_state.pop("log_cursor", None)
    st.session_state.pop("log_buffer", None)


# ── Job status fragment (auto-refreshes every 2 s while a job is active) ──

@st.fragment(run_every=2.0)
def _render_job_status() -> None:
    job_id = st.session_state.get("active_job_id")
    if not job_id:
        return
    job = _job_store().get_job(job_id)
    if not job:
        return

    status = job["status"]
    icons = {"pending": "⏳", "running": "⚙️", "done": "✅", "failed": "❌"}
    st.markdown(f"**{icons.get(status, '·')} ingestion job** — `{status}`")

    cur, tot = job["progress_current"], job["progress_total"]
    if tot > 0:
        st.progress(min(cur / tot, 1.0))
        st.caption(f"file {cur} / {tot}")
    elif status == "running":
        st.progress(0)

    # Stream new log lines incrementally
    last_rowid = st.session_state.get("log_cursor", 0)
    new_entries = _job_store().get_logs(job_id, after_rowid=last_rowid)
    if new_entries:
        rowids, messages = zip(*new_entries)
        st.session_state["log_cursor"] = rowids[-1]
        buf = st.session_state.get("log_buffer", "")
        st.session_state["log_buffer"] = buf + "\n".join(messages) + "\n"

    buf = st.session_state.get("log_buffer", "")
    if buf:
        with st.expander("processing log", expanded=(status == "running")):
            st.code(buf.strip(), language=None)

    if status == "done":
        st.success("Ingestion complete!")
    elif status == "failed":
        st.error("Ingestion failed — see log above.")


# Reconnect banner: if a job is already running for this collection but this
# session has no active_job_id (e.g. user reconnected after a tab closure),
# offer to reattach to the running job.
if "active_job_id" not in st.session_state:
    latest = _job_store().latest_job_for(qdrant_col_name)
    if latest and latest["status"] in ("running", "pending"):
        st.info(
            f"An ingestion is already running for **{collection_name}** "
            f"(started {latest['created_at'][:16]})."
        )
        if st.button("monitor this job"):
            _activate_job(latest["id"])
            st.rerun()

_render_job_status()

st.markdown("---")


# ── Remote download helper ────────────────────────────────────────────

def _remote_download_and_ingest(downloader: Callable[[str], None]) -> None:
    """Download from a remote source (sync, with live feedback), then submit ingestion as a background job."""
    user_corpus = os.path.join("downloaded_corpus", current_user())
    os.makedirs(user_corpus, exist_ok=True)
    download_dir = os.path.join(user_corpus, collection_name)

    downloader(download_dir)

    job_id = _submit_job(download_dir.encode("utf-8"))
    _activate_job(job_id)
    st.info("Download complete — ingestion is running in the background.")
    st.rerun()


# ══ LOCAL DIRECTORIES ══════════════════════════════════════════════════
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
                try:
                    job_id = _submit_job(uploaded.getvalue())
                    _activate_job(job_id)
                    st.rerun()
                except Exception as e:  # noqa: BLE001
                    st.error(f"Failed to submit job: {e}")
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
                upload_dir = os.path.join("user_data", current_user(), "uploads", collection_name)
                os.makedirs(upload_dir, exist_ok=True)
                try:
                    for uf in uploaded_files:
                        with open(os.path.join(upload_dir, uf.name), "wb") as fh:
                            fh.write(uf.getvalue())
                    job_id = _submit_job(upload_dir.encode("utf-8"))
                    _activate_job(job_id)
                    st.rerun()
                except Exception as e:  # noqa: BLE001
                    st.error(f"Failed to submit job: {e}")
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
                    try:
                        job_id = _submit_job(folder_path.encode("utf-8"))
                        _activate_job(job_id)
                        st.rerun()
                    except Exception as e:  # noqa: BLE001
                        st.error(f"Failed to submit job: {e}")
            else:
                st.warning("Provide a folder path and a collection name.")

    if os.path.exists(history_file):
        st.markdown("---")
        st.markdown(f"**Found progress history for '{collection_name}'**")
        if st.button("↺ resume embedding from history", type="secondary"):
            try:
                with open(history_file, "rb") as f:
                    job_id = _submit_job(f.read())
                _activate_job(job_id)
                st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(f"Failed to submit resume job: {e}")

# ══ GOOGLE DRIVE ═══════════════════════════════════════════════════════
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
            info_box("Share the Drive folder with your service account email (Viewer).")

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

# ══ SHAREPOINT ═════════════════════════════════════════════════════════
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
