"""Multi-user authentication backed by a JSON DB on disk."""
from __future__ import annotations

import hashlib
import json
import os

import streamlit as st

USER_DB_PATH = "user_data/users.json"
PUBLIC_PREFIX = "public_"


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def _is_v2(db: dict) -> bool:
    return "users" in db and isinstance(db.get("users"), dict)


def load_user_db() -> dict:
    """Load the user DB. Returns v2 format: {"admins": [...], "users": {username: hash}}."""
    if not os.path.exists(USER_DB_PATH):
        os.makedirs("user_data", exist_ok=True)
        initial: dict = {"admins": [], "users": {}}
        try:
            if "USERS" in st.secrets:
                initial["users"] = {u: hash_password(p) for u, p in st.secrets["USERS"].items()}
            elif "APP_PASSWORD" in st.secrets:
                initial["users"] = {
                    st.secrets["APP_USERNAME"].strip():
                        hash_password(st.secrets["APP_PASSWORD"].strip())
                }
        except Exception:
            pass
        with open(USER_DB_PATH, "w") as f:
            json.dump(initial, f, indent=2)
        return initial
    try:
        with open(USER_DB_PATH, "r") as f:
            db = json.load(f)
        if _is_v2(db):
            return db
        # Migrate v1 flat {user: hash} to v2
        return {"admins": [], "users": db}
    except Exception:
        return {"admins": [], "users": {}}


def save_user_db(db: dict) -> None:
    with open(USER_DB_PATH, "w") as f:
        json.dump(db, f, indent=2)


def check_password() -> bool:
    """Render the login UI and return True iff the user is authenticated."""
    db = load_user_db()
    users = db["users"]
    if not users:
        return True

    if not st.session_state.get("password_correct", False):
        _, mid, _ = st.columns([1, 1.2, 1])
        with mid:
            st.markdown("""
                <div style="text-align: center; margin-top: 5rem; margin-bottom: 2rem;">
                    <h1 style="font-family:'Playfair Display',serif; font-weight:900;
                               font-size:3rem; color:#2a1f18; margin-bottom:0;">eigenmind</h1>
                    <div style="font-family:'DM Mono',monospace; letter-spacing:0.2em;
                                color:#c44a28; font-size:0.8rem; text-transform:uppercase;">
                        accelerate clarity
                    </div>
                </div>
            """, unsafe_allow_html=True)
            with st.container(border=True):
                with st.form("login_form", clear_on_submit=False):
                    user = st.text_input("identity", key="login_username", placeholder="username")
                    pwd = st.text_input("credential", type="password",
                                        key="login_password", placeholder="password")
                    submitted = st.form_submit_button("sign in", use_container_width=True)
                if submitted:
                    user = (user or "").strip()
                    pwd = (pwd or "").strip()
                    if user in users and users[user] == hash_password(pwd):
                        st.session_state["password_correct"] = True
                        st.session_state["authenticated_user"] = user
                        st.session_state.pop("login_password", None)
                        st.session_state.pop("login_username", None)
                        st.rerun()
                    else:
                        st.session_state["password_correct"] = False
                        st.error("😕 Authentication failed. Access denied.")
                st.markdown("""
                    <div style="margin-top: 2rem; font-family:'DM Mono',monospace;
                                font-size:0.6rem; color:#8a6a50; text-align:center; opacity:0.6;">
                        SECURE ACCESS POINT · PX-V EIGENMIND v2.1 (Multi-User)
                    </div>
                """, unsafe_allow_html=True)
        return False
    return True


def current_user() -> str:
    return st.session_state.get("authenticated_user", "guest")


def is_admin() -> bool:
    """Return True if the current user has admin privileges."""
    db = load_user_db()
    return current_user() in db.get("admins", [])


def get_user_token_path() -> str:
    """Per-user file used to cache OAuth tokens."""
    user_dir = os.path.join("user_data", current_user())
    os.makedirs(user_dir, exist_ok=True)
    return os.path.join(user_dir, "gdrive_token.json")


def qdrant_collection_for(display_name: str) -> str:
    """Namespace a display name with the current user (private collection)."""
    return f"{current_user()}_{display_name}"


def public_qdrant_name_for(display_name: str) -> str:
    """Return the Qdrant collection name for a public collection."""
    return f"{PUBLIC_PREFIX}{display_name}"


def display_name_from(qdrant_name: str) -> str | None:
    """Return the display name if this collection belongs to the current user, else None."""
    prefix = f"{current_user()}_"
    if qdrant_name.startswith(prefix):
        return qdrant_name[len(prefix):]
    return None


def list_visible_collections(all_qdrant: list[str]) -> list[tuple[str, str]]:
    """Return (display_label, qdrant_name) for collections visible to the current user.

    Includes the user's own private collections and all public collections.
    Public collections are labeled with a '[public] ' prefix in the display label.
    """
    user = current_user()
    result: list[tuple[str, str]] = []
    for name in all_qdrant:
        if name.startswith(f"{user}_"):
            result.append((name[len(f"{user}_"):], name))
        elif name.startswith(PUBLIC_PREFIX):
            result.append((f"[public] {name[len(PUBLIC_PREFIX):]}", name))
    return sorted(result)
