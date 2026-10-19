"""
Login / Register for the RetailPulse dashboard.

Users are stored in a local SQLite file (data/users.db). Passwords are never
stored in plain text: each one is salted and hashed with PBKDF2-HMAC-SHA256.
After 5 wrong passwords an account is locked for 5 minutes.

Use in every dashboard page (first lines, right after set_page_config):

    from dashboard.auth import require_login
    require_login()          # shows the login/register screen and stops the page if not signed in

The storage functions (init_db, register_user, authenticate) have no Streamlit
dependency, so they can be unit-tested on their own.
"""

import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import time
from contextlib import closing

DB_PATH = os.environ.get("RETAILPULSE_USERS_DB", "data/users.db")

PBKDF2_ITERATIONS = 200_000
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_SECONDS = 300

USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,30}$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class AuthError(Exception):
    """Raised with a user-friendly message when registration or login fails."""


# --------------------------------------------------------------------------
# Storage layer (no Streamlit)
# --------------------------------------------------------------------------
def _connect(db_path: str = None) -> sqlite3.Connection:
    path = db_path or DB_PATH
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = None) -> None:
    with closing(_connect(db_path)) as conn, conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                username        TEXT NOT NULL UNIQUE COLLATE NOCASE,
                email           TEXT NOT NULL UNIQUE COLLATE NOCASE,
                full_name       TEXT NOT NULL DEFAULT '',
                password_hash   TEXT NOT NULL,
                salt            TEXT NOT NULL,
                failed_attempts INTEGER NOT NULL DEFAULT 0,
                locked_until    REAL NOT NULL DEFAULT 0,
                created_at      REAL NOT NULL
            )
            """
        )


def hash_password(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS).hex()


def validate_registration(username: str, email: str, password: str, confirm: str) -> None:
    if not USERNAME_RE.match(username or ""):
        raise AuthError("Username must be 3-30 characters: letters, numbers, dot, dash or underscore.")
    if not EMAIL_RE.match(email or ""):
        raise AuthError("Please enter a valid email address.")
    if len(password or "") < 8:
        raise AuthError("Password must be at least 8 characters.")
    if not (re.search(r"[A-Za-z]", password) and re.search(r"\d", password)):
        raise AuthError("Password must contain at least one letter and one number.")
    if password != confirm:
        raise AuthError("Passwords do not match.")


def register_user(username: str, email: str, password: str, confirm: str, full_name: str = "", db_path: str = None) -> dict:
    """Create an account. Raises AuthError (with a readable message) on any problem."""
    username, email, full_name = (username or "").strip(), (email or "").strip(), (full_name or "").strip()
    validate_registration(username, email, password, confirm)
    init_db(db_path)

    salt = secrets.token_bytes(16)
    pw_hash = hash_password(password, salt)
    try:
        with closing(_connect(db_path)) as conn, conn:
            conn.execute(
                "INSERT INTO users (username, email, full_name, password_hash, salt, created_at) VALUES (?,?,?,?,?,?)",
                (username, email, full_name, pw_hash, salt.hex(), time.time()),
            )
    except sqlite3.IntegrityError:
        raise AuthError("That username or email is already registered.")
    return {"username": username, "email": email, "full_name": full_name}


def authenticate(identifier: str, password: str, db_path: str = None) -> dict:
    """
    Check a login (username OR email + password). Returns the user dict on success,
    raises AuthError otherwise. The same message is used for "no such user" and
    "wrong password" so the form doesn't reveal which accounts exist.
    """
    init_db(db_path)
    identifier = (identifier or "").strip()
    generic = AuthError("Incorrect username/email or password.")

    with closing(_connect(db_path)) as conn, conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ? OR email = ?", (identifier, identifier)
        ).fetchone()
        if row is None:
            hash_password(password or "", b"\x00" * 16)  # keep timing similar for unknown users
            raise generic

        now = time.time()
        if row["locked_until"] > now:
            minutes = int((row["locked_until"] - now) // 60) + 1
            raise AuthError(f"Too many failed attempts. Try again in about {minutes} minute(s).")

        expected = hash_password(password or "", bytes.fromhex(row["salt"]))
        if not hmac.compare_digest(expected, row["password_hash"]):
            attempts = row["failed_attempts"] + 1
            locked_until = now + LOCKOUT_SECONDS if attempts >= MAX_FAILED_ATTEMPTS else 0
            if locked_until:
                attempts = 0
            conn.execute(
                "UPDATE users SET failed_attempts = ?, locked_until = ? WHERE id = ?",
                (attempts, locked_until, row["id"]),
            )
            conn.commit()  # persist the counter: raising inside `with conn` would roll it back
            raise generic

        conn.execute("UPDATE users SET failed_attempts = 0, locked_until = 0 WHERE id = ?", (row["id"],))
        return {"username": row["username"], "email": row["email"], "full_name": row["full_name"]}


# --------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------
def current_user():
    import streamlit as st

    return st.session_state.get("user")


def logout():
    import streamlit as st

    st.session_state.pop("user", None)
    st.rerun()


def _login_form(st):
    with st.form("login_form"):
        identifier = st.text_input("Username or email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in", width="stretch")
    if submitted:
        try:
            st.session_state["user"] = authenticate(identifier, password)
            st.rerun()
        except AuthError as e:
            st.error(str(e))


def _register_form(st):
    with st.form("register_form"):
        full_name = st.text_input("Full name (optional)")
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password", help="At least 8 characters, with a letter and a number.")
        confirm = st.text_input("Confirm password", type="password")
        submitted = st.form_submit_button("Create account", width="stretch")
    if submitted:
        try:
            user = register_user(username, email, password, confirm, full_name)
            st.session_state["user"] = user  # sign the new user straight in
            st.rerun()
        except AuthError as e:
            st.error(str(e))


def render_auth_page():
    import streamlit as st

    st.markdown("<h1 style='text-align:center'>📊 RetailPulse</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center;color:#5b6b8c'>Sign in to open the analytics dashboard</p>",
        unsafe_allow_html=True,
    )
    _, middle, _ = st.columns([1, 2, 1])
    with middle:
        login_tab, register_tab = st.tabs(["Log in", "Register"])
        with login_tab:
            _login_form(st)
        with register_tab:
            _register_form(st)


def require_login():
    """Call at the top of every page. Shows login/register if signed out, then stops the page."""
    import streamlit as st

    if current_user() is None:
        render_auth_page()
        st.stop()

    user = current_user()
    with st.sidebar:
        st.markdown(f"**👤 {user['full_name'] or user['username']}**")
        st.caption(user["email"])
        if st.button("Log out", width="stretch"):
            logout()
