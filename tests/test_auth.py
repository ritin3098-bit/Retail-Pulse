"""Tests for dashboard/auth.py storage logic (no Streamlit needed)."""

import os
import sqlite3
import tempfile

from dashboard import auth
from dashboard.auth import AuthError, register_user, authenticate


def _db():
    return os.path.join(tempfile.mkdtemp(), "users.db")


def _expect_error(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except AuthError as e:
        return str(e)
    raise AssertionError("expected AuthError")


def test_register_then_login_by_username_and_email():
    db = _db()
    register_user("alice", "alice@example.com", "secret123", "secret123", "Alice A", db_path=db)
    assert authenticate("alice", "secret123", db_path=db)["email"] == "alice@example.com"
    assert authenticate("ALICE@example.com", "secret123", db_path=db)["username"] == "alice"


def test_password_is_not_stored_in_plain_text():
    db = _db()
    register_user("bob", "bob@example.com", "secret123", "secret123", db_path=db)
    row = sqlite3.connect(db).execute("SELECT password_hash, salt FROM users").fetchone()
    assert "secret123" not in row[0] and len(row[0]) == 64 and len(row[1]) == 32


def test_duplicate_username_or_email_rejected():
    db = _db()
    register_user("carol", "carol@example.com", "secret123", "secret123", db_path=db)
    assert "already registered" in _expect_error(
        register_user, "Carol", "other@example.com", "secret123", "secret123", db_path=db
    )
    assert "already registered" in _expect_error(
        register_user, "carol2", "carol@example.com", "secret123", "secret123", db_path=db
    )


def test_registration_validation():
    db = _db()
    assert "Username" in _expect_error(register_user, "a!", "a@b.com", "secret123", "secret123", db_path=db)
    assert "email" in _expect_error(register_user, "dave", "not-an-email", "secret123", "secret123", db_path=db)
    assert "8 characters" in _expect_error(register_user, "dave", "d@b.com", "abc1", "abc1", db_path=db)
    assert "letter and one number" in _expect_error(register_user, "dave", "d@b.com", "abcdefgh", "abcdefgh", db_path=db)
    assert "do not match" in _expect_error(register_user, "dave", "d@b.com", "secret123", "secret124", db_path=db)


def test_wrong_password_and_unknown_user_share_one_message():
    db = _db()
    register_user("erin", "erin@example.com", "secret123", "secret123", db_path=db)
    m1 = _expect_error(authenticate, "erin", "wrongpass1", db_path=db)
    m2 = _expect_error(authenticate, "nobody", "secret123", db_path=db)
    assert m1 == m2


def test_account_locks_after_repeated_failures():
    db = _db()
    register_user("frank", "frank@example.com", "secret123", "secret123", db_path=db)
    for _ in range(auth.MAX_FAILED_ATTEMPTS):
        _expect_error(authenticate, "frank", "wrongpass1", db_path=db)
    # even the correct password is refused while locked
    assert "Too many failed attempts" in _expect_error(authenticate, "frank", "secret123", db_path=db)
