import os
import re
import secrets
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import threading
import uuid
import json
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path, PurePosixPath
from typing import Any

from flask import Flask, flash, g, jsonify, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = BASE_DIR / "instance" / "diary.db"

MAX_CODE_CHARS = int(os.getenv("CODE_MAX_CHARS", "20000"))
MAX_STDIN_CHARS = int(os.getenv("CODE_MAX_STDIN_CHARS", "8000"))
RUN_TIMEOUT = float(os.getenv("CODE_RUN_TIMEOUT", "5.0"))
CPP_COMPILE_TIMEOUT = float(os.getenv("CPP_COMPILE_TIMEOUT", "8.0"))
PYTHON_CMD = os.getenv("CODE_PYTHON_BIN") or sys.executable or shutil.which("python3") or shutil.which("python")
CPP_CMD = shutil.which("g++")
CONDA_CMD = os.getenv("CODE_CONDA_BIN") or os.getenv("CONDA_EXE") or shutil.which("conda")
CONDA_DISCOVERY_TIMEOUT = float(os.getenv("CONDA_DISCOVERY_TIMEOUT", "4.0"))
CONDA_RUN_OVERHEAD = float(os.getenv("CONDA_RUN_OVERHEAD", "10.0"))
IDE_WORKSPACE_ROOT = Path(os.getenv("IDE_WORKSPACE_ROOT", str(BASE_DIR / "instance" / "workspaces")))
IDE_MAX_FILE_CHARS = int(os.getenv("IDE_MAX_FILE_CHARS", "250000"))
IDE_MAX_TREE_ENTRIES = int(os.getenv("IDE_MAX_TREE_ENTRIES", "500"))
IDE_MAX_SEARCH_RESULTS = int(os.getenv("IDE_MAX_SEARCH_RESULTS", "120"))
IDE_MAX_SEARCH_CHARS = int(os.getenv("IDE_MAX_SEARCH_CHARS", "80000"))
TERMINAL_MAX_BUFFER_CHARS = int(os.getenv("TERMINAL_MAX_BUFFER_CHARS", "200000"))
TERMINAL_MAX_OUTPUT_CHARS = int(os.getenv("TERMINAL_MAX_OUTPUT_CHARS", "12000"))
TERMINAL_IDLE_SECONDS = int(os.getenv("TERMINAL_IDLE_SECONDS", "900"))
TERMINAL_MAX_SECONDS = int(os.getenv("TERMINAL_MAX_SECONDS", "900"))
TERMINAL_ALLOW_USER_SHELL = os.getenv("TERMINAL_ALLOW_USER_SHELL", "0") == "1"
GIT_CMD_TIMEOUT = float(os.getenv("IDE_GIT_TIMEOUT", "8.0"))

REGISTER_ENABLED = os.getenv("ENABLE_REGISTRATION", "1") == "1"
REGISTRATION_INVITE_CODE = os.getenv("REGISTRATION_INVITE_CODE", "").strip()
USERNAME_MIN_LENGTH = max(3, int(os.getenv("USERNAME_MIN_LENGTH", "3")))
USERNAME_MAX_LENGTH = min(32, max(USERNAME_MIN_LENGTH, int(os.getenv("USERNAME_MAX_LENGTH", "24"))))
PASSWORD_MIN_LENGTH = max(8, int(os.getenv("PASSWORD_MIN_LENGTH", "10")))

LOGIN_RATE_LIMIT = max(3, int(os.getenv("LOGIN_RATE_LIMIT", "12")))
LOGIN_RATE_WINDOW_SECONDS = max(60, int(os.getenv("LOGIN_RATE_WINDOW_SECONDS", "600")))
REGISTER_RATE_LIMIT = max(2, int(os.getenv("REGISTER_RATE_LIMIT", "6")))
REGISTER_RATE_WINDOW_SECONDS = max(300, int(os.getenv("REGISTER_RATE_WINDOW_SECONDS", "3600")))
RUNNER_RATE_LIMIT = max(5, int(os.getenv("RUNNER_RATE_LIMIT", "40")))
RUNNER_RATE_WINDOW_SECONDS = max(10, int(os.getenv("RUNNER_RATE_WINDOW_SECONDS", "60")))
AUTH_EVENT_RETENTION_SECONDS = max(3600, int(os.getenv("AUTH_EVENT_RETENTION_SECONDS", "86400")))

LOGIN_LOCK_THRESHOLD = max(3, int(os.getenv("LOGIN_LOCK_THRESHOLD", "5")))
LOGIN_LOCK_SECONDS = max(60, int(os.getenv("LOGIN_LOCK_SECONDS", "900")))

USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")
RESERVED_USERNAMES = {"root", "system", "support", "administrator"}
ENV_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
WORKSPACE_PATH_PATTERN = re.compile(r"^[A-Za-z0-9_./-]+$")
TAG_PATTERN = re.compile(r"^[A-Za-z0-9_+\-#\u4e00-\u9fff]+$")
MAX_TAG_COUNT = max(1, int(os.getenv("ENTRY_MAX_TAG_COUNT", "8")))
MAX_TAG_LENGTH = max(2, int(os.getenv("ENTRY_MAX_TAG_LENGTH", "24")))

_ENV_CACHE: dict[str, Any] = {"ts": 0.0, "envs": [], "conda_names": set()}
_TERMINAL_LOCK = threading.Lock()
_TERMINAL_SESSIONS: dict[str, dict[str, Any]] = {}
_TERMINAL_BY_USER: dict[int, str] = {}

app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.getenv("FLASK_SECRET_KEY", secrets.token_hex(32)),
    DATABASE=str(Path(os.getenv("DIARY_DB", DEFAULT_DB_PATH))),
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
)


def _now_text() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")


def _now_ts() -> int:
    return int(time.time())


def _admin_username() -> str:
    configured = os.getenv("ADMIN_USERNAME", "admin").strip()
    return configured or "admin"


def _admin_password_hash() -> str:
    configured_hash = os.getenv("ADMIN_PASSWORD_HASH", "").strip()
    if configured_hash:
        return configured_hash
    plain_password = os.getenv("ADMIN_PASSWORD", "change-me-now")
    return generate_password_hash(plain_password)


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _upsert_admin_user(conn: sqlite3.Connection) -> int:
    username = _admin_username()
    password_hash = _admin_password_hash()
    now_text = _now_text()
    row = conn.execute(
        "SELECT id FROM users WHERE lower(username) = lower(?)",
        (username,),
    ).fetchone()
    if row:
        user_id = int(row[0])
        conn.execute(
            "UPDATE users SET username = ?, password_hash = ?, is_admin = 1, is_active = 1 WHERE id = ?",
            (username, password_hash, user_id),
        )
        return user_id

    cursor = conn.execute(
        """
        INSERT INTO users (username, password_hash, created_at, failed_login_count, locked_until, is_admin, is_active)
        VALUES (?, ?, ?, 0, 0, 1, 1)
        """,
        (username, password_hash, now_text),
    )
    return int(cursor.lastrowid)


def _init_db() -> None:
    db_path = Path(app.config["DATABASE"])
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                failed_login_count INTEGER NOT NULL DEFAULT 0,
                locked_until INTEGER NOT NULL DEFAULT 0,
                is_admin INTEGER NOT NULL DEFAULT 0,
                is_active INTEGER NOT NULL DEFAULT 1
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS auth_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action TEXT NOT NULL,
                ip TEXT NOT NULL,
                created_at INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS entry_tags (
                entry_id INTEGER NOT NULL,
                tag_id INTEGER NOT NULL,
                PRIMARY KEY (entry_id, tag_id)
            )
            """
        )

        user_columns = _table_columns(conn, "users")
        if "failed_login_count" not in user_columns:
            conn.execute("ALTER TABLE users ADD COLUMN failed_login_count INTEGER NOT NULL DEFAULT 0")
        if "locked_until" not in user_columns:
            conn.execute("ALTER TABLE users ADD COLUMN locked_until INTEGER NOT NULL DEFAULT 0")
        if "is_admin" not in user_columns:
            conn.execute("ALTER TABLE users ADD COLUMN is_admin INTEGER NOT NULL DEFAULT 0")
        if "is_active" not in user_columns:
            conn.execute("ALTER TABLE users ADD COLUMN is_active INTEGER NOT NULL DEFAULT 1")

        entry_columns = _table_columns(conn, "entries")
        if "user_id" not in entry_columns:
            conn.execute("ALTER TABLE entries ADD COLUMN user_id INTEGER")

        admin_user_id = _upsert_admin_user(conn)
        conn.execute("UPDATE entries SET user_id = ? WHERE user_id IS NULL", (admin_user_id,))

        conn.execute("CREATE INDEX IF NOT EXISTS idx_entries_user_created ON entries(user_id, created_at DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_auth_events_action_ip ON auth_events(action, ip, created_at)")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username_lower ON users(lower(username))")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_tags_name_lower ON tags(lower(name))")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_entry_tags_entry ON entry_tags(entry_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_entry_tags_tag ON entry_tags(tag_id)")
        conn.commit()
    finally:
        conn.close()


def get_db() -> sqlite3.Connection:
    if "db" not in g:
        g.db = sqlite3.connect(app.config["DATABASE"])
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_error: BaseException | None) -> None:
    db = g.pop("db", None)
    if db is not None:
        db.close()


@app.before_request
def refresh_session_user() -> None:
    if _current_user_id() is not None:
        _load_current_user()


@app.context_processor
def inject_template_globals() -> dict[str, Any]:
    return {
        "registration_enabled": REGISTER_ENABLED,
    }


def _client_ip() -> str:
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()[:64] or "unknown"
    return (request.remote_addr or "unknown")[:64]


def _cleanup_auth_events(db: sqlite3.Connection, now_ts: int) -> None:
    db.execute(
        "DELETE FROM auth_events WHERE created_at < ?",
        (now_ts - AUTH_EVENT_RETENTION_SECONDS,),
    )


def _record_auth_event(action: str, ip: str) -> None:
    db = get_db()
    now_ts = _now_ts()
    _cleanup_auth_events(db, now_ts)
    db.execute(
        "INSERT INTO auth_events (action, ip, created_at) VALUES (?, ?, ?)",
        (action, ip, now_ts),
    )
    db.commit()


def _is_rate_limited(action: str, ip: str, limit: int, window_seconds: int) -> bool:
    db = get_db()
    now_ts = _now_ts()
    _cleanup_auth_events(db, now_ts)
    row = db.execute(
        """
        SELECT COUNT(1) AS cnt
        FROM auth_events
        WHERE action = ? AND ip = ? AND created_at >= ?
        """,
        (action, ip, now_ts - window_seconds),
    ).fetchone()
    return int(row["cnt"]) >= limit


def _get_user_by_username(username: str) -> sqlite3.Row | None:
    if not username:
        return None
    db = get_db()
    return db.execute(
        """
        SELECT id, username, password_hash, created_at, failed_login_count, locked_until, is_admin, is_active
        FROM users
        WHERE lower(username) = lower(?)
        """,
        (username,),
    ).fetchone()


def _get_user_by_id(user_id: int) -> sqlite3.Row | None:
    db = get_db()
    return db.execute(
        """
        SELECT id, username, password_hash, created_at, failed_login_count, locked_until, is_admin, is_active
        FROM users
        WHERE id = ?
        """,
        (user_id,),
    ).fetchone()


def _current_user_id() -> int | None:
    raw = session.get("user_id")
    if raw is None:
        return None
    try:
        user_id = int(raw)
    except (TypeError, ValueError):
        return None
    return user_id if user_id > 0 else None


def _start_session(user: sqlite3.Row) -> None:
    session.clear()
    session["logged_in"] = True
    session["user_id"] = int(user["id"])
    session["username"] = str(user["username"])
    session["is_admin"] = bool(user["is_admin"])


def _load_current_user() -> sqlite3.Row | None:
    cached = g.get("current_user")
    if cached is not None:
        return cached

    user_id = _current_user_id()
    if user_id is None:
        return None
    user = _get_user_by_id(user_id)
    if user is None or not bool(user["is_active"]):
        session.clear()
        return None

    session["username"] = str(user["username"])
    session["is_admin"] = bool(user["is_admin"])
    g.current_user = user
    return user


def _validate_username(username: str) -> str | None:
    if not (USERNAME_MIN_LENGTH <= len(username) <= USERNAME_MAX_LENGTH):
        return f"Username length must be between {USERNAME_MIN_LENGTH} and {USERNAME_MAX_LENGTH}."
    if not USERNAME_PATTERN.fullmatch(username):
        return "Username may only contain letters, digits, and underscore."
    lowered = username.lower()
    admin_name = _admin_username().lower()
    if lowered in RESERVED_USERNAMES and lowered != admin_name:
        return "This username is reserved."
    return None


def _validate_password(password: str) -> str | None:
    if len(password) < PASSWORD_MIN_LENGTH:
        return f"Password must be at least {PASSWORD_MIN_LENGTH} characters."
    score = 0
    if re.search(r"[a-z]", password):
        score += 1
    if re.search(r"[A-Z]", password):
        score += 1
    if re.search(r"\d", password):
        score += 1
    if re.search(r"[^\w\s]", password):
        score += 1
    if score < 3:
        return "Password must include at least three classes: upper, lower, digit, symbol."
    return None


def _is_invite_code_valid(invite_code: str) -> bool:
    if not REGISTRATION_INVITE_CODE:
        return False
    return secrets.compare_digest(invite_code.strip(), REGISTRATION_INVITE_CODE)


def _on_login_success(user_id: int) -> None:
    db = get_db()
    db.execute(
        "UPDATE users SET failed_login_count = 0, locked_until = 0 WHERE id = ?",
        (user_id,),
    )
    db.commit()


def _on_login_failure(user: sqlite3.Row) -> int:
    db = get_db()
    failed_login_count = int(user["failed_login_count"]) + 1
    lock_until = 0
    if failed_login_count >= LOGIN_LOCK_THRESHOLD:
        failed_login_count = 0
        lock_until = _now_ts() + LOGIN_LOCK_SECONDS

    db.execute(
        "UPDATE users SET failed_login_count = ?, locked_until = ? WHERE id = ?",
        (failed_login_count, lock_until, int(user["id"])),
    )
    db.commit()
    return lock_until


def _generate_temp_password(length: int = 12) -> str:
    upper = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    lower = "abcdefghijkmnopqrstuvwxyz"
    digits = "23456789"
    symbols = "!@#$%^&*"
    all_chars = upper + lower + digits + symbols

    # Keep all required classes in generated password.
    chars = [
        secrets.choice(upper),
        secrets.choice(lower),
        secrets.choice(digits),
        secrets.choice(symbols),
    ]
    chars.extend(secrets.choice(all_chars) for _ in range(max(0, length - len(chars))))
    secrets.SystemRandom().shuffle(chars)
    return "".join(chars)


def _user_workspace_root(user_id: int) -> Path:
    root = IDE_WORKSPACE_ROOT / f"user_{user_id}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _normalize_workspace_relpath(path_text: str) -> str:
    normalized = path_text.replace("\\", "/").strip().strip("/")
    if not normalized:
        raise ValueError("Path cannot be empty.")
    if not WORKSPACE_PATH_PATTERN.fullmatch(normalized):
        raise ValueError("Path contains invalid characters.")

    pure = PurePosixPath(normalized)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError("Access outside workspace is not allowed.")
    return pure.as_posix()


def _resolve_workspace_path(user_id: int, rel_path: str) -> Path:
    root = _user_workspace_root(user_id).resolve()
    safe_rel = _normalize_workspace_relpath(rel_path)
    resolved = (root / safe_rel).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError("Access outside workspace is not allowed.")
    return resolved


def _ensure_workspace_seed(user_id: int) -> None:
    root = _user_workspace_root(user_id)
    py_file = root / "main.py"
    cpp_file = root / "main.cpp"
    readme = root / "README.txt"

    if not py_file.exists():
        py_file.write_text(
            "def solve():\n"
            "    print('Hello from Python')\n\n"
            "if __name__ == '__main__':\n"
            "    solve()\n",
            encoding="utf-8",
        )
    if not cpp_file.exists():
        cpp_file.write_text(
            "#include <bits/stdc++.h>\n"
            "using namespace std;\n\n"
            "int main() {\n"
            "    ios::sync_with_stdio(false);\n"
            "    cin.tie(nullptr);\n"
            "    cout << \"Hello from C++\" << \"\\n\";\n"
            "    return 0;\n"
            "}\n",
            encoding="utf-8",
        )
    if not readme.exists():
        readme.write_text(
            "杩欐槸浣犵殑鍦ㄧ嚎 IDE 宸ヤ綔鍖恒€俓n"
            "- 鍙互鏂板缓/淇濆瓨鏂囦欢\n"
            "- 鍙互鍒囨崲杩愯鐜锛圫ystem / Conda锛塡n"
            "- 杩愯 Python 涓?C++\n",
            encoding="utf-8",
        )



def _current_user_is_admin() -> bool:
    return bool(session.get("is_admin"))


def _is_within_root(path: Path, root: Path) -> bool:
    try:
        resolved_path = path.resolve()
        resolved_root = root.resolve()
    except Exception:
        return False
    return resolved_path == resolved_root or resolved_root in resolved_path.parents


def _ide_root_dir(user_id: int, is_admin: bool) -> Path:
    if is_admin:
        if os.name == "nt":
            home = Path.home().resolve()
            anchor = home.anchor or str(home.drive) + "\\"
            return Path(anchor).resolve()
        return Path("/").resolve()
    root = _user_workspace_root(user_id).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _ide_home_dir(user_id: int, is_admin: bool) -> Path:
    if is_admin:
        preferred = Path("/home/user").resolve()
        if preferred.exists() and preferred.is_dir():
            return preferred
        home = Path.home().resolve()
        if home.exists() and home.is_dir():
            return home
        return Path("/").resolve()
    _ensure_workspace_seed(user_id)
    return _user_workspace_root(user_id).resolve()


def _resolve_ide_directory(user_id: int, dir_text: str | None) -> tuple[Path, Path, Path]:
    is_admin = _current_user_is_admin()
    root_dir = _ide_root_dir(user_id, is_admin)
    home_dir = _ide_home_dir(user_id, is_admin)

    session_dir_text = str(session.get("ide_cwd", "")).strip()
    session_dir: Path | None = None
    if session_dir_text:
        try:
            session_dir = Path(session_dir_text).expanduser().resolve()
        except Exception:
            session_dir = None
    if session_dir is None or not session_dir.exists() or not session_dir.is_dir() or not _is_within_root(session_dir, root_dir):
        session_dir = home_dir

    raw = str(dir_text or "").strip()
    if raw:
        candidate_raw = Path(raw.replace("\\", "/")).expanduser()
        if candidate_raw.is_absolute():
            candidate = candidate_raw.resolve()
        else:
            candidate = (session_dir / candidate_raw).resolve()
    else:
        candidate = session_dir

    if not candidate.exists() or not candidate.is_dir():
        raise ValueError("Directory does not exist.")
    if not _is_within_root(candidate, root_dir):
        raise ValueError("Access denied.")

    session["ide_cwd"] = str(candidate)
    return candidate, root_dir, home_dir


def _resolve_ide_target_path(user_id: int, cwd: Path, path_text: str) -> Path:
    raw = str(path_text or "").strip()
    if not raw:
        raise ValueError("Path is required.")

    is_admin = _current_user_is_admin()
    root_dir = _ide_root_dir(user_id, is_admin)
    candidate_raw = Path(raw.replace("\\", "/")).expanduser()
    if candidate_raw.is_absolute():
        candidate = candidate_raw.resolve()
    else:
        candidate = (cwd / candidate_raw).resolve()

    if not _is_within_root(candidate, root_dir):
        raise ValueError("Access denied.")
    return candidate


def _list_directory_entries(cwd: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    try:
        children = list(cwd.iterdir())
    except OSError as exc:
        raise ValueError(f"Cannot open directory: {exc}") from exc

    children.sort(key=lambda p: (0 if p.is_dir() else 1, p.name.lower()))
    for child in children:
        if len(entries) >= IDE_MAX_TREE_ENTRIES:
            break
        entry_type = "dir" if child.is_dir() else "file" if child.is_file() else ""
        if not entry_type:
            continue
        size = 0
        if entry_type == "file":
            try:
                size = child.stat().st_size
            except OSError:
                size = 0
        entries.append(
            {
                "name": child.name,
                "path": str(child),
                "type": entry_type,
                "size": size,
            }
        )
    return entries


def _ide_parent_dir(cwd: Path, root_dir: Path) -> str | None:
    if cwd == root_dir:
        return None
    parent = cwd.parent.resolve()
    if parent == cwd or not _is_within_root(parent, root_dir):
        return None
    return str(parent)

def _list_workspace_entries(user_id: int) -> list[dict[str, Any]]:
    root = _user_workspace_root(user_id)
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if len(entries) >= IDE_MAX_TREE_ENTRIES:
            break
        rel = path.relative_to(root).as_posix()
        if path.is_dir():
            entries.append({"path": rel, "type": "dir", "size": 0})
        elif path.is_file():
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            entries.append({"path": rel, "type": "file", "size": size})
    return entries


def _iter_files_recursive(base_dir: Path, max_items: int) -> list[Path]:
    files: list[Path] = []
    stack = [base_dir]
    while stack and len(files) < max_items:
        current = stack.pop()
        try:
            children = list(current.iterdir())
        except OSError:
            continue

        children.sort(key=lambda p: (0 if p.is_dir() else 1, p.name.lower()))
        for child in children:
            if child.name in {".git", ".venv", "__pycache__"}:
                continue
            if child.is_dir():
                stack.append(child)
                continue
            if child.is_file():
                files.append(child)
                if len(files) >= max_items:
                    break
    return files


def _search_workspace(cwd: Path, query: str) -> list[dict[str, Any]]:
    keyword = query.strip().lower()
    if not keyword:
        return []

    matches: list[dict[str, Any]] = []
    for file_path in _iter_files_recursive(cwd, max_items=IDE_MAX_SEARCH_RESULTS * 4):
        if len(matches) >= IDE_MAX_SEARCH_RESULTS:
            break

        name_hit = keyword in file_path.name.lower()
        text_hit = False
        line_preview = ""

        if not name_hit:
            try:
                stat = file_path.stat()
                if stat.st_size <= IDE_MAX_SEARCH_CHARS:
                    text = file_path.read_text(encoding="utf-8")
                    for idx, line in enumerate(text.splitlines(), start=1):
                        if keyword in line.lower():
                            text_hit = True
                            line_preview = f"{idx}: {line[:180]}"
                            break
            except Exception:
                text_hit = False

        if not name_hit and not text_hit:
            continue

        try:
            size = file_path.stat().st_size
        except OSError:
            size = 0
        matches.append(
            {
                "path": str(file_path),
                "name": file_path.name,
                "size": size,
                "kind": "name" if name_hit else "content",
                "preview": line_preview,
            }
        )

    return matches


def _run_git(cwd: Path, args: list[str], timeout: float | None = None) -> tuple[int, str, str]:
    command = ["git", *args]
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=timeout or GIT_CMD_TIMEOUT,
        )
    except FileNotFoundError:
        return 127, "", "git is not installed."
    except subprocess.TimeoutExpired:
        return 124, "", "git command timed out."
    return completed.returncode, completed.stdout or "", completed.stderr or ""


def _format_bytes(size_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    value = float(max(0, int(size_bytes)))
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return "0 B"


def _read_memory_status() -> dict[str, Any] | None:
    total_bytes: int | None = None
    available_bytes: int | None = None

    meminfo_path = Path("/proc/meminfo")
    if meminfo_path.exists():
        try:
            info: dict[str, int] = {}
            for line in meminfo_path.read_text(encoding="utf-8").splitlines():
                if ":" not in line:
                    continue
                key, value_part = line.split(":", 1)
                amount = value_part.strip().split()[0]
                if amount.isdigit():
                    info[key.strip()] = int(amount) * 1024
            total_bytes = info.get("MemTotal")
            available_bytes = info.get("MemAvailable", info.get("MemFree"))
        except Exception:
            total_bytes = None
            available_bytes = None

    if total_bytes is None:
        try:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            total_pages = int(os.sysconf("SC_PHYS_PAGES"))
            total_bytes = page_size * total_pages
            available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
            available_bytes = page_size * available_pages
        except Exception:
            total_bytes = None
            available_bytes = None

    if total_bytes is None or total_bytes <= 0:
        return None

    if available_bytes is None:
        available_bytes = 0

    used_bytes = max(total_bytes - available_bytes, 0)
    usage_percent = round((used_bytes / total_bytes) * 100.0, 1) if total_bytes else 0.0
    return {
        "total_bytes": total_bytes,
        "used_bytes": used_bytes,
        "available_bytes": available_bytes,
        "usage_percent": usage_percent,
        "total_human": _format_bytes(total_bytes),
        "used_human": _format_bytes(used_bytes),
        "available_human": _format_bytes(available_bytes),
    }


def _read_disk_status() -> dict[str, Any]:
    disk_anchor = IDE_WORKSPACE_ROOT
    if not disk_anchor.exists():
        disk_anchor = BASE_DIR
    usage = shutil.disk_usage(disk_anchor)
    used_bytes = max(usage.total - usage.free, 0)
    usage_percent = round((used_bytes / usage.total) * 100.0, 1) if usage.total else 0.0
    return {
        "path": str(disk_anchor),
        "total_bytes": usage.total,
        "used_bytes": used_bytes,
        "free_bytes": usage.free,
        "usage_percent": usage_percent,
        "total_human": _format_bytes(usage.total),
        "used_human": _format_bytes(used_bytes),
        "free_human": _format_bytes(usage.free),
    }


def _read_cpu_status() -> dict[str, Any] | None:
    cpu_count = os.cpu_count() or 1
    try:
        load_1, load_5, load_15 = os.getloadavg()
    except Exception:
        load_1 = load_5 = load_15 = 0.0

    usage_percent = round(min(100.0, (load_1 / max(1, cpu_count)) * 100.0), 1)
    return {
        "cpu_count": int(cpu_count),
        "load_1m": round(float(load_1), 2),
        "load_5m": round(float(load_5), 2),
        "load_15m": round(float(load_15), 2),
        "usage_percent": usage_percent,
    }


def _server_status_snapshot() -> dict[str, Any]:
    return {
        "server_time": _now_text(),
        "cpu": _read_cpu_status(),
        "memory": _read_memory_status(),
        "disk": _read_disk_status(),
    }


def _language_from_filename(filename: str) -> str:
    lower = filename.lower()
    if lower.endswith((".cpp", ".cc", ".cxx", ".hpp", ".h")):
        return "cpp"
    return "python"


def _detect_runtime_envs(force: bool = False) -> tuple[list[dict[str, Any]], set[str]]:
    global _ENV_CACHE

    now = time.time()
    cached_envs = _ENV_CACHE.get("envs") or []
    cached_names = _ENV_CACHE.get("conda_names") or set()
    if not force and cached_envs and now - float(_ENV_CACHE.get("ts", 0.0)) < 60:
        return cached_envs, set(cached_names)

    envs: list[dict[str, Any]] = [
        {
            "id": "system",
            "label": "System (default)",
            "python_available": bool(PYTHON_CMD),
            "cpp_available": bool(CPP_CMD),
        }
    ]
    conda_names: list[str] = []

    if CONDA_CMD:
        try:
            proc = subprocess.run(
                [CONDA_CMD, "env", "list", "--json"],
                text=True,
                capture_output=True,
                timeout=CONDA_DISCOVERY_TIMEOUT,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                payload = json.loads(proc.stdout)
                raw_envs = payload.get("envs") or []
                env_details = payload.get("envs_details") or {}
                conda_cmd_path = Path(CONDA_CMD).expanduser()
                conda_root_guess = conda_cmd_path.parent.parent
                seen: set[str] = set()
                for env_path in raw_envs:
                    env_path_text = str(env_path).strip()
                    if not env_path_text:
                        continue

                    detail = env_details.get(env_path_text, {})
                    name = str(detail.get("name") or "").strip()
                    if not name:
                        guess = Path(env_path_text)
                        if guess == conda_root_guess:
                            name = "base"
                        else:
                            name = guess.name.strip()
                    if name and name not in seen and ENV_ID_PATTERN.fullmatch(name):
                        seen.add(name)
                        conda_names.append(name)
        except Exception:
            conda_names = []

    for name in conda_names:
        envs.append(
            {
                "id": f"conda:{name}",
                "label": f"Conda / {name}",
                "python_available": True,
                "cpp_available": True,
            }
        )

    _ENV_CACHE = {
        "ts": now,
        "envs": envs,
        "conda_names": set(conda_names),
    }
    return envs, set(conda_names)


def _conda_env_dir(env_name: str) -> Path | None:
    env_key = str(env_name or "").strip()
    if not env_key or not CONDA_CMD:
        return None

    try:
        proc = subprocess.run(
            [CONDA_CMD, "env", "list", "--json"],
            text=True,
            capture_output=True,
            timeout=CONDA_DISCOVERY_TIMEOUT,
        )
    except Exception:
        return None

    if proc.returncode != 0 or not proc.stdout.strip():
        return None

    try:
        payload = json.loads(proc.stdout)
    except Exception:
        return None

    raw_envs = payload.get("envs") or []
    env_details = payload.get("envs_details") or {}
    conda_cmd_path = Path(CONDA_CMD).expanduser()
    conda_root_guess = conda_cmd_path.parent.parent

    for env_path in raw_envs:
        path_text = str(env_path).strip()
        if not path_text:
            continue
        path_obj = Path(path_text)
        detail = env_details.get(path_text, {})
        name = str(detail.get("name") or "").strip()
        if not name:
            try:
                if path_obj.resolve() == conda_root_guess.resolve():
                    name = "base"
                else:
                    name = path_obj.name.strip()
            except Exception:
                name = path_obj.name.strip()
        if name == env_key and path_obj.exists() and path_obj.is_dir():
            return path_obj

    return None


def _parse_runtime_env(env_id: str) -> tuple[str, str | None]:
    env_text = (env_id or "system").strip()
    if env_text in {"", "system"}:
        return "system", None

    if not env_text.startswith("conda:"):
        raise ValueError("Unknown runtime environment.")
    env_name = env_text.split(":", 1)[1].strip()
    if not env_name or not ENV_ID_PATTERN.fullmatch(env_name):
        raise ValueError("Conda environment name is invalid.")
    if not CONDA_CMD:
        raise ValueError("Conda is not configured on this server.")

    _, conda_names = _detect_runtime_envs(force=False)
    if env_name not in conda_names:
        raise ValueError(f"Conda environment does not exist: {env_name}")
    return "conda", env_name


def _effective_timeout(base_timeout: float, env_kind: str) -> float:
    if env_kind == "conda":
        return base_timeout + CONDA_RUN_OVERHEAD
    return base_timeout


def _build_python_command(env_id: str, code: str, *, unbuffered: bool = False) -> tuple[list[str], str]:
    env_kind, env_name = _parse_runtime_env(env_id)
    flags = ["-I", "-c", code]
    if unbuffered:
        flags = ["-u"] + flags
    if env_kind == "system":
        if not PYTHON_CMD:
            raise ValueError("Python interpreter is not configured.")
        return [PYTHON_CMD, *flags], env_kind

    # For interactive terminal mode, prefer the env's interpreter directly.
    # `conda run` may buffer or mishandle stdin in long-lived sessions.
    if unbuffered:
        env_dir = _conda_env_dir(env_name or "")
        if env_dir is not None:
            interpreter = env_dir / "python.exe" if os.name == "nt" else env_dir / "bin" / "python"
            if interpreter.exists():
                return [str(interpreter), *flags], env_kind

    return [CONDA_CMD, "run", "--no-capture-output", "-n", env_name or "", "python", *flags], env_kind


def _build_cpp_commands(
    env_id: str,
    source_path: Path,
    binary_path: Path,
) -> tuple[list[str], list[str], str]:
    env_kind, env_name = _parse_runtime_env(env_id)
    if env_kind == "system":
        if not CPP_CMD:
            raise ValueError("g++ is not available on this server.")
        compile_cmd = [CPP_CMD, "-std=c++17", "-O2", str(source_path), "-o", str(binary_path)]
        run_cmd = [str(binary_path)]
        return compile_cmd, run_cmd, env_kind

    # Fallback strategy:
    # - If system g++ exists, keep C++ experience stable even when selected conda env
    #   does not have compiler packages.
    # - Otherwise try compiler inside selected conda env.
    if CPP_CMD:
        compile_cmd = [CPP_CMD, "-std=c++17", "-O2", str(source_path), "-o", str(binary_path)]
        run_cmd = [str(binary_path)]
        return compile_cmd, run_cmd, "system"

    prefix = [CONDA_CMD, "run", "--no-capture-output", "-n", env_name or ""]
    compile_cmd = prefix + ["g++", "-std=c++17", "-O2", str(source_path), "-o", str(binary_path)]
    run_cmd = prefix + [str(binary_path)]
    return compile_cmd, run_cmd, env_kind


def _default_runtime_env() -> str:
    envs, _ = _detect_runtime_envs(force=False)
    if not envs:
        return "system"
    return str(envs[0].get("id") or "system")


def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        user = _load_current_user()
        if user is None:
            session.clear()
            return redirect(url_for("login"))
        return fn(*args, **kwargs)

    return wrapper


def admin_required(fn):
    @wraps(fn)
    @login_required
    def wrapper(*args, **kwargs):
        user = _load_current_user()
        if user is None or not bool(user["is_admin"]):
            flash("Administrator permission required.", "error")
            return redirect(url_for("workspace"))
        return fn(*args, **kwargs)

    return wrapper


def _sandbox_preexec(cpu_budget: float | None = None):
    if os.name != "posix":
        return None

    def _limits():
        import resource

        raw_budget = cpu_budget if cpu_budget is not None else RUN_TIMEOUT
        cpu_seconds = max(1, int(raw_budget))
        memory_bytes = int(os.getenv("CODE_MEMORY_MB", "256")) * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        resource.setrlimit(resource.RLIMIT_FSIZE, (2 * 1024 * 1024, 2 * 1024 * 1024))

    return _limits


def _normalize_output(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    lines = [line.rstrip() for line in stripped.splitlines()]
    return "\n".join(lines)


def _base_result() -> dict[str, Any]:
    return {
        "ok": False,
        "stage": "validation",
        "verdict": "Error",
        "stdout": "",
        "stderr": "",
        "runtime_ms": 0,
    }



def _terminal_cleanup_dir(temp_dir: str | None) -> None:
    if not temp_dir:
        return
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        return


def _terminal_append_output(state: dict[str, Any], text: str) -> None:
    if not text:
        return
    with state["lock"]:
        state["buffer"] += text
        overflow = len(state["buffer"]) - TERMINAL_MAX_BUFFER_CHARS
        if overflow > 0:
            state["buffer"] = state["buffer"][overflow:]
            state["buffer_start"] += overflow
        state["updated_at"] = time.time()


def _terminal_finalize(state: dict[str, Any], exit_code: int | None, note: str = "") -> None:
    if note:
        _terminal_append_output(state, note)
    with state["lock"]:
        state["done"] = True
        state["exit_code"] = exit_code
        state["updated_at"] = time.time()


def _terminal_reader(session_id: str) -> None:
    with _TERMINAL_LOCK:
        state = _TERMINAL_SESSIONS.get(session_id)
    if state is None:
        return

    process: subprocess.Popen[Any] = state["process"]
    stream = process.stdout
    if stream is None:
        return

    while True:
        try:
            chunk = stream.read(1024)
        except Exception as exc:
            _terminal_append_output(state, f"\n[terminal reader error] {exc}\n")
            break
        if not chunk:
            break
        if isinstance(chunk, bytes):
            decoded = chunk.decode("utf-8", errors="replace")
        else:
            decoded = str(chunk)
        _terminal_append_output(state, decoded)


def _terminal_waiter(session_id: str) -> None:
    with _TERMINAL_LOCK:
        state = _TERMINAL_SESSIONS.get(session_id)
    if state is None:
        return

    process: subprocess.Popen[Any] = state["process"]
    exit_code: int | None = None
    try:
        exit_code = process.wait()
    except Exception:
        exit_code = None

    _terminal_finalize(state, exit_code, f"\n\n[terminal exited: {exit_code}]\n")


def _terminal_gc() -> None:
    now = time.time()
    stale_ids: list[str] = []

    with _TERMINAL_LOCK:
        for sid, state in list(_TERMINAL_SESSIONS.items()):
            process: subprocess.Popen[Any] = state["process"]
            with state["lock"]:
                done = bool(state.get("done"))
                updated_at = float(state.get("updated_at", 0.0))
                deadline = float(state.get("deadline_at", 0.0))

            if not done and deadline > 0 and now > deadline and process.poll() is None:
                try:
                    process.terminate()
                except Exception:
                    pass
                _terminal_finalize(state, -9, "\n\n[terminal timed out and was stopped]\n")
                done = True

            if done and now - updated_at > TERMINAL_IDLE_SECONDS:
                stale_ids.append(sid)

        for sid in stale_ids:
            state = _TERMINAL_SESSIONS.pop(sid, None)
            if not state:
                continue
            user_id = int(state.get("user_id", 0))
            if _TERMINAL_BY_USER.get(user_id) == sid:
                _TERMINAL_BY_USER.pop(user_id, None)
            process: subprocess.Popen[Any] = state["process"]
            if process.poll() is None:
                try:
                    process.terminate()
                except Exception:
                    pass
            _terminal_cleanup_dir(state.get("temp_dir"))


def _terminal_stop_for_user(user_id: int, reason: str = "\n\n[terminal stopped]\n") -> None:
    with _TERMINAL_LOCK:
        session_id = _TERMINAL_BY_USER.get(user_id)
        state = _TERMINAL_SESSIONS.get(session_id or "")

    if not session_id or state is None:
        return

    process: subprocess.Popen[Any] = state["process"]
    if process.poll() is None:
        try:
            process.terminate()
            process.wait(timeout=1.5)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass

    _terminal_finalize(state, process.poll(), reason)

def _terminal_read_slice(state: dict[str, Any], cursor: int) -> tuple[str, int, bool]:
    with state["lock"]:
        buffer_start = int(state["buffer_start"])
        buffer_text = str(state["buffer"])

    truncated = cursor < buffer_start
    if truncated:
        cursor = buffer_start

    rel = max(0, cursor - buffer_start)
    chunk = buffer_text[rel:]
    next_cursor = buffer_start + len(buffer_text)

    if len(chunk) > TERMINAL_MAX_OUTPUT_CHARS:
        chunk = chunk[:TERMINAL_MAX_OUTPUT_CHARS]
        next_cursor = cursor + len(chunk)

    return chunk, next_cursor, truncated


def _build_terminal_process(user_id: int, language: str, code: str, runtime_env: str, cwd: Path) -> tuple[subprocess.Popen[Any], str | None, str]:
    if language == "shell":
        if os.name == "nt":
            shell_bin = os.getenv("IDE_SHELL_BIN", "cmd.exe")
            shell_cmd = [shell_bin]
        else:
            shell_bin = os.getenv("IDE_SHELL_BIN", "/bin/bash")
            shell_cmd = [shell_bin, "-i"]
        process = subprocess.Popen(
            shell_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
            text=False,
            cwd=str(cwd),
        )
        return process, None, "system"

    if language == "python":
        command, env_kind = _build_python_command(runtime_env, code, unbuffered=True)
        timeout_seconds = _effective_timeout(float(TERMINAL_MAX_SECONDS), env_kind)
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
            text=False,
            cwd=str(cwd),
            preexec_fn=_sandbox_preexec(min(timeout_seconds, 120.0)),
        )
        return process, None, env_kind

    if language != "cpp":
        raise ValueError("Only python and cpp are supported.")

    temp_dir = tempfile.mkdtemp(prefix="cpp-terminal-")
    temp_path = Path(temp_dir)
    source_path = temp_path / "main.cpp"
    binary_path = temp_path / "main.out"
    source_path.write_text(code, encoding="utf-8")

    compile_cmd, run_cmd, env_kind = _build_cpp_commands(runtime_env, source_path, binary_path)
    compile_timeout = _effective_timeout(CPP_COMPILE_TIMEOUT, env_kind)
    compile_process = subprocess.run(
        compile_cmd,
        text=True,
        capture_output=True,
        timeout=compile_timeout,
    )
    if compile_process.returncode != 0:
        _terminal_cleanup_dir(temp_dir)
        stderr = compile_process.stderr or compile_process.stdout or "Compilation failed."
        raise RuntimeError(stderr)

    timeout_seconds = _effective_timeout(float(TERMINAL_MAX_SECONDS), env_kind)
    process = subprocess.Popen(
        run_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
        text=False,
        cwd=str(cwd),
        preexec_fn=_sandbox_preexec(min(timeout_seconds, 120.0)),
    )
    return process, temp_dir, env_kind

def _run_python(code: str, stdin_data: str, runtime_env: str) -> dict[str, Any]:
    result = _base_result()
    result["stage"] = "run"
    try:
        command, env_kind = _build_python_command(runtime_env, code)
    except ValueError as exc:
        result.update({"verdict": "Runtime Error", "stderr": str(exc)})
        return result

    timeout_seconds = _effective_timeout(RUN_TIMEOUT, env_kind)
    started_at = time.perf_counter()

    try:
        completed = subprocess.run(
            command,
            input=stdin_data,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            preexec_fn=_sandbox_preexec(timeout_seconds),
        )
    except subprocess.TimeoutExpired as exc:
        result.update(
            {
                "verdict": "Time Limit Exceeded",
                "stderr": f"Execution exceeded {RUN_TIMEOUT:.1f}s and was terminated.",
                "stdout": exc.stdout or "",
                "runtime_ms": int(timeout_seconds * 1000),
            }
        )
        return result
    except Exception as exc:
        result.update({"verdict": "Runtime Error", "stderr": str(exc)})
        return result

    result["runtime_ms"] = int((time.perf_counter() - started_at) * 1000)
    result["stdout"] = completed.stdout
    result["stderr"] = completed.stderr
    if completed.returncode == 0:
        result["ok"] = True
        result["verdict"] = "Accepted"
    else:
        result["verdict"] = "Runtime Error"
    return result


def _run_cpp(code: str, stdin_data: str, runtime_env: str) -> dict[str, Any]:
    result = _base_result()

    with tempfile.TemporaryDirectory(prefix="cpp-runner-") as tmp_dir:
        tmp_path = Path(tmp_dir)
        source_path = tmp_path / "main.cpp"
        binary_path = tmp_path / "main.out"
        source_path.write_text(code, encoding="utf-8")

        try:
            compile_cmd, run_cmd, env_kind = _build_cpp_commands(runtime_env, source_path, binary_path)
        except ValueError as exc:
            result.update(
                {
                    "stage": "compile",
                    "verdict": "Compilation Error",
                    "stderr": str(exc),
                }
            )
            return result

        compile_timeout = _effective_timeout(CPP_COMPILE_TIMEOUT, env_kind)
        run_timeout = _effective_timeout(RUN_TIMEOUT, env_kind)

        try:
            compile_process = subprocess.run(
                compile_cmd,
                text=True,
                capture_output=True,
                timeout=compile_timeout,
            )
        except subprocess.TimeoutExpired:
            result.update(
                {
                    "stage": "compile",
                    "verdict": "Compilation Error",
                    "stderr": f"Compilation exceeded {CPP_COMPILE_TIMEOUT:.1f}s and was terminated.",
                }
            )
            return result
        except Exception as exc:
            result.update({"stage": "compile", "verdict": "Compilation Error", "stderr": str(exc)})
            return result

        if compile_process.returncode != 0:
            result.update(
                {
                    "stage": "compile",
                    "verdict": "Compilation Error",
                    "stderr": compile_process.stderr,
                    "stdout": compile_process.stdout,
                }
            )
            return result

        try:
            started_at = time.perf_counter()
            run_process = subprocess.run(
                run_cmd,
                input=stdin_data,
                text=True,
                capture_output=True,
                timeout=run_timeout,
                preexec_fn=_sandbox_preexec(run_timeout),
            )
        except subprocess.TimeoutExpired as exc:
            result.update(
                {
                    "stage": "run",
                    "verdict": "Time Limit Exceeded",
                    "stderr": f"Execution exceeded {RUN_TIMEOUT:.1f}s and was terminated.",
                    "stdout": exc.stdout or "",
                    "runtime_ms": int(run_timeout * 1000),
                }
            )
            return result
        except Exception as exc:
            result.update({"stage": "run", "verdict": "Runtime Error", "stderr": str(exc)})
            return result

    result["stage"] = "run"
    result["runtime_ms"] = int((time.perf_counter() - started_at) * 1000)
    result["stdout"] = run_process.stdout
    result["stderr"] = run_process.stderr
    if run_process.returncode == 0:
        result["ok"] = True
        result["verdict"] = "Accepted"
    else:
        result["verdict"] = "Runtime Error"
    return result


def _execute_code(language: str, code: str, stdin_data: str, runtime_env: str) -> dict[str, Any]:
    if language == "python":
        return _run_python(code, stdin_data, runtime_env)
    if language == "cpp":
        return _run_cpp(code, stdin_data, runtime_env)

    result = _base_result()
    result.update(
        {
            "stage": "validation",
            "verdict": "Error",
            "stderr": "Only python or cpp is supported.",
        }
    )
    return result


def _normalize_tag_name(tag_text: str) -> str:
    tag = " ".join(str(tag_text or "").strip().split())
    if not tag:
        raise ValueError("Tag cannot be empty.")
    if len(tag) > MAX_TAG_LENGTH:
        raise ValueError(f"Tag is too long (max {MAX_TAG_LENGTH}).")
    if not TAG_PATTERN.fullmatch(tag):
        raise ValueError("Tag can only contain letters, digits, underscore, dash, plus, hash, and Chinese characters.")
    return tag


def _parse_tags_input(raw_tags: str) -> list[str]:
    if not raw_tags.strip():
        return []

    seen: set[str] = set()
    parsed: list[str] = []
    normalized = raw_tags.replace("，", ",")
    pieces = re.split(r"[,\s]+", normalized.strip())
    for piece in pieces:
        if not piece:
            continue
        tag = _normalize_tag_name(piece)
        key = tag.lower()
        if key in seen:
            continue
        seen.add(key)
        parsed.append(tag)
        if len(parsed) >= MAX_TAG_COUNT:
            break
    return parsed


def _sync_entry_tags(db: sqlite3.Connection, entry_id: int, tags: list[str]) -> None:
    db.execute("DELETE FROM entry_tags WHERE entry_id = ?", (entry_id,))
    if not tags:
        return

    now_text = _now_text()
    for tag in tags:
        db.execute("INSERT OR IGNORE INTO tags (name, created_at) VALUES (?, ?)", (tag, now_text))
        row = db.execute("SELECT id FROM tags WHERE lower(name) = lower(?)", (tag,)).fetchone()
        if row is None:
            continue
        db.execute(
            "INSERT OR IGNORE INTO entry_tags (entry_id, tag_id) VALUES (?, ?)",
            (entry_id, int(row["id"])),
        )


def _list_user_tags(user_id: int) -> list[dict[str, Any]]:
    db = get_db()
    rows = db.execute(
        """
        SELECT t.name AS name, COUNT(et.entry_id) AS usage_count
        FROM tags t
        JOIN entry_tags et ON et.tag_id = t.id
        JOIN entries e ON e.id = et.entry_id
        WHERE e.user_id = ?
        GROUP BY t.id, t.name
        ORDER BY usage_count DESC, lower(t.name) ASC
        """,
        (user_id,),
    ).fetchall()
    return [{"name": str(row["name"]), "usage_count": int(row["usage_count"])} for row in rows]


def _entry_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    tags_csv = str(row["tags_csv"] or "")
    tags = [item for item in tags_csv.split(",") if item]
    return {
        "id": int(row["id"]),
        "title": str(row["title"]),
        "content": str(row["content"]),
        "created_at": str(row["created_at"]),
        "updated_at": str(row["updated_at"]),
        "tags": tags,
        "tags_input": ", ".join(tags),
    }


@app.route("/")
def landing():
    return render_template("landing.html", active_page="landing")


@app.route("/workspace")
@login_required
def workspace():
    return render_template("workspace.html", active_page="workspace")


@app.route("/diary")
@login_required
def diary_home():
    user_id = _current_user_id()
    if user_id is None:
        return redirect(url_for("login"))

    active_tag_raw = str(request.args.get("tag", "")).strip()
    active_tag = ""
    if active_tag_raw:
        try:
            active_tag = _normalize_tag_name(active_tag_raw)
        except ValueError:
            active_tag = ""

    db = get_db()
    raw_entries = db.execute(
        """
        SELECT
            e.id,
            e.title,
            e.content,
            e.created_at,
            e.updated_at,
            COALESCE(GROUP_CONCAT(t.name, ','), '') AS tags_csv
        FROM entries e
        LEFT JOIN entry_tags et ON et.entry_id = e.id
        LEFT JOIN tags t ON t.id = et.tag_id
        WHERE e.user_id = ?
          AND (
            ? = ''
            OR EXISTS (
                SELECT 1
                FROM entry_tags et2
                JOIN tags t2 ON t2.id = et2.tag_id
                WHERE et2.entry_id = e.id
                  AND lower(t2.name) = lower(?)
            )
          )
        GROUP BY e.id, e.title, e.content, e.created_at, e.updated_at
        ORDER BY e.created_at DESC
        """,
        (user_id, active_tag, active_tag),
    ).fetchall()
    entries = [_entry_row_to_dict(row) for row in raw_entries]
    all_tags = _list_user_tags(user_id)
    return render_template(
        "diary.html",
        entries=entries,
        all_tags=all_tags,
        active_tag=active_tag,
        active_page="diary",
    )


@app.route("/entry/new", methods=["POST"])
@login_required
def create_entry():
    title = request.form.get("title", "").strip()
    content = request.form.get("content", "").strip()
    tags_input = request.form.get("tags", "").strip()
    if not title or not content:
        flash("Title and content are required.", "error")
        return redirect(url_for("diary_home"))

    try:
        tags = _parse_tags_input(tags_input)
    except ValueError as exc:
        flash(str(exc), "error")
        return redirect(url_for("diary_home"))

    now_text = _now_text()
    user_id = _current_user_id()
    db = get_db()
    cursor = db.execute(
        "INSERT INTO entries (user_id, title, content, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        (user_id, title, content, now_text, now_text),
    )
    _sync_entry_tags(db, int(cursor.lastrowid), tags)
    db.commit()
    flash("Entry saved.", "success")
    return redirect(url_for("diary_home"))


@app.route("/entry/<int:entry_id>/edit", methods=["GET", "POST"])
@login_required
def edit_entry(entry_id: int):
    user_id = _current_user_id()
    db = get_db()
    raw_entry = db.execute(
        """
        SELECT
            e.id,
            e.title,
            e.content,
            e.created_at,
            e.updated_at,
            COALESCE(GROUP_CONCAT(t.name, ','), '') AS tags_csv
        FROM entries e
        LEFT JOIN entry_tags et ON et.entry_id = e.id
        LEFT JOIN tags t ON t.id = et.tag_id
        WHERE e.id = ? AND e.user_id = ?
        GROUP BY e.id, e.title, e.content, e.created_at, e.updated_at
        """,
        (entry_id, user_id),
    ).fetchone()
    if raw_entry is None:
        flash("Entry not found.", "error")
        return redirect(url_for("diary_home"))
    entry = _entry_row_to_dict(raw_entry)

    if request.method == "POST":
        title = request.form.get("title", "").strip()
        content = request.form.get("content", "").strip()
        tags_input = request.form.get("tags", "").strip()
        if not title or not content:
            flash("Title and content are required.", "error")
            entry["title"] = title
            entry["content"] = content
            entry["tags_input"] = tags_input
            return render_template("edit.html", entry=entry, active_page="diary")

        try:
            tags = _parse_tags_input(tags_input)
        except ValueError as exc:
            flash(str(exc), "error")
            entry["title"] = title
            entry["content"] = content
            entry["tags_input"] = tags_input
            return render_template("edit.html", entry=entry, active_page="diary")

        db.execute(
            "UPDATE entries SET title = ?, content = ?, updated_at = ? WHERE id = ? AND user_id = ?",
            (title, content, _now_text(), entry_id, user_id),
        )
        _sync_entry_tags(db, entry_id, tags)
        db.commit()
        flash("Entry updated.", "success")
        return redirect(url_for("diary_home"))

    return render_template("edit.html", entry=entry, active_page="diary")


@app.route("/entry/<int:entry_id>/delete", methods=["POST"])
@login_required
def delete_entry(entry_id: int):
    user_id = _current_user_id()
    db = get_db()
    db.execute("DELETE FROM entry_tags WHERE entry_id = ?", (entry_id,))
    db.execute("DELETE FROM entries WHERE id = ? AND user_id = ?", (entry_id, user_id))
    db.commit()
    flash("Entry deleted.", "success")
    return redirect(url_for("diary_home"))


@app.route("/playground")
@login_required
def playground():
    user_id = _current_user_id()
    if user_id is None:
        return redirect(url_for("login"))

    try:
        ide_cwd, ide_root, ide_home = _resolve_ide_directory(user_id, None)
    except ValueError:
        is_admin = _current_user_is_admin()
        ide_home = _ide_home_dir(user_id, is_admin)
        ide_root = _ide_root_dir(user_id, is_admin)
        ide_cwd = ide_home
        session["ide_cwd"] = str(ide_cwd)

    envs, _ = _detect_runtime_envs(force=False)
    current_env = str(session.get("runtime_env", "")).strip()
    env_ids = {str(item.get("id")) for item in envs}
    if current_env not in env_ids:
        current_env = _default_runtime_env()
        session["runtime_env"] = current_env

    return render_template(
        "playground.html",
        active_page="playground",
        runtime_envs=envs,
        current_runtime_env=current_env,
        max_code_chars=MAX_CODE_CHARS,
        max_stdin_chars=MAX_STDIN_CHARS,
        max_file_chars=IDE_MAX_FILE_CHARS,
        ide_cwd=str(ide_cwd),
        ide_root=str(ide_root),
        ide_home=str(ide_home),
        ide_can_browse_any=_current_user_is_admin(),
    )


@app.route("/api/runtime-envs", methods=["GET"])
@login_required
def list_runtime_envs():
    refresh = request.args.get("refresh", "0") == "1"
    envs, _ = _detect_runtime_envs(force=refresh)
    current_env = str(session.get("runtime_env", "")).strip()
    env_ids = {str(item.get("id")) for item in envs}
    if current_env not in env_ids:
        current_env = _default_runtime_env()
        session["runtime_env"] = current_env
    return jsonify({"ok": True, "envs": envs, "current_env": current_env})


@app.route("/api/ide/files", methods=["GET"])
@login_required
def ide_list_files():
    user_id = _current_user_id()
    if user_id is None:
        return jsonify({"ok": False, "error": "Not logged in."}), 401

    requested_dir = str(request.args.get("dir", "")).strip()
    try:
        cwd, root_dir, home_dir = _resolve_ide_directory(user_id, requested_dir or None)
        entries = _list_directory_entries(cwd)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    return jsonify(
        {
            "ok": True,
            "cwd": str(cwd),
            "root_dir": str(root_dir),
            "home_dir": str(home_dir),
            "parent_dir": _ide_parent_dir(cwd, root_dir),
            "entries": entries,
        }
    )


@app.route("/api/server-status", methods=["GET"])
@login_required
def api_server_status():
    try:
        snapshot = _server_status_snapshot()
    except Exception:
        return jsonify({"ok": False, "error": "Unable to fetch server status."}), 500
    return jsonify({"ok": True, **snapshot})


@app.route("/ops/pulse")
@admin_required
def ops_pulse():
    return render_template("ops_panel.html", active_page="ops")


@app.route("/api/admin/server-status", methods=["GET"])
@admin_required
def api_admin_server_status():
    try:
        snapshot = _server_status_snapshot()
    except Exception:
        return jsonify({"ok": False, "error": "Unable to fetch server status."}), 500
    return jsonify({"ok": True, **snapshot})


@app.route("/api/ide/file", methods=["GET", "POST"])
@login_required
def ide_file():
    user_id = _current_user_id()
    if user_id is None:
        return jsonify({"ok": False, "error": "Not logged in."}), 401

    if request.method == "GET":
        path_text = str(request.args.get("path", "")).strip()
        cwd_text = str(request.args.get("cwd", "")).strip()
        if not path_text:
            return jsonify({"ok": False, "error": "Missing file path."}), 400

        try:
            cwd, _, _ = _resolve_ide_directory(user_id, cwd_text or None)
            file_path = _resolve_ide_target_path(user_id, cwd, path_text)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

        if not file_path.exists() or not file_path.is_file():
            return jsonify({"ok": False, "error": "File not found."}), 404
        if file_path.stat().st_size > IDE_MAX_FILE_CHARS:
            return jsonify({"ok": False, "error": "File is too large to edit in browser."}), 400

        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return jsonify({"ok": False, "error": "Only UTF-8 text files can be edited."}), 400

        return jsonify(
            {
                "ok": True,
                "path": str(file_path),
                "content": content,
                "language": _language_from_filename(file_path.name),
            }
        )

    payload = request.get_json(silent=True) or {}
    path_text = str(payload.get("path", "")).strip()
    cwd_text = str(payload.get("cwd", "")).strip()
    content = str(payload.get("content", ""))
    if not path_text:
        return jsonify({"ok": False, "error": "Missing file path."}), 400
    if len(content) > IDE_MAX_FILE_CHARS:
        return jsonify({"ok": False, "error": f"File content exceeds limit ({IDE_MAX_FILE_CHARS} chars)."}), 400

    try:
        cwd, _, _ = _resolve_ide_directory(user_id, cwd_text or None)
        file_path = _resolve_ide_target_path(user_id, cwd, path_text)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")
    return jsonify({"ok": True, "path": str(file_path)})


@app.route("/api/ide/new-item", methods=["POST"])
@login_required
def ide_new_item():
    user_id = _current_user_id()
    if user_id is None:
        return jsonify({"ok": False, "error": "Not logged in."}), 401

    payload = request.get_json(silent=True) or {}
    path_text = str(payload.get("path", "")).strip()
    cwd_text = str(payload.get("cwd", "")).strip()
    item_type = str(payload.get("type", "file")).strip().lower()
    if not path_text:
        return jsonify({"ok": False, "error": "Missing path."}), 400
    if item_type not in {"file", "folder"}:
        return jsonify({"ok": False, "error": "type must be file or folder."}), 400

    try:
        cwd, _, _ = _resolve_ide_directory(user_id, cwd_text or None)
        target = _resolve_ide_target_path(user_id, cwd, path_text)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    if target.exists():
        return jsonify({"ok": False, "error": "Target already exists."}), 400

    if item_type == "folder":
        target.mkdir(parents=True, exist_ok=False)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        starter = ""
        lang = _language_from_filename(target.name)
        if lang == "python":
            starter = "print('New Python file')\n"
        elif lang == "cpp":
            starter = (
                "#include <bits/stdc++.h>\n"
                "using namespace std;\n\n"
                "int main() {\n"
                "    cout << \"New C++ file\" << \"\\n\";\n"
                "    return 0;\n"
                "}\n"
            )
        target.write_text(starter, encoding="utf-8")

    return jsonify({"ok": True, "path": str(target), "type": item_type})


@app.route("/api/ide/rename-item", methods=["POST"])
@login_required
def ide_rename_item():
    user_id = _current_user_id()
    if user_id is None:
        return jsonify({"ok": False, "error": "Not logged in."}), 401

    payload = request.get_json(silent=True) or {}
    old_path_text = str(payload.get("old_path", "")).strip()
    new_path_text = str(payload.get("new_path", "")).strip()
    cwd_text = str(payload.get("cwd", "")).strip()
    if not old_path_text or not new_path_text:
        return jsonify({"ok": False, "error": "old_path and new_path are required."}), 400

    try:
        cwd, _, _ = _resolve_ide_directory(user_id, cwd_text or None)
        source = _resolve_ide_target_path(user_id, cwd, old_path_text)
        target = _resolve_ide_target_path(user_id, cwd, new_path_text)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    if not source.exists():
        return jsonify({"ok": False, "error": "Source does not exist."}), 404
    if target.exists():
        return jsonify({"ok": False, "error": "Target already exists."}), 400

    target.parent.mkdir(parents=True, exist_ok=True)
    source.rename(target)
    return jsonify({"ok": True, "path": str(target)})


@app.route("/api/ide/delete-item", methods=["POST"])
@login_required
def ide_delete_item():
    user_id = _current_user_id()
    if user_id is None:
        return jsonify({"ok": False, "error": "Not logged in."}), 401

    payload = request.get_json(silent=True) or {}
    path_text = str(payload.get("path", "")).strip()
    cwd_text = str(payload.get("cwd", "")).strip()
    if not path_text:
        return jsonify({"ok": False, "error": "Missing path."}), 400

    try:
        cwd, root_dir, _ = _resolve_ide_directory(user_id, cwd_text or None)
        target = _resolve_ide_target_path(user_id, cwd, path_text)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    if not target.exists():
        return jsonify({"ok": False, "error": "Target does not exist."}), 404
    if target.resolve() == root_dir.resolve():
        return jsonify({"ok": False, "error": "Cannot delete workspace root."}), 400

    try:
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink(missing_ok=False)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500

    return jsonify({"ok": True})


@app.route("/api/ide/search", methods=["GET"])
@login_required
def ide_search():
    user_id = _current_user_id()
    if user_id is None:
        return jsonify({"ok": False, "error": "Not logged in."}), 401

    query = str(request.args.get("q", "")).strip()
    dir_text = str(request.args.get("dir", "")).strip()
    if len(query) < 2:
        return jsonify({"ok": True, "results": []})

    try:
        cwd, _, _ = _resolve_ide_directory(user_id, dir_text or None)
        results = _search_workspace(cwd, query)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    return jsonify({"ok": True, "results": results, "cwd": str(cwd)})


@app.route("/api/ide/git-status", methods=["GET"])
@login_required
def ide_git_status():
    user_id = _current_user_id()
    if user_id is None:
        return jsonify({"ok": False, "error": "Not logged in."}), 401

    cwd_text = str(request.args.get("cwd", "")).strip()
    try:
        cwd, _, _ = _resolve_ide_directory(user_id, cwd_text or None)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    code, top_out, top_err = _run_git(cwd, ["rev-parse", "--show-toplevel"])
    if code != 0:
        return jsonify({"ok": True, "repo_found": False, "error": (top_err or top_out).strip()})

    repo_root = top_out.strip()
    branch_code, branch_out, _ = _run_git(cwd, ["branch", "--show-current"])
    status_code, status_out, status_err = _run_git(cwd, ["status", "--porcelain"])
    if status_code != 0:
        return jsonify({"ok": False, "error": (status_err or status_out).strip() or "git status failed"}), 500

    changed: list[dict[str, Any]] = []
    for line in status_out.splitlines():
        if len(line) < 4:
            continue
        changed.append(
            {
                "xy": line[:2],
                "path": line[3:].strip(),
            }
        )

    return jsonify(
        {
            "ok": True,
            "repo_found": True,
            "repo_root": repo_root,
            "branch": branch_out.strip() if branch_code == 0 else "",
            "changed": changed,
        }
    )


@app.route("/api/ide/git-diff", methods=["GET"])
@login_required
def ide_git_diff():
    user_id = _current_user_id()
    if user_id is None:
        return jsonify({"ok": False, "error": "Not logged in."}), 401

    cwd_text = str(request.args.get("cwd", "")).strip()
    path_text = str(request.args.get("path", "")).strip()
    try:
        cwd, _, _ = _resolve_ide_directory(user_id, cwd_text or None)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    args = ["diff"]
    if path_text:
        args.extend(["--", path_text])
    code, out, err = _run_git(cwd, args, timeout=max(GIT_CMD_TIMEOUT, 12.0))
    if code != 0:
        return jsonify({"ok": False, "error": (err or out).strip() or "git diff failed"}), 500

    return jsonify({"ok": True, "diff": out})


@app.route("/api/ide/git-commit", methods=["POST"])
@login_required
def ide_git_commit():
    user_id = _current_user_id()
    if user_id is None:
        return jsonify({"ok": False, "error": "Not logged in."}), 401

    payload = request.get_json(silent=True) or {}
    cwd_text = str(payload.get("cwd", "")).strip()
    message = str(payload.get("message", "")).strip()
    add_all = bool(payload.get("add_all", True))
    if len(message) < 2:
        return jsonify({"ok": False, "error": "Commit message is too short."}), 400
    if len(message) > 200:
        return jsonify({"ok": False, "error": "Commit message is too long."}), 400

    try:
        cwd, _, _ = _resolve_ide_directory(user_id, cwd_text or None)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    if add_all:
        add_code, add_out, add_err = _run_git(cwd, ["add", "-A"])
        if add_code != 0:
            return jsonify({"ok": False, "error": (add_err or add_out).strip() or "git add failed"}), 500

    code, out, err = _run_git(cwd, ["commit", "-m", message], timeout=max(GIT_CMD_TIMEOUT, 15.0))
    if code != 0:
        text = (err or out).strip() or "git commit failed"
        return jsonify({"ok": False, "error": text}), 400

    return jsonify({"ok": True, "output": out.strip()})


@app.route("/api/terminal/start", methods=["POST"])
@login_required
def terminal_start():
    _terminal_gc()

    user_id = _current_user_id()
    if user_id is None:
        return jsonify({"ok": False, "error": "Not logged in."}), 401

    payload = request.get_json(silent=True) or {}
    language = str(payload.get("language", "")).strip().lower()
    code = str(payload.get("code", ""))
    runtime_env = str(payload.get("runtime_env", session.get("runtime_env", "system"))).strip() or "system"
    cwd_text = str(payload.get("cwd", "")).strip()

    if language not in {"python", "cpp", "shell"}:
        return jsonify({"ok": False, "stage": "validation", "error": "Only python/cpp/shell supported."}), 400

    if language != "shell":
        try:
            _parse_runtime_env(runtime_env)
        except ValueError as exc:
            return jsonify({"ok": False, "stage": "validation", "error": str(exc)}), 400

        if not code.strip():
            return jsonify({"ok": False, "stage": "validation", "error": "Code cannot be empty."}), 400
        if len(code) > MAX_CODE_CHARS:
            return jsonify({"ok": False, "stage": "validation", "error": f"Code exceeds {MAX_CODE_CHARS} chars."}), 400
    else:
        if not (_current_user_is_admin() or TERMINAL_ALLOW_USER_SHELL):
            return jsonify({"ok": False, "stage": "validation", "error": "Shell terminal is admin-only."}), 403

    try:
        cwd, _, _ = _resolve_ide_directory(user_id, cwd_text or None)
    except ValueError as exc:
        return jsonify({"ok": False, "stage": "validation", "error": str(exc)}), 400

    _terminal_stop_for_user(user_id, "\n\n[previous terminal stopped]\n")

    try:
        process, temp_dir, _ = _build_terminal_process(user_id, language, code, runtime_env, cwd)
    except subprocess.TimeoutExpired:
        return jsonify({"ok": False, "stage": "compile", "error": "Compilation timed out."}), 400
    except RuntimeError as exc:
        return jsonify({"ok": False, "stage": "compile", "error": str(exc)}), 400
    except ValueError as exc:
        return jsonify({"ok": False, "stage": "validation", "error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"ok": False, "stage": "start", "error": str(exc)}), 500

    session_id = uuid.uuid4().hex
    state = {
        "id": session_id,
        "user_id": user_id,
        "process": process,
        "temp_dir": temp_dir,
        "lock": threading.Lock(),
        "buffer": "",
        "buffer_start": 0,
        "done": False,
        "exit_code": None,
        "created_at": time.time(),
        "updated_at": time.time(),
        "deadline_at": time.time() + float(TERMINAL_MAX_SECONDS),
    }

    with _TERMINAL_LOCK:
        _TERMINAL_SESSIONS[session_id] = state
        _TERMINAL_BY_USER[user_id] = session_id

    threading.Thread(target=_terminal_reader, args=(session_id,), daemon=True).start()
    threading.Thread(target=_terminal_waiter, args=(session_id,), daemon=True).start()

    session["runtime_env"] = runtime_env

    return jsonify(
        {
            "ok": True,
            "session_id": session_id,
            "cwd": str(cwd),
        }
    )


@app.route("/api/terminal/poll", methods=["GET"])
@login_required
def terminal_poll():
    _terminal_gc()

    user_id = _current_user_id()
    if user_id is None:
        return jsonify({"ok": False, "error": "Not logged in."}), 401

    session_id = str(request.args.get("session_id", "")).strip()
    if not session_id:
        return jsonify({"ok": False, "error": "Missing session_id."}), 400

    cursor_text = str(request.args.get("cursor", "0")).strip()
    try:
        cursor = max(0, int(cursor_text))
    except ValueError:
        cursor = 0

    with _TERMINAL_LOCK:
        state = _TERMINAL_SESSIONS.get(session_id)

    if state is None or int(state.get("user_id", 0)) != user_id:
        return jsonify({"ok": False, "error": "Terminal session not found."}), 404

    chunk, next_cursor, truncated = _terminal_read_slice(state, cursor)
    with state["lock"]:
        done = bool(state.get("done"))
        exit_code = state.get("exit_code")

    return jsonify(
        {
            "ok": True,
            "chunk": chunk,
            "cursor": next_cursor,
            "truncated": truncated,
            "done": done,
            "exit_code": exit_code,
        }
    )


@app.route("/api/terminal/input", methods=["POST"])
@login_required
def terminal_input():
    user_id = _current_user_id()
    if user_id is None:
        return jsonify({"ok": False, "error": "Not logged in."}), 401

    payload = request.get_json(silent=True) or {}
    session_id = str(payload.get("session_id", "")).strip()
    data = str(payload.get("data", ""))
    if not session_id:
        return jsonify({"ok": False, "error": "Missing session_id."}), 400
    if not data:
        return jsonify({"ok": False, "error": "Input is empty."}), 400

    with _TERMINAL_LOCK:
        state = _TERMINAL_SESSIONS.get(session_id)

    if state is None or int(state.get("user_id", 0)) != user_id:
        return jsonify({"ok": False, "error": "Terminal session not found."}), 404

    process: subprocess.Popen[Any] = state["process"]
    if process.poll() is not None:
        with state["lock"]:
            state["done"] = True
            state["exit_code"] = process.returncode
        return jsonify({"ok": False, "error": "Terminal already exited."}), 400

    stream = process.stdin
    if stream is None:
        return jsonify({"ok": False, "error": "Terminal input stream closed."}), 400

    try:
        stream.write(data.encode("utf-8"))
        stream.flush()
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500

    with state["lock"]:
        state["updated_at"] = time.time()

    return jsonify({"ok": True})


@app.route("/api/terminal/stop", methods=["POST"])
@login_required
def terminal_stop():
    user_id = _current_user_id()
    if user_id is None:
        return jsonify({"ok": False, "error": "Not logged in."}), 401

    _terminal_stop_for_user(user_id, "\n\n[terminal stopped by user]\n")
    _terminal_gc()
    return jsonify({"ok": True})


@app.route("/run-code", methods=["POST"])
@login_required
def run_code():
    client_ip = _client_ip()
    if _is_rate_limited("run_code", client_ip, RUNNER_RATE_LIMIT, RUNNER_RATE_WINDOW_SECONDS):
        return jsonify(
            {
                "ok": False,
                "stage": "validation",
                "verdict": "Error",
                "stderr": "Run requests are too frequent. Please try again later.",
            }
        ), 429
    _record_auth_event("run_code", client_ip)

    payload = request.get_json(silent=True) or request.form
    language = str(payload.get("language", "")).strip().lower()
    code = str(payload.get("code", ""))
    stdin_data = str(payload.get("stdin", ""))
    expected_output = str(payload.get("expected_output", ""))
    runtime_env = str(payload.get("runtime_env", session.get("runtime_env", "system"))).strip() or "system"

    try:
        _parse_runtime_env(runtime_env)
    except ValueError as exc:
        return jsonify(
            {
                "ok": False,
                "stage": "validation",
                "verdict": "Error",
                "stderr": str(exc),
            }
        )
    session["runtime_env"] = runtime_env

    if not code.strip():
        return jsonify(
            {
                "ok": False,
                "stage": "validation",
                "verdict": "Error",
                "stderr": "Code cannot be empty.",
            }
        )
    if len(code) > MAX_CODE_CHARS:
        return jsonify(
            {
                "ok": False,
                "stage": "validation",
                "verdict": "Error",
                "stderr": f"Code exceeds limit ({MAX_CODE_CHARS} chars).",
            }
        )
    if len(stdin_data) > MAX_STDIN_CHARS:
        return jsonify(
            {
                "ok": False,
                "stage": "validation",
                "verdict": "Error",
                "stderr": f"stdin exceeds limit ({MAX_STDIN_CHARS} chars).",
            }
        )

    result = _execute_code(language, code, stdin_data, runtime_env)
    result["runtime_env"] = runtime_env

    if result.get("verdict") == "Accepted" and expected_output.strip():
        actual = _normalize_output(result.get("stdout", ""))
        expected = _normalize_output(expected_output)
        if actual != expected:
            result["ok"] = False
            result["verdict"] = "Wrong Answer"
            result["stderr"] = "Output does not match expected result."
            result["expected"] = expected_output
    return jsonify(result)


@app.route("/login", methods=["GET", "POST"])
def login():
    if _load_current_user() is not None:
        return redirect(url_for("workspace"))

    if request.method == "POST":
        client_ip = _client_ip()
        if _is_rate_limited("login", client_ip, LOGIN_RATE_LIMIT, LOGIN_RATE_WINDOW_SECONDS):
            flash("Too many login attempts. Please try again later.", "error")
            return render_template("login.html")
        _record_auth_event("login", client_ip)

        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = _get_user_by_username(username)

        if user is None:
            flash("Invalid username or password.", "error")
            return render_template("login.html")

        if not bool(user["is_active"]):
            flash("This account is disabled.", "error")
            return render_template("login.html")

        locked_until = int(user["locked_until"])
        now_ts = _now_ts()
        if locked_until > now_ts:
            remain = locked_until - now_ts
            flash(f"Account temporarily locked. Try again in {remain} seconds.", "error")
            return render_template("login.html")

        if check_password_hash(user["password_hash"], password):
            _on_login_success(int(user["id"]))
            refreshed_user = _get_user_by_username(username)
            if refreshed_user is None:
                flash("Login state error. Please retry.", "error")
                return render_template("login.html")
            _start_session(refreshed_user)
            return redirect(url_for("workspace"))

        lock_until_after_fail = _on_login_failure(user)
        if lock_until_after_fail:
            remain = lock_until_after_fail - _now_ts()
            flash(f"Too many failures. Account locked for {max(1, remain)} seconds.", "error")
        else:
            flash("Invalid username or password.", "error")

    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if _load_current_user() is not None:
        return redirect(url_for("workspace"))

    if not REGISTER_ENABLED:
        flash("Registration is currently disabled.", "error")
        return redirect(url_for("login"))

    if request.method == "POST":
        client_ip = _client_ip()
        if _is_rate_limited("register", client_ip, REGISTER_RATE_LIMIT, REGISTER_RATE_WINDOW_SECONDS):
            flash("Too many registration requests. Please try again later.", "error")
            return render_template("register.html")
        _record_auth_event("register", client_ip)

        honeypot = request.form.get("website", "").strip()
        if honeypot:
            flash("Registration failed. Please retry later.", "error")
            return render_template("register.html")

        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")
        invite_code = request.form.get("invite_code", "")

        username_error = _validate_username(username)
        if username_error:
            flash(username_error, "error")
            return render_template("register.html")

        password_error = _validate_password(password)
        if password_error:
            flash(password_error, "error")
            return render_template("register.html")

        if password != confirm_password:
            flash("Passwords do not match.", "error")
            return render_template("register.html")

        if not _is_invite_code_valid(invite_code):
            flash("Invalid invite code.", "error")
            return render_template("register.html")

        db = get_db()
        try:
            db.execute(
                """
                INSERT INTO users (username, password_hash, created_at, failed_login_count, locked_until, is_admin, is_active)
                VALUES (?, ?, ?, 0, 0, 0, 1)
                """,
                (username, generate_password_hash(password), _now_text()),
            )
            db.commit()
        except sqlite3.IntegrityError:
            flash("Username already exists.", "error")
            return render_template("register.html")

        user = _get_user_by_username(username)
        if user is None:
            flash("Registered, but auto-login failed. Please login manually.", "success")
            return redirect(url_for("login"))

        _start_session(user)
        flash("Registration successful.", "success")
        return redirect(url_for("workspace"))

    return render_template("register.html")


@app.route("/admin")
@admin_required
def admin_panel():
    db = get_db()
    current_user = _load_current_user()
    now_ts = _now_ts()

    raw_users = db.execute(
        """
        SELECT
            u.id,
            u.username,
            u.created_at,
            u.failed_login_count,
            u.locked_until,
            u.is_admin,
            u.is_active,
            COUNT(e.id) AS entries_count
        FROM users u
        LEFT JOIN entries e ON e.user_id = u.id
        GROUP BY u.id, u.username, u.created_at, u.failed_login_count, u.locked_until, u.is_admin, u.is_active
        ORDER BY u.created_at DESC
        """
    ).fetchall()

    users = []
    for row in raw_users:
        item = dict(row)
        locked_until = int(item["locked_until"])
        item["lock_remaining"] = max(0, locked_until - now_ts)
        users.append(item)

    stats = db.execute(
        """
        SELECT
            (SELECT COUNT(1) FROM users) AS total_users,
            (SELECT COUNT(1) FROM users WHERE is_active = 1) AS active_users,
            (SELECT COUNT(1) FROM users WHERE is_active = 0) AS disabled_users,
            (SELECT COUNT(1) FROM entries) AS total_entries
        """
    ).fetchone()

    return render_template(
        "admin.html",
        active_page="admin",
        users=users,
        stats=stats,
        current_admin_id=int(current_user["id"]) if current_user is not None else 0,
    )


@app.route("/admin/user/<int:user_id>/toggle-active", methods=["POST"])
@admin_required
def admin_toggle_user_active(user_id: int):
    db = get_db()
    current_user = _load_current_user()
    target = _get_user_by_id(user_id)
    if target is None:
        flash("User does not exist.", "error")
        return redirect(url_for("admin_panel"))

    target_id = int(target["id"])
    if current_user is not None and target_id == int(current_user["id"]) and bool(target["is_active"]):
        flash("You cannot disable your current admin account.", "error")
        return redirect(url_for("admin_panel"))

    if bool(target["is_admin"]) and current_user is not None and target_id != int(current_user["id"]):
        flash("Changing other admin accounts is blocked for safety.", "error")
        return redirect(url_for("admin_panel"))

    new_active = 0 if bool(target["is_active"]) else 1
    db.execute(
        "UPDATE users SET is_active = ?, failed_login_count = 0, locked_until = 0 WHERE id = ?",
        (new_active, target_id),
    )
    db.commit()
    flash(f"User {target['username']} has been {'enabled' if new_active else 'disabled'}.", "success")
    return redirect(url_for("admin_panel"))


@app.route("/admin/user/<int:user_id>/unlock", methods=["POST"])
@admin_required
def admin_unlock_user(user_id: int):
    db = get_db()
    current_user = _load_current_user()
    target = _get_user_by_id(user_id)
    if target is None:
        flash("User does not exist.", "error")
        return redirect(url_for("admin_panel"))

    if bool(target["is_admin"]) and current_user is not None and int(target["id"]) != int(current_user["id"]):
        flash("Unlocking other admin accounts is blocked.", "error")
        return redirect(url_for("admin_panel"))

    db.execute(
        "UPDATE users SET failed_login_count = 0, locked_until = 0 WHERE id = ?",
        (int(target["id"]),),
    )
    db.commit()
    flash(f"User {target['username']} has been unlocked.", "success")
    return redirect(url_for("admin_panel"))


@app.route("/admin/user/<int:user_id>/reset-password", methods=["POST"])
@admin_required
def admin_reset_password(user_id: int):
    db = get_db()
    current_user = _load_current_user()
    target = _get_user_by_id(user_id)
    if target is None:
        flash("User does not exist.", "error")
        return redirect(url_for("admin_panel"))

    if bool(target["is_admin"]) and current_user is not None and int(target["id"]) != int(current_user["id"]):
        flash("Resetting other admin passwords is blocked.", "error")
        return redirect(url_for("admin_panel"))

    temp_password = _generate_temp_password(12)
    db.execute(
        """
        UPDATE users
        SET password_hash = ?, failed_login_count = 0, locked_until = 0
        WHERE id = ?
        """,
        (generate_password_hash(temp_password), int(target["id"])),
    )
    db.commit()
    flash(f"Temporary password for {target['username']}: {temp_password}", "success")
    return redirect(url_for("admin_panel"))


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for("landing"))


_init_db()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=False)


