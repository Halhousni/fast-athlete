"""Storage: upload sessions in memory, profiles and results in SQLite.

Two deliberate choices:

* Uploads live only in this process. They used to be pickled to a temporary
  directory as well, which survived nothing (the directory is new on every
  start) and filled the container's writable layer. An upload belongs to one
  assessment; 30 minutes is enough.
* A profile PIN gates a short-lived session token. The token is what /history,
  /save and DELETE check, so the guard is on the server rather than in the
  browser.
"""
from __future__ import annotations

import datetime
import hashlib
import hmac
import os
import secrets
import sqlite3
import time
import uuid

import numpy as np

SESSION_TTL_S = 30 * 60
MAX_SESSIONS = 20
TOKEN_TTL_S = 60 * 60
PIN_ATTEMPTS = 5
PIN_WINDOW_S = 15 * 60
SCRYPT_N, SCRYPT_R, SCRYPT_P = 2 ** 14, 8, 1


class StoreError(Exception):
    pass


# ── upload sessions ──────────────────────────────────────────────────────────

class SessionStore:
    def __init__(self, ttl: int = SESSION_TTL_S, limit: int = MAX_SESSIONS):
        self._items: dict[str, dict] = {}
        self.ttl, self.limit = ttl, limit

    def put(self, channels: dict[str, np.ndarray], fs: float, filename: str,
            notes: list[str]) -> str:
        sid = uuid.uuid4().hex
        self._items[sid] = dict(channels=channels, fs=float(fs), filename=filename,
                                notes=notes, created=time.time())
        self._evict()
        return sid

    def get(self, sid: str) -> dict | None:
        item = self._items.get(sid)
        if not item:
            return None
        if time.time() - item['created'] > self.ttl:
            self._items.pop(sid, None)
            return None
        return item

    def _evict(self) -> None:
        now = time.time()
        for key in [k for k, v in self._items.items() if now - v['created'] > self.ttl]:
            self._items.pop(key, None)
        while len(self._items) > self.limit:
            oldest = min(self._items, key=lambda k: self._items[k]['created'])
            self._items.pop(oldest, None)

    def bytes_held(self) -> int:
        return sum(v['channels'][c].nbytes for v in self._items.values()
                   for c in v['channels'])


# ── PIN handling ─────────────────────────────────────────────────────────────

def hash_pin(pin: str, salt: bytes | None = None) -> tuple[str, str]:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(pin.encode(), salt=salt, n=SCRYPT_N, r=SCRYPT_R,
                            p=SCRYPT_P, dklen=32)
    return digest.hex(), salt.hex()


def verify_pin(pin: str, pin_hash: str, salt_hex: str) -> bool:
    try:
        salt = bytes.fromhex(salt_hex)
    except (TypeError, ValueError):
        return False
    candidate, _ = hash_pin(pin, salt)
    return hmac.compare_digest(candidate, pin_hash)


def legacy_pin_hash(pin: str) -> str:
    """The previous build's PIN hash, kept so existing profiles still open."""
    return hashlib.sha256(('fast|' + pin).encode()).hexdigest()


class _Attempts:
    """A small in-process throttle. Restarting the container clears it."""

    def __init__(self, limit: int = PIN_ATTEMPTS, window: int = PIN_WINDOW_S):
        self.limit, self.window = limit, window
        self._hits: dict[int, list[float]] = {}

    def blocked(self, profile_id: int) -> bool:
        now = time.time()
        hits = [t for t in self._hits.get(profile_id, []) if now - t < self.window]
        self._hits[profile_id] = hits
        return len(hits) >= self.limit

    def record(self, profile_id: int) -> None:
        self._hits.setdefault(profile_id, []).append(time.time())

    def clear(self, profile_id: int) -> None:
        self._hits.pop(profile_id, None)


# ── database ─────────────────────────────────────────────────────────────────

class Store:
    def __init__(self, data_dir: str):
        os.makedirs(data_dir, exist_ok=True)
        self.path = os.path.join(data_dir, 'fast.db')
        self.attempts = _Attempts()
        self._init()

    def _db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA foreign_keys = ON')
        conn.execute('PRAGMA journal_mode = WAL')
        conn.execute('PRAGMA busy_timeout = 5000')
        return conn

    def _init(self) -> None:
        with self._db() as conn:
            conn.executescript('''
            CREATE TABLE IF NOT EXISTS profiles(
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              name TEXT NOT NULL COLLATE NOCASE UNIQUE,
              pin_hash TEXT,
              pin_salt TEXT,
              created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS results(
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              profile_id INTEGER NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
              ts TEXT NOT NULL,
              avg_score REAL,
              status TEXT,
              detail TEXT,
              engine TEXT
            );
            CREATE TABLE IF NOT EXISTS tokens(
              token_hash TEXT PRIMARY KEY,
              profile_id INTEGER NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
              expires_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS results_by_profile ON results(profile_id, ts DESC);
            ''')

    # profiles
    def list_profiles(self) -> list[dict]:
        with self._db() as conn:
            rows = conn.execute('SELECT id, name, pin_hash IS NOT NULL AS has_pin '
                                'FROM profiles ORDER BY name').fetchall()
        return [dict(r) for r in rows]

    def create_profile(self, name: str, pin: str | None) -> dict:
        pin_hash = pin_salt = None
        if pin:
            pin_hash, pin_salt = hash_pin(pin)
        try:
            with self._db() as conn:
                cur = conn.execute(
                    'INSERT INTO profiles(name, pin_hash, pin_salt, created_at) '
                    'VALUES(?,?,?,?)',
                    (name, pin_hash, pin_salt, _now()))
                return dict(id=cur.lastrowid, name=name)
        except sqlite3.IntegrityError as exc:
            raise StoreError('That name is already taken') from exc

    def profile(self, profile_id: int) -> sqlite3.Row | None:
        with self._db() as conn:
            return conn.execute('SELECT * FROM profiles WHERE id=?',
                                (profile_id,)).fetchone()

    def delete_profile(self, profile_id: int) -> bool:
        with self._db() as conn:
            cur = conn.execute('DELETE FROM profiles WHERE id=?', (profile_id,))
            conn.execute('DELETE FROM tokens WHERE profile_id=?', (profile_id,))
            return cur.rowcount > 0

    # tokens
    def issue_token(self, profile_id: int) -> str:
        token = secrets.token_urlsafe(32)
        with self._db() as conn:
            conn.execute('INSERT INTO tokens(token_hash, profile_id, expires_at) '
                         'VALUES(?,?,?)',
                         (_token_hash(token), profile_id, time.time() + TOKEN_TTL_S))
        return token

    def token_profile(self, token: str | None) -> int | None:
        if not token:
            return None
        with self._db() as conn:
            row = conn.execute('SELECT profile_id, expires_at FROM tokens '
                               'WHERE token_hash=?', (_token_hash(token),)).fetchone()
        if not row or row['expires_at'] < time.time():
            return None
        return int(row['profile_id'])

    def revoke_token(self, token: str | None) -> None:
        if not token:
            return
        with self._db() as conn:
            conn.execute('DELETE FROM tokens WHERE token_hash=?', (_token_hash(token),))

    def upgrade_pin(self, profile_id: int, pin: str) -> None:
        """Replace a legacy unsalted hash with scrypt once the owner signs in."""
        pin_hash, pin_salt = hash_pin(pin)
        with self._db() as conn:
            conn.execute('UPDATE profiles SET pin_hash=?, pin_salt=? WHERE id=?',
                         (pin_hash, pin_salt, profile_id))

    # results
    def save_result(self, profile_id: int, avg_score, status: str,
                    detail: dict, engine: str) -> int:
        import json
        with self._db() as conn:
            cur = conn.execute(
                'INSERT INTO results(profile_id, ts, avg_score, status, detail, engine) '
                'VALUES(?,?,?,?,?,?)',
                (profile_id, _now(), avg_score, status, json.dumps(detail), engine))
            return int(cur.lastrowid)

    def history(self, profile_id: int, limit: int = 50) -> list[dict]:
        import json
        with self._db() as conn:
            rows = conn.execute(
                'SELECT id, ts, avg_score, status, detail FROM results '
                'WHERE profile_id=? ORDER BY ts DESC, id DESC LIMIT ?',
                (profile_id, limit)).fetchall()
        out = []
        for row in rows:
            try:
                detail = json.loads(row['detail'] or '{}')
            except json.JSONDecodeError:
                detail = {}
            out.append(dict(id=row['id'], ts=row['ts'], avg_score=row['avg_score'],
                            status=row['status'],
                            filename=detail.get('filename', ''),
                            results=detail.get('results', [])))
        return out


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()
