"""Storage: upload sessions in memory, accounts, sessions and results in SQLite.

Three deliberate choices:

* Uploads live only in this process. They used to be pickled to a temporary
  directory as well, which survived nothing (the directory is new on every
  start) and filled the container's writable layer. An upload belongs to one
  assessment; 30 minutes is enough.
* An account is a username and a password. The password is hashed with scrypt
  and never stored or logged; signing in issues a random session token whose
  sha256 is what the database holds, so a copy of the database cannot be turned
  into a set of working cookies.
* The token is what /save, /history and DELETE check, so the guard is on the
  server rather than in the browser.
"""
from __future__ import annotations

import datetime
import hashlib
import hmac
import os
import secrets
import sqlite3
import threading
import time
import uuid

import numpy as np

SESSION_TTL_S = 30 * 60          # an upload, not a person
MAX_SESSIONS = 20
TOKEN_TTL_S = 30 * 24 * 60 * 60  # a signed-in athlete, for a month
ATTEMPTS = 5
WINDOW_S = 15 * 60
SCRYPT_N, SCRYPT_R, SCRYPT_P = 2 ** 15, 8, 1
SCRYPT_MAXMEM = 96 * 1024 * 1024     # 32 MiB per hash at n=2^15, plus headroom
PASSWORD_MIN = 10
PASSWORD_MAX = 128
MAX_THROTTLE_KEYS = 5000             # attackers choose the keys; the dict must not grow


class StoreError(Exception):
    pass


# ── upload sessions ──────────────────────────────────────────────────────────

class SessionStore:
    """Uploads held in memory.

    Upload and analysis run in the threadpool, so more than one thread touches
    this at a time: every method takes the lock, and `_evict` is only ever
    called with it held.
    """

    def __init__(self, ttl: int = SESSION_TTL_S, limit: int = MAX_SESSIONS):
        self._items: dict[str, dict] = {}
        self._lock = threading.Lock()
        self.ttl, self.limit = ttl, limit

    def put(self, channels: dict[str, np.ndarray], fs: float, filename: str,
            notes: list[str]) -> str:
        sid = uuid.uuid4().hex
        with self._lock:
            self._items[sid] = dict(channels=channels, fs=float(fs), filename=filename,
                                    notes=notes, created=time.time())
            self._evict()
        return sid

    def get(self, sid: str) -> dict | None:
        with self._lock:
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
        with self._lock:
            return sum(v['channels'][c].nbytes for v in self._items.values()
                       for c in v['channels'])


# ── passwords ────────────────────────────────────────────────────────────────

def hash_password(password: str, n: int = SCRYPT_N, r: int = SCRYPT_R,
                  p: int = SCRYPT_P) -> str:
    """One string, carrying the cost as well as the salt.

    Storing the parameters with the hash is what makes the cost raiseable: a
    later build can hash at 2^16 and still verify every row written at 2^15.
    """
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode('utf-8'), salt=salt, n=n, r=r, p=p,
                            dklen=32, maxmem=SCRYPT_MAXMEM)
    return f'scrypt${n}${r}${p}${salt.hex()}${digest.hex()}'


def verify_password(password: str, stored: str | None) -> bool:
    if not stored:
        return False
    try:
        scheme, n, r, p, salt_hex, want = stored.split('$')
        if scheme != 'scrypt':
            return False
        salt = bytes.fromhex(salt_hex)
        digest = hashlib.scrypt(password.encode('utf-8'), salt=salt, n=int(n),
                                r=int(r), p=int(p), dklen=len(want) // 2,
                                maxmem=SCRYPT_MAXMEM)
    except (TypeError, ValueError):
        return False
    return hmac.compare_digest(digest.hex(), want)


# Run once per failed sign-in on an unknown username, so that "no such account"
# and "wrong password" take the same time to answer.
_DUMMY_HASH = hash_password('fast-dummy-password')


def burn_password_check() -> None:
    verify_password('wrong', _DUMMY_HASH)


class _Attempts:
    """A small in-process throttle. Restarting the container clears it."""

    def __init__(self, limit: int = ATTEMPTS, window: int = WINDOW_S):
        self.limit, self.window = limit, window
        self._hits: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def blocked(self, key: str) -> bool:
        now = time.time()
        with self._lock:
            hits = [t for t in self._hits.get(key, []) if now - t < self.window]
            if hits:
                self._hits[key] = hits
            else:
                self._hits.pop(key, None)
            return len(hits) >= self.limit

    def count(self, key: str) -> int:
        now = time.time()
        with self._lock:
            hits = [t for t in self._hits.get(key, []) if now - t < self.window]
            if hits:
                self._hits[key] = hits
            else:
                self._hits.pop(key, None)
            return len(hits)

    def record(self, key: str) -> None:
        with self._lock:
            if len(self._hits) >= MAX_THROTTLE_KEYS:
                # Drop the oldest quarter rather than refuse to record.
                oldest = sorted(self._hits, key=lambda k: self._hits[k][-1])
                for key_ in oldest[:MAX_THROTTLE_KEYS // 4]:
                    self._hits.pop(key_, None)
            self._hits.setdefault(key, []).append(time.time())

    def clear(self, key: str) -> None:
        with self._lock:
            self._hits.pop(key, None)


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
        conn.execute('PRAGMA secure_delete = ON')
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
            # An existing database predates accounts. Add the two columns in
            # place: the table holds the results of every check ever saved, and
            # dropping it to change its shape would take those with it.
            have = {row[1] for row in conn.execute('PRAGMA table_info(profiles)')}
            for column in ('password_hash', 'password_salt'):
                if column not in have:
                    conn.execute(f'ALTER TABLE profiles ADD COLUMN {column} TEXT')

    # accounts
    def create_account(self, username: str, password: str) -> dict:
        password_hash, password_salt = hash_password(password), None
        try:
            with self._db() as conn:
                cur = conn.execute(
                    'INSERT INTO profiles(name, password_hash, password_salt, created_at) '
                    'VALUES(?,?,?,?)',
                    (username, password_hash, password_salt, _now()))
                return dict(id=cur.lastrowid, name=username)
        except sqlite3.IntegrityError as exc:
            raise StoreError('That username is taken') from exc

    def account_by_username(self, username: str) -> sqlite3.Row | None:
        with self._db() as conn:
            return conn.execute('SELECT * FROM profiles WHERE name=?',
                                (username,)).fetchone()

    def account(self, account_id: int) -> sqlite3.Row | None:
        with self._db() as conn:
            return conn.execute('SELECT * FROM profiles WHERE id=?',
                                (account_id,)).fetchone()

    def set_password(self, account_id: int, password: str) -> bool:
        password_hash, password_salt = hash_password(password), None
        with self._db() as conn:
            cur = conn.execute('UPDATE profiles SET password_hash=?, password_salt=? '
                               'WHERE id=?', (password_hash, password_salt, account_id))
            return cur.rowcount > 0

    def delete_account(self, account_id: int) -> bool:
        with self._db() as conn:
            conn.execute('DELETE FROM tokens WHERE profile_id=?', (account_id,))
            conn.execute('DELETE FROM results WHERE profile_id=?', (account_id,))
            cur = conn.execute('DELETE FROM profiles WHERE id=?', (account_id,))
            return cur.rowcount > 0

    # tokens
    def issue_token(self, profile_id: int) -> str:
        token = secrets.token_urlsafe(32)
        now = time.time()
        with self._db() as conn:
            conn.execute('DELETE FROM tokens WHERE expires_at < ?', (now,))
            conn.execute('INSERT INTO tokens(token_hash, profile_id, expires_at) '
                         'VALUES(?,?,?)',
                         (_token_hash(token), profile_id, now + TOKEN_TTL_S))
        return token

    def revoke_all(self, profile_id: int) -> None:
        with self._db() as conn:
            conn.execute('DELETE FROM tokens WHERE profile_id=?', (profile_id,))

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


# ── command line ─────────────────────────────────────────────────────────────
# One job, for the operator: give an account a password, prompting on the tty.
# Nothing secret passes through a shell argument, a file, or a chat message.
#   docker compose exec signal-analyzer python3 store.py passwd hilal

if __name__ == '__main__':
    import getpass
    import sys

    args = sys.argv[1:]
    if len(args) != 2 or args[0] != 'passwd':
        print('usage: python3 store.py passwd <username>')
        raise SystemExit(2)

    st = Store(os.environ.get('FAST_DATA_DIR', '/data'))
    row = st.account_by_username(args[1].strip().lower())
    if not row:
        print('No account called', args[1])
        raise SystemExit(1)
    first = getpass.getpass('New password: ')
    if first != getpass.getpass('Again: '):
        print('They do not match. Nothing changed.')
        raise SystemExit(1)
    if len(first) < PASSWORD_MIN:
        print(f'At least {PASSWORD_MIN} characters. Nothing changed.')
        raise SystemExit(1)
    st.set_password(row['id'], first)
    st.revoke_all(row['id'])
    print('Set. Every existing session for', row['name'], 'has been signed out.')
