"""FAST: muscle fatigue from surface EMG.

The server is deliberately thin. Reading a file is `ingest.py`, the method is
`engine.py`, storage is `store.py`. This module only moves data between HTTP and
those three, and serves the page in `templates/index.html`.

Every route called by the page is listed here. Nothing else calls them.
"""
from __future__ import annotations

import os
import re
import threading
import time
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import engine
import ingest
import version
from store import (PASSWORD_MAX, PASSWORD_MIN, SessionStore, Store,
                   StoreError, TOKEN_TTL_S, burn_password_check,
                   verify_password)

BASE = Path(__file__).resolve().parent
DATA_DIR = os.environ.get('FAST_DATA_DIR', '/data')
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_MUSCLES = 16
MAX_ANALYSES_AT_ONCE = 2
COOKIE = 'fast_token'

app = FastAPI(title='FAST', docs_url=None, redoc_url=None, openapi_url=None)
app.mount('/static', StaticFiles(directory=BASE / 'static'), name='static')
store = Store(DATA_DIR)
sessions = SessionStore()
analyse_slots = threading.BoundedSemaphore(MAX_ANALYSES_AT_ONCE)
# Reading a file is the expensive part of an upload and it runs in the
# threadpool. One at a time: two 24 MB parses together exceed the container's
# 1 GB (measured 1.1-1.2 GB).
ingest_slots = threading.BoundedSemaphore(1)


# ── models ───────────────────────────────────────────────────────────────────

class MuscleIn(BaseModel):
    id: str = Field(max_length=64)
    column: str = Field(max_length=200)


class AnalyzeIn(BaseModel):
    session: str = Field(min_length=32, max_length=32)
    muscles: list[MuscleIn] = Field(default_factory=list, max_length=MAX_MUSCLES)


class ResultIn(BaseModel):
    id: str = Field(max_length=64)
    name: str = Field(default='', max_length=80)
    column: str = Field(default='', max_length=200)
    status: str = Field(default='grey', max_length=8)
    score: int = Field(default=0, ge=0, le=100)
    nfi: float | None = None
    tip: str = Field(default='', max_length=300)
    # What the verdict is read from.
    level: str = Field(default='', max_length=8)
    delta_mdf: float | None = None
    delta_rms: float | None = None


class SignupIn(BaseModel):
    username: str = Field(min_length=3, max_length=40)
    password: str = Field(min_length=1, max_length=PASSWORD_MAX)


class LoginIn(BaseModel):
    username: str = Field(max_length=40)
    password: str = Field(max_length=128)


class DeleteIn(BaseModel):
    password: str = Field(max_length=128)


class SaveIn(BaseModel):
    profile_id: int
    filename: str = Field(default='', max_length=200)
    results: list[ResultIn] = Field(default_factory=list, max_length=MAX_MUSCLES)


# ── helpers ──────────────────────────────────────────────────────────────────

def _json(payload: dict, status: int = 200) -> JSONResponse:
    return JSONResponse(payload, status_code=status)


def _profile_id(request: Request) -> int | None:
    return store.token_profile(request.cookies.get(COOKIE))


def _require_profile(request: Request) -> tuple[int | None, JSONResponse | None]:
    pid = _profile_id(request)
    if pid is None:
        return None, _json({'error': 'Sign in to do that'}, status=401)
    return pid, None


# ── pages and assets ─────────────────────────────────────────────────────────

_INDEX = {'mtime': None, 'text': None}


@app.get('/')
async def index():
    """Serve the page with this build's id on every asset.

    Without it the browser keeps the old app.js after a deploy and runs it
    against new markup, which looks like a broken feature rather than a stale
    cache.
    """
    global _INDEX
    page_file = BASE / 'templates' / 'index.html'
    stamp = page_file.stat().st_mtime
    if _INDEX['mtime'] != stamp:
        _INDEX = {'mtime': stamp, 'text': page_file.read_text(encoding='utf-8')}
    tag = version.build_id()
    page = _INDEX['text'].replace('{{v}}', tag)
    return HTMLResponse(page, headers={'Cache-Control': 'no-cache'})


@app.get('/demo_recording.csv')
async def demo_recording():
    return FileResponse(BASE / 'demo_recording.csv', media_type='text/csv',
                        filename='demo_recording.csv')


@app.get('/healthz')
async def healthz():
    return {'ok': True, **version.info(engine.ENGINE_VERSION)}


# ── upload and analysis ──────────────────────────────────────────────────────

@app.post('/upload')
def upload(file: UploadFile = File(...), fs: float | None = Form(default=None)):
    started = time.time()
    # The full name decides the format (.mat), so it goes to the parser; only
    # the stored and displayed copy is shortened.
    original = file.filename or ''
    filename = original[:120]
    chunks, total = [], 0
    while True:
        chunk = file.file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            return _json({'error': f'That file is larger than '
                                   f'{MAX_UPLOAD_BYTES // (1024 * 1024)} MB. '
                                   f'Export a shorter recording.'}, status=413)
        chunks.append(chunk)
    content = b''.join(chunks)
    chunks.clear()

    if not ingest_slots.acquire(timeout=60):
        return _json({'error': 'The server is busy reading files. Try again in a '
                               'moment.'}, status=503)
    try:
        # The athlete's rate goes in, so a file without one can still have its
        # gaps interpolated in seconds rather than in samples.
        parsed = ingest.load(content, original,
                             float(fs) if fs is not None else None)
    except ingest.IngestError as exc:
        return _json({'error': str(exc)}, status=400)
    finally:
        ingest_slots.release()

    rate, rate_source = parsed.fs, parsed.fs_source
    if rate is None:
        rate_source = 'none'

    usable = [c for c in parsed.channels if c['drop_reason'] is None]
    if not usable:
        return _json({'error': 'No channel in this file holds usable data'},
                     status=400)
    dropped = [{'name': c['name'], 'reason': c['drop_reason']}
               for c in parsed.channels if c['drop_reason'] is not None]

    duration = max(len(c['data']) for c in usable) / rate if rate else None
    sid = sessions.put({c['name']: c['data'].astype(np.float32) for c in usable},
                       rate or 0.0, filename, parsed.notes)

    return {
        'session': sid,
        'filename': filename,
        'fs': rate,
        'fs_source': rate_source,
        'cols': [c['name'] for c in usable],
        'dropped': dropped,
        'notes': parsed.notes,
        'duration': round(duration, 1) if duration else None,
        'analysed_seconds': round(min(duration, engine.MAX_WINDOW_S), 1) if duration else None,
        'truncated': bool(duration and duration > engine.MAX_WINDOW_S),
        'muscles': ingest.match_muscles([c['name'] for c in usable]),
        'seconds': round(time.time() - started, 2),
    }


@app.post('/analyze')
def analyze(body: AnalyzeIn):
    started = time.time()
    session = sessions.get(body.session)
    if not session:
        return _json({'error': 'That upload has expired. Please choose the file '
                               'again.'}, status=404)
    if not session['fs']:
        return _json({'error': 'This recording has no sampling rate. Set one and '
                               'upload it again.'}, status=400)
    if not body.muscles:
        return _json({'error': 'Pick at least one muscle'}, status=400)

    seen, chosen = set(), []
    for muscle in body.muscles:
        if muscle.column in session['channels'] and muscle.column not in seen:
            seen.add(muscle.column)
            chosen.append(muscle)
    if not chosen:
        return _json({'error': 'None of those channels is in this recording'},
                     status=400)

    if not analyse_slots.acquire(timeout=120):
        return _json({'error': 'The server is busy. Try again in a moment.'},
                     status=503)
    try:
        results: dict[str, dict] = {}
        for muscle in chosen:
            data = session['channels'][muscle.column]
            try:
                result = engine.analyse_channel(data, session['fs'], muscle.id)
            except engine.EngineError as exc:
                result = dict(status='grey', score=0, nfi=None, tip=str(exc),
                              analysed_seconds=round(len(data) / session['fs'], 1))
            result['id'] = muscle.id
            result['column'] = muscle.column
            # Keyed by column, which is unique per upload: two channels
            # can map to the same muscle id.
            results[muscle.column] = result
    finally:
        analyse_slots.release()

    return {'results': results,
            'overall': engine.overall(list(results.values())),
            'engine': engine.ENGINE_VERSION,
            'seconds': round(time.time() - started, 2)}


# ── accounts ─────────────────────────────────────────────────────────────────

USERNAME_RE = re.compile(r'^[a-z0-9][a-z0-9_.-]{2,19}$')
# Ten is the floor that stops the obvious guesses without making an athlete
# invent a symbol. A short blocklist covers the rest of the first hundred.
WEAK_PASSWORDS = {
    '1234567890', 'qwertyuiop', 'password12', 'password123', 'iloveyou12',
    'football12', 'baseball12', 'sunshine12', 'princess12', 'letmein123',
    'fastathlete', 'athlete123', 'musclestrong',
}


def _username(raw: str) -> str:
    return (raw or '').strip().lower()


def _password_problem(username: str, password: str) -> str | None:
    if len(password) < PASSWORD_MIN:
        return f'Use at least {PASSWORD_MIN} characters'
    if len(password) > PASSWORD_MAX:
        return f'That is longer than {PASSWORD_MAX} characters'
    if password.lower() == username:
        return 'The password cannot be your username'
    if password.lower() in WEAK_PASSWORDS:
        return 'That password is too common. Pick another.'
    return None


def _client(request: Request) -> str:
    return request.client.host if request.client else '?'


COOKIE_SECURE = os.environ.get('FAST_COOKIE_SECURE', '').strip() == '1'


def _sign_in_response(request: Request, account) -> JSONResponse:
    token = store.issue_token(account['id'])
    response = _json({'id': account['id'], 'name': account['name']})
    # Behind the proxy the app can see plain http even when the browser did not,
    # so production sets this explicitly rather than trusting the scheme.
    secure = COOKIE_SECURE or (request.url.scheme == 'https'
                               or request.headers.get('x-forwarded-proto') == 'https')
    response.set_cookie(COOKIE, token, max_age=TOKEN_TTL_S, httponly=True,
                        samesite='lax', path='/', secure=secure)
    return response


# These take plain `def` on purpose: scrypt is about 50 ms of CPU and would
# block every other request if it ran on the event loop.
@app.post('/auth/signup')
def signup(body: SignupIn, request: Request):
    username = _username(body.username)
    if not USERNAME_RE.match(username):
        return _json({'error': 'Usernames are 3 to 20 characters: letters, '
                               'numbers, dot, dash, underscore'}, status=400)
    problem = _password_problem(username, body.password)
    if problem:
        return _json({'error': problem}, status=400)

    key = f'signup:{_client(request)}'
    if store.attempts.blocked(key):
        return _json({'error': 'Too many attempts. Wait fifteen minutes.'}, status=429)
    store.attempts.record(key)

    try:
        created = store.create_account(username, body.password)
    except StoreError as exc:
        return _json({'error': str(exc)}, status=409)
    account = store.account(created['id'])
    return _sign_in_response(request, account)


@app.post('/auth/login')
def login(body: LoginIn, request: Request):
    username = _username(body.username)
    # Keyed by the pair, so one person guessing cannot lock the owner out from
    # another machine, and one machine cannot walk the whole user list.
    key = f'login:{username}:{_client(request)}'
    # Behind the proxy every request can look like it comes from the proxy, so
    # the pair alone would let one attacker lock an account for everyone else.
    # The name gets its own, looser cap; the pair stays tight.
    name_key = f'login-name:{username}'
    if store.attempts.blocked(key) or store.attempts.count(name_key) >= 12:
        return _json({'error': 'Too many attempts. Wait fifteen minutes.'}, status=429)

    row = store.account_by_username(username) if username else None
    store.attempts.record(name_key)
    if not row or not row['password_hash']:
        burn_password_check()          # same time as a real check
        store.attempts.record(key)
        # A stranger learns nothing about which half was wrong.
        return _json({'error': 'Wrong username or password'}, status=401)
    if not verify_password(body.password, row['password_hash']):
        store.attempts.record(key)
        return _json({'error': 'Wrong username or password'}, status=401)

    store.attempts.clear(key)
    store.attempts.clear(name_key)
    return _sign_in_response(request, row)


@app.post('/auth/logout')
async def logout(request: Request):
    store.revoke_token(request.cookies.get(COOKIE))
    response = _json({'ok': True})
    response.delete_cookie(COOKIE, path='/')
    return response


@app.get('/auth/me')
async def me(request: Request):
    pid = _profile_id(request)
    if pid is None:
        return _json({'error': 'Not signed in'}, status=401)
    row = store.account(pid)
    if not row:
        return _json({'error': 'Not signed in'}, status=401)
    return {'id': row['id'], 'name': row['name'], 'created_at': row['created_at']}


@app.delete('/auth/account')
def delete_account(body: DeleteIn, request: Request):
    """Deleting an account asks for the password again, then takes the results
    and every session with it."""
    pid = _profile_id(request)
    if pid is None:
        return _json({'error': 'Not signed in'}, status=401)
    row = store.account(pid)
    if not row:
        return _json({'error': 'Not signed in'}, status=401)
    if not verify_password(body.password, row['password_hash']):
        return _json({'error': 'Wrong password'}, status=403)
    if not store.delete_account(pid):
        return _json({'error': 'Account not found'}, status=404)
    response = _json({'ok': True})
    response.delete_cookie(COOKIE, path='/')
    return response


# ── results ──────────────────────────────────────────────────────────────────

@app.post('/save')
async def save_result(body: SaveIn, request: Request):
    own, error = _require_profile(request)
    if error:
        return error
    if own != body.profile_id:
        return _json({'error': 'That is not your profile'}, status=403)
    if not body.results:
        return _json({'error': 'Nothing to save'}, status=400)

    detail_results = [r.model_dump() for r in body.results]
    for row in detail_results:
        # The score drives the colour, always. A client that sends green with a
        # score of 5 is stored as red.
        row['status'] = ('grey' if row.get('nfi') is None
                         else engine.status_from_score(row['score']))
    verdict = engine.overall(detail_results)
    if verdict['status'] == 'grey':
        return _json({'error': 'Nothing was measured, so nothing was saved'},
                     status=400)

    rid = store.save_result(body.profile_id, verdict['mean_score'], verdict['status'],
                            dict(filename=body.filename[:120], results=detail_results),
                            engine.ENGINE_VERSION)
    return {'id': rid, 'avg_score': verdict['mean_score'], 'status': verdict['status']}


@app.get('/history')
async def history(request: Request, profile_id: int | None = None):
    own, error = _require_profile(request)
    if error:
        return error
    if profile_id is not None and profile_id != own:
        return _json({'error': 'That is not your profile'}, status=403)
    assert own is not None
    return {'history': store.history(own)}


# ── security headers ─────────────────────────────────────────────────────────

# Registered last, so it wraps everything else and runs before the multipart
# parser touches the body: Starlette spools a file part to a temporary file with
# no size limit, so a huge POST would fill the container before /upload could
# answer 413.
@app.middleware('http')
async def body_limit(request: Request, call_next):
    if request.url.path == '/upload':
        length = request.headers.get('content-length', '')
        if not length.isdigit():
            # Without a declared length the body is streamed past this check and
            # spooled to disk by the multipart parser. Browsers and curl -F
            # always send one.
            return _json({'error': 'Send the file with a Content-Length header.'},
                         status=411)
        if int(length) > MAX_UPLOAD_BYTES + 65536:
            return _json({'error': f'That file is larger than '
                                   f'{MAX_UPLOAD_BYTES // (1024 * 1024)} MB. '
                                   f'Export a shorter recording.'}, status=413)
    return await call_next(request)


@app.middleware('http')
async def security_headers(request: Request, call_next):
    # A request that changes something must come from a page on this host. The
    # endpoints take JSON, which a cross-site form cannot send, so this is a
    # second lock on the same door rather than the only one.
    if request.method in ('POST', 'PUT', 'PATCH', 'DELETE'):
        origin = request.headers.get('origin')
        if origin and origin != 'null' and origin.split('://')[-1] != request.headers.get('host', ''):
            return _json({'error': 'Cross-origin request refused'}, status=403)

    response = await call_next(request)
    if request.url.path.startswith(('/auth', '/history', '/save')):
        response.headers['Cache-Control'] = 'no-store'
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; img-src 'self' data:; media-src 'self'; "
        "style-src 'self'; script-src 'self'; connect-src 'self'; "
        "form-action 'self'; frame-ancestors 'none'; base-uri 'none'")
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Referrer-Policy'] = 'no-referrer'
    response.headers['X-Frame-Options'] = 'DENY'
    return response


if __name__ == '__main__':
    uvicorn.run(app, host=os.environ.get('FAST_BIND', '127.0.0.1'),
                port=int(os.environ.get('FAST_PORT', '8501')))
