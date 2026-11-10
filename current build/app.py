"""FAST: muscle fatigue from surface EMG.

The server is deliberately thin. Reading a file is `ingest.py`, the method is
`engine.py`, storage is `store.py`. This module only moves data between HTTP and
those three, and serves the page in `templates/index.html`.

Every route called by the page is listed here. Nothing else calls them.
"""
from __future__ import annotations

import hmac
import os
import threading
import time
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import engine
import ingest
import version
from store import (SessionStore, Store, StoreError, legacy_pin_hash,
                   verify_pin)

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
# threadpool, so it is capped as well as the analysis.
ingest_slots = threading.BoundedSemaphore(MAX_ANALYSES_AT_ONCE)


# ── models ───────────────────────────────────────────────────────────────────

class ProfileIn(BaseModel):
    name: str = Field(min_length=1, max_length=30)
    pin: str | None = Field(default=None, max_length=8)


class VerifyIn(BaseModel):
    id: int
    pin: str = Field(default='', max_length=8)


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
    centroid: float | None = None


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
        return None, _json({'error': 'Choose your profile and enter its PIN first'},
                           status=401)
    return pid, None


def _valid_pin(pin: str) -> bool:
    return bool(pin) and len(pin) == 4 and pin.isascii() and pin.isdigit()


def _valid_name(name: str) -> bool:
    return 1 <= len(name) <= 30 and all(ch.isprintable() for ch in name)


# ── pages and assets ─────────────────────────────────────────────────────────

@app.get('/')
async def index():
    return FileResponse(BASE / 'templates' / 'index.html', media_type='text/html')


@app.get('/demo.mp4')
async def demo_video():
    return FileResponse(BASE / 'demo.mp4', media_type='video/mp4')


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
    filename = (file.filename or '')[:120]
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

    if not ingest_slots.acquire(timeout=60):
        return _json({'error': 'The server is busy reading files. Try again in a '
                               'moment.'}, status=503)
    try:
        # The athlete's rate goes in, so a file without one can still have its
        # gaps interpolated in seconds rather than in samples.
        parsed = ingest.load(content, filename,
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


# ── profiles ─────────────────────────────────────────────────────────────────

@app.get('/profiles')
async def list_profiles():
    return {'profiles': store.list_profiles()}


@app.get('/profiles/me')
async def whoami(request: Request):
    pid = _profile_id(request)
    if pid is None:
        return _json({'error': 'Not signed in'}, status=401)
    row = store.profile(pid)
    if not row:
        return _json({'error': 'Not signed in'}, status=401)
    return {'id': row['id'], 'name': row['name']}


# These two routes are plain `def` on purpose: scrypt takes about 50 ms and
# would block every other request if it ran on the event loop.
@app.post('/profiles')
def create_profile(body: ProfileIn):
    name = body.name.strip()
    if not _valid_name(name):
        return _json({'error': 'Use a name of up to 30 characters'}, status=400)
    pin = (body.pin or '').strip() or None
    if pin and not _valid_pin(pin):
        return _json({'error': 'The PIN must be four digits'}, status=400)
    if store.profile_by_name(name):
        return _json({'error': 'That name is already taken'}, status=409)
    try:
        created = store.create_profile(name, pin)
    except StoreError as exc:
        return _json({'error': str(exc)}, status=409)
    return created


@app.post('/profiles/verify')
def verify_profile(body: VerifyIn, request: Request):
    row = store.profile(body.id)
    if not row:
        return _json({'error': 'Profile not found'}, status=404)
    # Keyed by client as well as profile, so one person guessing cannot lock the
    # owner out of their own profile.
    key = f'{body.id}:{request.client.host if request.client else "?"}'
    if store.attempts.blocked(key):
        return _json({'error': 'Too many wrong PINs. Wait fifteen minutes.'},
                     status=429)
    if row['pin_hash']:
        pin = (body.pin or '').strip()
        if not _valid_pin(pin):
            store.attempts.record(key)
            return _json({'error': 'Wrong PIN'}, status=403)
        if row['pin_salt']:
            ok = verify_pin(pin, row['pin_hash'], row['pin_salt'])
        else:
            # A profile created by the previous build: unsalted sha256. Accept
            # it once, then store the salted hash instead.
            ok = hmac.compare_digest(legacy_pin_hash(pin), row['pin_hash'])
            if ok:
                store.upgrade_pin(row['id'], pin)
        if not ok:
            store.attempts.record(key)
            return _json({'error': 'Wrong PIN'}, status=403)
    store.attempts.clear(key)
    token = store.issue_token(body.id)
    response = _json({'id': row['id'], 'name': row['name']})
    secure = (request.url.scheme == 'https'
              or request.headers.get('x-forwarded-proto') == 'https')
    response.set_cookie(COOKIE, token, max_age=3600, httponly=True,
                        samesite='lax', path='/', secure=secure)
    return response


@app.post('/profiles/logout')
async def logout(request: Request):
    store.revoke_token(request.cookies.get(COOKIE))
    response = _json({'ok': True})
    response.delete_cookie(COOKIE, path='/')
    return response


@app.delete('/profiles/{profile_id}')
async def delete_profile(profile_id: int, request: Request):
    own, error = _require_profile(request)
    if error:
        return error
    if own != profile_id:
        return _json({'error': 'You can only delete the profile you are signed in to'},
                     status=403)
    if not store.delete_profile(profile_id):
        return _json({'error': 'Profile not found'}, status=404)
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
        if length.isdigit() and int(length) > MAX_UPLOAD_BYTES + 65536:
            return _json({'error': f'That file is larger than '
                                   f'{MAX_UPLOAD_BYTES // (1024 * 1024)} MB. '
                                   f'Export a shorter recording.'}, status=413)
    return await call_next(request)


@app.middleware('http')
async def security_headers(request: Request, call_next):
    response = await call_next(request)
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
