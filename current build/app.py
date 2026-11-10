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
    name: str = ''
    column: str = ''
    status: str = 'grey'
    score: int = 0
    nfi: float | None = None
    tip: str = ''
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

    try:
        parsed = ingest.load(content, file.filename or '')
    except ingest.IngestError as exc:
        return _json({'error': str(exc)}, status=400)

    if fs is not None and ingest.FS_MIN <= fs <= ingest.FS_MAX:
        rate, rate_source = float(fs), 'provided'
    elif parsed.fs:
        rate, rate_source = parsed.fs, parsed.fs_source
    else:
        rate, rate_source = None, 'none'

    usable = [c for c in parsed.channels if c['drop_reason'] is None]
    if not usable:
        return _json({'error': 'No channel in this file holds usable data'},
                     status=400)
    dropped = [{'name': c['name'], 'reason': c['drop_reason']}
               for c in parsed.channels if c['drop_reason'] is not None]

    duration = max(len(c['data']) for c in usable) / rate if rate else None
    sid = sessions.put({c['name']: c['data'].astype(np.float32) for c in usable},
                       rate or 0.0, file.filename or '', parsed.notes)

    return {
        'session': sid,
        'filename': file.filename or '',
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
            results[muscle.id] = result
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


@app.post('/profiles')
async def create_profile(body: ProfileIn):
    name = body.name.strip()
    if not _valid_name(name):
        return _json({'error': 'Use a name of up to 30 characters'}, status=400)
    pin = (body.pin or '').strip() or None
    if pin and not _valid_pin(pin):
        return _json({'error': 'The PIN must be four digits'}, status=400)
    try:
        created = store.create_profile(name, pin)
    except StoreError as exc:
        return _json({'error': str(exc)}, status=409)
    return created


@app.post('/profiles/verify')
async def verify_profile(body: VerifyIn):
    row = store.profile(body.id)
    if not row:
        return _json({'error': 'Profile not found'}, status=404)
    if store.attempts.blocked(body.id):
        return _json({'error': 'Too many wrong PINs. Wait fifteen minutes.'},
                     status=429)
    if row['pin_hash']:
        pin = (body.pin or '').strip()
        if not _valid_pin(pin):
            store.attempts.record(body.id)
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
            store.attempts.record(body.id)
            return _json({'error': 'Wrong PIN'}, status=403)
    store.attempts.clear(body.id)
    token = store.issue_token(body.id)
    response = _json({'id': row['id'], 'name': row['name']})
    response.set_cookie(COOKIE, token, max_age=3600, httponly=True,
                        samesite='lax', path='/')
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
    verdict = engine.overall(detail_results)
    if verdict['status'] == 'grey':
        return _json({'error': 'Nothing was measured, so nothing was saved'},
                     status=400)

    rid = store.save_result(body.profile_id, verdict['mean_score'], verdict['status'],
                            dict(filename=body.filename, results=detail_results),
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
