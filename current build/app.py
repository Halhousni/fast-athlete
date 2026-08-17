"""FAST — Muscle Fatigue Check

Material Design 3 frontend + FastAPI backend.
Filter & Aggregate Synchrosqueezed Transform.
"""
from __future__ import annotations
import asyncio, io, time, re, itertools, base64, json, tempfile, pickle, os, sqlite3, hashlib, datetime
import numpy as np
import scipy.io
import pandas as pd
from scipy.signal import butter, filtfilt
from ssqueezepy import ssq_cwt
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
from bowen_centroid import run_bowen_pipeline

# ═══════════════════════════════════════════════════
# FAST ENGINE
# ═══════════════════════════════════════════════════

def _process_single_band(f1, x, fs, voices_per_octave=32, order=4):
    bandedge = 1.0
    filband1, filband2 = max(f1 - bandedge, 1.0), f1 + bandedge + 1.0
    nyq = 0.5 * fs
    if filband1 >= filband2 or filband2 >= nyq: return None
    b, a = butter(max(1, order // 2), [filband1 / nyq, filband2 / nyq], btype='band')
    Tx, _, _, _ = ssq_cwt(filtfilt(b, a, x), fs=fs, nv=32, wavelet=('morlet', {'mu': 6}))
    return Tx

def _fast_aggregate(x, fs, freq_steps, voices_per_octave=32):
    W_agg, n_valid = None, 0
    for f in freq_steps:
        Tx = _process_single_band(f, x, fs, voices_per_octave)
        if Tx is not None:
            if W_agg is None: W_agg = np.zeros_like(Tx, dtype=complex)
            W_agg += Tx; n_valid += 1
    if W_agg is not None and n_valid > 1: W_agg /= n_valid
    return W_agg

def _run_fast(sig, fs, f_min, f_max, df_step=1.0, voices_per_octave=32):
    from scipy.signal import decimate
    target_fs = 200.0
    if fs > target_fs:
        q = int(fs / target_fs); sig = decimate(sig, q, ftype='iir'); fs = target_fs
    freq_steps = np.arange(f_min, f_max, df_step)
    Tx_orig, _, ssq_freqs, _ = ssq_cwt(sig, fs=fs, nv=voices_per_octave, wavelet=('morlet', {'mu': 6}))
    W_agg = _fast_aggregate(sig, fs, freq_steps, voices_per_octave)
    mean_power = np.mean(np.abs(Tx_orig) ** 2)
    mask = (np.abs(Tx_orig) ** 2 > 0.9 * mean_power).astype(float)
    if W_agg is None: W_agg = np.zeros_like(Tx_orig, dtype=complex)
    return W_agg * mask, Tx_orig, ssq_freqs

def _load_file(content, name):
    from io import BytesIO
    detected_fs, data_df = None, None
    if name.endswith('.mat'):
        mat = scipy.io.loadmat(BytesIO(content))
        for k in mat.keys():
            if re.search(r'^fs$|samp|rate', k, re.I):
                try: detected_fs = float(np.array(mat[k]).ravel()[0])
                except Exception: pass
        valid = {k: mat[k].flatten() for k in mat
                 if not k.startswith('__') and mat[k].ndim <= 2
                 and np.issubdtype(mat[k].dtype, np.number)}
        if valid:
            ml = min(len(v) for v in valid.values())
            data_df = pd.DataFrame({k: v[:ml] for k, v in valid.items()})
    else:
        raw = content.decode('utf-8', errors='replace')
        for line in raw.splitlines()[:20]:
            m = re.search(r'(?:fs|freq|sample.?rate|sampling.?rate)[^\d]*(\d+(?:\.\d+)?)', line, re.I)
            if m: detected_fs = float(m.group(1)); break
        for sep in [',', '\t', ';']:
            for skip in range(6):
                try:
                    df_try = pd.read_csv(BytesIO(content), sep=sep, engine='python',
                                         comment='#', skiprows=skip, on_bad_lines='skip')
                    nc = df_try.select_dtypes(include=[np.number]).columns
                    if len(nc) >= 1: data_df = df_try[nc]; break
                except Exception: pass
            if data_df is not None: break
        if data_df is not None:
            data_df.columns = [c.replace('"', '').strip() for c in data_df.columns]
            data_df = data_df.apply(pd.to_numeric, errors='coerce').fillna(0)
    return data_df, detected_fs

def _muscle_columns(data_df):
    excl = re.compile(r'time|marker|trigger|sync|ref|event|frame|sample', re.I)
    return [c for c in data_df.columns if not excl.search(c)]

def _mif_time_series(Tx_fast, ssq_freqs, f_min=1.0, f_max=35.0):
    band = (ssq_freqs >= f_min) & (ssq_freqs <= f_max)
    freqs_b, power_b = ssq_freqs[band], np.abs(Tx_fast[band, :]) ** 2
    valid = power_b.sum(axis=0) > 0
    mif = np.full(power_b.shape[1], np.nan)
    mif[valid] = (power_b[:, valid] * freqs_b[:, None]).sum(axis=0) / power_b[:, valid].sum(axis=0)
    return mif

def _classify_fatigue(Tx_fast, ssq_freqs,
                      f_min=1.0, f_max=35.0, baseline_s=5.0,
                      green_nfi=15.0, amber_nfi=35.0):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mif_time = _mif_time_series(Tx_fast, ssq_freqs, f_min, f_max)
        valid_t = ~np.isnan(mif_time)
        if not valid_t.any():
            return dict(status='grey', score=0, tip='No signal energy in band')
        mif_median = float(np.median(mif_time[valid_t]))
        n_time = len(mif_time)
        n_baseline = max(3, min(int(baseline_s * 200), n_time))
        baseline_slice = mif_time[:n_baseline]
        bl_valid = ~np.isnan(baseline_slice)
        mif_baseline = float(np.median(baseline_slice[bl_valid])) if bl_valid.any() else mif_median
        nfi_mean = (mif_baseline - mif_median) / mif_baseline * 100.0 if mif_baseline > 0 else np.nan
        if np.isnan(nfi_mean):
            status, score, tip = 'grey', 0, 'Could not compute'
        elif nfi_mean < green_nfi:
            status, score, tip = 'green', int(100 - nfi_mean * 2), 'No fatigue detected'
        elif nfi_mean < amber_nfi:
            status, score, tip = 'amber', int(100 - nfi_mean * 2), 'Early fatigue signs'
        else:
            status, score, tip = 'red', max(0, int(100 - nfi_mean * 2)), 'Significant fatigue'
        return dict(status=status, score=max(0, min(100, score)), tip=tip)

# ═══════════════════════════════════════════════════
# MUSCLE DATA
# ═══════════════════════════════════════════════════

MUSCLES = [
    dict(id='VL', name='Vastus Lateralis', desc='Outer quad'),
    dict(id='VM', name='Vastus Medialis', desc='Inner quad'),
    dict(id='RF', name='Rectus Femoris', desc='Front quad'),
    dict(id='BF', name='Biceps Femoris', desc='Hamstring'),
    dict(id='ST', name='Semitendinosus', desc='Inner hamstring'),
    dict(id='TA', name='Tibialis Anterior', desc='Shin'),
    dict(id='GM', name='Gastrocnemius Med.', desc='Inner calf'),
    dict(id='GL', name='Gastrocnemius Lat.', desc='Outer calf'),
]

# ═══════════════════════════════════════════════════
# FASTAPI
# ═══════════════════════════════════════════════════

app = FastAPI(title='FAST')
session_store = {}
SESSION_DIR = tempfile.mkdtemp(prefix='fast_sessions_')

def _save_session(sid, data):
    with open(os.path.join(SESSION_DIR, sid), 'wb') as f:
        pickle.dump(data, f)
    session_store[sid] = data

def _load_session(sid):
    if sid in session_store:
        return session_store[sid]
    path = os.path.join(SESSION_DIR, sid)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        session_store[sid] = data
        return data
    return None

# ═══════════════════════════════════════════════════
# PROFILES + SAVED RESULTS (SQLite)
# ═══════════════════════════════════════════════════

DATA_DIR = os.environ.get('FAST_DATA_DIR', '/data')
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, 'fast.db')

def _db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA foreign_keys = ON')
    return conn

def _init_db():
    conn = _db()
    conn.executescript('''
    CREATE TABLE IF NOT EXISTS profiles(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT NOT NULL UNIQUE,
      pin_hash TEXT,
      created_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS results(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      profile_id INTEGER NOT NULL REFERENCES profiles(id),
      ts TEXT NOT NULL,
      avg_score REAL,
      status TEXT,
      detail TEXT
    );
    ''')
    conn.commit()
    conn.close()

_init_db()

def _hash_pin(pin):
    return hashlib.sha256(('fast|' + pin).encode()).hexdigest()

@app.get('/api/profiles')
async def list_profiles():
    conn = _db()
    rows = conn.execute(
        'SELECT id, name, pin_hash IS NOT NULL AS has_pin FROM profiles ORDER BY name'
    ).fetchall()
    conn.close()
    return {'profiles': [dict(r) for r in rows]}

@app.post('/api/profiles')
async def create_profile(data: dict):
    name = (data.get('name') or '').strip()
    if not name:
        return JSONResponse({'error': 'Name is required'}, status_code=400)
    pin = (data.get('pin') or '').strip()
    if pin and (not pin.isdigit() or len(pin) != 4):
        return JSONResponse({'error': 'PIN must be 4 digits'}, status_code=400)
    conn = _db()
    try:
        cur = conn.execute(
            'INSERT INTO profiles(name, pin_hash, created_at) VALUES(?,?,?)',
            (name, _hash_pin(pin) if pin else None,
             datetime.datetime.now().isoformat(timespec='seconds')))
        conn.commit()
        pid = cur.lastrowid
    except sqlite3.IntegrityError:
        conn.close()
        return JSONResponse({'error': 'That name is already taken'}, status_code=409)
    conn.close()
    return {'id': pid, 'name': name}

@app.post('/api/profiles/verify')
async def verify_profile(data: dict):
    pid = data.get('id')
    pin = (data.get('pin') or '').strip()
    conn = _db()
    row = conn.execute('SELECT name, pin_hash FROM profiles WHERE id=?', (pid,)).fetchone()
    conn.close()
    if not row:
        return JSONResponse({'error': 'Profile not found'}, status_code=404)
    if row['pin_hash'] and row['pin_hash'] != _hash_pin(pin):
        return JSONResponse({'error': 'Wrong PIN'}, status_code=403)
    return {'id': pid, 'name': row['name']}

@app.post('/api/save')
async def save_result(data: dict):
    pid = data.get('profile_id')
    if not pid:
        return JSONResponse({'error': 'No profile'}, status_code=400)
    detail = json.dumps({
        'filename': data.get('filename', ''),
        'results': data.get('results', []),
    })
    conn = _db()
    try:
        cur = conn.execute(
            'INSERT INTO results(profile_id, ts, avg_score, status, detail) VALUES(?,?,?,?,?)',
            (pid, datetime.datetime.now().isoformat(timespec='seconds'),
             data.get('avg_score'), data.get('status', ''), detail))
        conn.commit()
        rid = cur.lastrowid
    except sqlite3.IntegrityError:
        conn.close()
        return JSONResponse({'error': 'Profile no longer exists'}, status_code=404)
    conn.close()
    return {'id': rid}

@app.get('/api/history')
async def history(profile_id: int):
    conn = _db()
    rows = conn.execute(
        'SELECT id, ts, avg_score, status, detail FROM results '
        'WHERE profile_id=? ORDER BY ts DESC LIMIT 50', (profile_id,)).fetchall()
    conn.close()
    out = []
    for r in rows:
        d = json.loads(r['detail'] or '{}')
        out.append({
            'id': r['id'], 'ts': r['ts'],
            'filename': d.get('filename', ''),
            'avg_score': r['avg_score'], 'status': r['status'],
            'results': d.get('results', []),
        })
    return {'history': out}

@app.get('/', response_class=HTMLResponse)
async def index():
    return HTML_PAGE

@app.post('/api/upload')
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    try:
        df, fs = _load_file(content, file.filename)
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=400)
    if df is None or df.empty:
        return JSONResponse({'error': 'No numeric data'}, status_code=400)
    cols = _muscle_columns(df)
    if not cols:
        return JSONResponse({'error': 'No channels detected'}, status_code=400)
    sid = base64.urlsafe_b64encode(file.filename.encode()).decode()[:16]
    # Store as serializable types
    _save_session(sid, {
        'data': {c: df[c].tolist() for c in cols},
        'fs': int(fs) if fs else 1000,
        'cols': cols,
        'filename': file.filename,
    })
    return JSONResponse({
        'session': sid,
        'filename': file.filename,
        'fs': int(fs) if fs else 1000,
        'cols': cols,
        'duration': len(df) / (fs or 1000),
        'muscles': _match_columns(cols),
    })

@app.post('/api/analyze')
async def analyze(data: dict):
    print(f"ANALYZE called: session={data.get('session','?')[:8]}..., muscles={len(data.get('muscles',[]))}", flush=True)
    try:
        sid = data.get('session')
        selected = data.get('muscles', [])
        if not sid:
            return JSONResponse({'error': 'No session ID'}, status_code=400)
        st = _load_session(sid)
        if not st:
            return JSONResponse({'error': 'Session expired — please re-upload'}, status_code=404)
        results = {}
        for req in selected:
            col = req.get('column', '')
            mid = req.get('id', col)
            if not col or col not in st['data']:
                results[mid] = dict(status='grey', score=0, tip='Column not found')
                continue
            sig = np.array(st['data'][col], dtype=float)
            fs = st['fs']
            # Cap to 30 seconds
            max_samples = int(30 * fs)
            if len(sig) > max_samples:
                sig = sig[:max_samples]
            if np.isnan(sig).any() or np.isinf(sig).any() or np.std(sig) < 1e-10:
                results[mid] = dict(status='grey', score=0, tip='Bad or flat data')
                continue
            try:
                Tx_fast, _, ssq_freqs = _run_fast(sig, fs, 1.0, 35.0, 1.0, 32)
                result = _classify_fatigue(Tx_fast, ssq_freqs)
                bowen = run_bowen_pipeline(Tx_fast, ssq_freqs, mid, n_pad=0)
                result['centroid'] = (round(float(bowen.phase_mean_centroid), 1)
                                      if not np.isnan(bowen.phase_mean_centroid) else None)
                result['centroid_early'] = (round(float(bowen.early_centroid), 1)
                                            if not np.isnan(bowen.early_centroid) else None)
                result['centroid_late'] = (round(float(bowen.late_centroid), 1)
                                           if not np.isnan(bowen.late_centroid) else None)
                results[mid] = result
            except Exception as e:
                results[mid] = dict(status='grey', score=0, tip=str(e)[:80])
        return dict(results=results)
    except Exception as e:
        return dict(error=str(e), results={})

def _match_columns(cols):
    known = {m['id'].upper(): m for m in MUSCLES}
    matched, used = [], set()
    for c in cols:
        cu = c.upper().strip().replace('_', ' ').replace('-', ' ')
        for mid in known:
            if cu == mid or mid in cu or any(p == mid for p in cu.split()):
                if mid not in used:
                    matched.append(dict(id=mid, name=known[mid]['name'],
                                       desc=known[mid]['desc'], column=c))
                    used.add(mid)
                break
    colours = itertools.cycle(['#6750A4','#625B71','#9A25AE','#386A20',
                                '#BA1A1A','#AA3300','#00696B','#4A4458'])
    for c in cols:
        cu = c.upper().strip().replace('_', ' ').replace('-', ' ')
        if not any(mid in cu or cu == mid for mid in known):
            matched.append(dict(id=c, name=c, desc='Channel', column=c,
                               colour=next(colours)))
    return matched

# ═══════════════════════════════════════════════════
# MATERIAL DESIGN 3 HTML
# ═══════════════════════════════════════════════════
HTML_PAGE = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FAST ATHLETE</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:Inter,-apple-system,sans-serif;background:#0a0a0f;color:#e4e4ed;min-height:100vh;display:flex;flex-direction:column;-webkit-font-smoothing:antialiased}
:root{--cyan:#00E5FF;--green:#00E676;--amber:#FF9100;--red:#ef4444;--surface:#12151a;--card:#1a1f2e;--card-hover:#222836;--text:#e4e4ed;--text-dim:#8b92a0;--text-faint:#555b68;--border:rgba(255,255,255,0.06)}
.app-container{width:100%;max-width:480px;margin:0 auto;padding:0 16px 80px;min-height:100vh}
.hidden{display:none!important}

/* Header */
.app-header{display:flex;align-items:center;justify-content:space-between;padding:16px 0;position:sticky;top:0;z-index:10;background:#0a0a0f}
.logo{display:flex;align-items:center;gap:10px}
.logo-icon{width:36px;height:36px;background:linear-gradient(135deg,#00E5FF,#00B8D4);border-radius:10px;display:flex;align-items:center;justify-content:center}
.logo-wave{width:18px;height:14px;background:linear-gradient(90deg,transparent 0%,#fff 50%,transparent 100%);clip-path:polygon(0 100%,10% 30%,25% 70%,40% 20%,55% 80%,70% 30%,85% 70%,100% 40%,100% 100%)}
.logo-text{font-size:18px;font-weight:800;letter-spacing:-0.02em;color:#fff}
.logo-text span{color:var(--cyan)}
.header-right{display:flex;align-items:center;gap:16px}
.notif-btn{position:relative;background:none;border:none;color:var(--text-dim);cursor:pointer;font-size:20px;padding:4px}
.notif-badge{position:absolute;top:-2px;right:-4px;width:16px;height:16px;border-radius:50%;background:var(--red);color:#fff;font-size:10px;font-weight:700;display:flex;align-items:center;justify-content:center}
.avatar{width:32px;height:32px;border-radius:50%;background:linear-gradient(135deg,#00E5FF,#00B8D4);display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:700;color:#fff;cursor:pointer}

/* Buttons */
.btn-primary{display:flex;align-items:center;justify-content:center;gap:8px;width:100%;padding:16px;border:none;border-radius:14px;background:linear-gradient(135deg,#00E5FF,#00B8D4);color:#0a0a0f;font-family:Inter,sans-serif;font-size:15px;font-weight:700;cursor:pointer;transition:all .2s;letter-spacing:-.01em}
.btn-primary:hover{transform:translateY(-1px);box-shadow:0 8px 32px rgba(0,229,255,.25)}
.btn-primary:disabled{opacity:.4;cursor:not-allowed;transform:none;box-shadow:none}
.btn-outline{display:flex;align-items:center;gap:6px;padding:10px 20px;border:1px solid var(--border);border-radius:10px;background:transparent;color:var(--text-dim);font-family:Inter,sans-serif;font-size:13px;font-weight:500;cursor:pointer;transition:all .15s}
.btn-outline:hover{border-color:var(--cyan);color:var(--cyan)}

/* Cards */
.card{background:var(--card);border:1px solid var(--border);border-radius:16px;padding:20px;margin-bottom:12px}
.card-header{font-size:12px;font-weight:600;color:var(--text-dim);text-transform:uppercase;letter-spacing:.06em;margin-bottom:16px}

/* Gauge */
.gauge-wrap{display:flex;flex-direction:column;align-items:center;padding:8px 0}
.gauge-ring{position:relative;width:180px;height:180px}
.gauge-ring svg{transform:rotate(-90deg);width:100%;height:100%}
.gauge-ring .bg{fill:none;stroke:rgba(255,255,255,0.06);stroke-width:10}
.gauge-ring .fill{fill:none;stroke:var(--green);stroke-width:10;stroke-linecap:round;transition:stroke-dashoffset 1s ease}
.gauge-center{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);text-align:center}
.gauge-value{font-size:42px;font-weight:800;letter-spacing:-.03em;line-height:1}
.gauge-label{font-size:13px;color:var(--text-dim);margin-top:2px}
.gauge-status{display:flex;align-items:center;gap:6px;font-size:12px;font-weight:600;margin-top:4px}
.gauge-dot{width:8px;height:8px;border-radius:50%}

/* Status banner */
.status-banner{background:rgba(0,230,118,0.08);border:1px solid rgba(0,230,118,0.15);border-radius:14px;padding:16px 20px;text-align:center;margin-bottom:12px}
.status-banner h2{font-size:14px;font-weight:600;color:var(--green);text-transform:uppercase;letter-spacing:.06em}

/* Info row */
.info-row{display:flex;flex-direction:column;gap:8px}
.info-item{display:flex;justify-content:space-between;align-items:center;padding:10px 0;border-bottom:1px solid var(--border)}
.info-item:last-child{border-bottom:none}
.info-key{font-size:13px;color:var(--text-dim)}
.info-val{font-size:13px;font-weight:600;color:var(--text)}
.info-val.green{color:var(--green)}

/* Upload zone */
.upload-zone{border:2px dashed rgba(0,229,255,0.2);border-radius:16px;padding:40px 20px;text-align:center;cursor:pointer;transition:all .2s;margin-bottom:12px}
.upload-zone:hover,.upload-zone.dragover{border-color:var(--cyan);background:rgba(0,229,255,0.04)}
.upload-zone.has-file{border-style:solid;border-color:var(--green);background:rgba(0,230,118,0.04)}
.upload-icon{font-size:40px;margin-bottom:12px}
.upload-text{font-size:15px;font-weight:500;color:var(--text)}
.upload-formats{font-size:12px;color:var(--text-faint);margin-top:6px}
input[type=file]{display:none}

/* File chip */
.file-chip{display:inline-flex;align-items:center;gap:8px;padding:8px 14px;border-radius:10px;background:rgba(0,230,118,0.08);color:var(--green);font-size:13px;font-weight:600}
.file-meta{font-size:12px;color:var(--text-dim);margin-top:8px}

/* Muscle grid */
.muscle-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:12px 0}
.muscle-card{display:flex;align-items:center;gap:10px;padding:12px;background:var(--card);border:1px solid var(--border);border-radius:12px;cursor:pointer;transition:all .15s;position:relative}
.muscle-card:hover{background:var(--card-hover)}
.muscle-card.selected{border-color:var(--cyan);background:rgba(0,229,255,0.04)}
.muscle-card .body-svg{width:36px;height:64px;flex-shrink:0;border-radius:6px;overflow:hidden}
.muscle-name{font-size:13px;font-weight:600;color:var(--text)}
.muscle-desc{font-size:11px;color:var(--text-dim)}
.muscle-card .check-ring{position:absolute;top:10px;right:10px;width:20px;height:20px;border-radius:50%;flex-shrink:0;border:2px solid var(--text-faint);transition:all .2s}
.muscle-card.selected .check-ring{background:var(--cyan);border-color:var(--cyan)}
.muscle-card.selected .check-ring::after{content:'';position:absolute;top:3px;left:6px;width:5px;height:9px;border:solid #0a0a0f;border-width:0 2px 2px 0;transform:rotate(45deg)}

/* Count bar */
.count-bar{display:flex;align-items:center;justify-content:space-between;padding:12px 16px;background:var(--card);border:1px solid var(--border);border-radius:12px;position:sticky;bottom:16px}
.count-text{font-size:14px;color:var(--text-dim)}
.count-num{font-weight:700;color:var(--text)}

/* Results */
.summary-row{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:12px}
.summary-card{text-align:center;padding:16px 8px}
.summary-num{font-size:32px;font-weight:800;letter-spacing:-.02em}
.summary-label{font-size:11px;color:var(--text-dim);margin-top:4px}
.summary-card.green .summary-num{color:var(--green)}
.summary-card.amber .summary-num{color:var(--amber)}
.summary-card.red .summary-num{color:var(--red)}

.result-card{display:flex;align-items:center;gap:14px;padding:16px;border-bottom:1px solid var(--border)}
.result-card:last-child{border-bottom:none}
.result-score{font-size:28px;font-weight:800;letter-spacing:-.02em;min-width:50px;text-align:center}
.result-badge{font-size:11px;font-weight:600;padding:4px 12px;border-radius:8px;text-transform:uppercase;letter-spacing:.04em;white-space:nowrap}
.badge-green{background:rgba(0,230,118,0.1);color:var(--green)}
.badge-amber{background:rgba(255,145,0,0.1);color:var(--amber)}
.badge-red{background:rgba(239,68,68,0.1);color:var(--red)}
.result-tip{font-size:12px;color:var(--text-dim);margin-top:4px}

/* Trend chart */
.bar-chart{display:flex;align-items:flex-end;gap:6px;height:120px;padding:0 4px}
.bar-col{flex:1;display:flex;flex-direction:column;align-items:center;gap:4px}
.bar{width:100%;border-radius:4px 4px 0 0;transition:height .5s ease;min-height:4px}
.bar-label{font-size:10px;color:var(--text-faint);font-weight:600}

/* Progress */
.progress-section{text-align:center;padding:40px 20px}
.progress-ring{position:relative;width:80px;height:80px;margin:0 auto 20px}
.progress-ring svg{transform:rotate(-90deg)}
.progress-ring .bg{fill:none;stroke:rgba(255,255,255,0.06);stroke-width:6}
.progress-ring .fill{fill:none;stroke:var(--cyan);stroke-width:6;stroke-linecap:round;transition:stroke-dashoffset .3s ease}
.progress-label{font-size:16px;font-weight:600;color:var(--text)}
.progress-step{font-size:13px;color:var(--text-dim);margin-top:4px}

/* Muscle map */
.muscle-map{position:relative;display:flex;justify-content:center;padding:20px 0}
.muscle-map svg{width:min(300px,100%);height:auto}
.muscle-map .body-outline{fill:none;stroke:rgba(255,255,255,0.15);stroke-width:1.5}
.muscle-map .muscle-region{cursor:pointer;transition:opacity .3s}

/* Callouts */
.callout-row{display:flex;gap:12px;flex-wrap:wrap;margin-top:16px}
.callout{display:flex;align-items:center;gap:8px;font-size:12px;padding:8px 14px;border-radius:8px;background:var(--card);border:1px solid var(--border)}
.callout-dot{width:10px;height:10px;border-radius:50%;flex-shrink:0}

/* Bottom nav */
.bottom-nav{position:fixed;bottom:0;left:50%;transform:translateX(-50%);width:100%;max-width:480px;display:flex;justify-content:space-around;background:rgba(10,10,15,0.95);backdrop-filter:blur(20px);border-top:1px solid var(--border);padding:8px 0 12px;z-index:100}
.nav-item{display:flex;flex-direction:column;align-items:center;gap:4px;padding:4px 12px;border:none;background:none;cursor:pointer;color:var(--text-faint);font-family:Inter,sans-serif;transition:color .15s}
.nav-item.active{color:var(--cyan)}
.nav-item svg{width:22px;height:22px}
.nav-label{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.04em}

/* Welcome */
.welcome{text-align:center;padding:24px 0}
.welcome h1{font-size:28px;font-weight:800;letter-spacing:-.03em;margin-bottom:8px}
.welcome p{font-size:14px;color:var(--text-dim);max-width:360px;margin:0 auto 24px;line-height:1.6}

/* Animations */
@keyframes fadeIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
.fade-in{animation:fadeIn .4s ease}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
.pulse{animation:pulse 1.5s ease-in-out infinite}

/* Section toggle */
.section{display:none}
.section.active{display:block;animation:fadeIn .3s ease}

@media(max-width:380px){.muscle-grid{grid-template-columns:1fr}}

/* ═══ Responsive layout & home content ═══ */
.tl-dot{width:12px;height:12px;border-radius:50%;display:inline-block;flex-shrink:0}
.tl-green{background:#00E676;box-shadow:0 0 6px rgba(0,230,118,0.3)}
.tl-amber{background:#FF9100;box-shadow:0 0 6px rgba(255,145,0,0.3)}
.tl-red{background:#ef4444;box-shadow:0 0 6px rgba(239,68,68,0.3)}

.home-grid{display:block}
.home-left{display:flex;flex-direction:column;gap:12px}
.home-right{display:flex;flex-direction:column;gap:12px;margin-top:12px}
.home-left .card,.home-right .card{margin-bottom:0}
.steps{display:flex;flex-direction:column;gap:14px}
.step{display:flex;align-items:flex-start;gap:12px}
.step-icon{width:36px;height:36px;border-radius:10px;background:rgba(255,255,255,0.05);border:1px solid var(--border);display:flex;align-items:center;justify-content:center;color:var(--cyan);flex-shrink:0}
.step-icon svg{width:18px;height:18px}
.step-title{font-size:14px;font-weight:600;color:var(--text)}
.step-desc{font-size:12px;color:var(--text-dim);margin-top:2px;line-height:1.5}
.example-card{opacity:.55}
.example-row{display:flex;align-items:center;gap:12px;padding:10px 0;border-bottom:1px solid var(--border)}
.example-info{flex:1}
.example-name{font-size:14px;font-weight:600;color:var(--text)}
.example-meta{font-size:12px;color:var(--text-dim);margin-top:2px}
.example-score{font-size:24px;font-weight:800;color:var(--green)}
.example-foot{font-size:12px;color:var(--text-dim);padding-top:10px;line-height:1.5}
.tag{font-size:9px;font-weight:700;background:rgba(255,255,255,0.08);border:1px solid var(--border);color:var(--text-dim);padding:2px 6px;border-radius:6px;letter-spacing:.08em;vertical-align:2px;margin-left:6px}
.tip-body{font-size:13px;color:var(--text-dim);line-height:1.6}

@media(min-width:640px){
  .app-container{max-width:720px;padding:0 24px 88px}
  .bottom-nav{max-width:720px}
  .bottom-nav .nav-item{flex:1;max-width:170px}
  .muscle-grid{grid-template-columns:repeat(3,1fr)}
  .gauge-ring{width:210px;height:210px}
  .upload-zone{max-width:560px;margin-left:auto;margin-right:auto}
}
@media(min-width:900px){
  #screen-results.active{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;align-items:start}
  #screen-results .card{margin-bottom:0}
  #status-banner-container{grid-column:1/-1}
  #muscle-map-card{grid-column:2}
  .results-footer{grid-column:1/-1}
}
@media(min-width:1024px){
  .app-container{max-width:1080px;padding:0 32px 96px}
  .bottom-nav{max-width:1080px}
  .home-grid{flex:1;display:grid;grid-template-columns:1fr 1fr;gap:12px 24px;align-items:stretch;align-content:center}
  .home-left,.home-right{justify-content:center}
  .home-right{margin-top:0}
  #screen-welcome.active{display:flex;flex-direction:column}
  .welcome h1{font-size:32px}
  .welcome p{max-width:480px}
  .muscle-grid{grid-template-columns:repeat(4,1fr)}
  .gauge-ring{width:240px;height:240px}
  .gauge-value{font-size:52px}
  .steps{flex-direction:row;gap:12px}
  .step{flex:1;flex-direction:column}
  .muscle-map svg{width:min(360px,100%)}
  #results-list{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:0 20px}
  .result-card>div:nth-child(2){min-width:0}
}

/* ═══ Profiles ═══ */
.profile-chip{display:flex;align-items:center;gap:8px;background:var(--card);border:1px solid var(--border);border-radius:999px;padding:4px 12px 4px 4px;cursor:pointer;transition:border-color .15s;font-family:Inter,sans-serif}
.profile-chip:hover{border-color:var(--cyan)}
.profile-avatar{width:26px;height:26px;border-radius:50%;background:linear-gradient(135deg,#00E5FF,#00B8D4);display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:800;color:#0a0a0f;flex-shrink:0}
.profile-name{font-size:12px;font-weight:600;color:var(--text);max-width:110px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}

.sheet-overlay{position:fixed;inset:0;background:rgba(0,0,0,0.6);backdrop-filter:blur(4px);z-index:200;display:flex;align-items:flex-end;justify-content:center}
.sheet-overlay.hidden{display:none}
.sheet{width:100%;max-width:480px;background:var(--card);border:1px solid var(--border);border-radius:20px 20px 0 0;padding:20px 20px 28px;max-height:70vh;overflow-y:auto;animation:fadeIn .25s ease}
.sheet-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:12px}
.sheet-title{font-size:12px;font-weight:700;color:var(--text-dim);letter-spacing:.08em}
.sheet-close{background:none;border:none;color:var(--text-dim);font-size:16px;cursor:pointer;padding:4px}
.profile-row{display:flex;align-items:center;gap:12px;padding:12px;background:var(--surface);border:1px solid var(--border);border-radius:12px;margin-bottom:8px;cursor:pointer;transition:all .15s}
.profile-row:hover{border-color:var(--cyan)}
.profile-row-name{font-size:14px;font-weight:600;color:var(--text);flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.profile-active{font-size:9px;font-weight:800;color:var(--cyan);letter-spacing:.08em}
.sheet-divider{height:1px;background:var(--border);margin:16px 0}
.sheet-input{width:100%;background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:12px 14px;color:var(--text);font-family:Inter,sans-serif;font-size:14px;outline:none;margin-bottom:8px}
.sheet-input:focus{border-color:var(--cyan)}
.sheet-pin-row{display:flex;gap:8px;align-items:center}
.sheet-pin-row .sheet-input{margin-bottom:0}
.sheet-hint{font-size:11px;color:var(--text-faint);margin-top:8px;line-height:1.5}

/* Toast */
.toast{position:fixed;bottom:96px;left:50%;transform:translateX(-50%) translateY(20px);background:var(--card);border:1px solid rgba(0,229,255,0.3);color:var(--text);font-size:13px;font-weight:600;padding:12px 20px;border-radius:12px;opacity:0;pointer-events:none;transition:all .3s;z-index:300;max-width:90vw;text-align:center;box-shadow:0 8px 32px rgba(0,0,0,0.4)}
.toast.show{opacity:1;transform:translateX(-50%) translateY(0)}

/* History */
.hist-head{display:flex;align-items:center;gap:12px;margin-bottom:10px}
.hist-score{font-size:24px;font-weight:800;letter-spacing:-.02em;min-width:56px;text-align:center}
.hist-date{font-size:12px;color:var(--text-dim)}
.hist-file{font-size:11px;color:var(--text-faint);margin-top:2px;word-break:break-all}
.hist-status{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.05em;white-space:nowrap}
.hist-chips{display:flex;flex-wrap:wrap;gap:6px}
.hist-chip{display:inline-flex;align-items:center;gap:6px;font-size:11px;font-weight:600;color:var(--text-dim);background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:4px 10px}
.hist-chip .tl-dot{width:8px;height:8px;box-shadow:none}
</style>
</head>
<body>
<div class="app-container">
<svg style="display:none" aria-hidden="true">
<defs>
  <path id="body-front" d="M 309.48 168.91 Q 305.84 164.32 303.32 169.76 C 298.49 180.21 308.31 200.03 314.51 208.74 C 316.34 211.31 318.01 208.95 318.58 207.26 A 0.67 0.66 57.6 0 1 319.87 207.55 C 319.06 215.09 318.68 227.40 324.34 232.47 C 327.22 235.05 326.97 235.88 326.92 239.51 Q 326.68 255.16 323.97 266.82 Q 323.85 267.35 323.48 267.73 Q 308.61 282.73 290.26 293.23 C 278.34 300.05 267.53 299.26 253.00 298.03 Q 237.49 296.72 224.74 305.21 C 208.71 315.86 190.95 335.73 189.24 355.50 Q 186.95 381.81 190.53 412.66 C 190.79 414.92 190.69 417.49 191.02 419.92 Q 191.09 420.43 190.88 420.90 C 187.89 427.65 183.99 434.89 181.93 441.29 C 177.25 455.76 176.31 470.23 176.20 486.02 Q 176.20 486.51 175.90 486.90 C 159.84 507.69 147.56 529.29 141.49 554.95 Q 140.10 560.80 138.16 574.66 Q 131.28 623.74 118.11 671.52 C 115.99 679.21 112.98 690.29 104.08 693.63 Q 90.70 698.65 79.29 707.27 C 73.17 711.89 69.48 719.95 66.12 726.62 C 62.44 733.91 47.57 737.30 49.20 746.00 C 49.75 748.96 51.89 750.13 54.75 750.02 Q 67.27 749.50 74.18 740.00 C 76.03 737.45 77.93 736.62 80.54 735.24 Q 81.02 734.98 81.24 735.48 Q 84.59 743.00 80.47 750.73 Q 71.41 767.75 62.21 784.70 Q 60.53 787.81 59.49 791.20 C 57.52 797.69 65.78 800.84 69.45 795.20 C 76.80 783.92 82.72 773.30 92.55 762.52 Q 93.00 762.04 92.84 762.67 Q 87.89 783.24 79.07 802.44 C 77.36 806.17 75.64 812.30 79.19 815.18 C 89.50 823.53 107.08 773.44 109.24 767.88 A 0.37 0.36 -30.3 0 1 109.94 768.06 C 108.51 777.44 106.43 787.14 105.28 796.13 C 104.34 803.43 103.67 808.49 104.41 814.32 C 105.40 822.00 112.74 817.15 114.09 812.77 C 118.56 798.32 120.41 781.74 125.18 766.21 A 0.55 0.55 0.0 0 1 125.93 765.87 C 131.64 768.40 126.65 796.54 133.38 803.49 A 1.35 1.35 0.0 0 0 134.16 803.90 C 138.40 804.59 139.71 797.34 140.15 793.73 Q 141.74 780.80 142.58 767.76 Q 142.86 763.46 144.07 759.34 Q 150.39 737.64 154.77 715.46 Q 156.15 708.50 155.48 697.76 Q 154.48 681.63 161.99 665.46 Q 180.58 625.46 201.25 586.52 C 213.64 563.18 218.66 541.14 220.65 514.18 C 221.24 506.18 223.22 502.59 228.42 495.84 C 237.76 483.72 242.73 464.92 246.12 450.19 Q 246.24 449.64 246.75 449.42 L 250.30 447.82 A 0.49 0.49 0.0 0 1 250.99 448.23 Q 252.78 470.14 257.44 487.01 C 259.04 492.80 264.20 498.21 265.32 505.20 C 265.91 508.82 266.99 512.44 267.11 516.00 Q 267.57 529.33 266.95 540.50 C 265.58 565.32 263.85 592.20 259.98 619.13 C 258.39 630.19 253.14 640.55 250.52 651.43 Q 245.19 673.62 242.32 696.24 C 239.63 717.56 236.59 740.02 236.04 757.75 Q 234.98 791.48 237.98 842.55 Q 239.43 867.18 244.64 891.26 Q 247.76 905.70 255.88 917.90 Q 256.15 918.31 256.08 918.79 C 254.89 926.25 257.03 933.47 255.60 940.95 Q 252.28 958.32 251.77 975.98 C 251.55 983.43 252.85 991.28 253.67 998.93 Q 253.99 1001.95 253.29 1005.00 C 239.19 1067.03 246.93 1130.64 261.77 1190.07 C 266.01 1207.06 266.47 1222.37 264.71 1240.03 C 263.85 1248.62 262.10 1260.41 264.24 1268.75 C 266.05 1275.80 267.54 1287.46 261.78 1293.28 C 256.71 1298.39 242.40 1310.55 240.72 1316.98 C 239.19 1322.86 235.04 1332.26 242.29 1333.71 Q 242.69 1333.79 243.08 1333.66 L 244.23 1333.29 Q 245.05 1333.02 244.81 1333.85 C 242.95 1340.16 249.20 1340.52 253.77 1340.86 C 256.46 1341.06 257.37 1343.60 259.30 1344.71 Q 263.13 1346.91 267.14 1344.43 Q 267.59 1344.15 267.92 1344.56 Q 271.17 1348.61 276.21 1349.09 C 278.90 1349.35 281.27 1347.36 283.62 1346.09 Q 284.10 1345.82 284.44 1346.26 Q 288.33 1351.29 294.72 1351.38 C 295.77 1351.39 297.65 1351.62 298.54 1350.79 Q 301.20 1348.30 306.57 1341.58 C 312.04 1334.74 311.14 1328.85 310.29 1320.16 C 309.43 1311.33 311.17 1303.41 313.76 1295.20 C 315.84 1288.56 313.35 1280.06 314.07 1273.15 C 314.57 1268.39 315.80 1263.68 315.01 1259.02 C 314.06 1253.42 311.98 1247.60 311.31 1242.66 Q 309.57 1229.80 309.57 1219.75 Q 309.57 1192.29 313.54 1161.94 C 315.34 1148.21 319.24 1136.08 324.12 1123.46 Q 325.66 1119.48 326.10 1115.72 C 330.14 1081.34 326.20 1048.44 320.65 1013.26 C 319.84 1008.17 319.39 1002.54 321.72 997.72 C 328.03 984.68 329.28 969.38 329.07 954.15 C 329.01 949.50 327.95 944.55 327.58 939.63 C 327.13 933.64 329.28 925.78 330.82 919.80 C 334.72 904.69 337.76 888.96 341.43 874.30 Q 348.95 844.25 355.42 813.95 C 358.50 799.49 357.70 784.78 357.75 768.06 Q 357.78 756.80 356.36 748.81 Q 356.26 748.24 356.77 748.50 L 363.71 751.99 A 1.07 1.07 0.0 0 0 364.67 751.99 L 371.53 748.56 Q 372.07 748.29 371.98 748.89 C 369.47 765.94 370.28 783.04 371.30 800.17 Q 371.86 809.54 372.73 813.51 C 378.37 839.12 384.90 864.49 390.59 890.08 Q 394.83 909.20 399.51 928.22 C 400.58 932.58 401.13 937.66 400.58 941.57 C 398.11 958.92 398.53 982.22 407.11 998.54 C 408.41 1001.01 408.74 1005.35 408.31 1008.09 C 402.82 1043.75 398.07 1079.22 402.19 1115.33 Q 402.65 1119.34 404.21 1123.44 C 410.53 1140.06 413.55 1150.61 415.25 1164.75 C 418.31 1190.26 420.52 1218.43 416.79 1244.33 C 415.56 1252.86 411.78 1258.57 413.63 1267.80 Q 415.33 1276.21 414.16 1284.74 C 413.11 1292.39 415.65 1298.68 417.31 1305.89 C 419.02 1313.32 418.11 1320.99 417.47 1328.50 C 416.71 1337.55 423.74 1344.86 430.17 1350.90 A 1.48 1.46 -18.7 0 0 430.95 1351.28 Q 439.25 1352.41 444.03 1346.06 Q 444.40 1345.57 444.87 1345.96 Q 453.39 1352.89 460.49 1344.48 Q 460.81 1344.11 461.23 1344.37 C 469.09 1349.37 469.89 1340.80 474.98 1340.71 C 479.52 1340.64 485.21 1340.09 483.54 1333.77 Q 483.38 1333.17 483.97 1333.35 C 488.25 1334.67 490.66 1331.94 490.06 1327.75 C 489.09 1321.04 487.50 1314.41 483.44 1310.30 Q 474.77 1301.53 466.05 1292.83 C 461.19 1287.98 462.25 1276.40 463.74 1270.47 C 466.27 1260.35 464.49 1248.06 463.03 1236.25 C 461.04 1220.05 463.22 1204.28 467.41 1187.04 C 481.60 1128.60 488.89 1065.20 475.23 1006.07 C 473.92 1000.37 475.00 995.00 475.76 989.36 C 477.88 973.68 475.72 958.50 473.08 942.76 C 471.70 934.55 473.60 926.56 472.20 918.79 Q 472.11 918.30 472.39 917.89 C 483.07 902.63 486.53 880.99 488.49 863.25 C 492.12 830.38 492.47 797.34 492.26 764.31 C 492.11 741.56 488.80 719.07 486.12 696.53 C 484.30 681.19 480.76 664.32 477.47 649.99 C 474.89 638.73 469.69 628.87 468.04 617.25 C 465.37 598.45 464.19 580.92 462.40 556.31 Q 460.86 535.06 461.01 522.74 Q 461.13 512.05 463.22 504.00 C 464.54 498.90 468.30 493.91 469.91 489.46 C 474.50 476.74 476.10 461.71 477.56 448.28 Q 477.62 447.74 478.13 447.94 L 481.73 449.35 A 0.77 0.77 0.0 0 1 482.19 449.89 Q 486.03 466.84 492.52 482.96 C 494.16 487.04 496.63 491.75 500.12 495.79 C 505.75 502.32 507.17 507.95 508.00 517.24 C 509.72 536.47 512.15 552.06 518.89 569.24 Q 521.60 576.16 527.50 587.28 Q 543.57 617.60 558.56 648.47 C 566.04 663.89 571.90 675.54 572.85 690.59 Q 572.98 692.57 572.55 700.88 Q 572.12 709.31 573.99 718.25 Q 577.87 736.78 582.37 752.38 C 585.15 761.98 586.32 769.32 586.71 778.53 C 586.92 783.46 587.58 803.53 593.41 804.06 C 599.41 804.61 599.71 774.61 600.39 768.08 A 1.12 1.12 0.0 0 1 600.80 767.33 Q 601.30 766.93 601.62 766.30 A 1.39 1.00 59.0 0 1 603.70 767.19 C 607.27 782.50 609.43 797.55 614.25 812.25 C 615.52 816.12 618.33 820.08 622.81 817.38 A 1.18 1.17 -8.4 0 0 623.35 816.66 Q 624.98 810.32 624.13 803.72 Q 621.83 785.89 618.23 768.64 A 0.53 0.53 0.0 0 1 619.24 768.34 C 622.72 777.06 636.06 814.20 645.24 816.03 C 650.64 817.10 652.13 811.12 650.95 807.31 C 648.59 799.74 644.42 791.59 642.09 784.69 Q 638.29 773.46 635.22 761.98 A 0.15 0.14 -73.3 0 1 635.47 761.84 Q 640.35 767.61 644.90 773.66 C 649.45 779.70 653.60 787.18 658.03 793.93 Q 660.09 797.07 661.70 797.82 C 665.53 799.62 670.61 795.77 669.00 791.28 C 666.63 784.66 661.63 776.66 659.33 772.19 Q 654.22 762.29 648.82 752.53 C 645.43 746.40 644.71 741.93 646.89 735.59 Q 647.08 735.05 647.60 735.27 C 650.55 736.50 652.37 737.45 654.44 740.27 Q 661.27 749.61 673.53 749.92 C 681.25 750.12 680.47 740.89 676.20 738.28 C 671.33 735.31 664.61 731.14 661.97 725.94 C 657.98 718.11 654.62 711.26 649.21 707.28 Q 637.40 698.62 623.76 693.40 C 619.45 691.75 615.12 686.26 613.76 682.47 Q 608.42 667.65 602.70 641.81 Q 594.90 606.62 590.85 578.90 Q 588.46 562.58 587.74 559.15 C 582.02 531.75 569.74 509.81 552.98 487.61 C 551.81 486.06 551.91 485.12 551.97 483.26 Q 552.48 466.57 548.70 449.61 C 546.27 438.71 541.82 430.32 537.44 420.82 Q 537.22 420.36 537.28 419.85 C 539.40 398.94 540.83 377.68 539.05 356.70 C 537.31 336.13 521.34 317.28 504.86 306.23 C 494.75 299.45 485.77 296.97 473.93 298.16 Q 464.41 299.12 453.63 298.41 C 438.05 297.39 418.32 280.58 407.40 270.35 C 405.82 268.87 404.57 267.56 404.10 265.32 Q 401.24 251.68 401.26 237.76 Q 401.26 233.73 404.68 232.04 Q 405.14 231.82 405.39 231.38 C 409.76 223.86 408.77 215.16 408.75 206.85 A 0.38 0.38 0.0 0 1 409.48 206.69 C 410.36 208.62 412.01 211.62 414.22 208.45 C 421.05 198.67 427.45 183.93 425.97 172.00 C 425.49 168.15 422.83 165.91 418.91 167.68" fill="none" stroke="currentColor" stroke-width="2" vector-effect="non-scaling-stroke"/>
  <path id="body-back" d="M 1028.14 166.45 Q 1021.22 166.96 1021.73 176.02 C 1022.38 187.38 1027.41 200.00 1034.70 209.56 A 0.95 0.95 0.0 0 0 1035.77 209.88 Q 1037.97 209.08 1038.42 206.75 Q 1038.48 206.41 1038.79 206.56 C 1039.50 206.91 1039.29 219.51 1039.32 221.19 C 1039.41 225.63 1041.33 230.61 1045.48 233.58 A 1.48 1.46 -79.2 0 1 1046.03 234.40 C 1047.33 239.56 1046.14 264.59 1042.52 268.26 Q 1027.38 283.59 1008.53 293.99 C 997.30 300.18 985.80 298.88 972.00 298.05 C 960.16 297.34 951.79 300.13 941.86 307.09 C 927.96 316.83 911.37 335.39 909.24 353.00 C 906.85 372.86 908.46 396.71 910.58 417.97 Q 910.78 420.04 909.97 421.91 C 907.17 428.36 903.51 435.29 901.56 441.28 Q 895.91 458.72 896.11 477.26 Q 896.15 480.50 895.88 486.15 Q 895.86 486.66 895.55 487.06 C 879.06 508.02 866.67 530.27 860.84 556.43 Q 859.72 561.44 857.62 576.15 C 853.15 607.45 846.97 639.64 837.96 670.48 C 835.37 679.35 832.82 690.15 824.31 693.38 Q 811.21 698.35 799.91 706.70 C 793.05 711.77 790.22 717.94 785.68 726.75 C 782.37 733.16 764.38 739.29 769.45 747.77 C 771.01 750.37 774.09 750.14 776.79 749.81 Q 787.25 748.51 793.13 740.83 C 795.42 737.84 797.13 736.50 800.36 735.31 A 0.63 0.63 0.0 0 1 801.16 735.68 C 803.48 741.92 802.81 745.80 799.51 751.90 Q 789.51 770.39 779.78 789.01 C 775.87 796.49 784.57 802.15 789.55 794.51 C 796.72 783.50 802.47 773.20 812.06 762.59 Q 812.62 761.98 812.43 762.79 Q 807.49 783.70 798.01 804.03 Q 795.79 808.79 797.53 813.47 C 798.35 815.65 800.88 816.85 802.95 815.95 C 807.95 813.78 812.74 805.60 815.08 800.58 Q 820.51 788.92 825.23 776.95 Q 827.37 771.52 829.06 768.26 A 0.34 0.34 0.0 0 1 829.69 768.47 C 828.65 774.94 819.92 813.84 825.80 817.66 C 829.47 820.04 832.91 815.52 833.80 812.51 Q 838.73 795.91 842.08 776.75 C 842.69 773.31 843.62 770.03 844.54 766.92 A 1.49 1.49 0.0 0 1 847.45 767.13 C 849.06 778.16 848.17 788.91 850.91 799.85 C 851.57 802.48 854.41 806.12 856.99 802.69 C 861.32 796.92 861.47 780.19 861.98 770.25 C 862.50 760.22 866.62 750.03 868.70 741.28 C 871.57 729.16 876.10 714.64 875.42 700.50 C 874.79 687.46 876.48 676.40 882.00 664.53 Q 899.81 626.31 920.51 587.27 C 928.60 572.01 933.68 558.17 937.01 542.00 Q 938.40 535.24 940.57 511.31 C 941.06 506.01 943.33 501.94 947.04 497.29 C 957.02 484.77 962.25 465.95 965.86 450.00 Q 965.97 449.54 966.40 449.37 L 969.87 447.93 Q 970.39 447.72 970.44 448.27 C 972.08 465.19 974.18 483.97 982.58 498.42 Q 985.25 503.01 985.69 509.45 C 985.76 510.51 986.43 511.70 986.49 512.50 C 986.89 517.68 987.09 525.23 986.82 531.50 Q 985.00 573.11 980.47 614.52 C 978.98 628.13 972.65 640.33 969.66 653.60 C 966.01 669.78 963.02 685.46 961.19 702.45 C 959.24 720.52 956.19 739.39 955.83 756.75 C 954.96 797.57 955.28 842.51 962.96 884.21 C 965.15 896.11 968.33 907.72 975.37 917.40 A 1.48 1.46 27.9 0 1 975.65 918.29 C 975.42 926.20 976.32 934.21 975.03 942.01 C 971.89 960.94 969.95 978.86 973.41 997.96 C 973.70 999.53 973.58 1001.87 973.23 1003.42 C 959.26 1065.20 965.77 1130.76 981.86 1191.82 C 985.51 1205.68 986.32 1220.46 984.96 1234.92 C 984.02 1244.98 982.27 1255.20 983.30 1265.30 C 984.08 1272.87 988.23 1284.18 983.14 1291.21 C 978.75 1297.25 969.45 1303.98 963.07 1312.35 C 960.11 1316.25 952.52 1335.31 964.02 1333.54 Q 964.55 1333.46 964.42 1333.98 C 962.73 1340.59 969.52 1340.54 974.36 1340.95 Q 974.88 1341.00 975.24 1341.37 C 978.64 1344.83 981.89 1347.54 986.66 1344.41 Q 987.11 1344.12 987.46 1344.52 C 992.32 1350.09 997.09 1350.27 1003.06 1346.11 Q 1003.50 1345.80 1003.93 1346.12 C 1005.34 1347.18 1006.20 1348.82 1007.59 1349.58 Q 1011.98 1351.98 1017.08 1351.27 A 1.56 1.56 0.0 0 0 1017.93 1350.86 Q 1024.28 1344.70 1027.72 1339.46 C 1032.14 1332.71 1030.13 1325.67 1029.71 1317.92 C 1029.27 1309.96 1031.28 1302.44 1033.52 1294.97 C 1034.58 1291.42 1034.05 1286.50 1033.60 1282.59 Q 1032.89 1276.40 1034.01 1270.28 C 1034.95 1265.11 1035.75 1261.39 1034.60 1257.67 Q 1029.90 1242.46 1029.51 1227.25 Q 1028.64 1193.94 1033.40 1159.73 C 1035.13 1147.30 1038.92 1136.76 1043.43 1124.47 Q 1045.16 1119.75 1045.73 1115.31 C 1050.32 1079.07 1044.60 1044.51 1039.86 1008.73 C 1038.66 999.61 1043.98 993.60 1045.54 987.51 C 1048.41 976.36 1049.80 959.10 1047.93 945.66 C 1046.88 938.09 1047.48 931.84 1049.21 924.99 C 1053.15 909.35 1056.75 892.75 1059.78 880.01 Q 1066.27 852.63 1072.60 825.22 Q 1075.98 810.55 1076.49 805.75 Q 1077.50 796.31 1077.72 775.82 Q 1077.85 764.16 1076.54 752.58 Q 1076.32 750.58 1075.99 749.61 Q 1075.45 748.03 1076.95 748.78 L 1083.35 752.00 A 1.10 1.08 44.4 0 0 1084.32 752.00 L 1091.50 748.31 A 0.24 0.24 0.0 0 1 1091.84 748.59 Q 1090.49 753.63 1090.36 758.75 C 1089.82 779.99 1089.54 802.24 1094.28 822.45 Q 1101.55 853.47 1108.92 884.46 C 1111.25 894.25 1114.60 910.13 1117.95 922.87 C 1119.13 927.36 1119.75 931.95 1120.50 936.49 C 1121.14 940.42 1119.45 945.92 1119.24 949.53 Q 1118.26 966.73 1121.38 983.68 C 1121.98 986.96 1123.21 991.52 1124.54 993.96 C 1128.10 1000.50 1128.52 1004.24 1127.36 1012.10 C 1122.34 1046.29 1118.51 1078.84 1121.48 1113.50 C 1121.72 1116.32 1122.66 1120.49 1123.91 1123.73 C 1131.43 1143.10 1134.58 1156.98 1136.42 1177.99 C 1138.35 1200.12 1139.52 1222.20 1136.35 1244.60 Q 1135.88 1247.88 1134.29 1252.69 C 1132.00 1259.62 1132.37 1264.14 1133.83 1271.98 C 1135.50 1280.93 1132.17 1288.45 1134.90 1297.66 C 1136.88 1304.36 1138.19 1310.69 1137.87 1317.88 C 1137.58 1324.48 1135.49 1332.56 1139.15 1338.36 Q 1142.72 1344.04 1149.63 1350.84 Q 1149.97 1351.18 1150.46 1351.25 Q 1158.71 1352.49 1163.67 1346.15 A 0.64 0.64 0.0 0 1 1164.58 1346.04 Q 1173.02 1352.85 1180.03 1344.60 Q 1180.37 1344.20 1180.83 1344.46 Q 1186.12 1347.40 1190.08 1343.66 Q 1192.28 1341.58 1193.29 1341.22 C 1197.87 1339.60 1204.81 1341.71 1203.29 1333.67 A 0.39 0.39 0.0 0 1 1203.82 1333.23 L 1204.86 1333.62 Q 1205.25 1333.77 1205.65 1333.71 C 1212.46 1332.65 1209.17 1324.33 1208.00 1319.87 C 1205.32 1309.62 1192.63 1299.79 1185.30 1292.30 C 1180.77 1287.68 1182.22 1274.71 1183.62 1269.06 C 1186.76 1256.35 1182.79 1239.97 1182.29 1230.50 C 1181.63 1217.80 1182.70 1204.60 1185.99 1191.35 C 1200.90 1131.35 1208.58 1067.26 1194.98 1006.22 C 1193.56 999.84 1194.88 994.32 1195.73 987.24 C 1197.46 972.87 1195.00 955.62 1192.39 940.62 C 1191.27 934.14 1192.32 927.30 1192.25 920.69 Q 1192.25 920.23 1192.09 919.80 L 1191.79 918.97 Q 1191.59 918.45 1191.92 918.00 C 1199.57 907.39 1203.42 893.36 1205.50 881.25 C 1212.13 842.49 1212.38 800.86 1211.97 761.04 C 1211.76 739.76 1208.12 718.12 1205.90 696.81 Q 1204.13 679.89 1197.85 652.94 C 1194.73 639.58 1188.50 627.37 1187.05 613.69 Q 1183.04 575.93 1181.17 542.06 Q 1180.56 530.97 1180.85 518.01 C 1180.96 512.91 1182.20 504.08 1184.51 499.52 C 1186.81 494.98 1189.81 490.71 1191.01 485.74 Q 1195.45 467.32 1197.09 448.35 A 0.55 0.55 0.0 0 1 1197.86 447.90 L 1201.25 449.41 Q 1201.74 449.63 1201.86 450.16 C 1205.49 466.08 1210.60 484.96 1221.09 497.82 C 1229.48 508.13 1227.82 523.50 1229.73 535.92 C 1232.46 553.65 1237.66 569.19 1246.25 585.54 Q 1262.47 616.39 1284.56 662.22 Q 1292.50 678.70 1292.52 695.41 Q 1292.52 695.47 1292.20 701.94 C 1291.63 713.32 1294.91 723.91 1297.35 734.87 C 1300.01 746.89 1305.13 759.34 1305.74 772.33 C 1305.98 777.24 1306.66 804.29 1313.58 804.01 A 1.29 1.29 0.0 0 0 1314.41 803.66 C 1321.43 797.06 1316.55 769.02 1321.52 766.22 A 1.20 1.19 -21.2 0 1 1323.27 766.99 C 1326.58 781.35 1329.25 795.81 1332.92 809.99 C 1334.01 814.20 1338.07 821.55 1342.84 816.86 Q 1343.20 816.50 1343.28 816.00 Q 1344.28 809.42 1343.76 805.00 Q 1341.60 786.63 1337.95 768.42 A 0.48 0.48 0.0 0 1 1338.86 768.15 C 1342.31 776.96 1355.85 815.37 1366.03 816.16 C 1370.51 816.50 1371.54 810.41 1370.44 807.06 C 1367.79 798.97 1363.64 790.62 1361.28 783.45 Q 1357.86 773.08 1355.02 762.60 A 0.28 0.28 0.0 0 1 1355.50 762.34 Q 1359.72 767.36 1363.75 772.57 C 1368.83 779.14 1373.25 787.32 1378.17 794.66 Q 1379.99 797.36 1381.66 797.98 C 1384.30 798.97 1389.15 796.58 1388.99 793.50 Q 1388.85 790.72 1386.66 786.58 Q 1378.13 770.40 1369.24 754.42 C 1365.36 747.45 1364.08 743.12 1366.68 735.63 Q 1366.81 735.24 1367.20 735.38 Q 1371.90 736.99 1372.91 738.60 Q 1379.67 749.28 1393.03 749.97 C 1401.07 750.38 1400.13 741.50 1395.34 738.12 C 1390.41 734.62 1384.54 731.36 1381.93 726.55 C 1378.04 719.37 1374.79 711.78 1368.18 706.82 Q 1357.23 698.60 1343.50 693.43 C 1335.51 690.42 1332.54 680.64 1330.25 672.50 C 1321.70 642.22 1315.13 611.45 1310.75 580.29 Q 1308.97 567.62 1308.28 563.74 C 1302.89 533.66 1289.99 510.94 1272.05 486.75 Q 1271.76 486.36 1271.76 485.88 C 1271.89 470.59 1270.82 455.36 1265.92 440.80 C 1263.95 434.94 1260.59 428.46 1257.79 422.38 Q 1256.94 420.52 1257.10 418.48 C 1258.73 398.21 1260.25 378.73 1258.88 358.36 C 1257.39 336.36 1241.06 316.98 1223.33 305.40 C 1213.33 298.87 1205.11 297.32 1193.06 298.08 C 1179.40 298.94 1169.27 299.86 1157.52 293.24 Q 1139.58 283.12 1124.50 267.54 Q 1124.15 267.19 1124.04 266.70 Q 1121.33 254.82 1121.08 242.66 C 1120.97 237.52 1120.38 234.21 1124.51 231.78 Q 1124.95 231.52 1125.21 231.07 C 1128.92 224.63 1129.03 215.40 1128.17 207.76 Q 1128.08 207.01 1128.59 206.65 Q 1128.95 206.40 1129.15 206.78 L 1130.41 209.10 A 1.80 1.79 -42.1 0 0 1133.47 209.25 C 1138.33 202.11 1153.60 172.22 1141.68 166.80 Q 1141.16 166.57 1140.69 166.88 L 1138.38 168.39" fill="none" stroke="currentColor" stroke-width="2" vector-effect="non-scaling-stroke"/>
  <path id="quad-left-0" d="M297.69 1008.37c-7.27 7.29-16.34 3.42-19.64-5.18q-6.18-16.11-9.57-30.68c-1.99-8.6-2.24-19.68 9.72-19.91q13.12-.24 26.05 2.15 1.71.32 3.29 1.02a1.17 1.15 4.2 01.63.72c3.17 10.27 2.5 23.36.05 33.69q-2.37 10.01-10.53 18.19z" fill="currentColor" stroke="none"/>
  <path id="quad-left-1" d="M288.03 1059.54c-6.99-5.81 13.75-46.43 17.3-53.91q7.3-15.38 10.9-32.01c.74-3.42 2-6.31 4.18-8.64a1.36 1.35 54.7 012.23.39c3.97 9.09 1.66 13.86-1.67 24.65q-10.23 33.19-27.2 63.57-1.8 3.23-4.2 5.84a1.13 1.12-49 01-1.54.11z" fill="currentColor" stroke="none"/>
  <path id="quad-right-0" d="M430.44 1008.31c-12.92-12.62-14.34-33.49-10.92-50.31.31-1.53 1.09-2.53 2.73-2.86q11.44-2.25 23.08-2.59c14.13-.42 17.31 5.67 14.54 18.63q-3.13 14.69-9.12 30.37c-3.45 9.03-11.63 15.25-20.31 6.76z" fill="currentColor" stroke="none"/>
  <path id="quad-right-1" d="M438.96 1059.52q-2.25-1.89-3.8-4.64-20.15-35.92-31.06-75.66-2.11-7.68 1.95-14.16a1.16 1.16 0 011.91-.08c2.26 3.06 3.4 5.4 4.26 9.37 3.98 18.54 10.94 32.53 20.07 51.09 3.51 7.14 11.38 26.16 8.5 33.61a1.16 1.16 0 01-1.83.47z" fill="currentColor" stroke="none"/>
  <path id="tib-left-0" d="M252.09 1032.57c.24-3.71 2.14-22.17 4.63-24.18a1.03 1.02-17.9 011.67.85c-.45 7.89-1.27 16-1.49 23.45q-.57 18.93-.66 37.88-.02 3.63.34 6.85c2.08 18.76 5.56 37.32 9.3 55.8 3.82 18.84 9.13 37.64 13.11 56.63q2.44 11.68 2.08 17.95c-.32 5.7-3.08 20.49-8.51 23.92a.62.62 0 01-.84-.16q-1.2-1.65-.95-3.55c.92-7.26 1.45-14.15-.3-21.52q-8.25-34.74-13.62-59.06c-1.86-8.44-3.17-17.18-3.93-26.3q-3.69-44.24-.83-88.56z" fill="currentColor" stroke="none"/>
  <path id="tib-left-1" d="M315.01 1025.17a.16.16 0 01.32.02c4.06 25.75 8.98 52.72 8.71 77.81q-.13 12.06-5.74 26.31c-7.2 18.3-8.93 38.57-15.95 56.93q-.18.48-.21-.03c-1.87-34.47-5.67-65.91-8.56-103.28q-.97-12.49 4.44-23.14 7.47-14.69 15.14-29.29c.81-1.55 1.35-3.62 1.85-5.33z" fill="currentColor" stroke="none"/>
  <path id="tib-right-0" d="M455.5 1231.67c-7.13-5.81-9.23-24.34-8.2-31.86 1.41-10.32 4.63-23.14 7.98-36.33q9.54-37.46 15.15-75.74c2.86-19.5 1.53-40.15.75-59.8-.22-5.67-.98-12.51-1.23-18.75a.97.97 0 011.87-.4c.35.86.92 1.76 1.12 2.68q2.96 14.31 3.31 20.53 2.37 43.28-.49 84.75-1.21 17.42-5.43 35.77-6.33 27.51-12.84 54.98-2.01 8.49-.11 18.36c.36 1.9.11 3.95-.68 5.55a.79.79 0 01-1.2.26z" fill="currentColor" stroke="none"/>
  <path id="tib-right-1" d="M412.77 1025.44a.14.14 0 01.27-.04c4.88 11.62 10.93 22.01 17.28 34.78 4.07 8.19 4.71 14.41 4.1 24.25-2.13 34.3-6.27 68.85-8.45 101.59q-.05.69-.31.05-1.48-3.67-2.28-6.75c-4.34-16.75-8.78-38.38-16.39-57.57q-1.4-3.55-2.2-10.11c-1.78-14.73-.2-31.24 2.04-45.88q3.06-20.02 5.94-40.32z" fill="currentColor" stroke="none"/>
  <path id="ham-left-0" d="M982.69 1149.31c-3.07-2.23-3.98-6.24-5.24-11.03-7.19-27.14-7.88-53.18-6.67-82.78q1.03-25.29 9.23-47.45c4.77-12.89 15.33-24.77 23.79-36q.82-1.09.74.27c-1.37 22.86-2.72 45.67-3.11 68.49-.52 30.56-1.51 61.11-.42 91.68.24 6.83-2.77 16.29-10.08 18.37q-4.39 1.25-8.24-1.55z" fill="currentColor" stroke="none"/>
  <path id="ham-left-1" d="M983.99 1163.56c7.15-5.59 16.16-.63 17 8.23q4.31 45.02 5.22 90.26c.16 8.25-.8 15.79-2.19 23.65q-.45 2.52-1.43 3.66-.95 1.11-1.22-.33c-5.03-26.7-8.28-53.49-11.87-80.36q-1.68-12.52-3.24-18.71-2.04-8.12-5.53-18.24c-1.03-3 .8-6.25 3.26-8.16z" fill="currentColor" stroke="none"/>
  <path id="ham-left-2" d="M1013.69 1150.31c-4.8-2.61-4.66-16.17-4.36-20.75 2.34-36.49 3.44-73.94 1.04-110.45-1.03-15.55.02-31.49.62-47.06q.03-.66.25-.03c2.28 6.45 4.52 12.88 7.39 19.11 5.12 11.14 11.5 22.91 14.83 33.92q2.34 7.74 3.97 16.46 5.3 28.43 5.62 56.09c.2 18.32-7.9 40-22.63 51.79q-3.42 2.73-6.73.92z" fill="currentColor" stroke="none"/>
  <path id="ham-left-3" d="M1014.14 1164.37c7-1.83 14.1 2.2 14.11 9.95q.06 29.04-5.62 57.41c-3.87 19.28-6.24 38.23-8.43 57.48a.37.37 0 01-.74-.01q-3.12-43.48-3.58-86.64-.15-14.16.76-28.3c.18-2.83.02-8.98 3.5-9.89z" fill="currentColor" stroke="none"/>
  <path id="ham-right-0" d="M1172.94 1149.31c-6.06-4.56-6.94-11.4-6.8-19.4.96-52.67-.49-105.31-3.54-157.9q-.04-.72.41-.16 7.96 10.07 15.43 20.44c9.11 12.64 13.61 28.98 15.78 44.21 4.96 34.71 3.75 72.94-5.97 106.5-1.97 6.82-9.18 10.93-15.31 6.31z" fill="currentColor" stroke="none"/>
  <path id="ham-right-1" d="M1144.41 1147.33q-17.19-17.37-20.08-40.86-.89-7.22-.13-19.97 1.18-20.06 4.69-41.33c2.33-14.1 5.8-25.22 12.41-38.61q8.19-16.59 14.35-34.15a.14.13-37.7 01.26.03q1.01 15.71 1.26 31.44c.18 11.61-1.34 24.91-1.58 36.43-.72 34.7 1.22 62.05 2.06 93.19.17 6.32-1.1 26.1-13.24 13.83z" fill="currentColor" stroke="none"/>
  <path id="ham-right-2" d="M1173.74 1161.73c6.88-2 14.34 3.23 11.98 10.91-2.24 7.3-4.78 14.44-5.99 21.96-5.07 31.52-8.04 63.18-14.13 94.6a.72.71-61.9 01-1.21.37c-.14-.14-.35-.39-.4-.59q-3.53-13.58-3.19-28.23 1.04-44.67 5.06-87.04c.58-6.1 1.93-10.25 7.88-11.98z" fill="currentColor" stroke="none"/>
  <path id="ham-right-3" d="M1154.32 1165a1.58 1.57-84.6 01.97 1.18c.79 4.42 1.42 8.78 1.57 13.4.96 29.17-.47 62.66-2.04 90.23q-.78 13.79-1.39 19.52a.23.23 0 01-.45 0c-2.79-21.25-5.41-41.99-9.64-63.03-3.44-17.08-4.29-34.91-4.68-52.3-.19-8.37 8.99-11.61 15.66-9z" fill="currentColor" stroke="none"/>
  <path id="calf-left-0" d="M998.25 1320.52c-4.62.24-8.17-1.08-8.78-6.28-1.6-13.81-.75-28.85-2.16-42.41q-.39-3.74.24-7.03a.69.69 0 011.23-.28c2.35 3.15 4.22 5.75 5.14 9.66 1.54 6.57 1.91 22.57 9.97 24.09q13.33 2.5 15.93-10.47c.92-4.57 1-12.33 5.05-17.25q.42-.51.42.15c.11 14.39.4 30.86-3.08 44.54-.79 3.13-3.31 4.23-6.51 4.4q-8.73.45-17.45.88z" fill="currentColor" stroke="none"/>
  <path id="calf-right-0" d="M1149.5 1319.51c-6.93-.63-6.82-18.08-7.14-23.7q-.73-12.53-.59-25.09.01-.71.45-.15 2.74 3.49 3.29 7.17c1.67 11.25 3.21 25.34 19.7 19.99 4.87-1.58 7.03-18.57 7.89-23.21.79-4.2 2.74-7 5.28-10.13a.56.56 0 01.98.22c1.12 4.6.04 12.39-.37 17.26-.92 10.77-.32 21.48-1.52 32.37q-.7 6.23-7.01 6.18-12.13-.11-20.96-.91z" fill="currentColor" stroke="none"/>
  <path id="glute-left-0" d="M1070.06 785.19c2.95 1.36 1.8 10.43 1.49 13.04q-3.98 33.27-14.66 64.61a.39.39 0 01-.76-.17c.9-7.05 2.31-14.29 2.16-20.92q-.68-30.14-18.71-54.52-.29-.39.18-.49c7.42-1.52 23.53-4.69 30.3-1.55z" fill="currentColor" stroke="none"/>
  <path id="glute-right-0" d="M1127.24 787.66c-15.99 21.49-22.3 48.51-16.08 74.83a.47.46-63.2 01-.88.29q-1.99-4.69-3.65-10.24-8.29-27.75-11.6-56.54c-.65-5.71-1.1-11.77 6.87-11.9q13-.19 25.68 2.83a.31.24 41.2 01.1.53q-.12.01-.27.07-.1.04-.17.13z" fill="currentColor" stroke="none"/>
</defs>
</svg>

<!-- ═══ HEADER ═══ -->
<header class="app-header">
  <div class="logo">
    <div class="logo-icon"><div class="logo-wave"></div></div>
    <div class="logo-text">FAST <span>ATHLETE</span></div>
  </div>
  <div class="header-right">
    <button class="profile-chip" id="profile-chip" onclick="openProfileSheet()">
      <span class="profile-avatar" id="profile-avatar">G</span>
      <span class="profile-name" id="profile-name">Guest</span>
    </button>
  </div>
</header>

<!-- ═══ PROFILE SHEET ═══ -->
<div class="sheet-overlay hidden" id="profile-sheet" onclick="if(event.target===this)closeProfileSheet()">
  <div class="sheet">
    <div class="sheet-header">
      <span class="sheet-title">WHO IS CHECKING?</span>
      <button class="sheet-close" onclick="closeProfileSheet()">✕</button>
    </div>
    <div id="profile-list"></div>
    <div class="hidden" id="pin-box" style="margin-top:4px">
      <div class="sheet-pin-row" style="margin-bottom:8px">
        <input class="sheet-input" id="pin-input" placeholder="Enter 4-digit PIN" inputmode="numeric" maxlength="4" autocomplete="off">
        <button class="btn-primary" style="width:auto;padding:12px 18px;font-size:13px" onclick="confirmPin()">OK</button>
      </div>
      <div style="font-size:11px;color:var(--text-faint)">This profile is protected by a PIN.</div>
    </div>
    <div class="sheet-divider"></div>
    <div class="sheet-add">
      <input class="sheet-input" id="new-profile-name" placeholder="Your name" maxlength="30" autocomplete="off">
      <div class="sheet-pin-row">
        <input class="sheet-input" id="new-profile-pin" placeholder="PIN (optional)" inputmode="numeric" maxlength="4" autocomplete="off">
        <button class="btn-primary" style="width:auto;padding:12px 18px;font-size:13px" onclick="createProfile()">Add</button>
      </div>
      <div class="sheet-hint">Profiles keep your results saved on this server. A 4-digit PIN is optional.</div>
    </div>
  </div>
</div>

<div class="toast" id="toast"></div>

<!-- ═══ SCREEN 1: WELCOME / DASHBOARD ═══ -->
<div class="section active" id="screen-welcome">
  <div class="welcome">
    <h1>How are your muscles?</h1>
    <p>Upload an sEMG recording. FAST analyses each muscle for fatigue — giving you a clear green, amber, or red status.</p>
  </div>

  <div class="home-grid">
    <div class="home-left">
      <div class="card" id="gauge-card" style="display:none">
        <div class="card-header">FATIGUE ASSESSMENT</div>
        <div class="gauge-wrap">
          <div class="gauge-ring">
            <svg width="180" height="180" viewBox="0 0 180 180">
              <circle class="bg" cx="90" cy="90" r="78"/>
              <circle class="fill" id="gauge-fill" cx="90" cy="90" r="78"
                stroke-dasharray="490" stroke-dashoffset="490"/>
            </svg>
            <div class="gauge-center">
              <div class="gauge-value" id="gauge-value" style="color:var(--green)">--</div>
              <div class="gauge-label">recovery</div>
              <div class="gauge-status"><span class="gauge-dot" id="gauge-dot" style="background:var(--green)"></span><span id="gauge-status-label">Optimal</span></div>
            </div>
          </div>
        </div>
      </div>

      <div class="card" id="last-assessment" style="display:none">
        <div class="card-header">LAST ASSESSMENT</div>
        <div class="info-row">
          <div class="info-item"><span class="info-key">Muscle</span><span class="info-val" id="la-muscle">—</span></div>
          <div class="info-item"><span class="info-key">Date</span><span class="info-val" id="la-date">—</span></div>
          <div class="info-item"><span class="info-key">Result</span><span class="info-val green" id="la-result">—</span></div>
        </div>
      </div>

      <button class="btn-primary" onclick="startNewAssessment()">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="6" r="4"/><path d="M6 20v-2a6 6 0 0112 0v2"/></svg>
        START NEW FAST ASSESSMENT
      </button>

      <div class="card">
        <div class="card-header">QUICK TIP</div>
        <div class="tip-body">Record a steady 30-second contraction — that gives the most reliable results.</div>
      </div>
    </div>

    <div class="home-right">
      <div class="card">
        <div class="card-header">HOW IT WORKS</div>
        <div class="steps">
          <div class="step">
            <div class="step-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 17v3a2 2 0 002 2h14a2 2 0 002-2v-3"/><path d="M12 3v13"/><path d="M7 8l5-5 5 5"/></svg></div>
            <div>
              <div class="step-title">Upload your recording</div>
              <div class="step-desc">Drop your sEMG file — .mat, .csv, or .txt</div>
            </div>
          </div>
          <div class="step">
            <div class="step-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M8 12.5l2.5 2.5L16 9.5"/></svg></div>
            <div>
              <div class="step-title">Pick your muscles</div>
              <div class="step-desc">Choose the ones you want to check</div>
            </div>
          </div>
          <div class="step">
            <div class="step-icon"><svg viewBox="0 0 24 24"><circle cx="12" cy="6" r="3.2" fill="#00E676"/><circle cx="12" cy="12" r="3.2" fill="#FF9100"/><circle cx="12" cy="18" r="3.2" fill="#ef4444"/></svg></div>
            <div>
              <div class="step-title">Get your status</div>
              <div class="step-desc">Clear green, amber, or red for each muscle</div>
            </div>
          </div>
        </div>
      </div>

      <div class="card example-card">
        <div class="card-header">EXAMPLE RESULT<span class="tag">PREVIEW</span></div>
        <div class="example-row">
          <span class="tl-dot tl-green"></span>
          <div class="example-info">
            <div class="example-name">Vastus Lateralis</div>
            <div class="example-meta">Score 92 — Not fatigued</div>
          </div>
          <div class="example-score">92</div>
        </div>
        <div class="example-foot">Every muscle gets its own status and a simple tip.</div>
      </div>
    </div>
  </div>
</div>

<!-- ═══ SCREEN 2: UPLOAD ═══ -->
<div class="section" id="screen-upload">
  <div class="card" style="padding:0;overflow:hidden">
    <label class="upload-zone" id="upload-zone" for="file-input">
      <div class="upload-icon">📁</div>
      <div class="upload-text" id="upload-main-text">Drop your sEMG recording here</div>
      <div class="upload-formats">.mat · .csv · .txt</div>
    </label>
    <input type="file" id="file-input" accept=".mat,.csv,.txt">
  </div>

  <div id="file-info-section" class="hidden">
    <div class="card" style="margin-bottom:8px">
      <div class="file-chip" id="file-chip-name">—</div>
      <div class="file-meta" id="file-chip-meta"></div>
    </div>

    <div id="muscle-selection" class="hidden">
      <div style="font-size:14px;font-weight:600;margin:12px 0 8px">Select muscles to check</div>
      <div class="muscle-grid" id="muscle-grid"></div>
      <div id="selection-count" style="font-size:12px;color:var(--text-dim);margin-bottom:8px">0 selected</div>
    </div>
  </div>

  <div class="count-bar" id="analyse-bar" style="display:none">
    <span class="count-text"><span class="count-num" id="analyse-count">0</span> muscles selected</span>
    <button class="btn-primary" style="width:auto;padding:12px 24px;font-size:13px" onclick="runAnalysis()">Check Fatigue</button>
  </div>
</div>

<!-- ═══ SCREEN 3: PROGRESS ═══ -->
<div class="section" id="screen-progress">
  <div class="progress-section">
    <div class="progress-ring">
      <svg width="80" height="80" viewBox="0 0 80 80">
        <circle class="bg" cx="40" cy="40" r="34"/>
        <circle class="fill" id="prog-fill" cx="40" cy="40" r="34"
          stroke-dasharray="213.6" stroke-dashoffset="213.6"/>
      </svg>
    </div>
    <div class="progress-label" id="prog-label">Running FAST analysis…</div>
    <div class="progress-step" id="prog-step"></div>
  </div>
</div>

<!-- ═══ SCREEN 4: RESULTS ═══ -->
<div class="section" id="screen-results">
  <div id="status-banner-container"></div>

  <div class="card" id="results-gauge-card">
    <div class="card-header">FATIGUE ASSESSMENT</div>
    <div class="gauge-wrap">
      <div class="gauge-ring">
        <svg width="180" height="180" viewBox="0 0 180 180">
          <circle class="bg" cx="90" cy="90" r="78"/>
          <circle class="fill" id="results-gauge-fill" cx="90" cy="90" r="78"
            stroke-dasharray="490" stroke-dashoffset="490"/>
        </svg>
        <div class="gauge-center">
          <div class="gauge-value" id="results-gauge-value">--</div>
          <div class="gauge-label">recovery</div>
          <div class="gauge-status"><span class="gauge-dot" id="results-gauge-dot"></span><span id="results-gauge-status">—</span></div>
        </div>
      </div>
    </div>
  </div>

  <div class="card" id="muscle-map-card">
    <div class="card-header">CURRENT MUSCLE MAP</div>
    <div class="muscle-map" id="muscle-map-svg"></div>
    <div class="callout-row" id="callout-row"></div>
  </div>

  <div class="card" id="trend-card">
    <div class="card-header">RECENT FATIGUE TREND</div>
    <div class="bar-chart" id="trend-chart">
      <div class="bar-col"><div class="bar" style="height:45%;background:var(--amber)"></div><div class="bar-label">M</div></div>
      <div class="bar-col"><div class="bar" style="height:40%;background:var(--amber)"></div><div class="bar-label">T</div></div>
      <div class="bar-col"><div class="bar" style="height:60%;background:var(--amber)"></div><div class="bar-label">W</div></div>
      <div class="bar-col"><div class="bar" style="height:70%;background:var(--green)"></div><div class="bar-label">T</div></div>
      <div class="bar-col"><div class="bar" style="height:75%;background:var(--green)"></div><div class="bar-label">F</div></div>
      <div class="bar-col"><div class="bar" style="height:80%;background:var(--green)"></div><div class="bar-label">S</div></div>
      <div class="bar-col"><div class="bar" style="height:92%;background:var(--green)" id="today-bar"></div><div class="bar-label" style="color:var(--cyan)">Today</div></div>
    </div>
  </div>

  <div class="card" id="results-detail">
    <div class="card-header">MUSCLE BREAKDOWN</div>
    <div class="summary-row" id="results-summary">
      <div class="summary-card green"><div class="summary-num" id="count-green">0</div><div class="summary-label">Not fatigued</div></div>
      <div class="summary-card amber"><div class="summary-num" id="count-amber">0</div><div class="summary-label">Some fatigue</div></div>
      <div class="summary-card red"><div class="summary-num" id="count-red">0</div><div class="summary-label">Needs rest</div></div>
    </div>
    <div id="results-list"></div>
  </div>

  <div class="results-footer" style="text-align:center;margin-top:12px">
    <button class="btn-outline" onclick="startNewAssessment()">New Assessment</button>
  </div>
</div>

<!-- ═══ SCREEN 5: HISTORY ═══ -->
<div class="section" id="screen-history">
  <div style="font-size:12px;font-weight:700;color:var(--text-dim);text-transform:uppercase;letter-spacing:.06em;margin:16px 0 12px">SAVED ASSESSMENTS</div>
  <div id="history-list"></div>
</div>

<!-- ═══ BOTTOM NAV ═══ -->
<nav class="bottom-nav" id="bottom-nav">
  <button class="nav-item active" data-screen="welcome" onclick="navTo('welcome')">
    <svg viewBox="0 0 24 24" fill="currentColor"><path d="M10 20v-6h4v6h5v-8h3L12 3 2 12h3v8z"/></svg>
    <span class="nav-label">Home</span>
  </button>
  <button class="nav-item" data-screen="upload" onclick="navTo('upload')">
    <svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/></svg>
    <span class="nav-label">Assess</span>
  </button>
  <button class="nav-item" data-screen="history" onclick="navTo('history')">
    <svg viewBox="0 0 24 24" fill="currentColor"><path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/></svg>
    <span class="nav-label">History</span>
  </button>
  <button class="nav-item" data-screen="results" onclick="navTo('results')">
    <svg viewBox="0 0 24 24" fill="currentColor"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2" fill="none" stroke="currentColor" stroke-width="2"/></svg>
    <span class="nav-label">Train</span>
  </button>
  <button class="nav-item" data-screen="results" onclick="navTo('results')">
    <svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/></svg>
    <span class="nav-label">Profile</span>
  </button>
</nav>

</div>

<script>
let sessionData = null;
let muscles = [];
let lastResults = null;
let currentScreen = 'welcome';
let profile = null;
let profileList = [];
let pinTarget = null;

function navTo(screen) {
  document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  const target = document.getElementById('screen-' + screen);
  if (target) target.classList.add('active');
  const navBtn = document.querySelector('.nav-item[data-screen="' + screen + '"]');
  if (navBtn) navBtn.classList.add('active');
  currentScreen = screen;
  if (screen === 'history') loadHistory();
}

function startNewAssessment() {
  navTo('upload');
  document.getElementById('upload-zone').classList.remove('has-file');
  document.getElementById('upload-main-text').textContent = 'Drop your sEMG recording here';
  document.getElementById('file-info-section').classList.add('hidden');
  document.getElementById('muscle-selection').classList.add('hidden');
  document.getElementById('analyse-bar').style.display = 'none';
  document.getElementById('muscle-grid').innerHTML = '';
}

// ═══ UPLOAD ═══
const uploadZone = document.getElementById('upload-zone');
const fileInput = document.getElementById('file-input');

['dragenter','dragover'].forEach(e => {
  uploadZone.addEventListener(e, ev => { ev.preventDefault(); uploadZone.classList.add('dragover'); });
});
['dragleave','drop'].forEach(e => {
  uploadZone.addEventListener(e, ev => { ev.preventDefault(); uploadZone.classList.remove('dragover'); });
});
uploadZone.addEventListener('drop', ev => {
  const file = ev.dataTransfer.files[0];
  if (file) handleFile(file);
});
fileInput.addEventListener('change', () => {
  const file = fileInput.files[0];
  if (file) handleFile(file);
});

async function handleFile(file) {
  document.getElementById('upload-main-text').textContent = 'Reading file...';
  const formData = new FormData();
  formData.append('file', file);
  try {
    const resp = await fetch('/api/upload', { method: 'POST', body: formData });
    const data = await resp.json();
    if (data.error) { alert(data.error); return; }
    sessionData = data;
    muscles = data.muscles;
    uploadZone.classList.add('has-file');
    document.getElementById('upload-main-text').textContent = file.name;
    document.getElementById('file-info-section').classList.remove('hidden');
    document.getElementById('file-chip-name').textContent = file.name;
    document.getElementById('file-chip-meta').textContent = data.duration.toFixed(0) + 's, ' + data.fs + ' Hz, ' + data.cols.length + ' channels';
    buildMuscleGrid();
  } catch(e) { alert('Upload failed: ' + e.message); }
}

function buildMuscleGrid() {
  document.getElementById('muscle-selection').classList.remove('hidden');
  const grid = document.getElementById('muscle-grid');
  grid.innerHTML = '';
  muscles.forEach((m, i) => {
    const svg = muscleThumbSVG(m.id || m.column, m.colour || '#00E5FF');
    grid.innerHTML += '<div class="muscle-card selected" data-id="' + (m.id || m.column) + '" data-col="' + m.column + '" onclick="toggleMuscle(this)"><div class="body-svg">' + svg + '</div><div style="flex:1"><div class="muscle-name">' + (m.id || m.column) + ' - ' + m.name + '</div><div class="muscle-desc">' + m.desc + '</div></div><div class="check-ring"></div></div>';
  });
  document.getElementById('analyse-bar').style.display = 'flex';
  updateCount();
}

function toggleMuscle(el) {
  el.classList.toggle('selected');
  updateCount();
}

function updateCount() {
  const n = document.querySelectorAll('.muscle-card.selected').length;
  document.getElementById('analyse-count').textContent = n;
  document.getElementById('selection-count').textContent = n + ' of ' + muscles.length + ' selected';
}

function muscleThumbSVG(mid, colour) {
  const frontMuscles = ['VL','VM','RF','TA'];
  const isFront = frontMuscles.includes(mid);
  const vb = isFront ? '0 700 724 748' : '724 700 724 748';
  const outlineId = isFront ? 'body-front' : 'body-back';
  let musclePathId = 'quad-left-0';
  if (['VL','VM','RF'].includes(mid)) musclePathId = 'quad-left-0';
  else if (['BF','ST'].includes(mid)) musclePathId = 'ham-left-0';
  else if (['GM','GL'].includes(mid)) musclePathId = 'calf-left-0';
  else if (mid === 'TA') musclePathId = 'tib-left-0';
  return '<svg viewBox="' + vb + '" width="36" height="64"><rect width="724" height="748" fill="#1a1f2e"/><use href="#' + outlineId + '" stroke="rgba(255,255,255,0.08)" stroke-width="2"/><use href="#' + musclePathId + '" fill="' + colour + '33" stroke="' + colour + '88" stroke-width="1"/></svg>';
}

// ═══ ANALYSIS ═══
async function runAnalysis() {
  const cards = document.querySelectorAll('.muscle-card.selected');
  const selected = Array.from(cards).map(c => {
    const mid = c.dataset.id;
    const col = c.dataset.col;
    const m = muscles.find(x => (x.id || x.column) === mid);
    return m || {id: mid, column: col, name: col, desc: 'Channel'};
  }).filter(Boolean);

  if (!selected.length) { alert('Select at least one muscle'); return; }

  navTo('progress');
  document.getElementById('prog-label').textContent = 'Running FAST analysis...';
  document.getElementById('prog-step').textContent = selected.length + ' muscle' + (selected.length>1?'s':'') + ' selected';
  document.getElementById('prog-fill').style.strokeDashoffset = '213.6';

  let progress = 0;
  const interval = setInterval(() => {
    progress = Math.min(progress + 3, 90);
    const offset = 213.6 - (213.6 * progress / 100);
    document.getElementById('prog-fill').style.strokeDashoffset = offset;
  }, 100);

  try {
    const resp = await fetch('/api/analyze', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ session: sessionData.session, muscles: selected })
    });
    const text = await resp.text();
    let data;
    try { data = JSON.parse(text); } catch(e) { data = {error:'Server error',results:{}}; }
    clearInterval(interval);
    document.getElementById('prog-fill').style.strokeDashoffset = '0';

    if (data.error && Object.keys(data.results||{}).length === 0) {
      document.getElementById('prog-label').textContent = 'Error: ' + data.error;
      return;
    }

    for (const m of selected) {
      m.result = (data.results || {})[m.id || m.column] || {status:'grey',score:0,tip:'No result'};
    }
    lastResults = selected;
    showResults(selected);
    navTo('results');
  } catch(e) {
    clearInterval(interval);
    for (const m of selected) m.result = {status:'grey',score:0,tip:e.message};
    lastResults = selected;
    showResults(selected);
    navTo('results');
  }
}

// ═══ RESULTS ═══
function showResults(selected) {
  let green = 0, amber = 0, red = 0;
  let totalScore = 0, nValid = 0;
  selected.forEach(m => {
    const s = (m.result||{}).status;
    if (s==='green') green++;
    else if (s==='amber') amber++;
    else if (s==='red') red++;
    if (s&&s!=='grey') { totalScore += m.result.score||0; nValid++; }
  });

  const avgScore = nValid > 0 ? Math.round(totalScore / nValid) : 0;
  let overallStatus, gaugeColor, gaugeRgba;
  if (avgScore >= 70) { overallStatus = 'RECOVERED'; gaugeColor = 'var(--green)'; gaugeRgba = '0,230,118'; }
  else if (avgScore >= 40) { overallStatus = 'SOME FATIGUE'; gaugeColor = 'var(--amber)'; gaugeRgba = '255,145,0'; }
  else { overallStatus = 'FATIGUED'; gaugeColor = 'var(--red)'; gaugeRgba = '239,68,68'; }

  document.getElementById('status-banner-container').innerHTML =
    '<div class="status-banner" style="background:rgba(' + gaugeRgba + ',0.08);border-color:rgba(' + gaugeRgba + ',0.15)">' +
    '<div style="font-size:12px;color:var(--text-dim);text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px">MUSCLE FATIGUE STATUS</div>' +
    '<h2 style="color:' + gaugeColor + ';margin:0">' + overallStatus + '</h2></div>';

  const gaugeOffset = 490 - (490 * avgScore / 100);
  const gf = document.getElementById('results-gauge-fill');
  gf.setAttribute('stroke', gaugeColor);
  gf.setAttribute('stroke-dashoffset', gaugeOffset);
  document.getElementById('results-gauge-value').textContent = avgScore + '%';
  document.getElementById('results-gauge-value').style.color = gaugeColor;
  document.getElementById('results-gauge-dot').style.background = gaugeColor;
  document.getElementById('results-gauge-status').textContent = avgScore >= 70 ? 'Optimal' : avgScore >= 40 ? 'Moderate' : 'Needs rest';

  document.getElementById('count-green').textContent = green;
  document.getElementById('count-amber').textContent = amber;
  document.getElementById('count-red').textContent = red;

  buildMuscleMap(selected);

  let listHTML = '';
  selected.forEach(m => {
    const r = m.result||{};
    const status = r.status||'grey';
    const score = r.score||0;
    let color;
    if (status==='green') color = 'var(--green)';
    else if (status==='amber') color = 'var(--amber)';
    else if (status==='red') color = 'var(--red)';
    else color = 'var(--text-dim)';
    const badgeClass = 'badge-' + status;
    const badgeText = status==='green'?'Not fatigued':status==='amber'?'Some fatigue':status==='red'?'Needs rest':'No data';
    const tip = r.tip||'';
    const cleanColor = color.replace('var(','').replace(')','');
    const svg = muscleThumbSVG(m.id||m.column, cleanColor);
    listHTML += '<div class="result-card"><div class="body-svg">' + svg + '</div><div style="flex:1"><div class="muscle-name">' + (m.id||m.column) + ' - ' + m.name + '</div><div class="muscle-desc">' + m.desc + '</div><div class="result-tip">' + tip + '</div>' + (r.centroid != null ? '<div style="font-size:11px;color:#555b68;margin-top:3px">LOW centroid: ' + r.centroid + ' Hz (8–23 Hz)</div>' : '') + '</div><div class="result-score" style="color:' + color + '">' + score + '</div><span class="result-badge ' + badgeClass + '">' + badgeText + '</span></div>';
  });
  document.getElementById('results-list').innerHTML = listHTML;

  // Update welcome screen
  document.getElementById('gauge-fill').setAttribute('stroke', gaugeColor);
  document.getElementById('gauge-fill').setAttribute('stroke-dashoffset', gaugeOffset);
  document.getElementById('gauge-value').textContent = avgScore + '%';
  document.getElementById('gauge-value').style.color = gaugeColor;
  document.getElementById('gauge-dot').style.background = gaugeColor;
  document.getElementById('gauge-status-label').textContent = avgScore >= 70 ? 'Optimal' : avgScore >= 40 ? 'Moderate' : 'Needs rest';
  document.getElementById('gauge-card').style.display = 'block';
  document.getElementById('last-assessment').style.display = 'block';
  if (selected.length) {
    const m = selected[0];
    document.getElementById('la-muscle').textContent = (m.id||m.column) + ' - ' + m.name;
    document.getElementById('la-date').textContent = new Date().toLocaleDateString('en-US',{month:'short',day:'numeric'});
    document.getElementById('la-result').textContent = (m.result||{}).status==='green'?'Low Fatigue':'Moderate';
  }

  const todayBar = document.getElementById('today-bar');
  todayBar.style.height = avgScore + '%';
  todayBar.style.background = gaugeColor;

  saveAssessment(avgScore, overallStatus);
}

function buildMuscleMap(selected) {
  const calloutRow = document.getElementById('callout-row');
  calloutRow.innerHTML = '';
  const REGIONS = {
    'quad-right': {view:'front', ids:['quad-right-0','quad-right-1'], muscles:['VL','VM','RF']},
    'tib-right':  {view:'front', ids:['tib-right-0','tib-right-1'],  muscles:['TA']},
    'ham-right':  {view:'back',  ids:['ham-right-0','ham-right-1','ham-right-2','ham-right-3'], muscles:['BF','ST']},
    'calf-right': {view:'back',  ids:['calf-right-0'], muscles:['GM','GL']}
  };
  const regionState = {};
  selected.forEach(m => {
    const r = m.result||{};
    const status = r.status||'grey';
    const mid = m.id||m.column;
    let regionKey = null;
    for (const k in REGIONS) { if (REGIONS[k].muscles.indexOf(mid) >= 0) { regionKey = k; break; } }
    if (regionKey) {
      if (!regionState[regionKey]) regionState[regionKey] = {minScore: Infinity, hasScore: false};
      if (r.score != null && status !== 'grey') {
        regionState[regionKey].hasScore = true;
        if (r.score < regionState[regionKey].minScore) regionState[regionKey].minScore = r.score;
      }
    }
    const colorVar = status==='green'?'var(--green)':status==='amber'?'var(--amber)':status==='red'?'var(--red)':'var(--text-dim)';
    const label = status==='green'?'Recovered':status==='amber'?'Moderate fatigue':status==='red'?'Fatigued':'No data';
    calloutRow.innerHTML += '<div class="callout"><span class="callout-dot" style="background:' + colorVar + '"></span><span>' + mid + ': <b>' + label + '</b></span></div>';
  });
  let frontUses = '', backUses = '';
  for (const k in REGIONS) {
    const reg = REGIONS[k];
    if (!regionState[k]) continue;
    let fillColor, strokeColor;
    if (!regionState[k].hasScore) { fillColor='rgba(255,255,255,0.05)'; strokeColor='rgba(255,255,255,0.1)'; }
    else {
      const c = heatColor(regionState[k].minScore);
      fillColor = 'rgba(' + c + ',0.32)';
      strokeColor = 'rgba(' + c + ',0.65)';
    }
    let uses = '';
    for (let i = 0; i < reg.ids.length; i++) {
      uses += '<use href="#' + reg.ids[i] + '" fill="' + fillColor + '" stroke="' + strokeColor + '" stroke-width="1.5"/>';
    }
    if (reg.view === 'front') frontUses += uses; else backUses += uses;
  }
  document.getElementById('muscle-map-svg').innerHTML =
    '<div style="display:flex;flex-direction:column;align-items:center;gap:10px">' +
    '<svg viewBox="0 700 1448 748"><use href="#body-front" class="body-outline"/>' + frontUses +
    '<use href="#body-back" class="body-outline"/>' + backUses + '</svg>' +
    '<div style="width:min(240px,80%);display:flex;flex-direction:column;gap:4px">' +
    '<div style="height:8px;border-radius:4px;background:linear-gradient(90deg,#ef4444,#FF9100,#00E676)"></div>' +
    '<div style="display:flex;justify-content:space-between;font-size:10px;color:var(--text-dim)">' +
    '<span>Fatigued</span><span>Some fatigue</span><span>Recovered</span></div></div></div>';
}

function heatColor(score) {
  const s = Math.max(0, Math.min(100, Number(score) || 0));
  const stops = [[0,239,68,68],[35,255,145,0],[70,0,230,118],[100,0,230,118]];
  for (let i = 0; i < stops.length - 1; i++) {
    const s0 = stops[i][0], s1 = stops[i+1][0];
    if (s >= s0 && s <= s1) {
      const t = (s - s0) / (s1 - s0 || 1);
      const r = Math.round(stops[i][1] + (stops[i+1][1] - stops[i][1]) * t);
      const g = Math.round(stops[i][2] + (stops[i+1][2] - stops[i][2]) * t);
      const b = Math.round(stops[i][3] + (stops[i+1][3] - stops[i][3]) * t);
      return r + ',' + g + ',' + b;
    }
  }
  return '239,68,68';
}

// ═══ PROFILES ═══
try { profile = JSON.parse(localStorage.getItem('fast_profile') || 'null'); } catch(e) { profile = null; }

function openProfileSheet() { loadProfiles(); document.getElementById('profile-sheet').classList.remove('hidden'); }
function closeProfileSheet() { document.getElementById('profile-sheet').classList.add('hidden'); }

async function loadProfiles() {
  try {
    const resp = await fetch('/api/profiles');
    const data = await resp.json();
    profileList = data.profiles || [];
    const list = document.getElementById('profile-list');
    list.innerHTML = '';
    profileList.forEach(p => {
      const isActive = profile && profile.id === p.id;
      list.innerHTML += '<div class="profile-row"' + (isActive ? ' style="border-color:var(--cyan)"' : '') + ' onclick="onProfileClick(' + p.id + ')">' +
        '<span class="profile-avatar">' + p.name.charAt(0).toUpperCase() + '</span>' +
        '<span class="profile-row-name">' + p.name + '</span>' +
        (p.has_pin ? '<span style="font-size:10px;color:var(--text-faint)">PIN</span>' : '') +
        (isActive ? '<span class="profile-active">ACTIVE</span>' : '') + '</div>';
    });
    if (!profileList.length) {
      list.innerHTML = '<div style="font-size:13px;color:var(--text-faint);padding:8px 0">No profiles yet — add one below.</div>';
    }
  } catch(e) {}
}

async function selectProfile(id) {
  const p = profileList.find(x => x.id === id);
  if (!p) return;
  profile = {id: p.id, name: p.name};
  localStorage.setItem('fast_profile', JSON.stringify(profile));
  updateProfileChip();
  closeProfileSheet();
  toast('Profile: ' + p.name);
}

function onProfileClick(id) {
  const p = profileList.find(x => x.id === id);
  if (!p) return;
  if (p.has_pin) {
    pinTarget = id;
    document.getElementById('pin-box').classList.remove('hidden');
    const input = document.getElementById('pin-input');
    input.value = '';
    input.focus();
    return;
  }
  selectProfile(id);
}

async function confirmPin() {
  const input = document.getElementById('pin-input');
  const pin = input.value.trim();
  if (!pin) { input.focus(); return; }
  const resp = await fetch('/api/profiles/verify', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({id: pinTarget, pin: pin})
  });
  const data = await resp.json();
  if (data.error) { alert(data.error); input.value = ''; input.focus(); return; }
  document.getElementById('pin-box').classList.add('hidden');
  pinTarget = null;
  profile = {id: data.id, name: data.name};
  localStorage.setItem('fast_profile', JSON.stringify(profile));
  updateProfileChip();
  closeProfileSheet();
  toast('Profile: ' + data.name);
}

async function createProfile() {
  const name = document.getElementById('new-profile-name').value.trim();
  const pin = document.getElementById('new-profile-pin').value.trim();
  if (!name) { alert('Enter a name'); return; }
  const resp = await fetch('/api/profiles', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({name: name, pin: pin || undefined})
  });
  const data = await resp.json();
  if (data.error) { alert(data.error); return; }
  document.getElementById('new-profile-name').value = '';
  document.getElementById('new-profile-pin').value = '';
  profile = {id: data.id, name: data.name};
  localStorage.setItem('fast_profile', JSON.stringify(profile));
  updateProfileChip();
  await loadProfiles();
  toast('Profile added: ' + data.name);
}

function updateProfileChip() {
  document.getElementById('profile-name').textContent = profile ? profile.name : 'Guest';
  document.getElementById('profile-avatar').textContent = profile ? profile.name.charAt(0).toUpperCase() : 'G';
}

// ═══ SAVE + HISTORY ═══
async function saveAssessment(avgScore, overallStatus) {
  if (!profile) { toast('Add a profile to save results'); return; }
  if (!lastResults || !lastResults.length) return;
  const results = lastResults.map(m => ({
    id: m.id || m.column, name: m.name,
    status: (m.result || {}).status, score: (m.result || {}).score,
    centroid: (m.result || {}).centroid
  }));
  try {
    const resp = await fetch('/api/save', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        profile_id: profile.id,
        filename: sessionData ? sessionData.filename : '',
        avg_score: avgScore, status: overallStatus, results: results
      })
    });
    const data = await resp.json();
    if (!data.error) toast('Saved to ' + profile.name);
  } catch(e) {}
}

async function loadHistory() {
  const list = document.getElementById('history-list');
  if (!profile) {
    list.innerHTML = '<div class="card" style="text-align:center;color:var(--text-dim);font-size:13px;padding:32px 20px">Pick a profile to see saved assessments.</div>';
    return;
  }
  list.innerHTML = '<div class="progress-step">Loading…</div>';
  try {
    const resp = await fetch('/api/history?profile_id=' + profile.id);
    const data = await resp.json();
    const h = data.history || [];
    if (!h.length) {
      list.innerHTML = '<div class="card" style="text-align:center;color:var(--text-dim);font-size:13px;padding:32px 20px">No saved assessments yet for <b>' + profile.name + '</b>.</div>';
      return;
    }
    list.innerHTML = '';
    h.forEach(item => {
      const d = new Date(item.ts);
      const dateStr = d.toLocaleDateString('en-US', {month: 'short', day: 'numeric'}) + ' · ' + d.toLocaleTimeString('en-US', {hour: 'numeric', minute: '2-digit'});
      const st = (item.status || '').toUpperCase();
      let color = 'var(--green)';
      if (st.indexOf('SOME') >= 0) color = 'var(--amber)';
      else if (st.indexOf('FATIGUED') >= 0) color = 'var(--red)';
      let chips = '';
      (item.results || []).forEach(r => {
        let c = 'var(--text-dim)';
        if (r.status === 'green') c = 'var(--green)';
        else if (r.status === 'amber') c = 'var(--amber)';
        else if (r.status === 'red') c = 'var(--red)';
        chips += '<span class="hist-chip"><span class="tl-dot" style="background:' + c + '"></span>' + r.id + '</span>';
      });
      list.innerHTML += '<div class="card"><div class="hist-head">' +
        '<span class="hist-score" style="color:' + color + '">' + (item.avg_score != null ? Math.round(item.avg_score) + '%' : '—') + '</span>' +
        '<div style="flex:1"><div class="hist-date">' + dateStr + '</div><div class="hist-file">' + (item.filename || '') + '</div></div>' +
        '<span class="hist-status" style="color:' + color + '">' + st + '</span></div>' +
        '<div class="hist-chips">' + chips + '</div></div>';
    });
  } catch(e) {
    list.innerHTML = '<div class="card" style="text-align:center;color:var(--text-dim)">Could not load history.</div>';
  }
}

// ═══ TOAST ═══
let toastTimer = null;
function toast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove('show'), 2600);
}

document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('gauge-fill').setAttribute('stroke-dashoffset', '490');
  updateProfileChip();
});

document.getElementById('new-profile-name').addEventListener('keydown', e => { if (e.key === 'Enter') createProfile(); });
document.getElementById('new-profile-pin').addEventListener('keydown', e => { if (e.key === 'Enter') createProfile(); });
document.getElementById('pin-input').addEventListener('keydown', e => { if (e.key === 'Enter') confirmPin(); });
</script>
</body>
</html>'''

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8501)