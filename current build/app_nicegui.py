"""
FAST — Muscle Fatigue Check  (NiceGUI edition)
Filter & Aggregate Synchrosqueezed Transform
"""
from __future__ import annotations
import asyncio
import io
import time
import re

import numpy as np
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.signal import butter, filtfilt
from ssqueezepy import ssq_cwt

from nicegui import ui, run

# ═══════════════════════════════════════════════════════════════════════
# PARULA COLORMAP
# ═══════════════════════════════════════════════════════════════════════

_PARULA = [
    [0.2081,0.1663,0.5292],[0.2116,0.1898,0.5777],[0.2123,0.2138,0.627],
    [0.2081,0.2386,0.6771],[0.1959,0.2645,0.7279],[0.1707,0.2919,0.7792],
    [0.1253,0.3242,0.8303],[0.0591,0.3598,0.8683],[0.0117,0.3875,0.882],
    [0.006,0.4086,0.8828],[0.0165,0.4266,0.8786],[0.0329,0.443,0.872],
    [0.0498,0.4586,0.8641],[0.0629,0.4735,0.8554],[0.0723,0.4886,0.8467],
    [0.0779,0.504,0.8384],[0.0793,0.52,0.8312],[0.0749,0.5375,0.8263],
    [0.0641,0.557,0.824],[0.0488,0.5772,0.8228],[0.0343,0.5966,0.8199],
    [0.0265,0.6137,0.8135],[0.0239,0.6287,0.8038],[0.0231,0.6418,0.7913],
    [0.0228,0.6535,0.7768],[0.0267,0.6642,0.7607],[0.0384,0.6743,0.7436],
    [0.059,0.6838,0.7254],[0.0843,0.6928,0.7062],[0.1133,0.7015,0.6859],
    [0.1453,0.7098,0.6646],[0.1801,0.7177,0.6424],[0.2178,0.725,0.6193],
    [0.2586,0.7317,0.5954],[0.3022,0.7376,0.5712],[0.3482,0.7424,0.5473],
    [0.3953,0.7459,0.5244],[0.442,0.7481,0.5033],[0.4871,0.7491,0.484],
    [0.53,0.7491,0.4661],[0.5709,0.7485,0.4494],[0.6099,0.7473,0.4337],
    [0.6473,0.7456,0.4188],[0.6834,0.7435,0.4044],[0.7184,0.7411,0.3905],
    [0.7525,0.7384,0.3768],[0.7858,0.7356,0.3633],[0.8185,0.7327,0.3498],
    [0.8507,0.7299,0.336],[0.8824,0.7274,0.3217],[0.9139,0.7258,0.3063],
    [0.945,0.7261,0.2886],[0.9739,0.7314,0.2666],[0.9938,0.7455,0.2403],
    [0.999,0.7653,0.2164],[0.9955,0.7861,0.1967],[0.988,0.8066,0.1794],
    [0.9789,0.8271,0.1633],[0.9697,0.8481,0.1475],[0.9626,0.8705,0.1309],
    [0.9589,0.8949,0.1132],[0.9598,0.9218,0.0948],[0.9661,0.9514,0.0755],
    [0.9763,0.9831,0.0538]
]
parula_cmap = mcolors.LinearSegmentedColormap.from_list('parula', _PARULA)

# ═══════════════════════════════════════════════════════════════════════
# FAST ENGINE  (unchanged from Streamlit version)
# ═══════════════════════════════════════════════════════════════════════

def _process_single_band(f1, x, fs, voices_per_octave=32, order=4):
    bandedge = 1.0
    filband1 = max(f1 - bandedge, 1.0)
    filband2 = f1 + bandedge + 1.0
    nyq = 0.5 * fs
    if filband1 >= filband2 or filband2 >= nyq:
        return None
    scipy_order = max(1, order // 2)
    b, a = butter(scipy_order, [filband1 / nyq, filband2 / nyq], btype='band')
    Tx, _, _, _ = ssq_cwt(filtfilt(b, a, x), fs=fs,
                           nv=voices_per_octave, wavelet=('morlet', {'mu': 6}))
    return Tx


def _fast_aggregate(x, fs, freq_steps, voices_per_octave=32):
    """Average filtered WSST transforms. Returns (W_agg, n_valid)."""
    W_agg, n_valid = None, 0
    for f in freq_steps:
        Tx = _process_single_band(f, x, fs, voices_per_octave)
        if Tx is not None:
            if W_agg is None:
                W_agg = np.zeros_like(Tx, dtype=complex)
            W_agg += Tx
            n_valid += 1
    if W_agg is not None and n_valid > 1:
        W_agg /= n_valid
    return W_agg


def _run_fast(sig, fs, f_min, f_max, df_step=1.0, voices_per_octave=32):
    """FAST pipeline -> (Tx_fast, Tx_orig, ssq_freqs)."""
    freq_steps = np.arange(f_min, f_max, df_step)
    Tx_orig, _, ssq_freqs, _ = ssq_cwt(
        sig, fs=fs, nv=voices_per_octave, wavelet=('morlet', {'mu': 6}))
    W_agg = _fast_aggregate(sig, fs, freq_steps, voices_per_octave)
    mean_power = np.mean(np.abs(Tx_orig) ** 2)
    mask = (np.abs(Tx_orig) ** 2 > 0.9 * mean_power).astype(float)
    if W_agg is None:
        W_agg = np.zeros_like(Tx_orig, dtype=complex)
    return W_agg * mask, Tx_orig, ssq_freqs


# ═══════════════════════════════════════════════════════════════════════
# FILE LOADER
# ═══════════════════════════════════════════════════════════════════════

def _load_file(uploaded_file):
    """Returns (data_df, detected_fs). Handles .mat and CSV/TXT via bytes."""
    from io import BytesIO
    detected_fs = None
    data_df = None
    name = uploaded_file.name
    content = uploaded_file.content.read()  # raw bytes

    if name.endswith('.mat'):
        mat = scipy.io.loadmat(BytesIO(content))
        for k in mat.keys():
            if re.search(r'^fs$|samp|rate', k, re.I):
                try:
                    detected_fs = float(np.array(mat[k]).ravel()[0])
                except Exception:
                    pass
        valid = {k: mat[k].flatten() for k in mat
                 if not k.startswith('__')
                 and mat[k].ndim <= 2
                 and np.issubdtype(mat[k].dtype, np.number)}
        if valid:
            ml = min(len(v) for v in valid.values())
            data_df = pd.DataFrame({k: v[:ml] for k, v in valid.items()})
    else:
        raw = content.decode('utf-8', errors='replace')
        for line in raw.splitlines()[:20]:
            m = re.search(
                r'(?:fs|freq|sample.?rate|sampling.?rate)[^\d]*(\d+(?:\.\d+)?)',
                line, re.I)
            if m:
                detected_fs = float(m.group(1))
                break
        df = None
        for sep in [',', '\t', ';']:
            # Auto-detect skip rows for files with metadata headers
            for skip in range(6):
                try:
                    df_try = pd.read_csv(BytesIO(content), sep=sep,
                                         engine='python', comment='#',
                                         skiprows=skip, on_bad_lines='skip')
                    nc = df_try.select_dtypes(include=[np.number]).columns
                    if len(nc) >= 1:
                        df = df_try[nc]
                        break
                except Exception:
                    pass
            if df is not None:
                break
        if df is not None:
            df.columns = [c.replace('"', '').strip() for c in df.columns]
            data_df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

    return data_df, detected_fs


def _muscle_columns(data_df):
    excl = re.compile(r'time|marker|trigger|sync|ref|event|frame|sample', re.I)
    return [c for c in data_df.columns if not excl.search(c)]


# ═══════════════════════════════════════════════════════════════════════
# FATIGUE CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════

def _classify_fatigue(Tx_fast, ssq_freqs, green_thresh=18.0, amber_thresh=12.0,
                     f_min=1.0, f_max=35.0):
    band = (ssq_freqs >= f_min) & (ssq_freqs <= f_max)
    freqs_b = ssq_freqs[band]
    power_b = np.abs(Tx_fast[band, :]) ** 2
    total_p = power_b.sum(axis=0)
    valid = total_p > 0
    if not valid.any():
        return np.nan, 'grey', 'No signal energy in band'
    wmf = (power_b[:, valid] * freqs_b[:, None]).sum(axis=0) / total_p[valid]
    mif = float(np.median(wmf))
    if mif >= green_thresh:
        return mif, 'green', f'MIF = {mif:.1f} Hz'
    elif mif >= amber_thresh:
        return mif, 'amber', f'MIF = {mif:.1f} Hz'
    else:
        return mif, 'red', f'MIF = {mif:.1f} Hz'


def _traffic_light_html(label, status, detail):
    cfg = {
        'green': ('tl-off','tl-off','tl-green-on', 'fl-green','Not fatigued'),
        'amber': ('tl-off','tl-amber-on','tl-off',  'fl-amber','Some fatigue'),
        'red':   ('tl-red-on','tl-off','tl-off',    'fl-red',  'Fatigued'),
        'grey':  ('tl-off','tl-off','tl-off',        'fl-grey', 'No data'),
    }
    r, a, g, lc, lt = cfg.get(status, cfg['grey'])
    return f"""
    <div class="muscle-card">
        <div class="muscle-name" title="{label}">{label}</div>
        <div class="tl-housing">
            <div class="tl-bulb {r}"></div>
            <div class="tl-bulb {a}"></div>
            <div class="tl-bulb {g}"></div>
        </div>
        <div class="fatigue-label {lc}">{lt}</div>
        <div class="metric-val">{detail}</div>
    </div>"""


# ═══════════════════════════════════════════════════════════════════════
# CPU-BOUND WORKER  (module-level — required by NiceGUI run.cpu_bound)
# ═══════════════════════════════════════════════════════════════════════

def _compute_one_muscle(sig_bytes, fs, label):
    """Compute FAST for one muscle. Pickleable for run.cpu_bound."""
    sig = np.frombuffer(sig_bytes, dtype=np.float64)
    try:
        Tx_fast, _, ssq_freqs = _run_fast(sig, fs, 1.0, 35.0, 1.0)
        mif, status, detail = _classify_fatigue(Tx_fast, ssq_freqs)
        return label, Tx_fast, ssq_freqs, mif, status, detail
    except Exception as e:
        return label, None, None, np.nan, 'grey', str(e)[:60]


# ═══════════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════════

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500&display=swap');
html, body { font-family: 'DM Sans', sans-serif; }

.page-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.7rem 0 1rem; border-bottom: 1.5px solid #e2e8f0;
    margin-bottom: 1.5rem;
}
.page-header-title {
    font-family: 'DM Serif Display', serif; font-size: 1.5rem; color: #0f172a;
}
.page-header-title span { color: #2563eb; font-style: italic; }

.fatigue-grid {
    display: flex; flex-wrap: wrap; gap: 1.1rem; margin: 1.4rem 0;
}
.muscle-card {
    background: #fff; border: 1.5px solid #e2e8f0;
    border-radius: 14px; padding: 1.3rem 1.6rem;
    text-align: center; flex: 1 1 160px; max-width: 210px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.muscle-name {
    font-size: 0.78rem; font-weight: 500; color: #475569;
    text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 0.8rem;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.tl-housing {
    background: #1e293b; border-radius: 40px; width: 44px; height: 120px;
    margin: 0 auto 0.8rem; display: flex; flex-direction: column;
    align-items: center; justify-content: space-around; padding: 7px 0;
}
.tl-bulb { width: 26px; height: 26px; border-radius: 50%; }
.tl-red-on    { background: #ef4444; box-shadow: 0 0 10px #ef4444aa; }
.tl-amber-on  { background: #f59e0b; box-shadow: 0 0 10px #f59e0baa; }
.tl-green-on  { background: #22c55e; box-shadow: 0 0 10px #22c55eaa; }
.tl-off       { background: #334155; }
.fatigue-label { font-size: 0.83rem; font-weight: 500; margin-top: 0.15rem; }
.fl-green { color: #16a34a; }
.fl-amber { color: #d97706; }
.fl-red   { color: #dc2626; }
.fl-grey  { color: #94a3b8; }
.metric-val {
    font-size: 0.72rem; color: #94a3b8; margin-top: 0.25rem;
    font-variant-numeric: tabular-nums;
}
.legend { font-size: 0.8rem; color: #64748b; }
</style>
"""


# ═══════════════════════════════════════════════════════════════════════
# MAIN PAGE
# ═══════════════════════════════════════════════════════════════════════

@ui.page('/')
def main_page():
    ui.add_head_html(GLOBAL_CSS)

    app = dict(data_df=None, fs=1000, muscle_cols=[], results={}, crop=(0, 30))

    # ── HEADER ──
    ui.html("""
    <div class="page-header">
        <div class="page-header-title">FA<span>ST</span></div>
        <div style="font-size:0.82rem;color:#94a3b8;letter-spacing:0.06em;
                    text-transform:uppercase">Muscle Fatigue Check</div>
    </div>""", sanitize=False)

    # ── DYNAMIC SECTIONS ──
    upload_zone = ui.column().classes('w-full')
    after_upload = ui.column().classes('w-full')
    results_zone = ui.column().classes('w-full')

    # ── UPLOAD ──
    async def on_upload(e):
        upload_zone.clear()
        with upload_zone:
            ui.spinner(size='lg')
            ui.label('Reading your file…')

        data = await e.file.read()
        wrapper = type('W', (), {
            'name': e.file.name,
            'content': type('C', (), {'read': lambda self=None: data})()
        })()

        try:
            df, det_fs = _load_file(wrapper)
        except Exception as ex:
            upload_zone.clear()
            with upload_zone:
                ui.label(f'Error: {ex}').classes('text-red-500')
            return

        if df is None or df.empty:
            upload_zone.clear()
            with upload_zone:
                ui.label('No numeric data found').classes('text-red-500')
            return

        app['data_df'] = df
        app['fs'] = int(det_fs) if det_fs else 1000
        app['muscle_cols'] = _muscle_columns(df)
        app['results'] = {}
        app['crop'] = (0.0, min(30.0, len(df) / app['fs']))

        _build_controls(app, upload_zone, after_upload, results_zone)

    upload_zone.clear()
    with upload_zone:
        ui.upload(
            label='Drop your sEMG recording here',
            on_upload=on_upload,
            auto_upload=True,
        ).classes('w-full').props('accept=.csv,.txt,.mat')


def _build_controls(app, upload_zone, after_upload, results_zone):
    cols = app['muscle_cols']
    fs = app['fs']
    ref_len = len(app['data_df'][cols[0]].values)
    total_dur = ref_len / fs
    crop = app['crop']

    upload_zone.clear()
    with upload_zone:
        ui.label(f'Loaded {len(cols)} muscles · {fs} Hz · {total_dur:.0f}s')
        ui.label(', '.join(cols[:4]) + (f' +{len(cols)-4}' if len(cols) > 4 else ''))

    # ── TIME WINDOW ──
    after_upload.clear()
    results_zone.clear()

    with after_upload:
        ui.label('Select time window:').classes('font-medium mt-2')
        slider = ui.range_slider(
            min=0, max=total_dur,
            value={'min': crop[0], 'max': crop[1]},
            step=0.5
        ).classes('w-full')

        status = ui.label().classes('text-sm text-gray-500')

        def _update_status():
            v = slider.value
            app['crop'] = (v['min'], v['max'])
            status.set_text(f'{v["min"]:.0f}s – {v["max"]:.0f}s  ({v["max"] - v["min"]:.0f}s)')

        slider.on('change', _update_status)
        _update_status()

        # ── BIG RUN BUTTON ──
        async def _on_run():
            results_zone.clear()
            s0 = max(int(app['crop'][0] * fs), 0)
            s1 = min(int(app['crop'][1] * fs), ref_len)
            if s0 >= s1:
                s0, s1 = 0, min(5000, ref_len)
            app['s0'] = s0
            app['s1'] = s1
            app['crop_used'] = app['crop']

            # Show progress area
            with results_zone:
                progress = ui.linear_progress(0).classes('w-full')
                msg = ui.label('Starting…').classes('text-sm text-gray-600')

            results = {}
            n = len(cols)

            for i, col in enumerate(cols):
                sig = app['data_df'][col].values[s0:s1].astype(float)
                if np.isnan(sig).any() or np.isinf(sig).any():
                    results[col] = dict(Tx=None, ssq=None, mif=np.nan, status='grey',
                                        detail='Bad data', label=col)
                elif np.std(sig) < 1e-10:
                    results[col] = dict(Tx=None, ssq=None, mif=np.nan, status='grey',
                                        detail='Flat', label=col)
                else:
                    msg.set_text(f'Analysing {col} ({i+1}/{n})…')
                    await asyncio.sleep(0)  # flush UI
                    try:
                        Tx_fast, _, ssq_freqs = await run.cpu_bound(
                            _run_fast, sig, fs, 1.0, 35.0, 1.0, 32)
                        mif, st, detail = _classify_fatigue(Tx_fast, ssq_freqs)
                        results[col] = dict(Tx=Tx_fast, ssq=ssq_freqs, mif=mif,
                                            status=st, detail=detail, label=col)
                    except Exception as e:
                        results[col] = dict(Tx=None, ssq=None, mif=np.nan, status='grey',
                                            detail=str(e)[:60], label=col)
                progress.set_value((i + 1) / n)
                await asyncio.sleep(0)  # flush UI

            app['results'] = results
            results_zone.clear()
            _show_results(app, results_zone)

        ui.button('Check My Muscles', on_click=_on_run)\
            .props('color=primary size=xl').classes('w-full mt-4')


def _show_results(app, container):
    results = app['results']
    with container:
        ui.markdown('## Results')
        html = '<div class="fatigue-grid">'
        for col, r in results.items():
            html += _traffic_light_html(r['label'], r['status'], r['detail'])
        html += '</div>'
        ui.html(html, sanitize=False)

        ui.html("""
        <div class="legend" style="display:flex;gap:1.5rem;flex-wrap:wrap;margin-top:0.5rem;
                    font-size:0.8rem;color:#64748b">
            <span>🟢 Green = No fatigue</span>
            <span>🟡 Amber = Some fatigue</span>
            <span>🔴 Red   = Fatigued</span>
            <span>⚫ Grey  = No data</span>
        </div>""", sanitize=False)

        # Spectrograms
        with ui.expansion('View spectrograms', value=False):
            t_axis = np.arange(app['s1'] - app['s0']) / app['fs'] + app['crop_used'][0]
            for col, r in results.items():
                if r['Tx'] is None:
                    ui.label(f'{r["label"]}: skipped').classes('text-xs text-gray-400')
                    continue
                band = (r['ssq'] >= 1.0) & (r['ssq'] <= 35.0)
                fp = r['ssq'][band]
                e = np.abs(r['Tx'][band, :]) ** 2
                e /= (e.max() + 1e-12)
                fig, ax = plt.subplots(figsize=(10, 2.2))
                ax.pcolormesh(t_axis, fp, e, norm=mcolors.PowerNorm(gamma=0.3),
                              cmap=parula_cmap, shading='auto')
                ax.set_ylabel('Hz', fontsize=9)
                ax.set_ylim(1, 35)
                ax.set_title(r['label'], fontsize=10, fontweight='bold')
                plt.tight_layout()
                ui.pyplot(fig, close_figure=True)

        # Download CSV
        rows = [{'Muscle': r['label'],
                 'MIF (Hz)': f"{r['mif']:.2f}" if not np.isnan(r['mif']) else 'N/A',
                 'Status': r['status'].capitalize()}
                for r in results.values()]
        buf = io.StringIO()
        pd.DataFrame(rows).to_csv(buf, index=False)
        ui.download(buf.getvalue().encode(), 'fatigue_report.csv', 'text/csv')\
            .props('label="Download Report (CSV)" flat')


# ═══════════════════════════════════════════════════════════════════════
# ENTRY
# ═══════════════════════════════════════════════════════════════════════

if __name__ in {"__main__", "__mp_main__"}:
    ui.run(host='0.0.0.0', port=8501, title='FAST — Muscle Fatigue Check', favicon='💪')
