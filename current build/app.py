# FAST — Filter & Aggregate Synchrosqueezed Transform
# Version: 2025-07-10  ·  git push → auto-deploy
import streamlit as st
import numpy as np
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import io
import time
import re
from scipy.signal import butter, filtfilt

from ssqueezepy import ssq_cwt

# ──────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLE
# ──────────────────────────────────────────────────────────────────────────────

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
#MainMenu, footer { visibility: hidden; }

/* ── WHITE BACKGROUND ── */
.stApp, [data-testid="stAppViewContainer"], .main, section[data-testid="stSidebar"] {
    background-color: #ffffff;
}
.stApp > header { background-color: #ffffff; }

/* ── SPLASH ── */
.splash-wrap {
    min-height: 80vh; display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    text-align: center; padding: 2rem;
}
.splash-logo {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(3.5rem, 8vw, 5.5rem);
    letter-spacing: -0.02em; color: #0f172a;
    line-height: 1; margin-bottom: 0.2rem;
}
.splash-logo span { color: #2563eb; font-style: italic; }
.splash-sub {
    font-size: 0.95rem; color: #64748b; font-weight: 300;
    letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 2.8rem;
}
.splash-divider {
    width: 40px; height: 2px; background: #2563eb;
    margin: 0 auto 2.4rem;
}
.mode-card {
    background: #fff; border: 1.5px solid #e2e8f0;
    border-radius: 16px; padding: 2rem 2.2rem; text-align: left;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    margin-bottom: 0.6rem;
}
.mode-card-icon { font-size: 1.8rem; margin-bottom: 0.6rem; }
.mode-card-title {
    font-family: 'DM Serif Display', serif; font-size: 1.35rem;
    color: #0f172a; margin-bottom: 0.4rem;
}
.mode-card-desc { font-size: 0.86rem; color: #64748b; line-height: 1.55; }

/* ── HEADER ── */
.page-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.7rem 0 1rem; border-bottom: 1.5px solid #e2e8f0;
    margin-bottom: 1.5rem;
}
.page-header-title {
    font-family: 'DM Serif Display', serif; font-size: 1.5rem; color: #0f172a;
}
.page-header-title span { color: #2563eb; font-style: italic; }

/* ── TRAFFIC LIGHTS ── */
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

/* ── DECISION BOX ── */
.decision-box {
    background: #f8fafc; border: 1.5px dashed #cbd5e1;
    border-radius: 12px; padding: 1.3rem 1.6rem; margin: 1rem 0 1.4rem;
}
.decision-box-title {
    font-family: 'DM Serif Display', serif; font-size: 1.05rem;
    color: #0f172a; margin-bottom: 0.6rem;
    display: flex; align-items: center; gap: 0.5rem;
}
.pending-badge {
    font-size: 0.67rem; background: #fef3c7; color: #92400e;
    border: 1px solid #fcd34d; border-radius: 4px;
    padding: 0.1rem 0.4rem; font-family: 'DM Sans', sans-serif;
    font-weight: 500; letter-spacing: 0.05em; text-transform: uppercase;
}
</style>
"""

# ──────────────────────────────────────────────────────────────────────────────
# PARULA COLORMAP
# ──────────────────────────────────────────────────────────────────────────────

_parula = [
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
parula_cmap = mcolors.LinearSegmentedColormap.from_list('parula', _parula)


# ──────────────────────────────────────────────────────────────────────────────
# FAST ENGINE  (shared by both modes)
# ──────────────────────────────────────────────────────────────────────────────

def process_single_band(f1, x, fs, voices_per_octave=32, order=4):
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


def fast_sequential_aggregate(x, fs, freq_steps, voices_per_octave=32,
                               progress_callback=None):
    """Average filtered WSST transforms over valid bands (paper §III-B)."""
    W_agg, n_valid = None, 0
    n_bands = len(freq_steps)
    for i, f in enumerate(freq_steps):
        Tx = process_single_band(f, x, fs, voices_per_octave)
        if Tx is not None:
            if W_agg is None:
                W_agg = np.zeros_like(Tx, dtype=complex)
            W_agg += Tx
            n_valid += 1
        if progress_callback:
            progress_callback((i + 1) / n_bands,
                              f"Band {i+1}/{n_bands}  ({f:.1f} Hz)")
    if W_agg is not None and n_valid > 1:
        W_agg /= n_valid
    return W_agg


def run_fast(sig, fs, f_min, f_max, df_step=1.0,
             voices_per_octave=32, progress_callback=None):
    """Full FAST pipeline → (Tx_fast, Tx_orig, ssq_freqs)."""
    freq_steps = np.arange(f_min, f_max, df_step)
    Tx_orig, _, ssq_freqs, _ = ssq_cwt(
        sig, fs=fs, nv=voices_per_octave, wavelet=('morlet', {'mu': 6}))
    W_agg = fast_sequential_aggregate(
        sig, fs, freq_steps, voices_per_octave=voices_per_octave,
        progress_callback=progress_callback)
    mean_power = np.mean(np.abs(Tx_orig) ** 2)
    mask = (np.abs(Tx_orig) ** 2 > 0.9 * mean_power).astype(float)
    if W_agg is None:
        W_agg = np.zeros_like(Tx_orig, dtype=complex)
    return W_agg * mask, Tx_orig, ssq_freqs


# ──────────────────────────────────────────────────────────────────────────────
# FILE LOADER
# ──────────────────────────────────────────────────────────────────────────────

def load_file(uploaded_file):
    """Returns (data_df, detected_fs). Auto-detects fs and muscle names."""
    detected_fs = None
    data_df = None

    if uploaded_file.name.endswith('.mat'):
        mat = scipy.io.loadmat(uploaded_file)
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
        raw = uploaded_file.read().decode('utf-8', errors='replace')
        for line in raw.splitlines()[:20]:
            m = re.search(
                r'(?:fs|sample.?rate|sampling.?rate)[^\d]*(\d+(?:\.\d+)?)',
                line, re.I)
            if m:
                detected_fs = float(m.group(1))
                break
        uploaded_file.seek(0)
        df = None
        for sep in [',', '\t', ';']:
            try:
                uploaded_file.seek(0)
                df_try = pd.read_csv(uploaded_file, sep=sep, engine='python',
                                     comment='#')
                nc = df_try.select_dtypes(include=[np.number]).columns
                if len(nc) >= 1:
                    df = df_try[nc]
                    break
            except Exception:
                pass
        if df is not None:
            df.columns = [c.replace('"', '').strip() for c in df.columns]
            data_df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

    return data_df, detected_fs


def muscle_columns(data_df):
    excl = re.compile(r'time|marker|trigger|sync|ref|event|frame|sample', re.I)
    return [c for c in data_df.columns if not excl.search(c)]


# ──────────────────────────────────────────────────────────────────────────────
# FATIGUE METRIC  (placeholder — awaiting Prof Samit's criteria)
# ──────────────────────────────────────────────────────────────────────────────

def classify_fatigue(Tx_fast, ssq_freqs, green_thresh, amber_thresh,
                     f_min=1.0, f_max=35.0):
    """
    Compute power-weighted mean instantaneous frequency (MIF) within the
    analysis band, then classify using user-supplied thresholds.

    ⚠️  PROVISIONAL — thresholds and metric choice pending confirmation
    from Prof Samit Chakrabarty (University of Leeds).

    Returns
    -------
    mif    : float  — median power-weighted MIF across time frames (Hz)
    status : str    — 'green' | 'amber' | 'red' | 'grey'
    detail : str    — short human-readable explanation
    """
    band   = (ssq_freqs >= f_min) & (ssq_freqs <= f_max)
    freqs_b = ssq_freqs[band]
    power_b = np.abs(Tx_fast[band, :]) ** 2
    total_p = power_b.sum(axis=0)
    valid   = total_p > 0

    if not valid.any():
        return np.nan, 'grey', 'No signal energy in band'

    wmf = (power_b[:, valid] * freqs_b[:, None]).sum(axis=0) / total_p[valid]
    mif = float(np.median(wmf))

    if   mif >= green_thresh:
        return mif, 'green', f'MIF = {mif:.1f} Hz'
    elif mif >= amber_thresh:
        return mif, 'amber', f'MIF = {mif:.1f} Hz'
    else:
        return mif, 'red',   f'MIF = {mif:.1f} Hz'


def traffic_light_html(label, status, detail):
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


# ──────────────────────────────────────────────────────────────────────────────
# MAIN PAGE  (athlete-focused — single flow)
# ──────────────────────────────────────────────────────────────────────────────

def page_main():
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

    # ── HEADER ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="page-header">
        <div class="page-header-title">FA<span>ST</span></div>
        <div style="font-size:0.82rem;color:#94a3b8;letter-spacing:0.06em;
                    text-transform:uppercase">Muscle Fatigue Check</div>
    </div>""", unsafe_allow_html=True)

    # ── UPLOAD ────────────────────────────────────────────────────────────────
    uf = st.file_uploader(
        "Drop your sEMG recording here",
        type=["csv", "txt", "mat"],
        key="main_uf", label_visibility="collapsed")

    if not uf:
        st.markdown("""
        <div style="text-align:center;padding:3rem 1rem;color:#94a3b8">
            <div style="font-size:3rem;margin-bottom:1rem">📂</div>
            <p style="font-size:1.05rem;margin-bottom:0.3rem">
                Upload a CSV, TXT, or MAT file
            </p>
            <p style="font-size:0.82rem">
                Your muscle names are detected automatically.<br>
                Nothing else to configure — just drop and check.
            </p>
        </div>""", unsafe_allow_html=True)
        _coach_tools_expander(None, None)
        return

    # ── LOAD DATA ─────────────────────────────────────────────────────────────
    with st.spinner("Reading your file…"):
        try:
            data_df, det_fs = load_file(uf)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            return

    if data_df is None or data_df.empty:
        st.error("No numeric data found. Check the file format.")
        return

    all_cols = list(data_df.columns)
    muscle_cols = muscle_columns(data_df)
    default_fs = int(det_fs) if det_fs else 1000

    # ── QUICK SUMMARY ─────────────────────────────────────────────────────────
    ref_len = len(data_df[muscle_cols[0]].values) if muscle_cols else len(data_df)
    total_dur = ref_len / default_fs

    c_s1, c_s2, c_s3, c_s4 = st.columns(4)
    c_s1.metric("Muscles found", len(muscle_cols))
    c_s2.metric("Sampling rate", f"{default_fs} Hz")
    c_s3.metric("Duration", f"{total_dur:.1f} s")
    c_s4.metric("Channels", ", ".join(muscle_cols[:3]) +
                (f" +{len(muscle_cols)-3}" if len(muscle_cols) > 3 else ""))

    # ── TIME WINDOW ───────────────────────────────────────────────────────────
    st.markdown("#### Which part of the recording?")
    crop = st.slider(
        "Time range", 0.0, float(total_dur),
        (0.0, min(float(total_dur), 30.0)),
        key="main_crop",
        help="Choose the section to analyse. Default is the first 30 seconds.")
    s0 = max(int(crop[0] * default_fs), 0)
    s1 = min(int(crop[1] * default_fs), ref_len)
    if s0 >= s1:
        s0, s1 = 0, min(5000, ref_len)
    window_dur = (s1 - s0) / default_fs

    st.caption(f"Analysing {crop[0]:.0f}s – {crop[1]:.0f}s  "
               f"({window_dur:.1f}s of data)")

    # ── RUN ───────────────────────────────────────────────────────────────────
    if not st.button("🏃 Check My Muscles", type="primary",
                     use_container_width=True, key="main_run"):
        _coach_tools_expander(data_df, default_fs)
        return

    results = {}
    overall = st.progress(0, text="Starting analysis…")

    for m_idx, col in enumerate(muscle_cols):
        sig = data_df[col].values[s0:s1].astype(float)

        if np.isnan(sig).any() or np.isinf(sig).any():
            results[col] = dict(Tx=None, ssq=None, mif=np.nan,
                                status='grey', detail='NaN/Inf in data',
                                label=col)
            continue
        if np.std(sig) < 1e-10:
            results[col] = dict(Tx=None, ssq=None, mif=np.nan,
                                status='grey', detail='Flat signal',
                                label=col)
            continue

        bprog = st.progress(0, text=f"Processing {col}…")

        def _cb(frac, txt, _bp=bprog, _lb=col):
            _bp.progress(frac, text=f"{_lb}: {txt}")

        try:
            Tx_fast, _, ssq_freqs = run_fast(
                sig, default_fs, f_min=1.0, f_max=35.0, df_step=1.0,
                progress_callback=_cb)

            mif, status, detail = classify_fatigue(
                Tx_fast, ssq_freqs, 18.0, 12.0)

            results[col] = dict(Tx=Tx_fast, ssq=ssq_freqs,
                                mif=mif, status=status,
                                detail=detail, label=col)
        except Exception as e:
            results[col] = dict(Tx=None, ssq=None, mif=np.nan,
                                status='grey', detail=str(e)[:60],
                                label=col)
        bprog.empty()
        overall.progress((m_idx + 1) / len(muscle_cols),
                         text=f"Completed: {col}")

    overall.empty()

    # ── RESULTS ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## Your Muscle Status")

    html = '<div class="fatigue-grid">'
    for col, r in results.items():
        html += traffic_light_html(r['label'], r['status'], r['detail'])
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

    # Legend
    st.markdown("""
    <div style="display:flex;gap:1.5rem;flex-wrap:wrap;margin-top:0.5rem;
                font-size:0.8rem;color:#64748b">
        <span>🟢 Green = No fatigue detected</span>
        <span>🟡 Amber = Some fatigue</span>
        <span>🔴 Red   = Fatigued</span>
        <span>⚫ Grey  = Could not analyse</span>
    </div>""", unsafe_allow_html=True)

    # Spectrograms
    t_axis = np.arange(s1 - s0) / default_fs + crop[0]
    with st.expander("📊 See detailed spectrograms"):
        for col, r in results.items():
            if r['Tx'] is None:
                st.caption(f"{r['label']}: skipped — {r['detail']}")
                continue
            band = (r['ssq'] >= 1.0) & (r['ssq'] <= 35.0)
            fp = r['ssq'][band]
            e = np.abs(r['Tx'][band, :]) ** 2
            e /= (e.max() + 1e-12)
            fig, ax = plt.subplots(figsize=(10, 2.6))
            ax.pcolormesh(t_axis, fp, e,
                          norm=mcolors.PowerNorm(gamma=0.3),
                          cmap=parula_cmap, shading='auto')
            ax.set_ylabel('Frequency (Hz)', fontsize=9)
            ax.set_xlabel('Time (s)', fontsize=9)
            ax.set_ylim(1, 35)
            ax.set_title(f"FAST — {r['label']}", fontsize=10,
                         fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    # Download
    rows = [{'Muscle': r['label'],
             'MIF (Hz)': f"{r['mif']:.2f}" if not np.isnan(r['mif']) else 'N/A',
             'Status': r['status'].capitalize(),
             'Detail': r['detail']}
            for r in results.values()]
    buf = io.StringIO()
    pd.DataFrame(rows).to_csv(buf, index=False)
    st.download_button("⬇ Download Report (CSV)",
                       buf.getvalue().encode(),
                       "fatigue_report.csv", "text/csv")

    # ── COACH TOOLS ───────────────────────────────────────────────────────────
    _coach_tools_expander(data_df, default_fs)


# ──────────────────────────────────────────────────────────────────────────────
# COACH TOOLS  (collapsed expander — researcher / clinician access)
# ──────────────────────────────────────────────────────────────────────────────

def _coach_tools_expander(data_df, samp_rate):
    with st.expander("🔬 Coach Tools — Advanced Analysis", expanded=False):
        st.caption(
            "For coaches, clinicians, and researchers. "
            "Tune FAST parameters, test with synthetic signals, "
            "and compare WSST vs FAST spectrograms.")

        st.markdown("---")
        src = st.radio("Signal source", ["Use uploaded data", "Synthetic test"],
                       key="ct_src", horizontal=True)

        if src == "Synthetic test":
            dur = st.slider("Duration (s)", 1, 5, 2, key="ct_dur")
            sr = st.number_input("Sampling rate (Hz)", value=1000,
                                 min_value=100, key="ct_fs")
            t_s = np.linspace(0, dur, int(dur * sr))
            sig = (0.5 * np.sin(2 * np.pi * 20 * t_s) +
                   0.8 * np.sin(2 * np.pi * 25 * t_s) +
                   0.6 * np.sin(2 * np.pi * 60 * t_s))
            final_signals = {"Synthetic": sig}
            t_arr = t_s
            sr_used = sr
        else:
            if data_df is None:
                st.info("Upload a file above first.")
                return
            cols = muscle_columns(data_df)
            sel = st.multiselect("Channel", cols,
                                 default=cols[:1] if cols else [],
                                 key="ct_chan")
            if not sel:
                return
            ref = data_df[sel[0]].values
            tdur = len(ref) / samp_rate
            cr = st.slider("Time range (s)", 0.0, float(tdur),
                           (0.0, min(float(tdur), 5.0)), key="ct_crop")
            s0 = max(int(cr[0] * samp_rate), 0)
            s1 = min(int(cr[1] * samp_rate), len(ref))
            if s0 >= s1:
                s0, s1 = 0, min(1000, len(ref))
            t_arr = np.arange(s1 - s0) / samp_rate + cr[0]
            final_signals = {c: data_df[c].values[s0:s1] for c in sel}
            sr_used = samp_rate

        c1, c2, c3 = st.columns(3)
        with c1:
            f_min = st.number_input("Min freq (Hz)", value=8, min_value=1,
                                    key="ct_fmin")
        with c2:
            f_max = st.number_input("Max freq (Hz)", value=100, min_value=2,
                                    key="ct_fmax")
        with c3:
            df_step = st.number_input("Step (Hz)", value=1.0, min_value=0.5,
                                      step=0.5, key="ct_step")
        st.caption("Filter band: f−1 to f+2 Hz  ·  Powered by ssqueezepy")

        if not st.button("🚀 Run Advanced Analysis", key="ct_run"):
            return

        freq_steps = np.arange(f_min, f_max, df_step)
        n_plots = len(final_signals) * 2
        plt.rcParams.update({"font.family": "sans-serif",
                             "savefig.bbox": "tight"})
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 5 * n_plots),
                                 sharex=True, constrained_layout=True)
        if n_plots == 1:
            axes = [axes]

        t0 = time.time()
        for idx, (col, sig) in enumerate(final_signals.items()):
            try:
                if np.isnan(sig).any() or np.isinf(sig).any():
                    st.warning(f"Skipping '{col}': NaN/Inf")
                    continue
                if np.std(sig) < 1e-10:
                    st.warning(f"Skipping '{col}': flatline")
                    continue

                ph = st.empty()
                ph.text(f"WSST for {col}…")
                Tx_orig, _, ssq, _ = ssq_cwt(
                    sig, fs=sr_used, nv=32,
                    wavelet=('morlet', {'mu': 6}))
                ph.empty()

                pb = st.progress(0, text=f"{col}: 0/{len(freq_steps)} bands")

                def _cb_adv(frac, txt, _pb=pb, _c=col):
                    _pb.progress(frac, text=f"{_c}: {txt}")

                W_agg = fast_sequential_aggregate(
                    sig, sr_used, freq_steps,
                    voices_per_octave=32, progress_callback=_cb_adv)
                pb.empty()

                mp = np.mean(np.abs(Tx_orig) ** 2)
                mask = (np.abs(Tx_orig) ** 2 > 0.9 * mp).astype(float)
                Tx_f = (W_agg if W_agg is not None
                        else np.zeros_like(Tx_orig, dtype=complex)) * mask
                mf = (ssq >= f_min) & (ssq <= f_max)
                fp = ssq[mf]

                ax0 = axes[idx * 2]
                e0 = np.abs(Tx_orig[mf, :]) ** 2
                e0 /= e0.max() + 1e-12
                ax0.pcolormesh(t_arr, fp, e0,
                               norm=mcolors.PowerNorm(0.3),
                               cmap=parula_cmap, shading='auto')
                ax0.set_ylabel('Frequency (Hz)', fontsize=11)
                ax0.set_title(f"Standard WSST: {col}", fontsize=12,
                              fontweight='bold')
                ax0.set_ylim(f_min, f_max)

                ax1 = axes[idx * 2 + 1]
                e1 = np.abs(Tx_f[mf, :]) ** 2
                e1 /= e1.max() + 1e-12
                ax1.pcolormesh(t_arr, fp, e1,
                               norm=mcolors.PowerNorm(0.3),
                               cmap=parula_cmap, shading='auto')
                ax1.set_ylabel('Frequency (Hz)', fontsize=11)
                ax1.set_title(f"FAST Result: {col}", fontsize=12,
                              fontweight='bold')
                ax1.set_ylim(f_min, f_max)
                if idx * 2 + 1 == n_plots - 1:
                    ax1.set_xlabel('Time (s)', fontsize=11)

            except Exception as e:
                st.error(f"Error — {col}: {e}")
                import traceback
                st.code(traceback.format_exc())

        st.success(f"{len(freq_steps)} bands · {time.time() - t0:.2f}s")
        st.pyplot(fig, use_container_width=True)

        dl1, dl2 = st.columns(2)
        with dl1:
            b = io.BytesIO()
            fig.savefig(b, format="pdf", dpi=300)
            b.seek(0)
            st.download_button("📄 PDF", b, "FAST_advanced.pdf",
                               "application/pdf")
        with dl2:
            b = io.BytesIO()
            fig.savefig(b, format="png", dpi=300)
            b.seek(0)
            st.download_button("🖼️ PNG", b, "FAST_advanced.png",
                               "image/png")


# ──────────────────────────────────────────────────────────────────────────────
# APP ENTRY
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    st.set_page_config(
        page_title="FAST — Muscle Fatigue Check",
        page_icon="💪",
        layout="wide"
    )
    page_main()
