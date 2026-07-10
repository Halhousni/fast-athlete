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

import custom_wsst

# ──────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLE
# ──────────────────────────────────────────────────────────────────────────────

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
#MainMenu, footer { visibility: hidden; }

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
    Tx, _, _, _ = custom_wsst.ssq_cwt(filtfilt(b, a, x), fs=fs,
                                       voices_per_octave=voices_per_octave)
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
    Tx_orig, _, ssq_freqs, _ = custom_wsst.ssq_cwt(
        sig, fs=fs, voices_per_octave=voices_per_octave)
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
# PAGE: SPLASH
# ──────────────────────────────────────────────────────────────────────────────

def page_splash():
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    st.markdown("""
    <div class="splash-wrap">
        <div class="splash-logo">FA<span>ST</span></div>
        <div class="splash-sub">Filter &amp; Aggregate Synchrosqueezed Transform</div>
        <div class="splash-divider"></div>
        <p style="color:#475569;max-width:440px;margin-bottom:2.5rem;
                  font-size:0.93rem;line-height:1.75">
            High-resolution time–frequency analysis for muscle activity signals.<br>
            Based on Chakrabarty <em>et al.</em> (2021).
        </p>
    </div>""", unsafe_allow_html=True)

    _, col_b, col_a, _ = st.columns([1, 1.3, 1.3, 1])

    with col_b:
        st.markdown("""
        <div class="mode-card">
            <div class="mode-card-icon">🟢</div>
            <div class="mode-card-title">Basic</div>
            <div class="mode-card-desc">
                Upload sEMG data and get an instant muscle fatigue
                report — no signal-processing knowledge required.
            </div>
        </div>""", unsafe_allow_html=True)
        if st.button("Open Basic Mode", key="btn_basic",
                     use_container_width=True, type="primary"):
            st.session_state.page = 'basic'
            st.rerun()

    with col_a:
        st.markdown("""
        <div class="mode-card">
            <div class="mode-card-icon">🔬</div>
            <div class="mode-card-title">Advanced</div>
            <div class="mode-card-desc">
                Full control over frequency bands, filter order,
                and WSST parameters for expert analysis.
            </div>
        </div>""", unsafe_allow_html=True)
        if st.button("Open Advanced Mode", key="btn_adv",
                     use_container_width=True):
            st.session_state.page = 'advanced'
            st.rerun()

    binfo = custom_wsst.backend_info()
    tag = (f"⚡ {binfo['device']} · {binfo['backend']} · {binfo['memory_mb']} MB"
           if binfo['backend'] != 'numpy' else "🖥️ CPU (no GPU detected)")
    st.markdown(f"<p style='text-align:center;color:#94a3b8;font-size:0.78rem;"
                f"margin-top:1.2rem'>{tag}</p>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# PAGE: BASIC
# ──────────────────────────────────────────────────────────────────────────────

def page_basic():
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

    st.markdown("""
    <div class="page-header">
        <div class="page-header-title">FA<span>ST</span> — Fatigue Assessment</div>
    </div>""", unsafe_allow_html=True)

    if st.button("← Home", key="back_basic"):
        st.session_state.page = 'splash'
        st.rerun()

    # ── STEP 1: UPLOAD ────────────────────────────────────────────────────────
    st.markdown("### 1 · Upload sEMG data")
    st.caption("CSV, TXT, or MAT file. Column headers become muscle names "
               "automatically.")

    uf = st.file_uploader("Drop file here", type=["csv","txt","mat"],
                          key="basic_uf", label_visibility="collapsed")
    if not uf:
        st.info("👆 Upload an sEMG file to begin.")
        return

    with st.spinner("Reading file…"):
        try:
            data_df, det_fs = load_file(uf)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            return

    if data_df is None or data_df.empty:
        st.error("No numeric data found. Please check the file format.")
        return

    # ── STEP 2: CONFIRM SETTINGS ──────────────────────────────────────────────
    st.markdown("### 2 · Confirm settings")

    all_cols    = list(data_df.columns)
    muscle_cols = muscle_columns(data_df)

    c1, c2 = st.columns(2)
    with c1:
        default_fs = int(det_fs) if det_fs else 1000
        samp_rate  = st.number_input(
            "Sampling rate (Hz)", min_value=50, max_value=10000,
            value=default_fs, step=1,
            help="Auto-detected where possible — please verify.")
        if det_fs:
            st.caption(f"✓ Auto-detected from file: {det_fs:.0f} Hz")

    with c2:
        sel_muscles = st.multiselect(
            "Muscles to analyse", options=all_cols,
            default=(muscle_cols or all_cols)[:6],
            help="Time/Marker columns are filtered out automatically.")

    if not sel_muscles:
        st.warning("Select at least one muscle channel.")
        return

    with st.expander("✏️  Rename muscles (optional)"):
        muscle_labels = {}
        cols_ui = st.columns(min(len(sel_muscles), 3))
        for i, col in enumerate(sel_muscles):
            with cols_ui[i % len(cols_ui)]:
                muscle_labels[col] = st.text_input(
                    f"Label for '{col}'", value=col, key=f"lbl_{col}")

    ref_len  = len(data_df[sel_muscles[0]].values)
    total_dur = ref_len / samp_rate
    with st.expander("⏱  Select time window (optional)", expanded=False):
        crop = st.slider("Time range (s)", 0.0, float(total_dur),
                         (0.0, min(float(total_dur), 30.0)),
                         key="basic_crop")
    s0 = max(int(crop[0] * samp_rate), 0)
    s1 = min(int(crop[1] * samp_rate), ref_len)
    if s0 >= s1:
        s0, s1 = 0, min(5000, ref_len)

    # ── STEP 3: DECISION CRITERIA ─────────────────────────────────────────────
    st.markdown("### 3 · Fatigue classification criteria")
    st.markdown("""
    <div class="decision-box">
        <div class="decision-box-title">
            🔬 Spectral thresholds
            <span class="pending-badge">Pending — Prof Samit</span>
        </div>
        <p style="font-size:0.86rem;color:#475569;margin:0 0 0.4rem">
            These thresholds map the FAST spectral output to a traffic-light
            fatigue status. The values below are <strong>provisional
            placeholders</strong>. Update them here once clinical criteria
            are confirmed by Prof Samit Chakrabarty (University of Leeds).
        </p>
    </div>""", unsafe_allow_html=True)

    dc1, dc2, dc3 = st.columns(3)
    with dc1:
        metric_info = st.selectbox(
            "Spectral metric",
            ["Mean instantaneous frequency (MIF)"],
            help="FAST-derived spectral measure used for classification. "
                 "Additional options can be added once criteria are confirmed.")
    with dc2:
        green_thresh = st.number_input(
            "🟢 Green threshold (Hz)", min_value=1.0, max_value=35.0,
            value=18.0, step=0.5,
            help="MIF at or above this value → Not fatigued")
    with dc3:
        amber_thresh = st.number_input(
            "🟡 Amber threshold (Hz)", min_value=1.0, max_value=35.0,
            value=12.0, step=0.5,
            help="MIF between amber and green → Some fatigue. "
                 "Below amber → Fatigued.")

    if amber_thresh >= green_thresh:
        st.warning("⚠️ Amber threshold must be lower than the Green threshold.")
        return

    st.caption("Analysis band is fixed to 1–35 Hz (sEMG fatigue range).")

    # ── STEP 4: RUN ───────────────────────────────────────────────────────────
    st.markdown("### 4 · Run analysis")
    if not st.button("▶  Analyse muscles", type="primary",
                     use_container_width=True, key="basic_run"):
        return

    results = {}   # col → dict
    overall = st.progress(0, text="Starting…")

    for m_idx, col in enumerate(sel_muscles):
        label = muscle_labels.get(col, col)
        sig   = data_df[col].values[s0:s1].astype(float)

        if np.isnan(sig).any() or np.isinf(sig).any():
            results[col] = dict(Tx=None, ssq=None, mif=np.nan,
                                status='grey', detail='NaN/Inf in data',
                                label=label)
            continue
        if np.std(sig) < 1e-10:
            results[col] = dict(Tx=None, ssq=None, mif=np.nan,
                                status='grey', detail='Flat signal',
                                label=label)
            continue

        bprog = st.progress(0, text=f"Processing {label}…")

        def _cb(frac, txt, _bp=bprog, _lb=label):
            _bp.progress(frac, text=f"{_lb}: {txt}")

        try:
            Tx_fast, _, ssq_freqs = run_fast(
                sig, samp_rate, f_min=1.0, f_max=35.0, df_step=1.0,
                progress_callback=_cb)

            mif, status, detail = classify_fatigue(
                Tx_fast, ssq_freqs, green_thresh, amber_thresh)

            results[col] = dict(Tx=Tx_fast, ssq=ssq_freqs,
                                mif=mif, status=status,
                                detail=detail, label=label)
        except Exception as e:
            results[col] = dict(Tx=None, ssq=None, mif=np.nan,
                                status='grey', detail=str(e)[:60],
                                label=label)
        bprog.empty()
        overall.progress((m_idx + 1) / len(sel_muscles),
                         text=f"Completed: {label}")

    overall.empty()

    # ── RESULTS: TRAFFIC LIGHTS ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## Results")

    html = '<div class="fatigue-grid">'
    for col, r in results.items():
        html += traffic_light_html(r['label'], r['status'], r['detail'])
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

    t_axis = np.arange(s1 - s0) / samp_rate + crop[0]

    with st.expander("📊 View FAST spectrograms"):
        for col, r in results.items():
            if r['Tx'] is None:
                st.caption(f"{r['label']}: skipped — {r['detail']}")
                continue
            band = (r['ssq'] >= 1.0) & (r['ssq'] <= 35.0)
            fp   = r['ssq'][band]
            e    = np.abs(r['Tx'][band, :]) ** 2
            e   /= (e.max() + 1e-12)
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

    # Summary CSV download
    rows = [{'Muscle': r['label'],
             'MIF (Hz)': f"{r['mif']:.2f}" if not np.isnan(r['mif']) else 'N/A',
             'Status': r['status'].capitalize(),
             'Detail': r['detail']}
            for r in results.values()]
    buf = io.StringIO()
    pd.DataFrame(rows).to_csv(buf, index=False)
    st.download_button("⬇  Download summary CSV",
                       buf.getvalue().encode(),
                       "fatigue_summary.csv", "text/csv")


# ──────────────────────────────────────────────────────────────────────────────
# PAGE: ADVANCED
# ──────────────────────────────────────────────────────────────────────────────

def page_advanced():
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    binfo = custom_wsst.backend_info()

    st.markdown("""
    <div class="page-header">
        <div class="page-header-title">FA<span>ST</span> — Advanced Analysis</div>
    </div>""", unsafe_allow_html=True)

    if st.button("← Home", key="back_adv"):
        st.session_state.page = 'splash'
        st.rerun()

    gpu_tag = (f"⚡ {binfo['device']}  ({binfo['backend']}, {binfo['memory_mb']} MB)"
               if binfo['backend'] != 'numpy' else "🖥️ CPU (no GPU detected)")
    st.caption(gpu_tag)

    # ── SIDEBAR ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("1. Data")
        src = st.radio("Source", ["Synthetic", "Upload File"], key="adv_src")
        data_df, samp_rate = None, 1000

        if src == "Synthetic":
            dur       = st.slider("Duration (s)", 1, 5, 2)
            samp_rate = st.number_input("Sampling rate (Hz)",
                                        value=1000, min_value=100)
            t_s = np.linspace(0, dur, int(dur * samp_rate))
            sig = (0.5 * np.sin(2*np.pi*20*t_s) +
                   0.8 * np.sin(2*np.pi*25*t_s) +
                   0.6 * np.sin(2*np.pi*60*t_s))
            data_df = pd.DataFrame({"Synthetic (20,25,60 Hz)": sig})
        else:
            uf = st.file_uploader("Upload", type=["mat","csv","txt"],
                                  key="adv_uf")
            samp_rate = st.number_input("Sampling rate (Hz)",
                                        value=1000, min_value=100)
            if uf:
                try:
                    data_df, det_fs = load_file(uf)
                    if det_fs:
                        st.caption(f"✓ Auto-detected fs: {det_fs:.0f} Hz")
                except Exception as e:
                    st.error(f"Load error: {e}")

        final_signals, t = {}, np.array([])
        if data_df is not None:
            all_cols = list(data_df.columns)
            sel = st.multiselect("Channels", all_cols,
                                 default=(muscle_columns(data_df) or
                                          all_cols)[:1])
            if sel:
                ref  = data_df[sel[0]].values
                tdur = len(ref) / samp_rate
                cr   = st.slider("Time range (s)", 0.0, float(tdur),
                                 (0.0, min(float(tdur), 5.0)))
                s0   = max(int(cr[0]*samp_rate), 0)
                s1   = min(int(cr[1]*samp_rate), len(ref))
                if s0 >= s1: s0, s1 = 0, min(1000, len(ref))
                t    = np.arange(s1-s0)/samp_rate + cr[0]
                final_signals = {c: data_df[c].values[s0:s1] for c in sel}

        st.markdown("---")
        with st.form("adv_form"):
            st.header("2. FAST Settings")
            st.info(f"Engine: {binfo['backend']}")
            c1, c2 = st.columns(2)
            f1 = c1.number_input("Min freq (Hz)", value=8,  min_value=1)
            f2 = c2.number_input("Max freq (Hz)", value=100, min_value=2)
            dfs = st.number_input("Step (Hz)", value=1.0, min_value=0.5,
                                  step=0.5)
            st.caption("Filter: f−1 to f+2 Hz  (MATLAB V4 match)")
            go = st.form_submit_button("🚀 Run FAST", type="primary")

    # ── ANALYSIS ──────────────────────────────────────────────────────────────
    if go and final_signals:
        freq_steps = np.arange(f1, f2, dfs)
        n_plots    = len(final_signals) * 2
        plt.rcParams.update({"font.family": "sans-serif",
                             "savefig.bbox": "tight"})
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 5*n_plots),
                                 sharex=True, constrained_layout=True)
        if n_plots == 1:
            axes = [axes]

        t0 = time.time()
        for idx, (col, sig) in enumerate(final_signals.items()):
            try:
                if np.isnan(sig).any() or np.isinf(sig).any():
                    st.warning(f"Skipping '{col}': NaN/Inf"); continue
                if np.std(sig) < 1e-10:
                    st.warning(f"Skipping '{col}': flatline"); continue

                ph = st.empty()
                ph.text(f"WSST for {col}…")
                Tx_orig, _, ssq, _ = custom_wsst.ssq_cwt(
                    sig, fs=samp_rate, voices_per_octave=32)
                ph.empty()

                pb = st.progress(0, text=f"{col}: 0/{len(freq_steps)} bands")

                def _cb(frac, txt, _pb=pb, _c=col):
                    _pb.progress(frac, text=f"{_c}: {txt}")

                W_agg = fast_sequential_aggregate(
                    sig, samp_rate, freq_steps,
                    voices_per_octave=32, progress_callback=_cb)
                pb.empty()

                mp    = np.mean(np.abs(Tx_orig)**2)
                mask  = (np.abs(Tx_orig)**2 > 0.9*mp).astype(float)
                Tx_f  = (W_agg if W_agg is not None
                         else np.zeros_like(Tx_orig, complex)) * mask
                mf    = (ssq >= f1) & (ssq <= f2)
                fp    = ssq[mf]

                ax0   = axes[idx*2]
                e0    = np.abs(Tx_orig[mf,:])**2
                e0   /= e0.max() + 1e-12
                ax0.pcolormesh(t, fp, e0, norm=mcolors.PowerNorm(0.3),
                               cmap=parula_cmap, shading='auto')
                ax0.set_ylabel('Frequency (Hz)', fontsize=11)
                ax0.set_title(f"Standard WSST: {col}", fontsize=12,
                              fontweight='bold')
                ax0.set_ylim(f1, f2)

                ax1   = axes[idx*2+1]
                e1    = np.abs(Tx_f[mf,:])**2
                e1   /= e1.max() + 1e-12
                ax1.pcolormesh(t, fp, e1, norm=mcolors.PowerNorm(0.3),
                               cmap=parula_cmap, shading='auto')
                ax1.set_ylabel('Frequency (Hz)', fontsize=11)
                ax1.set_title(f"FAST Result: {col}", fontsize=12,
                              fontweight='bold')
                ax1.set_ylim(f1, f2)
                if idx*2+1 == n_plots-1:
                    ax1.set_xlabel('Time (s)', fontsize=11)

            except Exception as e:
                st.error(f"Error — {col}: {e}")
                import traceback; st.code(traceback.format_exc())

        st.success(f"✅ {len(freq_steps)} bands · {time.time()-t0:.2f}s"
                   f"  ({binfo['backend']})")
        st.pyplot(fig, use_container_width=True)

        dl1, dl2 = st.columns(2)
        with dl1:
            b = io.BytesIO(); fig.savefig(b, format="pdf", dpi=300); b.seek(0)
            st.download_button("📄 PDF", b, "FAST.pdf", "application/pdf")
        with dl2:
            b = io.BytesIO(); fig.savefig(b, format="png", dpi=300); b.seek(0)
            st.download_button("🖼️ PNG", b, "FAST.png", "image/png")


# ──────────────────────────────────────────────────────────────────────────────
# ROUTER
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    st.set_page_config(
        page_title="FAST — Muscle Fatigue Analysis",
        page_icon="🔬",
        layout="wide"
    )
    if 'page' not in st.session_state:
        st.session_state.page = 'splash'

    {
        'splash':   page_splash,
        'basic':    page_basic,
        'advanced': page_advanced,
    }.get(st.session_state.page, page_splash)()
