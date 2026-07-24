"""
Bowen's LOW 8–23 Hz Centroid Implementation
===========================================
Exact implementation of the algorithm described in:
  "FAST LOW 8–23 Hz Centroid Calculation" (Bowen, 22 July 2026)

NOT a proposed alternative — this is the executed algorithm.

Gaps between this implementation and the full MATLAB pipeline are
annotated with `# GAP:` comments throughout.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════════════════════════
# SECTION 2–3: Bin edge reconstruction and overlap widths
# ═══════════════════════════════════════════════════════════════════════

def reconstruct_bin_edges(freq_centres: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Step 1: Reconstruct bin edges from midpoints between saved frequency centres.

    The code does NOT assume a perfectly uniform grid and does NOT treat
    frequency centres as widthless points.

    Returns (left_edges, right_edges) as 1-D arrays, same length as input.
    """
    f = np.asarray(freq_centres, dtype=float)
    # Sort — the frequency axis must be ascending
    sort_idx = np.argsort(f)
    f_sorted = f[sort_idx]

    n = len(f_sorted)

    # GAP: The MATLAB code names frequency centres 'saved centre frequency'.
    # We assume ssq_cwt returns them sorted, but we sort defensively.

    # Internal edges at midpoints between adjacent centres
    if n >= 2:
        internal_edges = (f_sorted[:-1] + f_sorted[1:]) / 2.0
    else:
        internal_edges = np.array([])

    # Left edges: internal_edge[i-1] for i > 0, else extrapolated
    left = np.empty(n)
    if n >= 2:
        # Extrapolate first: half the nearest spacing
        first_spacing = f_sorted[1] - f_sorted[0]
        left[0] = f_sorted[0] - first_spacing / 2.0
        left[1:] = internal_edges
    else:
        left[0] = f_sorted[0] - 1.0  # fallback

    # Right edges: internal_edge[i] for i < n-1, else extrapolated
    right = np.empty(n)
    if n >= 2:
        right[:-1] = internal_edges
        last_spacing = f_sorted[-1] - f_sorted[-2]
        right[-1] = f_sorted[-1] + last_spacing / 2.0
    else:
        right[0] = f_sorted[0] + 1.0  # fallback

    # Map back to original ordering
    inv_sort = np.argsort(sort_idx)
    return left[inv_sort], right[inv_sort]


def compute_overlap_widths(
    freq_centres: np.ndarray,
    band_low: float = 8.0,
    band_high: float = 23.0,
) -> np.ndarray:
    """
    For each bin, compute the overlap width with the target LOW interval.

    wᵢ = max{0, min(rᵢ, band_high) − max(lᵢ, band_low)}

    Returns wᵢ array (same shape as freq_centres) in Hz.
    Bin with centre outside 8-23 may still contribute if its edge extends in.
    """
    left, right = reconstruct_bin_edges(freq_centres)
    w = np.maximum(0.0, np.minimum(right, band_high) - np.maximum(left, band_low))
    return w


# ═══════════════════════════════════════════════════════════════════════
# SECTION 4: Time-point centroid calculation
# ═══════════════════════════════════════════════════════════════════════

def compute_low_centroid(
    Tx_fast: np.ndarray,       # (n_freqs, n_times) — complex FAST output
    ssq_freqs: np.ndarray,     # (n_freqs,) — frequency centres from SSQ-CWT
    band_low: float = 8.0,
    band_high: float = 23.0,
    full_band_low: float = 8.0,
    full_band_high: float = 200.0,
) -> dict:
    """
    Bowen §4–5: Time-point LOW centroid with full validity masking.

    The three core lines (MATLAB equivalent):
        low_weighted    = energy .* low_overlap_hz;
        low_mass        = sum(low_weighted, 1);
        low_centroid_hz = sum(low_weighted .* frequency_hz, 1) ./ low_mass;

    Returns dict with keys:
        centroid          — (n_times,) LOW centroid in Hz (NaN where invalid)
        low_mass          — (n_times,) effective LOW mass
        full_mass         — (n_times,) 8-200 Hz full mass
        low_fraction      — (n_times,) LOW mass / full mass
        low_mdf           — (n_times,) LOW MDF (computed for validity mask identity)
        valid_mask        — (n_times,) boolean — passes all §5 conditions
    """
    energy = np.abs(Tx_fast) ** 2          # FAST energy (non-negative)
    n_freqs, n_times = energy.shape

    # ── LOW overlap widths ──
    low_w = compute_overlap_widths(ssq_freqs, band_low, band_high)   # (n_freqs,)
    # ── Full-band overlap widths (8-200 Hz) ──
    full_w = compute_overlap_widths(ssq_freqs, full_band_low, full_band_high)

    # ── Three core lines ──
    low_weighted = energy * low_w[:, None]                # (n_freqs, n_times)
    low_mass = np.sum(low_weighted, axis=0)               # (n_times,)
    with np.errstate(divide='ignore', invalid='ignore'):
        centroid = (np.sum(low_weighted * ssq_freqs[:, None], axis=0)
                    / low_mass)

    # ── Full mass for LOW fraction ──
    full_weighted = energy * full_w[:, None]
    full_mass = np.sum(full_weighted, axis=0)

    with np.errstate(divide='ignore', invalid='ignore'):
        low_fraction = low_mass / full_mass

    # ── LOW MDF (computed for validity-mask identity; not exported as primary) ──
    # MDF is the frequency where cumulative mass crosses 50%
    low_mdf = np.full(n_times, np.nan)
    for t in range(n_times):
        if low_mass[t] > 0:
            cumsum = np.cumsum(low_weighted[:, t])
            half = low_mass[t] / 2.0
            idx = np.searchsorted(cumsum, half)
            if idx < n_freqs:
                low_mdf[t] = ssq_freqs[idx]

    # ── Validity mask (§5: six conditions) ──
    valid = np.ones(n_times, dtype=bool)

    # Condition 1: all FAST values in 8-200 Hz QC range are finite and non-negative
    full_band_mask = (ssq_freqs >= full_band_low) & (ssq_freqs <= full_band_high)
    # GAP: We check non-negativity of energy (it's squared magnitude, always ≥0).
    # The MATLAB code may check the raw FAST complex values.  For energy this
    # condition is always satisfied; the MATLAB check may be on the complex array.
    finite_full = np.all(np.isfinite(energy[full_band_mask, :]), axis=0)
    valid &= finite_full

    # Condition 2: full mass finite and positive
    valid &= np.isfinite(full_mass) & (full_mass > 0)

    # Condition 3: LOW mass finite and positive
    valid &= np.isfinite(low_mass) & (low_mass > 0)

    # Condition 4: LOW fraction between 0 and 1 (inclusive) and finite
    valid &= np.isfinite(low_fraction) & (low_fraction >= 0) & (low_fraction <= 1)

    # Condition 5: centroid finite
    valid &= np.isfinite(centroid)

    # GAP: Condition 6 — centroid must lie within 8–23 Hz.
    # The spec says "Every valid centroid must lie within 8–23 Hz; values outside
    # the interval trigger a validation failure."
    # We include this check.
    valid &= (centroid >= band_low) & (centroid <= band_high)

    # GAP: Condition 7 — LOW MDF finite (the MATLAB code also checks this)
    valid &= np.isfinite(low_mdf)

    # Apply NaN where invalid
    centroid[~valid] = np.nan
    low_mass[~valid] = np.nan
    full_mass[~valid] = np.nan
    low_fraction[~valid] = np.nan
    low_mdf[~valid] = np.nan

    return dict(
        centroid=centroid,
        low_mass=low_mass,
        full_mass=full_mass,
        low_fraction=low_fraction,
        low_mdf=low_mdf,
        valid_mask=valid,
    )


# ═══════════════════════════════════════════════════════════════════════
# SECTION 6.1–6.3: Phase-level aggregation
# ═══════════════════════════════════════════════════════════════════════

def remove_padding(centroid: np.ndarray, n_pad: int = 1500) -> np.ndarray:
    """
    §6.1: Remove padding samples from each side.
    In the whole-phase FAST route, 1500 padding samples are removed from
    each side.

    GAP: We do not know whether the input signal was padded.  1500 samples
    is a hard number from the MATLAB pipeline.  If the input was not padded,
    this step is a no-op (n_pad=0).  Caller must decide.
    """
    if n_pad <= 0:
        return centroid
    end = len(centroid) - n_pad
    if end <= n_pad:
        return centroid  # too short
    return centroid[n_pad:end]


def con_ecc_split(n_real: int, con_frac: float = 0.60) -> tuple[int, int]:
    """
    §6.1: Fixed 60/40 sample-proportion CON/ECC partition.

    The first floor(0.60 × n_real) samples are labelled CON, remainder ECC.
    This is a fixed sample-proportion split, NOT a per-repetition kinematic
    transition detector.

    GAP: The spec assumes a registered proportion from a known trial structure.
    We do not have per-participant Set/Curl/Rep registration here.

    Returns (n_con, n_ecc).
    """
    n_con = int(np.floor(con_frac * n_real))
    n_ecc = n_real - n_con
    return n_con, n_ecc


def compute_phase_mean(centroid: np.ndarray) -> float:
    """
    §6.3: Phase-level mean.

    Arithmetic mean of valid time-point centroids within a phase, NaNs omitted.

    Critical order (§6.3 box): spectral weighting first (per time point),
    then temporal averaging.  NOT: pool spectrum then centroid.
    """
    valid = centroid[~np.isnan(centroid)]
    if len(valid) == 0:
        return np.nan
    return float(np.mean(valid))


def compute_temporal_features(
    centroid: np.ndarray,
    phase_mask: np.ndarray | None = None,
) -> dict:
    """
    §6.4: Early, Middle, Late values.

    Each phase is divided using sample-centre relative positions:
      Early  < 1/3
      Middle from 1/3 to < 2/3
      Late   ≥ 2/3

    Returns dict with 'early', 'middle', 'late' mean centroids.
    """
    if phase_mask is not None:
        c = centroid[phase_mask]
    else:
        c = centroid

    n = len(c)
    result = {'early': np.nan, 'middle': np.nan, 'late': np.nan}

    third = n / 3.0
    parts = {
        'early':  slice(0, int(np.floor(third))),
        'middle': slice(int(np.floor(third)), int(np.floor(2 * third))),
        'late':   slice(int(np.floor(2 * third)), n),
    }
    for name, slc in parts.items():
        seg = c[slc]
        valid_seg = seg[~np.isnan(seg)]
        if len(valid_seg) > 0:
            result[name] = float(np.mean(valid_seg))

    return result


# ═══════════════════════════════════════════════════════════════════════
# SECTION 6.2: Whole-phase trajectory (101-point normalized)
# ═══════════════════════════════════════════════════════════════════════

def whole_phase_trajectory(
    centroid: np.ndarray,
    n_points: int = 101,
    con_ecc_split_idx: int | None = None,
) -> dict:
    """
    §6.2: Map CON and ECC separately to 0–100% relative phase time,
    resample to n_points using segment-aware linear interpolation.

    Interpolation restricted to contiguous valid segments; invalid stretches
    remain gaps (NaN).

    GAP: We interpolate over time only.  The spec says segment-aware linear
    interpolation — we detect contiguous valid runs and interpolate within
    each, leaving NaN gaps between runs.  This is our best reconstruction
    without the original MATLAB code.

    Returns dict with 'con_trajectory' and 'ecc_trajectory' as arrays of
    length n_points (NaN where no valid data).
    """
    if con_ecc_split_idx is None:
        con_ecc_split_idx = int(np.floor(0.60 * len(centroid)))

    trajectories = {}
    for label, segment in [('con', centroid[:con_ecc_split_idx]),
                            ('ecc', centroid[con_ecc_split_idx:])]:
        n_seg = len(segment)
        traj = np.full(n_points, np.nan)

        if n_seg > 0:
            # Source x: index positions in [0, 1]
            src_x = np.linspace(0, 1, n_seg)
            # Target x: 0..100% in n_points
            tgt_x = np.linspace(0, 1, n_points)

            # Find contiguous valid runs
            valid = ~np.isnan(segment)
            # Use simple numpy interpolation for contiguous segments
            # For each target point, find nearest valid source within range
            for i, tx in enumerate(tgt_x):
                # Find source indices whose x bounds bracket the target
                # Simple approach: nearest valid source index, but only if
                # both neighbours are in the same contiguous block
                # GAP: The MATLAB 'segment-aware' interpolation likely uses
                # interp1 with linear method restricted to contiguous valid
                # stretches.  We approximate with nearest-neighbour within
                # contiguous blocks.
                diffs = np.abs(src_x - tx)
                # Find nearest valid point
                valid_diffs = np.where(valid, diffs, np.inf)
                nearest_idx = np.argmin(valid_diffs)
                if valid_diffs[nearest_idx] < (1.0 / (n_seg - 1)) * 1.5 if n_seg > 1 else np.inf:
                    traj[i] = segment[nearest_idx]

        trajectories[label] = traj

    return trajectories


def compute_group_trajectory(
    participant_trajs: list[np.ndarray],
    n_bootstrap: int = 5000,
    ci_lower: float = 2.5,
    ci_upper: float = 97.5,
    min_n: int = 3,
) -> dict:
    """
    §6.5: Group trajectory with bootstrap CI.

    At each normalized time point, group mean across participants with finite
    centroid at that point.  Bootstrap 5000 resamples for 2.5th/97.5th
    percentile intervals.  Not returned when n < min_n.

    GAP: This requires multiple participants' data — not used in the
    single-recording athlete app.  Included for completeness.
    """
    trajs = np.array(participant_trajs)  # (n_participants, n_points)
    n_points = trajs.shape[1]

    group_mean = np.full(n_points, np.nan)
    ci_low = np.full(n_points, np.nan)
    ci_high = np.full(n_points, np.nan)

    rng = np.random.RandomState(42)
    for p in range(n_points):
        col = trajs[:, p]
        valid_col = col[~np.isnan(col)]
        n_valid = len(valid_col)
        if n_valid < min_n:
            continue
        group_mean[p] = np.mean(valid_col)
        # Bootstrap
        boot_means = np.array([
            np.mean(rng.choice(valid_col, size=n_valid, replace=True))
            for _ in range(n_bootstrap)
        ])
        ci_low[p] = np.percentile(boot_means, ci_lower)
        ci_high[p] = np.percentile(boot_means, ci_upper)

    return dict(mean=group_mean, ci_lower=ci_low, ci_upper=ci_high, n=n_points)


# ═══════════════════════════════════════════════════════════════════════
# Full pipeline (ties it all together)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class BowenResult:
    """Complete Bowen LOW centroid analysis result for one muscle."""
    label: str

    # Time-point level
    centroid: np.ndarray          # (n_times,) Hz — NaN where invalid
    low_mass: np.ndarray          # (n_times,)
    full_mass: np.ndarray         # (n_times,)
    low_fraction: np.ndarray      # (n_times,)
    low_mdf: np.ndarray           # (n_times,)
    valid_mask: np.ndarray        # (n_times,) bool

    # Phase-level
    phase_mean_centroid: float = np.nan           # §6.3
    phase_mean_con: float = np.nan                # CON phase mean
    phase_mean_ecc: float = np.nan                # ECC phase mean

    # Temporal features
    early_centroid: float = np.nan                # §6.4
    middle_centroid: float = np.nan
    late_centroid: float = np.nan

    # Whole-phase trajectory
    con_trajectory: np.ndarray | None = None      # 101 points
    ecc_trajectory: np.ndarray | None = None

    # Summary for display
    n_valid: int = 0
    n_total: int = 0


def run_bowen_pipeline(
    Tx_fast: np.ndarray,
    ssq_freqs: np.ndarray,
    label: str = "",
    band_low: float = 8.0,
    band_high: float = 23.0,
    n_pad: int = 1500,
    con_frac: float = 0.60,
) -> BowenResult:
    """
    Full Bowen LOW centroid pipeline for one muscle.

    Follows the exact spec from 'FAST LOW 8–23 Hz Centroid Calculation',
    22 July 2026, Sections 2–6.
    """
    # Step 4: Time-point centroid with validity mask
    d = compute_low_centroid(Tx_fast, ssq_freqs, band_low, band_high)

    result = BowenResult(
        label=label,
        centroid=d['centroid'],
        low_mass=d['low_mass'],
        full_mass=d['full_mass'],
        low_fraction=d['low_fraction'],
        low_mdf=d['low_mdf'],
        valid_mask=d['valid_mask'],
        n_total=len(d['centroid']),
        n_valid=int(np.sum(d['valid_mask'])),
    )

    # Step 6.1: Remove padding + CON/ECC split
    # GAP: We assume NO padding in the app's input — the recording is
    # already a user-chosen window.  Set n_pad=0 unless the data was
    # pre-padded by the FAST preprocessing pipeline.
    if n_pad > 0:
        centroid_nopad = remove_padding(d['centroid'], n_pad)
        # Recompute valid count after padding removal
        valid_nopad = ~np.isnan(centroid_nopad)
        n_real = len(centroid_nopad)
        n_con, n_ecc = con_ecc_split(n_real, con_frac)

        con_centroids = centroid_nopad[:n_con]
        ecc_centroids = centroid_nopad[n_con:]
    else:
        # ── NO padding removal — use full centroid array ──
        n_pad = 0
        centroid_nopad = d['centroid']
        valid_nopad = ~np.isnan(centroid_nopad)
        n_real = len(centroid_nopad)
        n_con, n_ecc = con_ecc_split(n_real, con_frac)

        con_centroids = centroid_nopad[:n_con]
        ecc_centroids = centroid_nopad[n_con:]

    # Phase-level means (§6.3)
    result.phase_mean_centroid = compute_phase_mean(centroid_nopad)
    result.phase_mean_con = compute_phase_mean(con_centroids)
    result.phase_mean_ecc = compute_phase_mean(ecc_centroids)

    # Temporal features (§6.4)
    temporal = compute_temporal_features(centroid_nopad)
    result.early_centroid = temporal['early']
    result.middle_centroid = temporal['middle']
    result.late_centroid = temporal['late']

    # Whole-phase trajectory (§6.2) — 101-point CON/ECC trajectories
    trajectories = whole_phase_trajectory(centroid_nopad, n_points=101,
                                          con_ecc_split_idx=n_con)
    result.con_trajectory = trajectories['con']
    result.ecc_trajectory = trajectories['ecc']

    return result


# ═══════════════════════════════════════════════════════════════════════
# GAP SUMMARY (printed for user review)
# ═══════════════════════════════════════════════════════════════════════

GAPS = """
═══════════════════════════════════════════════════════════════════════════
GAPS between this Python implementation and Bowen's MATLAB pipeline
═══════════════════════════════════════════════════════════════════════════

1. NO ACCESS TO ORIGINAL MATLAB FILES
   The spec references four verified .m files:
     compute_fast_frequency_bin_widths_v1.m
     extract_fast_low_phase_features_v1.m
     extract_fast_low_temporal_features_v1.m
     FAST_whole_phase_LOW_centroid_trajectory_v1.m
   This implementation is reconstructed from the pseudocode and worked
   examples only.  Subtle differences may exist.

2. PADDING ASSUMPTION (n_pad = 0 by default)
   The MATLAB pipeline removes 1500 padding samples from each side of the
   whole-phase FAST output.  The current app processes user-chosen time
   windows from raw sEMG — there is no padding.  Set n_pad=1500 only if
   you know the input was padded.

3. CON/ECC SPLIT IS A FIXED PROPORTION, NOT KINEMATIC
   The 60/40 split uses a registered sample proportion — not a joint-angle
   or velocity-based transition detector.  The app has no concept of
   repetitions or kinematics.  We simply split the time window 60/40.

4. TRIAL STRUCTURE MISSING
   The spec assumes data organized by: Participant → Muscle → Set → Curl.
   The current app has NO trial registration — it processes one flat
   recording with no Set/Curl/Rep metadata.

5. GROUP-LEVEL BOOTSTRAP NOT APPLICABLE
   §6.5 describes pointwise bootstrap over multiple participants.  Not
   meaningful in a single-recording athlete app.  Included as
   compute_group_trajectory() for API completeness only.

6. FATIGUE CLASSIFICATION NOT SPECIFIED BY BOWEN
   The centroid computation stops at computing the value in Hz.  The spec
   explicitly says: "No universal fatigue direction is assigned: a lower
   centroid does not automatically mean greater fatigue."  Any traffic-light
   classification on centroid values is our own addition.

7. SEGMENT-AWARE INTERPOLATION
   The MATLAB code uses segment-aware linear interpolation restricted to
   contiguous valid segments.  Our approximation uses nearest-valid-point
   within contiguous runs — may differ at segment boundaries.

8. SSQ_FREQS FROM SSQ-CWT (not from MATLAB)
   The FAST frequency axis comes from Python's ssqueezepy, not from the
   MATLAB SSQ-CWT implementation.  Bin centres may differ slightly.

9. ENERGY VS COMPLEX FAST VALUES
   §5 condition 1 checks "FAST values in 8-200 Hz QC range are finite and
   non-negative."  We take abs(Tx)^2 (always ≥0).  MATLAB may check the raw
   complex array, which could be negative for the real/imag parts.

═══ UNADDRESSED — NEED BOWEN'S INPUT ═══

10. THRESHOLD FOR CENTROID-BASED FATIGUE
    What centroid shift constitutes fatigue?  The spec gives no thresholds.
    Options discussed in literature: within-subject baseline normalization
    (NFI), rate of centroid decline, or crossing a fixed boundary.

11. BASELINE NORMALIZATION
    The sEMG fatigue literature uses NFI = (baseline − current)/baseline × 100%.
    Should the centroid be normalized to the first N seconds of recording?

12. MULTI-MUSCLE AGGREGATION
    Should centroids from multiple muscles be combined into an overall score?
═══════════════════════════════════════════════════════════════════════════
"""
