"""FAST engine: one recording channel in, one fatigue verdict out.

The method, in the order it happens:

1. Resample the channel to 200 Hz with a rational polyphase filter. The rate
   returned by `resample_to_200` is the true rate of the returned samples, so
   every frequency below is in real hertz. Nothing is relabelled.
2. Filter and aggregate, the way FAST is defined: the channel is split into 34
   overlapping bands between 1 and 35 Hz, each band gets its own
   synchrosqueezed transform, and the band powers are summed back onto one
   frequency grid. Each band is decimated to eight times its top edge first, so
   a 1-3 Hz band is transformed at 24 Hz rather than 200 Hz; a band cannot hold
   content above its own edge, so nothing is lost. Bands run in parallel.
3. Keep only time-frequency points whose power exceeds 0.9 times the mean power
   of the plane, which suppresses the noise floor before any frequency statistic
   is computed.
4. The fatigue call follows Zhao, Charoonsrisawad, Ichiyama & Chakrabarty
   (2025): the change in median frequency and the change in RMS between the
   start of the recording and its end.
5. The mean instantaneous frequency, and the NFI it gives, is still computed and
   reported as a second number; it no longer decides the verdict.

Thresholds are provisional and await clinical validation. They are not a
diagnosis.
"""
from __future__ import annotations

import base64
import os
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction

import numpy as np
import wsstpy
from scipy.signal import butter, decimate, filtfilt, resample_poly


TARGET_FS = 200.0
MAX_WINDOW_S = 30.0        # a longer recording is cut to its first 30 s
MIN_WINDOW_S = 15.0        # below this the estimate is not reported
BASELINE_S = 5.0
BAND_LOW, BAND_HIGH = 1.0, 35.0
MASK_FACTOR = 0.9
VOICES_PER_OCTAVE = 32
WAVELET = 'amor'

# Filter and aggregate. The bands overlap: a band centred at c covers c-1 to
# c+2 Hz, which is how the method was specified.
BAND_STEP_HZ = 1.0
BAND_BELOW_HZ = 1.0
BAND_ABOVE_HZ = 2.0
FA_VOICES = 16
FA_WORKERS = int(os.environ.get('FAST_FA_WORKERS', '6'))
GRID_STEP_HZ = 0.5
METHOD = os.environ.get('FAST_TRANSFORM', 'aggregate').lower()

# NFI thresholds, in per cent.
NFI_GREEN = 15.0
NFI_AMBER = 35.0

STATUS_ORDER = {'green': 2, 'amber': 1, 'red': 0, 'grey': -1}

ENGINE_VERSION = f'wsstpy-{wsstpy.__version__}'

FLAT = 1e-10


class EngineError(ValueError):
    """A channel that cannot be analysed, with a message for the athlete."""


def resample_to_200(sig: np.ndarray, fs: float) -> tuple[np.ndarray, float]:
    """Resample to 200 Hz and return the true rate of the result.

    Downsampling only: an EMG recorded below 200 Hz is left alone and keeps its
    own rate. The ratio is a rational approximation of 200/fs, so the returned
    rate is fs * up / down, which for any plausible device rate is within
    0.01 % of 200 Hz.
    """
    sig = np.asarray(sig, dtype=float)
    if fs <= TARGET_FS * 1.01:
        return sig, float(fs)
    ratio = Fraction(TARGET_FS / fs).limit_denominator(20000)
    up, down = ratio.numerator, ratio.denominator
    out = resample_poly(sig, up, down)
    return out, float(fs) * up / down


def _grid(lo: float = BAND_LOW, hi: float = BAND_HIGH) -> np.ndarray:
    """The frequency grid every band is summed onto, in hertz."""
    return np.arange(lo, hi + 1e-9, GRID_STEP_HZ)


def _band_power(sig: np.ndarray, fs: float, centre: float,
                grid: np.ndarray, frames: int) -> np.ndarray | None:
    """One band: filter, decimate, transform, and scatter onto the grid.

    A band from f1 to f2 cannot hold anything above f2, so it is transformed at
    eight times f2 (never below 20 Hz). Without this a 1-3 Hz band would be
    transformed at 200 Hz for nothing, and the aggregate would cost minutes.
    """
    f1 = max(centre - BAND_BELOW_HZ, BAND_LOW)
    f2 = centre + BAND_ABOVE_HZ
    if f1 >= f2 or f2 >= 0.5 * fs:
        return None
    b, a = butter(2, [f1 / (0.5 * fs), f2 / (0.5 * fs)], btype='band')
    band = filtfilt(b, a, sig)
    factor = int(max(1, min(round(fs / max(20.0, 8.0 * f2)), 64)))
    if factor > 1:
        band = decimate(band, factor, ftype='iir', zero_phase=True)
        rate = fs / factor
    else:
        rate = fs
    if len(band) < 32:
        return None
    sst, freqs = wsstpy.wsst(band, fs=rate, wavelet=WAVELET,
                             voices_per_octave=FA_VOICES, extend_signal=True)
    power = np.abs(sst) ** 2
    if power.shape[1] != frames:
        # A decimated band has fewer frames than the recording. Put them back on
        # the recording's time base so every band describes the same instants.
        t_out = np.linspace(0.0, 1.0, frames)
        t_in = np.linspace(0.0, 1.0, power.shape[1])
        power = np.stack([np.interp(t_out, t_in, row) for row in power])

    # Linear scatter onto the shared grid; rows outside the band are dropped.
    inside = (freqs >= f1) & (freqs <= f2)
    if inside.sum() < 2:
        return None
    fb = freqs[inside]
    pb = power[inside]
    out = np.zeros((len(grid), pb.shape[1]))
    upper = np.clip(np.searchsorted(grid, fb), 1, len(grid) - 1)
    lower = upper - 1
    span = grid[upper] - grid[lower]
    weight = np.clip((fb - grid[lower]) / np.where(span == 0, 1, span), 0.0, 1.0)
    out[lower] += pb * (1.0 - weight)[:, None]
    out[upper] += pb * weight[:, None]
    return out


def transform_fa(sig: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Filter and aggregate: 34 band transforms summed onto one grid.

    Returns (power, freqs) where power is real, freqs low to high.
    """
    grid = _grid()
    frames = len(sig)
    centres = np.arange(BAND_LOW, BAND_HIGH, BAND_STEP_HZ)
    total = np.zeros((len(grid), frames))
    pieces = 0
    with ThreadPoolExecutor(max_workers=max(1, min(FA_WORKERS, len(centres)))) as pool:
        for piece in pool.map(lambda c: _band_power(sig, fs, c, grid, frames), centres):
            if piece is not None:
                total += piece
                pieces += 1
    if not pieces:
        raise EngineError('No band of this channel could be transformed.')
    mask = total > MASK_FACTOR * float(np.mean(total))
    return total * mask, grid


def transform_single(sig: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """One whole-band transform, kept for comparison. See ENGINE-NOTES.md."""
    sst, freqs = wsstpy.wsst(sig, fs=fs, wavelet=WAVELET,
                             voices_per_octave=VOICES_PER_OCTAVE,
                             extend_signal=True)
    power = np.abs(sst) ** 2
    mask = power > MASK_FACTOR * float(np.mean(power))
    return power * mask, freqs


def transform(sig: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """The pipeline's transform: filter and aggregate by default."""
    if METHOD == 'single':
        return transform_single(sig, fs)
    return transform_fa(sig, fs)


def _power(arr: np.ndarray) -> np.ndarray:
    """The plane may arrive as power (real) or as a transform (complex)."""
    return np.abs(arr) ** 2 if np.iscomplexobj(arr) else arr


def mean_instantaneous_frequency(Tx: np.ndarray, freqs: np.ndarray,
                                 lo: float = BAND_LOW,
                                 hi: float = BAND_HIGH) -> np.ndarray:
    """Power-weighted mean frequency per time frame, NaN where the band is empty."""
    band = (freqs >= lo) & (freqs <= hi)
    power = _power(Tx[band, :])
    fb = freqs[band]
    out = np.full(power.shape[1], np.nan)
    total = power.sum(axis=0)
    ok = total > 0
    if ok.any():
        out[ok] = (power[:, ok] * fb[:, None]).sum(axis=0) / total[ok]
    return out


def score_from_nfi(nfi: float) -> int:
    """Score 0-100 from NFI in per cent. Higher is more recovered.

    Under 15 % maps linearly from 100 down to 70, 15-35 % from 69 down to 40,
    and 35 % or more from 39 down to 0, reaching 0 at 60 %.
    """
    if not np.isfinite(nfi):
        return 0
    n = max(nfi, 0.0)
    if n < NFI_GREEN:
        score = 100.0 - 2.0 * n
    elif n < NFI_AMBER:
        frac = (n - NFI_GREEN) / (NFI_AMBER - NFI_GREEN)
        score = 69.0 - 29.0 * frac
    else:
        frac = min(1.0, (n - NFI_AMBER) / 25.0)
        score = 39.0 - 39.0 * frac
    return int(max(0, min(100, round(score))))


def status_from_score(score: int) -> str:
    """The traffic light is derived from the score, never computed separately."""
    if score >= 70:
        return 'green'
    if score >= 40:
        return 'amber'
    return 'red'


_TIPS = {
    'green': 'No fatigue detected',
    'amber': 'Early fatigue signs',
    'red': 'Significant fatigue',
}


# ── fatiguability from MDF and RMS ──────────────────────────────────────────
# Zhao, Charoonsrisawad, Ichiyama & Chakrabarty (2025), "A non-invasive,
# quantitative method for detection of muscle fatigue and neural mechanism of
# compensation during muscle fatigue": fatigue is read from the change in median
# frequency and the change in RMS, not from where the spectrum sits.
#
#   low fatigue     RMS change < 10 % and MDF change < 5 %
#   medium fatigue  RMS change > 10 % and MDF change >  5 %  (MDF falling)
#   high fatigue    the same, with the trend consistent over the recording
#
# The slope thresholds in the poster (K_RMS >= 0.05, K_MDF <= -0.05) are quoted
# in the paper's own units, which are not defined there, so slopes are reported
# in percent per second and used only to say whether the trend is consistent.

EPOCH_S = 0.5
REFERENCE_S = 5.0
RMS_LIMIT_PCT = 10.0
MDF_LIMIT_PCT = 5.0

LEVEL_TIPS = {
    'low': 'Fatigue is not detectable in this recording',
    'medium': 'Some fatigue: the spectrum fell and the amplitude rose',
    'high': 'Fatigue is clear and grows through the recording',
    'mixed': 'One indicator moved without the other',
}


def _epochs(n: int, fs: float) -> list[tuple[int, int]]:
    step = max(1, int(round(EPOCH_S * fs)))
    return [(start, min(start + step, n)) for start in range(0, n, step)]


def median_frequency_series(Tx: np.ndarray, freqs: np.ndarray, fs: float) -> np.ndarray:
    """Median frequency per epoch, from the transform's frequency marginal."""
    power = _power(Tx)
    out = []
    for start, stop in _epochs(power.shape[1], fs):
        spectrum = power[:, start:stop].mean(axis=1)
        total = float(spectrum.sum())
        if not np.isfinite(total) or total <= 0:
            out.append(np.nan)
            continue
        cumulative = np.cumsum(spectrum) / total
        index = min(int(np.searchsorted(cumulative, 0.5)), len(freqs) - 1)
        if index == 0:
            out.append(float(freqs[0]))
            continue
        low, high = cumulative[index - 1], cumulative[index]
        frac = 0.0 if high == low else (0.5 - low) / (high - low)
        out.append(float(freqs[index - 1] + frac * (freqs[index] - freqs[index - 1])))
    return np.asarray(out, dtype=float)


def rms_series(sig: np.ndarray, fs: float) -> np.ndarray:
    """Root mean square per epoch, from the signal itself."""
    step = max(1, int(round(EPOCH_S * fs)))
    out = [float(np.sqrt(np.mean(sig[i:i + step] ** 2)))
           for i in range(0, len(sig) - step + 1, step)]
    return np.asarray(out, dtype=float)


def _slope(values: np.ndarray, fs: float) -> float:
    """Percent change per second, from a straight line through the series."""
    if len(values) < 2 or values[0] <= 0:
        return 0.0
    t = np.arange(len(values)) * EPOCH_S
    pct = values / values[0] * 100.0
    return float(np.polyfit(t, pct, 1)[0])


def fatiguability(sig: np.ndarray, Tx: np.ndarray, freqs: np.ndarray,
                  fs: float) -> dict:
    """Change in MDF and RMS from the start of the recording to its end."""
    mdf = median_frequency_series(Tx, freqs, fs)
    rms = rms_series(sig, fs)
    if len(rms) != len(mdf) and len(rms) > 1 and len(mdf) > 1:
        # The aggregate's frames follow its slowest band; put RMS on the same
        # time base so the two series describe the same instants.
        xp = np.linspace(0.0, 1.0, len(rms))
        x = np.linspace(0.0, 1.0, len(mdf))
        rms = np.interp(x, xp, rms)
    n = min(len(mdf), len(rms))
    mdf, rms = mdf[:n], rms[:n]
    ok = np.isfinite(mdf) & np.isfinite(rms) & (rms > 0) & (mdf > 0)
    if ok.sum() < 6:
        return dict(level='grey', status='grey', score=0, r=None,
                    tip='Not enough clean signal to measure a change')

    window = max(2, int(round(REFERENCE_S / EPOCH_S)))
    start_ok = ok[:window]
    end_ok = ok[max(0, n - window):]
    if start_ok.sum() < 2 or end_ok.sum() < 2:
        return dict(level='grey', status='grey', score=0, r=None,
                    tip='Not enough clean signal at the start or the end')

    mdf0 = float(np.median(mdf[:window][start_ok]))
    mdf1 = float(np.median(mdf[max(0, n - window):][end_ok]))
    rms0 = float(np.median(rms[:window][start_ok]))
    rms1 = float(np.median(rms[max(0, n - window):][end_ok]))
    if mdf0 <= 0 or rms0 <= 0:
        return dict(level='grey', status='grey', score=0, r=None,
                    tip='Could not read a starting value')

    delta_mdf = (mdf1 - mdf0) / mdf0 * 100.0
    delta_rms = (rms1 - rms0) / rms0 * 100.0

    # One number, r, is how far the muscle has moved past the two thresholds.
    # Both count: the method reads fatigue from the pair, so a frequency drop
    # with no amplitude compensation is half the evidence, not all of it.
    r = max(0.0, (delta_rms / RMS_LIMIT_PCT - delta_mdf / MDF_LIMIT_PCT) / 2.0)
    if r <= 1.0:
        score = 100.0 - 30.0 * r
    elif r <= 2.0:
        score = 70.0 - 30.0 * (r - 1.0)
    else:
        score = 40.0 - 40.0 * min(1.0, (r - 2.0) / 2.0)
    score = int(max(0, min(100, round(score))))

    moved = (delta_rms > RMS_LIMIT_PCT) and (delta_mdf < -MDF_LIMIT_PCT)
    if r <= 1.0:
        level = 'low'
    elif moved:
        level = 'high' if r > 2.0 else 'medium'
    else:
        level = 'mixed'
    status = status_from_score(score)

    k_rms = _slope(rms[ok], fs)
    k_mdf = _slope(mdf[ok], fs)
    consistent = ((k_rms > 0) == (delta_rms > 0)) and ((k_mdf < 0) == (delta_mdf < 0))

    return dict(level=level, status=status, score=score, r=round(r, 2),
                mdf_start=round(mdf0, 1), mdf_end=round(mdf1, 1),
                rms_start=round(rms0, 4), rms_end=round(rms1, 4),
                delta_mdf=round(delta_mdf, 1), delta_rms=round(delta_rms, 1),
                k_mdf=round(k_mdf, 3), k_rms=round(k_rms, 3),
                consistent=bool(consistent),
                tip=LEVEL_TIPS['high' if level == 'high' and consistent else level])


def classify(Tx: np.ndarray, freqs: np.ndarray, fs: float) -> dict:
    """NFI, score and traffic light for one masked transform."""
    mif = mean_instantaneous_frequency(Tx, freqs)
    valid = ~np.isnan(mif)
    n = len(mif)
    duration = n / fs

    if duration < MIN_WINDOW_S:
        return dict(status='grey', score=0, nfi=None, n_valid_frames=int(valid.sum()),
                    valid_fraction=0.0,
                    tip=f'Recording is {duration:.0f} s. FAST needs at least '
                        f'{MIN_WINDOW_S:.0f} s.')
    if not valid.any():
        return dict(status='grey', score=0, nfi=None, n_valid_frames=0,
                    valid_fraction=0.0, tip='No signal energy in the analysis band')

    n_baseline = max(3, min(int(round(BASELINE_S * fs)), n))
    baseline = mif[:n_baseline]
    base_valid = ~np.isnan(baseline)
    base_fraction = float(base_valid.sum()) / n_baseline
    if base_fraction < 0.5:
        return dict(status='grey', score=0, nfi=None, n_valid_frames=int(valid.sum()),
                    valid_fraction=0.0,
                    tip='Not enough clean signal in the first five seconds to '
                        'set a baseline')

    mif_baseline = float(np.median(baseline[base_valid]))
    mif_median = float(np.median(mif[valid]))
    if mif_baseline <= 0:
        return dict(status='grey', score=0, nfi=None, n_valid_frames=int(valid.sum()),
                    valid_fraction=base_fraction, tip='Could not compute a baseline')

    nfi = (mif_baseline - mif_median) / mif_baseline * 100.0
    score = score_from_nfi(nfi)
    status = status_from_score(score)
    return dict(status=status, score=score, nfi=round(nfi, 1),
                n_valid_frames=int(valid.sum()), valid_fraction=round(base_fraction, 2),
                tip=_TIPS[status])


STRIP_COLS, STRIP_ROWS = 120, 68
STRIP_FLOOR_DB = -40.0


def strip_payload(power: np.ndarray, freqs: np.ndarray, seconds: float,
                  cols: int = STRIP_COLS, rows: int = STRIP_ROWS) -> dict:
    """The plane as the home screen draws it: 120 x 68, 0-255, plus the MDF trace.

    The client only needs a small grid and one line, so the plane is reduced here
    rather than shipped whole.
    """
    power = np.asarray(power, dtype=float)
    if power.ndim != 2 or power.size == 0:
        return {}
    n_frames = power.shape[1]
    cols = max(2, min(cols, n_frames))
    t_idx = np.unique(np.linspace(0, n_frames, cols + 1).astype(int))
    f_idx = np.unique(np.linspace(0, len(freqs), rows + 1).astype(int))
    # reduceat sums each block; the divisor is the block width.
    by_time = np.add.reduceat(power, t_idx[:-1], axis=1) / np.diff(t_idx)
    block = np.add.reduceat(by_time, f_idx[:-1], axis=0) / np.diff(f_idx)[:, None]
    block = np.nan_to_num(block, nan=0.0, posinf=0.0, neginf=0.0)
    rows = block.shape[0]

    peak = float(block.max()) or 1.0
    db = 10.0 * np.log10(np.maximum(block, peak * 1e-9) / peak)
    value = np.clip((db - STRIP_FLOOR_DB) / -STRIP_FLOOR_DB, 0.0, 1.0) ** 0.85

    f_lo_edges = freqs[np.clip(f_idx[:-1], 0, len(freqs) - 1)]
    f_hi_edges = freqs[np.clip(f_idx[1:] - 1, 0, len(freqs) - 1)]
    mdf = []
    for j in range(block.shape[1]):
        spectrum = block[:, j]
        total = float(spectrum.sum())
        if total <= 0:
            mdf.append(None)
            continue
        cumulative = np.cumsum(spectrum) / total
        k = min(int(np.searchsorted(cumulative, 0.5)), len(freqs) - 1)
        low = cumulative[k - 1] if k else 0.0
        span = cumulative[k] - low
        frac = 0.0 if span <= 0 else (0.5 - low) / span
        f_lo = float(f_lo_edges[min(k, len(f_lo_edges) - 1)])
        f_hi = float(f_hi_edges[min(k, len(f_hi_edges) - 1)])
        mdf.append(round(f_lo + frac * (f_hi - f_lo), 2))

    return {'cols': int(block.shape[1]), 'rows': int(rows),
            'f0': float(freqs[0]), 'f1': float(f_hi_edges[-1]),
            'seconds': round(float(seconds), 1), 'mdf': mdf,
            'grid': base64.b64encode(
                (value * 255).astype(np.uint8).tobytes()).decode()}


def analyse_channel(sig: np.ndarray, fs: float, muscle_id: str) -> dict:
    """Full analysis for one channel. Input must be finite and non-flat."""
    sig = np.asarray(sig, dtype=float)
    if sig.size == 0:
        raise EngineError('The channel is empty')
    if not np.isfinite(sig).all():
        raise EngineError('The channel still contains missing or infinite samples')
    if float(np.std(sig)) < FLAT:
        raise EngineError('The channel has no variation (flat signal)')
    cut = sig[:int(round(MAX_WINDOW_S * fs))]
    resampled, true_fs = resample_to_200(cut, fs)
    Tx, freqs = transform(resampled, true_fs)

    # The duration and validity guard, and the mean-frequency fall that used to
    # be the whole verdict.
    result = dict(classify(Tx, freqs, true_fs))
    if result['status'] != 'grey':
        fatigue = fatiguability(resampled, Tx, freqs, true_fs)
        if fatigue['status'] != 'grey':
            # The call comes from MDF and RMS now, not from the mean frequency.
            result['status'] = fatigue['status']
            result['score'] = fatigue['score']
            result['tip'] = fatigue['tip']
        for key in ('level', 'r', 'mdf_start', 'mdf_end', 'rms_start', 'rms_end',
                    'delta_mdf', 'delta_rms', 'k_mdf', 'k_rms', 'consistent'):
            if key in fatigue:
                result[key] = fatigue[key]
    result['analysed_seconds'] = round(len(cut) / fs, 1)
    result['strip'] = strip_payload(Tx, freqs, len(resampled) / true_fs)
    return result


def overall(results: list[dict]) -> dict:
    """Overall verdict. The worst muscle caps it; the mean is reported separately."""
    scored = [r for r in results if r.get('status') in ('green', 'amber', 'red')]
    if not scored:
        return dict(status='grey', mean_score=None, worst=None)
    worst = min(scored, key=lambda r: STATUS_ORDER[r['status']])
    return dict(status=worst['status'],
                mean_score=int(round(sum(r['score'] for r in scored) / len(scored))),
                worst=worst.get('id') or worst.get('column'))
