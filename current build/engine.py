"""FAST engine: one recording channel in, one fatigue verdict out.

The method, in the order it happens:

1. Resample the channel to 200 Hz with a rational polyphase filter. The rate
   returned by `resample_to_200` is the true rate of the returned samples, so
   every frequency below is in real hertz. Nothing is relabelled.
2. Take one wavelet synchrosqueezed transform of the whole channel (wsstpy,
   analytic Morlet, 32 voices per octave, signal extension on). This replaces
   the 35 band-filtered transforms the app used to run: on the bundled demo
   recording the two agree on the verdict for all eight channels and differ by
   at most one point of score, at 1/37 of the cost. See ENGINE-NOTES.md.
3. Keep only time-frequency points whose power exceeds 0.9 times the mean
   power of the plane, which suppresses the noise floor before any frequency
   statistic is computed.
4. Per time frame, compute the power-weighted mean instantaneous frequency
   over 1-35 Hz.
5. Compare the median of the whole recording with the median of the first
   five seconds of the same contraction:

       NFI = (MIF_baseline - MIF_median) / MIF_baseline * 100

   A positive NFI means the spectrum moved down during the recording, which is
   the usual signature of fatigue.
6. Map NFI to a score, and the score to a traffic light. There is one
   function for each, so the number and the colour cannot disagree.

Thresholds are provisional and await clinical validation. They are not a
diagnosis.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import wsstpy
from scipy.signal import resample_poly

import bowen_centroid

TARGET_FS = 200.0
MAX_WINDOW_S = 30.0        # a longer recording is cut to its first 30 s
MIN_WINDOW_S = 15.0        # below this the estimate is not reported
BASELINE_S = 5.0
BAND_LOW, BAND_HIGH = 1.0, 35.0
MASK_FACTOR = 0.9
VOICES_PER_OCTAVE = 32
WAVELET = 'amor'

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


def transform(sig: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """One masked synchrosqueezed transform. Returns (Tx, freqs), rows low to high."""
    sst, freqs = wsstpy.wsst(sig, fs=fs, wavelet=WAVELET,
                             voices_per_octave=VOICES_PER_OCTAVE,
                             extend_signal=True)
    power = np.abs(sst) ** 2
    mask = power > MASK_FACTOR * float(np.mean(power))
    return sst * mask, freqs


def mean_instantaneous_frequency(Tx: np.ndarray, freqs: np.ndarray,
                                 lo: float = BAND_LOW,
                                 hi: float = BAND_HIGH) -> np.ndarray:
    """Power-weighted mean frequency per time frame, NaN where the band is empty."""
    band = (freqs >= lo) & (freqs <= hi)
    power = np.abs(Tx[band, :]) ** 2
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
    result = classify(Tx, freqs, true_fs)

    bowen = bowen_centroid.run_bowen_pipeline(Tx, freqs, muscle_id, n_pad=0)
    for key, value in (('centroid', bowen.phase_mean_centroid),
                       ('centroid_early', bowen.early_centroid),
                       ('centroid_late', bowen.late_centroid)):
        result[key] = round(float(value), 1) if np.isfinite(value) else None
    result['analysed_seconds'] = round(len(cut) / fs, 1)
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
