"""Reproduce the equivalence claim in ENGINE-NOTES.md.

    python3 scripts/compare_transform.py [recording.csv]

Prints, for every channel, the fatigue statistic from the old 35-band aggregate
and from a single transform, with the resulting status and score.
"""
import os
import sys

import numpy as np
from scipy.signal import butter, decimate, filtfilt
from ssqueezepy import ssq_cwt
import wsstpy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))
import engine  # noqa: E402
import ingest  # noqa: E402

TARGET_FS = 200.0


def aggregate(sig, fs, lo=1.0, hi=35.0, step=1.0):
    """The pipeline's previous construction: band-filter, transform, sum."""
    total, n_valid = None, 0
    for centre in np.arange(lo, hi, step):
        f1, f2 = max(centre - 1.0, 1.0), centre + 2.0
        if f1 >= f2 or f2 >= 0.5 * fs:
            continue
        b, a = butter(2, [f1 / (0.5 * fs), f2 / (0.5 * fs)], btype='band')
        Tx, _, _, _ = ssq_cwt(filtfilt(b, a, sig), fs=fs, nv=32,
                              wavelet=('morlet', {'mu': 6}))
        if total is None:
            total = np.zeros_like(Tx, dtype=complex)
        total += Tx
        n_valid += 1
    if total is not None and n_valid > 1:
        total /= n_valid
    Tx_orig, _, freqs, _ = ssq_cwt(sig, fs=fs, nv=32, wavelet=('morlet', {'mu': 6}))
    power = np.abs(Tx_orig) ** 2
    mask = (power > 0.9 * np.mean(power)).astype(float)
    if total is None:
        total = np.zeros_like(Tx_orig, dtype=complex)
    return total * mask, freqs


def main(path: str) -> None:
    content = open(path, 'rb').read()
    parsed = ingest.load(content, os.path.basename(path))
    fs = parsed.fs or 1000.0
    print(f'{path}: fs={parsed.fs} (assumed {fs:.0f} if absent)')
    print(f"{'muscle':8s} {'aggregate':>26s} {'single (wsstpy)':>26s}")
    print(f"{'':8s} {'NFI%  status  score':>26s} {'NFI%  status  score':>26s}")
    for channel in parsed.channels:
        if channel['drop_reason']:
            continue
        sig = channel['data'][:int(engine.MAX_WINDOW_S * fs)]
        sig = decimate(sig, max(1, int(fs / TARGET_FS)), ftype='iir') if fs > TARGET_FS else sig
        rate = fs / max(1, int(fs / TARGET_FS)) if fs > TARGET_FS else fs

        Tx_a, freqs_a = aggregate(sig, rate)
        nfi_a = engine.classify(Tx_a, freqs_a, rate)
        Tx_s, freqs_s = engine.transform(sig, rate)
        nfi_s = engine.classify(Tx_s, freqs_s, rate)

        def fmt(r):
            nfi = '—' if r['nfi'] is None else f"{r['nfi']:.1f}"
            return f"{nfi:>5} {r['status']:>7} {r['score']:>5}"

        print(f"{channel['name']:8s} {fmt(nfi_a):>26s} {fmt(nfi_s):>26s}")


if __name__ == '__main__':
    default = os.path.join(os.path.dirname(__file__), '..', 'app',
                           'demo_recording.csv')
    main(sys.argv[1] if len(sys.argv) > 1 else default)
