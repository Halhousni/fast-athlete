"""Build the strip the home screen shows.

    python3 scripts/make_strip.py [recording.csv] [channel]

Runs the real engine on one channel of the demo recording, keeps the
time-frequency power plane, and writes static/demo-strip.json: 120 columns of
time (0.25 s each), 68 rows of frequency (0.5 Hz each from 1 to 35 Hz), the
median-frequency trace, and nothing else. The client colours it through a
lookup table, so no image is stored and the strip is the measurement itself.
"""
import base64
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'current build'))
import engine  # noqa: E402
import ingest  # noqa: E402

COLS, ROWS = 120, 68
FLOOR_DB = -40.0


def main(path: str, channel: str) -> None:
    parsed = ingest.load(open(path, 'rb').read(), os.path.basename(path),
                         fs=1000.0)
    found = next((c for c in parsed.channels if c['name'].lower() == channel.lower()),
                 None)
    if found is None:
        raise SystemExit(f'no channel {channel!r}; have '
                         f'{[c["name"] for c in parsed.channels]}')

    sig = found['data'][:int(engine.MAX_WINDOW_S * parsed.fs)]
    resampled, fs = engine.resample_to_200(sig, parsed.fs)
    power, freqs = engine.transform(resampled, fs)
    seconds = len(resampled) / fs

    # Block-average onto the 120 x 68 grid.
    t_edges = np.linspace(0, power.shape[1], COLS + 1).astype(int)
    f_edges = np.linspace(0, len(freqs), ROWS + 1).astype(int)
    grid = np.zeros((ROWS, COLS))
    for i in range(ROWS):
        for j in range(COLS):
            block = power[f_edges[i]:max(f_edges[i] + 1, f_edges[i + 1]),
                          t_edges[j]:max(t_edges[j] + 1, t_edges[j + 1])]
            grid[i, j] = block.mean() if block.size else 0.0

    # Decibels under the loudest cell, so the ridge survives the quiet parts.
    peak = float(grid.max()) or 1.0
    db = 10.0 * np.log10(np.maximum(grid, peak * 1e-9) / peak)
    value = np.clip((db - FLOOR_DB) / -FLOOR_DB, 0.0, 1.0)
    value = value ** 0.85
    print(f'lit fraction (v>0.25): {(value > 0.25).mean():.2f}')

    # Median frequency per column, from the same plane.
    mdf = []
    for j in range(COLS):
        spectrum = grid[:, j]
        total = spectrum.sum()
        if total <= 0:
            mdf.append(None)
            continue
        cumulative = np.cumsum(spectrum) / total
        k = min(int(np.searchsorted(cumulative, 0.5)), ROWS - 1)
        lo = cumulative[k - 1] if k else 0.0
        span = cumulative[k] - lo
        frac = 0.0 if span <= 0 else (0.5 - lo) / span
        f_lo = freqs[f_edges[k]]
        f_hi = freqs[max(f_edges[k], f_edges[k + 1] - 1)]
        mdf.append(round(float(f_lo + frac * (f_hi - f_lo)), 2))

    meta = next((m for m in ingest.MUSCLES if m['id'] == found['name'].upper()), {})
    payload = {
        'channel': found['name'],
        'label': (meta.get('name') or found['name']).lower(),
        'seconds': round(seconds, 1),
        'cols': COLS, 'rows': ROWS,
        'f0': float(freqs[0]), 'f1': float(freqs[f_edges[-1] - 1]),
        'baseline_s': engine.BASELINE_S,
        'mdf': mdf,
        'grid': base64.b64encode((value * 255).astype(np.uint8).tobytes()).decode(),
    }
    out = os.path.join(os.path.dirname(__file__), '..', 'current build',
                       'static', 'demo-strip.json')
    with open(out, 'w') as fh:
        json.dump(payload, fh, separators=(',', ':'))
    print(f'wrote {out} ({os.path.getsize(out) / 1024:.1f} KB)')
    print(f'mdf {min(v for v in mdf if v):.1f} to {max(v for v in mdf if v):.1f} Hz')


if __name__ == '__main__':
    default = os.path.join(os.path.dirname(__file__), '..', 'current build',
                           'demo_recording.csv')
    main(sys.argv[1] if len(sys.argv) > 1 else default,
         sys.argv[2] if len(sys.argv) > 2 else 'GM')
