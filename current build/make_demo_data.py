#!/usr/bin/env python3
"""Generate demo_recording.csv — the FAST app's built-in demo recording.

One synthetic 30 s / 1000 Hz recording, 8 right-leg channels, designed so a
single assessment shows every status:
  fresh               (green): VL, VM, RF, TA
  some fatigue        (amber): BF, ST
  significant fatigue (red):   GM, GL

Same signal model as the Commercial demo generator (BIPE-style harmonic
carrier whose instantaneous frequency follows a piecewise-linear
trajectory), plus the amplitude compensation that goes with it: as the
spectrum falls the recording gets louder, which is what Zhao et al. (2025)
read alongside the frequency shift. Fast twitch units drop out first, so the
amplitude rises while the median frequency falls.

Re-run this file to regenerate demo_recording.csv, then verify the statuses
through the real pipeline before shipping (see the git-push-deploy skill,
references/faaST-demo-data.md).
"""
import os
import numpy as np

MUSCLES = ['VL', 'VM', 'RF', 'BF', 'ST', 'TA', 'GM', 'GL']
BASE = {'VL': 22.0, 'VM': 21.5, 'RF': 21.0, 'BF': 20.5, 'ST': 20.0,
        'TA': 22.5, 'GM': 19.5, 'GL': 19.0}
STATUS = {'VL': 'fresh', 'VM': 'fresh', 'RF': 'fresh', 'TA': 'fresh',
          'BF': 'partial', 'ST': 'partial', 'GM': 'fatigued', 'GL': 'fatigued'}
TRAJ = {  # (time_s, frequency multiplier) trajectories
    'fresh':    [(0, 1.0), (30, 1.0)],
    'partial':  [(0, 1.0), (5, 1.0), (12, 0.90), (30, 0.90)],
    'fatigued': [(0, 1.0), (5, 1.0), (10, 0.58), (30, 0.58)],
}
GAIN = {  # amplitude compensation over the same window
    'fresh':    [(0, 1.0), (30, 1.0)],
    'partial':  [(0, 1.0), (5, 1.0), (12, 1.14), (30, 1.14)],
    'fatigued': [(0, 1.0), (5, 1.0), (10, 1.45), (30, 1.45)],
}
FS, DUR = 1000.0, 30.0


def make_channel(traj, gain, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(0, DUR, 1 / FS)
    f = np.interp(t, [p[0] for p in traj], [p[1] for p in traj])
    a = np.interp(t, [p[0] for p in gain], [p[1] for p in gain])
    phase = 2 * np.pi * np.cumsum(f) / FS
    x = np.sin(phase) + 0.35 * np.sin(1.48 * phase) \
        + 0.18 * np.sin(0.52 * phase) + 0.08 * np.sin(2.01 * phase)
    return a * (0.5 * x + 0.02 * rng.standard_normal(len(t)))


def main():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'demo_recording.csv')
    cols = {}
    for i, m in enumerate(MUSCLES):
        traj = [(tt, ff * BASE[m]) for tt, ff in TRAJ[STATUS[m]]]
        cols[m] = make_channel(traj, GAIN[STATUS[m]], seed=700 + i)
    with open(out, 'w') as fh:
        fh.write(','.join(MUSCLES) + '\n')
        for row in zip(*[cols[m] for m in MUSCLES]):
            fh.write(','.join(f'{v:.6f}' for v in row) + '\n')
    print(f'wrote {out} ({len(MUSCLES)} channels, {DUR:.0f}s, {FS:.0f} Hz)')


if __name__ == '__main__':
    main()
