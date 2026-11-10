# Engine notes

Two changes were made to the method in the working copy. Both are recorded here
with the measurement that justifies them, because for a research tool the method
is part of the result.

## 1. One transform instead of thirty-five

The previous `_run_fast` built its time-frequency matrix by band-pass filtering
the signal 34 times, from 1 Hz to 35 Hz, taking a synchrosqueezed CWT of each
band, and summing the results. It then took one more transform of the whole
signal to build a power mask. A channel cost about 4.5 s.

Only one transform is needed. The band filters shape the spectrum; the transform
already resolves frequency, so the bands can be selected on its frequency axis.

Measured on the bundled `demo_recording.csv`, every channel, 30 s at 1000 Hz,
through the old aggregate and through one transform:

| muscle | aggregate NFI | status | score | single NFI | status | score |
|---|---|---|---|---|---|---|
| VL | -0.0 % | green | 100 | 0.0 % | green | 100 |
| VM | -0.0 % | green | 100 | 0.0 % | green | 100 |
| RF | 0.0 % | green | 99 | 0.0 % | green | 100 |
| BF | 25.1 % | amber | 54 | 25.7 % | amber | 54 |
| ST | 24.7 % | amber | 54 | 25.5 % | amber | 54 |
| TA | 0.0 % | green | 99 | 0.1 % | green | 100 |
| GM | 53.9 % | red | 9 | 53.4 % | red | 10 |
| GL | 53.8 % | red | 9 | 53.6 % | red | 10 |

Status agrees on all eight channels; the score moves by at most one point. Total
time for the eight channels fell from 33.1 s to 5.7 s.

The mask is kept: time-frequency points whose power is below 0.9 times the mean
power of the plane are set to zero before any frequency statistic is computed.

## 2. wsstpy instead of ssqueezepy

`wsstpy` (this account's own library) implements MATLAB's `wsst` and agrees with
MATLAB R2025b to 2.6e-15 on the transform matrix. `DESIGN.md` previously listed
"SSQ_FREQS FROM SSQ-CWT (not from MATLAB)" as a known gap between this
implementation and the MATLAB reference; using wsstpy closes it.

It also removes a dependency: `ssqueezepy` pulls in `numba`, which lags NumPy
releases and made the container fragile. wsstpy needs only NumPy and SciPy.

Costs and benefits, measured on one 30 s channel at 200 Hz:

| library | one transform | rows x columns | MATLAB agreement |
|---|---|---|---|
| ssqueezepy `ssq_cwt` | 0.38 s | 293 x 6000 | not established |
| wsstpy `wsst` | 0.45 s | 324 x 6000 | 2.6e-15 |

Signal extension is on (`extend_signal=True`), which is MATLAB's default and
reduces edge artefacts in the first seconds, where the baseline is taken.

## What did not change

The statistic itself: power-weighted mean instantaneous frequency over 1–35 Hz,
compared median-of-recording against median-of-first-five-seconds. The NFI
thresholds (15 % and 35 %) are unchanged and remain provisional.

The score mapping is now written once, as `score_from_nfi`, and the traffic light
is derived from the score by `status_from_score`, so the number and the colour
cannot disagree. The mapping reproduces the previous piecewise formula.

`bowen_centroid.py` is unchanged and still runs alongside, on the same transform.
Its 8–23 Hz centroid is reported to the athlete but does not drive the traffic
light; `DESIGN.md` says why.

## How to re-run the comparison

`tests/test_engine.py` pins the behaviour that matters (a 20 Hz tone must read
20 Hz at any device rate, and the colour must match the score). The equivalence
table above was produced with the probe scripts used during review; the short
version is:

```bash
python3 - <<'PY'
# compare aggregate vs single transform on any channel
PY
```

If the aggregate ever needs to come back, it is one function: restore the
band-filter loop in `engine.transform`.

## Filter and aggregate, restored

The pipeline now runs the method as named: 34 overlapping bands between 1 and
35 Hz (a band centred at c covers c−1 to c+2 Hz), each band gets its own
synchrosqueezed transform, and the band powers are summed onto one 0.5 Hz grid.
The single whole-band transform described above is still in the file as
`transform_single`, selectable with `FAST_TRANSFORM=single`, and is what the
equivalence table in `scripts/compare_transform.py` compares against.

Two things make it affordable. Each band is decimated to eight times its own
top edge before transforming — a 1–3 Hz band is transformed at 24 Hz, not
200 Hz, and a band cannot hold content above its edge, so nothing is lost. And
the bands are transformed in parallel (6 workers by default,
`FAST_FA_WORKERS`).

Measured on the bundled demo, eight channels, 30 s at 1 kHz:

| pipeline | per channel | four minutes of audio |
| --- | --- | --- |
| 35-band reference (ssqueezepy, full rate, 32 voices) | ~17 s | — |
| filter and aggregate, as shipped (wsstpy, decimated, 16 voices) | 2.0 s | 8 channels in 16.5 s |

Agreement with the reference on the demo, NFI in per cent:

| muscle | reference | shipped |
| --- | --- | --- |
| VL | −0.0 | −0.0 |
| VM | −0.0 | −0.0 |
| RF | 0.0 | 0.0 |
| BF | 8.0 | 8.3 |
| ST | 9.5 | 12.8 |
| TA | 0.0 | −0.0 |
| GM | 41.9 | 40.6 |
| GL | 41.5 | 42.7 |

Same status on all eight channels. ST is the largest difference at 3.3 points,
which does not move its class.

The aggregate's frames follow the slowest band, so each band is put back on the
recording's own time base before summing; otherwise a 30 s recording would be
measured as 3.75 s long and rejected by the duration guard.
