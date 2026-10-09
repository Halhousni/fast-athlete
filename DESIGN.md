# FAST ATHLETE — Design Decisions

Filter and Aggregate Synchrosqueezed Transform. This file records how the app
works now and why the parts were chosen. It is not a user guide, see README.md
for that.

## The pipeline

`_run_fast()` in `app.py` does the following per muscle channel.

1. Decimate to 200 Hz with an IIR decimator if the recording is faster.
2. Take one synchrosqueezed CWT of the whole decimated signal, Morlet wavelet
   with mu = 6 and 32 voices per octave.
3. Build an aggregate transform by walking centre frequencies from 1 Hz to 35 Hz
   in 1 Hz steps. For each centre, bandpass the signal, run a second SSQ-CWT over
   that narrow band, and accumulate. The accumulated transform is divided by the
   number of valid bands.
4. Mask the aggregate by the full-band transform, keeping only points where the
   full-band power exceeds 0.9 times its mean. This suppresses the low-power
   noise floor before any frequency statistic is computed.

The output is a time by frequency complex matrix plus the frequency axis.

## Fatigue classification

`_classify_fatigue()` computes the power-weighted mean instantaneous frequency
per time frame over 1 to 35 Hz, then compares the median of the whole recording
against the first 5 seconds.

```
NFI = (MIF_baseline - MIF_median) / MIF_baseline * 100
```

A downward shift of the EMG spectrum is the standard signature of fatigue, so a
positive NFI means the spectrum moved down during the recording.

| NFI | status | score |
|-----|--------|-------|
| under 15 % | green, no fatigue detected | `100 - 2 * NFI` |
| 15 to 35 % | amber, early fatigue signs | `100 - 2 * NFI` |
| over 35 % | red, significant fatigue | `100 - 2 * NFI`, floored at 0 |

The score is clamped to 0 to 100, and the frontend reads it as 70 and above
recovered, 40 to 69 some fatigue, below 40 needs rest.

The baseline is the subject's own opening seconds of the same contraction, so the
method is within-subject. It does not compare an athlete to a population.

**These thresholds are provisional.** They await clinical validation from Prof
Samit Chakrabarty at the University of Leeds.

## Bowen LOW centroid

`bowen_centroid.py` implements the LOW 8 to 23 Hz centroid described in "FAST LOW
8–23 Hz Centroid Calculation" (Bowen, 22 July 2026). It runs alongside the MIF
classification through `run_bowen_pipeline()` and returns a `BowenResult`.

It computes per time point the LOW centroid, LOW mass, LOW fraction and LOW MDF,
with a full validity mask, then derives phase means, split at a 60 percent
contraction fraction into CON and ECC phases, early, middle and late centroid
values, and whole-phase trajectories sampled at 101 points. A 1500 sample pad is
removed from the centroid before the phase split.

The MIF path gives a single status per muscle. The Bowen path gives the shape of
the change over the contraction, which is what the trend and trajectory views use.

## Why decimation to 200 Hz

The fatigue statistic uses only content below 37 Hz after bandpass filtering, so
Nyquist requires just over 74 Hz. Decimating 1500 Hz to 200 Hz leaves nearly 3x
headroom and drops no information in the band of interest. Lowpass and downsample
before spectral estimation is standard practice in the EMG fatigue literature. The
measured speedup is about 7.5x.

## Why ssqueezepy and not a self-built GPU transform

The original build carried a hand-written WSST with a GPU backend
the GPU. On that card, an older generation with few shader cores, it only offloaded the Morlet
kernel and left every FFT on the CPU, so kernel launch overhead likely ate the
gain. ssqueezepy offers a published implementation with a DOI and CI, several
wavelet families, and a multi-threaded CPU path that measured faster on the same
practice. The GPU code and its container dependencies have since been removed
entirely and the app runs CPU only.

## Why the bands run sequentially

A `ThreadPoolExecutor` with four workers was tried for concurrent band processing
and crashed with "process terminated abruptly". NumPy plus forked processes cause
BLAS threading deadlocks. Speed comes instead from decimation and from analyzing
fewer muscles, not from parallelism.

## Frontend

The entire interface is one HTML page embedded in `app.py` as `HTML_PAGE`. There
is no template engine, no build step, no JavaScript framework and no bundler. The
page is a plain Python string, so it cannot be an f-string, and dynamic markup is
assembled in JavaScript.

Material 3 dark theme by default, with a light variant driven by
`:root[data-theme="light"]` token overrides. The choice persists in
`localStorage` under `fast_theme`, and a small inline script in the head applies
it before first paint so there is no flash. Inter is loaded from Google Fonts.

Mobile first, capped at 480 px and widened in two breakpoints. Navigation is a
fixed bottom bar with five screens, welcome, upload, history, train and profile.

The muscle map is a canvas widget from MuscleMapJS, bundled into the page as a
minified IIFE with esbuild. The local copy carries one added method,
`highlightSide(muscle, side, color)`, so a single leg can be coloured rather than
both. Channel to region mapping treats vastus lateralis and vastus medialis as
distinct regions, which is the reason for choosing this library over the
alternatives. The smaller thumbnails on the selection grid and result cards use
inline SVG paths instead.

## Profiles and storage

Athlete profiles and saved assessments live in SQLite through stdlib `sqlite3`
only, no extra dependency. The database sits at `$FAST_DATA_DIR/fast.db`, which
defaults to `/data`, and in the container that path is a named volume.

A profile PIN is stored as `sha256('fast|' + pin)` and is optional. `has_pin` is
derived from whether the hash is null rather than stored separately.

Uploaded signals are held in a temporary directory and an in-memory dictionary,
keyed by a hash of the filename. They are deliberately ephemeral and do not
survive a container restart, since an upload belongs to one assessment.

## File format support

- `.mat`, read with `scipy.io`, sampling rate taken from a variable matching
  `fs`, `samp` or `rate`
- `.txt` and `.csv`, sampling rate parsed from the first 20 lines by regex, so
  headers reading `Frequency` as well as `fs` and `sample rate` work
- separator detection across comma, tab and semicolon
- up to 5 leading metadata rows skipped, which covers the four-row MR32 header
- muscle channels chosen by excluding time, marker, trigger, sync, ref, event,
  frame, sample and unit-only columns
- column names are matched to muscles by full name first, then by abbreviation,
  and an abbreviation only matches as a whole token, so `ST` cannot match inside
  `VASTUS LATERALIS`

## Limits

- 30 seconds per channel. Longer recordings are truncated, not resampled.
- Thresholds are provisional and not clinically validated.
- The score is relative to the opening seconds of the same recording, so a
  recording that starts already fatigued will read as recovered.
- The interface is built for two legs and seven arm channels. Other channel sets
  will analyze correctly but may not map to a highlighted region.

## Deployment

Not documented in this repository. The app is deployed as a Docker container
serving port 8501, built from `current build/Dockerfile`.
