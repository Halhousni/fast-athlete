# FAST Web App — Design Decisions

## Why decimation to 200 Hz

The fatigue metric uses only the 1–35 Hz frequency band. After bandpass filtering around each centre frequency, the signal contains content exclusively below 37 Hz. Nyquist requires >74 Hz. Decimating from 1500 Hz to 200 Hz gives Nyquist = 100 Hz — nearly 3× headroom. Zero information loss in the fatigue band. EMG fatigue papers standardly lowpass-filter and downsample before computing spectral metrics. The speedup is ~7.5× on the the build server.

## Why ssqueezepy over a self-built transform

| | ssqueezepy | self-built |
|---|---|---|
| Validation | 799 ★, Zenodo DOI, CI | None (self-built) |
| Wavelet types | GMW, Morlet, bump, custom | Morlet only |
| CPU parallelism | Built-in multi-threaded | Single-thread NumPy |
| GPU (discrete) | Yes | Yes |
| GPU (older cards) | No | Yes |

The self-built backend was the whole point of the original. That backend offloaded the Morlet kernel to GPU — all FFTs stayed on CPU. The kernel launch overhead likely erased any gain. ssqueezepy's multi-threaded CPU path measured faster in practice.

## Why NiceGUI over Streamlit

| | Streamlit | NiceGUI |
|---|---|---|
| Re-renders | Full page on every interaction | Element-level updates |
| State | session_state (fragile) | Native Python variables |
| Memory per user | ~200 MB | ~50 MB |
| Dependencies | ~50 packages | ~5 packages |
| Production server | Needs proxy | Built-in FastAPI |

## Why sequential bands (not parallel)

Attempted `ThreadPoolExecutor` with 4 workers for concurrent band processing. Crashed with "process terminated abruptly" — NumPy + forked processes in `run.cpu_bound` cause BLAS threading deadlocks. Speedup is instead achieved through: (a) decimation to 200 Hz, (b) muscle selection (analyze fewer muscles).

## Fatigue classification

Current: power-weighted mean instantaneous frequency (MIF) across 1–35 Hz, median across time frames. Hard thresholds:
- MIF ≥ 18 Hz → green (not fatigued)
- MIF ≥ 12 Hz → amber (some fatigue)
- MIF < 12 Hz → red (fatigued)

These are provisional placeholders pending clinical validation from Prof Samit Chakrabarty (University of Leeds). Recommended future approach: within-subject baseline normalization (compare current MIF to subject's own unfatigued MIF from first 5s, classify by % drop).

## File format support

- .mat (MATLAB): auto-detects fs from variable named 'fs'/'samp'/'rate'
- .txt/.csv: auto-detects fs from header lines containing 'Frequency', 'fs', 'sample rate'
- Auto-detects metadata skip rows (MR32 format with 4-line header)
- Separator auto-detection: tab, comma, semicolon
- Muscle columns detected by excluding time/marker/trigger/sync/ref/event/frame/sample

## Architecture

```
app_nicegui.py  (517 lines)
├── Engine (lines 1-96)
│   ├── _process_single_band()    — bandpass filter + WSST per frequency
│   ├── _fast_aggregate()         — averages filtered WSST across bands
│   ├── _run_fast()               — full pipeline incl. decimation
│   ├── _load_file()              — .mat/.txt/.csv parser
│   ├── _muscle_columns()         — auto-detect muscle channels
│   ├── _classify_fatigue()       — MIF + traffic light thresholds
│   └── _traffic_light_html()     — CSS traffic light card
├── CSS (lines 200-280)
│   └── GLOBAL_CSS                — DM Sans/Serif fonts, cards, traffic lights
├── UI (lines 284-503)
│   ├── main_page()               — single @ui.page, upload → controls → results
│   ├── _build_controls()         — muscle checkboxes, time slider, run button
│   └── _show_results()           — traffic lights, SVG spectrograms, CSV download
└── Entry (lines 506-510)
    └── ui.run()                   — host 0.0.0.0:8501
```

## Deployment pipeline

See PIPELINE.md. Summary: `git push deploy` → post-receive hook → `docker compose up --build -d`. The app serves on port 8501.

**Note — this file describes an earlier build.** The structure above is the NiceGUI
version. The current app is FastAPI serving one embedded HTML page. See README.md
for the current layout.

## Known limitations

- MIF thresholds are provisional placeholders
- No within-subject baseline normalization
- Spectrograms are researcher-oriented; not athlete-facing
- No historical tracking or session comparison
- Server build takes ~5 min on the build server (cached: ~30s for code-only changes)
