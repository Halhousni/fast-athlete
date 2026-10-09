# FAST ATHLETE

FAST estimates muscle fatigue from surface EMG recordings. An athlete uploads a
recording, picks the muscles, and gets a green, amber, or red status for each one
with a recovery score out of 100.

**This is a research prototype.** The method is still under study and the scores
are not a clinical or diagnostic measurement.

Live instance: https://fast.ptrehab.icu

## How it works

FAST tracks the downward shift in the EMG power spectrum that happens as a muscle
fatigues. The signal goes through a synchrosqueezed continuous wavelet transform
(SSQ-CWT), the spectrum is aggregated into bands, and the low-frequency centroid
of that band power gives the fatigue index. Recordings are capped at 30 seconds
per channel. `current build/METHOD.md` has the full description.

Results feed a per-muscle traffic light. 70 and above reads as recovered, 40 to
69 as some fatigue, below 40 as needing rest.

## Running locally

```bash
python3 -m venv /tmp/fastvenv
/tmp/fastvenv/bin/pip install fastapi uvicorn python-multipart numpy scipy pandas ssqueezepy
FAST_DATA_DIR=/tmp/fastdata /tmp/fastvenv/bin/python3 "current build/app.py"
```

The app serves on http://localhost:8501. Set `FAST_DATA_DIR` every time. The
default is `/data`, which fails without root.

## Running with Docker

```bash
cd "current build"
docker compose up --build -d
```

The named volume `fast_data` holds the SQLite database at `/data/fast.db`.
Without that volume every saved profile and result disappears on the next deploy.

## Routes

The page is the only client. Everything below is called by the app's own
JavaScript, on the same origin. Nothing else calls it, and the server publishes
no route listing of its own.

- `POST /upload` reads a recording and returns the detected channels
- `POST /analyze` runs FAST on the selected muscles
- `GET /profiles`, `POST /profiles`, `POST /profiles/verify`,
  `DELETE /profiles/{id}` handle athlete profiles and their optional 4 digit PIN
- `POST /save` stores one assessment against a profile
- `GET /history?profile_id=N` returns the last 50 assessments, newest first
- `GET /demo.mp4` and `GET /demo_recording.csv` serve the assets behind the home
  screen demo and the "Try demo data" button

Profiles live in SQLite, created with stdlib `sqlite3` only. The PIN is stored as
`sha256('fast|' + pin)`.

## Layout

```
current build/
  app.py              the server, with the whole UI embedded in it
  bowen_centroid.py   Bowen centroid and the FAST band pipeline
  make_demo_data.py   regenerates the synthetic demo recording
  METHOD.md           the method, in full
  Dockerfile          Python 3.12, CPU only
  docker-compose.yml  container signal-analyzer, port 8501
  demo.mp4            14 s screen recording on the home screen
  demo_recording.csv  synthetic recording behind "Try demo data"
```

## Deployment

The app runs as a Docker container serving port 8501, built from
`current build/Dockerfile`. The deployment setup for this instance is not
documented in this repository.

## Third-party

The muscle map uses MuscleMapJS, MIT licensed and bundled into `app.py`. See
`THIRD_PARTY_NOTICES.md`.

## Licence

MIT. See `LICENSE`.
