"""Reading an athlete's recording: bytes in, channels and a sampling rate out.

Everything that can go wrong with a device export happens here, so the rules
are explicit:

* The sampling rate is never invented. It comes from a value the athlete
  supplies, from a key-value line in the file, or from a time column. If none
  of those exist, `Ingest.fs` is None and the caller must ask for it.
* A time column is recognised and used, not silently analysed as a muscle.
* Channels that are obviously not EMG (accelerometer, gyroscope, force, angle,
  marker, trigger) are dropped instead of being offered for analysis.
* Short gaps are interpolated. A channel is only rejected when a substantial
  part of it is missing, and the reason is reported.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.io

MAX_GAP_FRACTION = 0.01     # interpolate up to 1 % of samples
MAX_GAP_SECONDS = 0.05      # ... and never bridge a gap longer than 50 ms

MUSCLES = [
    dict(id='VL', name='Vastus Lateralis', desc='Outer quad'),
    dict(id='VM', name='Vastus Medialis', desc='Inner quad'),
    dict(id='RF', name='Rectus Femoris', desc='Front quad'),
    dict(id='BF', name='Biceps Femoris', desc='Hamstring'),
    dict(id='ST', name='Semitendinosus', desc='Inner hamstring'),
    dict(id='TA', name='Tibialis Anterior', desc='Shin'),
    dict(id='GM', name='Gastrocnemius Med.', desc='Inner calf'),
    dict(id='GL', name='Gastrocnemius Lat.', desc='Outer calf'),
    dict(id='MD', name='Middle Deltoid', desc='Shoulder'),
    dict(id='BB', name='Biceps Brachii', desc='Front arm'),
    dict(id='BR', name='Brachioradialis', desc='Outer forearm'),
    dict(id='FCR', name='Flexor Carpi Radialis', desc='Inner forearm'),
    dict(id='ECR', name='Extensor Carpi Radialis', desc='Outer forearm'),
    dict(id='FDI', name='First Dorsal Interosseous', desc='Hand'),
    dict(id='FDS', name='Flexor Digitorum Superficialis', desc='Forearm'),
]

# Full names are checked before abbreviations so 'VASTUS LATERALIS' matches VL
# instead of the 'ST' inside it. Abbreviations match only as a whole token or
# anchored to digits ('VL1', '1VL').
MUSCLE_ALIASES = [
    ('VL', ['VASTUS LATERALIS', 'VL']),
    ('VM', ['VASTUS MEDIALIS', 'VM']),
    ('RF', ['RECTUS FEMORIS', 'RF']),
    ('BF', ['BICEPS FEMORIS', 'BF']),
    ('ST', ['SEMITENDINOSUS', 'ST']),
    ('TA', ['TIBIALIS ANTERIOR', 'TA']),
    ('GM', ['GASTROCNEMIUS MEDIALIS', 'GASTROCNEMIUS MED', 'GM']),
    ('GL', ['GASTROCNEMIUS LATERALIS', 'GASTROCNEMIUS LAT', 'GL']),
    ('MD', ['MIDDLE DELTOID', 'MID DELTOID', 'MID DELT', 'MD']),
    ('BB', ['BICEPS BRACHII', 'BICEPS BR', 'BB']),
    ('BR', ['BRACHIORADIALIS', 'BRACHIORAD', 'BR']),
    ('FCR', ['FLEXOR CARPI RADIALIS', 'FLEX CARP R', 'FCR']),
    ('ECR', ['EXTENSOR CARPI RADIALIS', 'EXT CARP RAD', 'EXT CARP R', 'ECR']),
    ('FDI', ['FIRST DORSAL INTEROSSEOUS', 'FDI']),
    ('FDS', ['FLEXOR DIGITORUM SUPERFICIALIS', 'FDS']),
]

# Columns that are not EMG. Time-like columns are detected separately, then
# excluded from analysis but kept for rate estimation.
NON_EMG = re.compile(
    r'\btime\b|\bsec\b|^t$|^x$|^x\[s\]$|_time|time_|marker|trigger|sync|\bref\b|'
    r'event|frame|\bsample\b|acc|gyro|imu|\bforce\b|torque|angle|\bmag\b|'
    r'heart|\becg\b|resp|^(?:µV|uV|mV|V)(?:[._ ]?\d+)?$',
    re.I)

TIME_NAMES = re.compile(r'^(time|t|sec|secs|seconds|timestamp|x|x\[s\]|'
                        r'time\s*\(s\)|time_?s?|elapsed)$', re.I)

FS_KEYS = re.compile(r'^(fs|f_s|fs_hz|srate|sr|sampling_?rate|sample_?rate|rate)$', re.I)
FS_IN_TEXT = re.compile(
    r'(?:^|\b)(?:fs|f_s|sampling\s*(?:rate|freq(?:uency)?)|sample\s*(?:rate|freq(?:uency)?))'
    r'\s*[:=]?\s*(\d+(?:\.\d+)?)', re.I)

FS_MIN, FS_MAX = 100.0, 20000.0


class IngestError(ValueError):
    """A recording that cannot be used, with a message meant for the athlete."""


@dataclass
class Ingest:
    channels: list[dict] = field(default_factory=list)   # {name, data, dropped}
    fs: float | None = None
    fs_source: str = 'none'      # 'file' | 'time-column' | 'none'
    n_samples: int = 0
    notes: list[str] = field(default_factory=list)


def _norm_col(s) -> str:
    return re.sub(r'\s+', ' ', str(s).upper().strip()
                  .replace('_', ' ').replace('-', ' ').replace('.', ' ')
                  .replace('(', ' ').replace(')', ' ').replace(',', ' ')).strip()


def _decode(content: bytes) -> str:
    for enc in ('utf-8-sig', 'utf-16', 'utf-8', 'latin-1'):
        try:
            return content.decode(enc)
        except UnicodeDecodeError:
            continue
    return content.decode('utf-8', errors='replace')


def _fs_from_text(text: str) -> float | None:
    """A key-value line in the first 20 lines, e.g. 'Sampling rate: 2000'."""
    for line in text.splitlines()[:20]:
        m = FS_IN_TEXT.search(line)
        if m:
            try:
                value = float(m.group(1))
            except ValueError:
                continue
            if FS_MIN <= value <= FS_MAX:
                return value
    return None


def _read_table(content: bytes) -> pd.DataFrame:
    """Pick the parse that yields the most numeric columns, cheaply.

    Every separator is scored rather than the first one that parses, because a
    comma-decimal semicolon file also parses as one bad comma-separated column.
    The C engine is used with a few skip rows; the slower python engine is only
    reached when nothing numeric comes back. A units row under the header is
    handled by the caller, which retries with one more skipped row.
    """
    text = _decode(content)
    head = text[:4096]
    counts = {sep: head.count(sep) for sep in (',', '\t', ';')}
    if not any(counts.values()):
        raise IngestError('No numeric columns found in this file. Save the '
                          'recording as CSV, TXT or MAT.')
    order = sorted(counts, key=lambda s: -counts[s])

    best, best_score = None, (0, 0)
    for sep in order:
        for skip in range(0, 4):
            try:
                df = pd.read_csv(io.BytesIO(content), sep=sep, skiprows=skip,
                                 on_bad_lines='skip',
                                 decimal=',' if sep != ',' else '.')
            except Exception:
                continue
            numeric = df.select_dtypes(include=[np.number])
            score = (len(numeric.columns), len(numeric))
            if score > best_score:
                best, best_score = df, score

    if best is not None and best_score[0] >= 1 and best_score[1] > 1:
        return best

    # Fall back: a units row or ragged metadata makes every column an object.
    for skip in range(0, 7):
        try:
            df = pd.read_csv(io.BytesIO(content), sep=None, engine='python',
                             skiprows=skip, on_bad_lines='skip')
        except Exception:
            continue
        df = df.apply(pd.to_numeric, errors='coerce')
        numeric = df.dropna(axis=1, how='all')
        if len(numeric.columns) and len(numeric) > 1:
            return numeric
    raise IngestError('No numeric columns found in this file')


def _from_mat(content: bytes) -> tuple[pd.DataFrame, float | None, list[str]]:
    notes = []
    try:
        mat = scipy.io.loadmat(io.BytesIO(content))
    except Exception as exc:
        raise IngestError(f'Could not read the .mat file ({exc})') from exc

    fs = None
    for key in mat:
        if key.startswith('__'):
            continue
        if FS_KEYS.match(key):
            try:
                value = float(np.asarray(mat[key]).ravel()[0])
            except Exception:
                continue
            if FS_MIN <= value <= FS_MAX:
                fs = value
                break

    series = {}
    for key, value in mat.items():
        if key.startswith('__') or not np.issubdtype(np.asarray(value).dtype, np.number):
            continue
        arr = np.asarray(value)
        if arr.size <= 1:
            continue                      # a scalar carries no signal
        if arr.ndim == 1:
            series[key] = arr
        elif arr.ndim == 2:
            # (samples, channels) when the long axis is first, else transpose
            block = arr if arr.shape[0] >= arr.shape[1] else arr.T
            for i in range(block.shape[1]):
                series[f'{key}_{i + 1}'] = block[:, i]
            if block.shape[1] > 1:
                notes.append(f'{key} was read as {block.shape[1]} channels')
        else:
            notes.append(f'{key} has {arr.ndim} dimensions and was skipped')

    if not series:
        raise IngestError('No numeric channels found in the .mat file')
    length = min(len(v) for v in series.values())
    df = pd.DataFrame({k: v[:length] for k, v in series.items()})
    return df, fs, notes


def _clean_series(s: pd.Series, fs: float | None) -> tuple[np.ndarray, str | None]:
    """Interpolate short gaps; return the data and a drop reason if unusable."""
    arr = pd.to_numeric(s, errors='coerce').to_numpy(dtype=float)
    missing = ~np.isfinite(arr)
    n_missing = int(missing.sum())
    if n_missing == 0:
        return arr, None

    fraction = n_missing / len(arr)
    max_gap = 1
    if fs:
        max_gap = max(1, int(MAX_GAP_SECONDS * fs))
    run = longest = 0
    for flag in missing:
        run = run + 1 if flag else 0
        longest = max(longest, run)

    if fraction > MAX_GAP_FRACTION or longest > max_gap:
        return arr, (f'{n_missing} of {len(arr)} samples are missing '
                     f'(longest run {longest})')
    filled = pd.Series(arr).interpolate(limit_direction='both').to_numpy()
    return filled, None


def _fs_from_time_column(df: pd.DataFrame) -> tuple[float | None, str | None]:
    """Find a time column and read the rate off it."""
    for col in df.columns:
        name = str(col).strip()
        numeric = pd.to_numeric(df[col], errors='coerce').to_numpy(dtype=float)
        if np.isfinite(numeric).sum() < 3:
            continue
        diffs = np.diff(numeric[np.isfinite(numeric)])
        if len(diffs) == 0:
            continue
        positive = diffs[diffs > 0]
        if len(positive) < 0.9 * len(diffs):
            continue
        median_step = float(np.median(positive))
        uniform = np.allclose(positive, median_step, rtol=0.02)
        if not (TIME_NAMES.match(name) or (uniform and median_step < 1.0)):
            continue
        if median_step <= 0:
            continue
        fs = 1.0 / median_step
        if FS_MIN <= fs <= FS_MAX:
            return fs, col
    return None, None


def load(content: bytes, filename: str) -> Ingest:
    """Parse a recording. Raises IngestError with a message fit for the athlete."""
    name = (filename or '').lower()
    notes: list[str] = []
    fs = None
    fs_source = 'none'

    if name.endswith('.mat'):
        df, fs, mat_notes = _from_mat(content)
        notes += mat_notes
        if fs is not None:
            fs_source = 'file'
    else:
        text = _decode(content)
        fs = _fs_from_text(text)
        if fs is not None:
            fs_source = 'file'
        df = _read_table(content)

    df.columns = [str(c).replace('"', '').strip() for c in df.columns]
    df = df.dropna(axis=1, how='all')
    if df.empty or len(df) < 10:
        raise IngestError('The file holds fewer than ten rows of data')

    # A units row under the header leaves columns called mV, mV.1, ... Read
    # again with that row skipped.
    if all(NON_EMG.search(str(c)) or TIME_NAMES.match(str(c).strip())
           for c in df.columns) and not name.endswith('.mat'):
        for extra in (1, 2):
            try:
                # skiprows=[1] drops the units row and keeps the header.
                retry = pd.read_csv(io.BytesIO(content), skiprows=list(range(1, extra + 1)),
                                    on_bad_lines='skip', sep=None, engine='python')
            except Exception:
                break
            retry.columns = [str(c).replace('"', '').strip() for c in retry.columns]
            if not all(NON_EMG.search(str(c)) or TIME_NAMES.match(str(c).strip())
                       for c in retry.columns):
                df = retry
                break

    if fs is None:
        fs, time_col = _fs_from_time_column(df)
        if fs is not None:
            fs_source = 'time-column'
            notes.append(f'Read {fs:.0f} Hz from the "{time_col}" column')

    # Drop time and non-EMG columns now that the rate has been read.
    keep = []
    for col in df.columns:
        if TIME_NAMES.match(str(col).strip()) or NON_EMG.search(str(col)):
            continue
        if pd.to_numeric(df[col], errors='coerce').notna().sum() < len(df) * 0.5:
            continue
        keep.append(col)
    if not keep:
        raise IngestError('No EMG channels found. Columns named time, marker, '
                          'trigger or accelerometer are skipped.')

    channels = []
    for col in keep:
        data, reason = _clean_series(df[col], fs)
        channels.append(dict(name=str(col), data=data, drop_reason=reason))
    usable = [c for c in channels if c['drop_reason'] is None]
    if not usable:
        raise IngestError('None of the channels holds usable data')

    return Ingest(channels=channels, fs=fs, fs_source=fs_source,
                  n_samples=max(len(c['data']) for c in usable), notes=notes)


def match_muscles(names: list[str]) -> list[dict]:
    """Known muscles in file order, then unmatched channels as plain channels."""
    known = {m['id'].upper(): m for m in MUSCLES}
    alias_order = sorted(((mid, a) for mid, aliases in MUSCLE_ALIASES for a in aliases),
                         key=lambda x: -len(x[1]))
    matched, used = [], set()
    for name in names:
        cu = _norm_col(name)
        if not cu:
            continue
        for mid, alias in alias_order:
            if ' ' in alias:
                hit = (cu == alias or cu.replace(' ', '') == alias.replace(' ', '')
                       or alias in cu)
            else:
                hit = (cu == alias or any(t == alias for t in cu.split())
                       or (cu.startswith(alias)
                           and (len(cu) == len(alias) or cu[len(alias)].isdigit()))
                       or (cu.endswith(alias) and len(cu) > len(alias)
                           and cu[len(cu) - len(alias) - 1].isdigit()))
            if hit:
                if mid in used:
                    break
                used.add(mid)
                matched.append(dict(id=mid, name=known[mid]['name'],
                                    desc=known[mid]['desc'], column=name,
                                    recognised=True))
                break
    for name in names:
        if any(m['column'] == name for m in matched):
            continue
        matched.append(dict(id=name, name='Unrecognised channel', desc=name,
                            column=name, recognised=False))
    return matched
