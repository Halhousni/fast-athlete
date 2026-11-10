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

# Above this size the slow python parser is not tried at all: it is the only
# part of reading a file whose cost grows with the whole file, and one
# unauthenticated upload must not be able to stall the container.
MAX_FALLBACK_BYTES = 2_000_000

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

# A header cell that is only a unit ('mV', '(mV)', 's', 'X[s]') means the real
# header is one row above: many exports put units under the channel names.
UNIT_ONLY = re.compile(
    r'^[\s(\[]*(?:µ?V|uV|mV|kV|A|mA|N|Nm|N·m|deg|°|rad|s|ms|samples?|%|force|'
    r'm?\/?s\^?2?|g|Hz)[\s)\]]*(?:[._]?\d+)?$', re.I)

# Columns that are not EMG. Time-like columns are detected separately, then
# excluded from analysis but kept for rate estimation.
NON_EMG = re.compile(
    r'\btime\b|\bsec\b|^t$|^x$|^x\[s\]$|_time|time_|marker|trigger|sync|\bref\b|'
    r'event|frame|\bsample\b|acc|gyro|imu|\bforce\b|torque|angle|\bmag\b|'
    r'heart|\becg\b|resp|^(?:µV|uV|mV|V)(?:[._ ]?\d+)?$',
    re.I)

TIME_NAMES = re.compile(r'^(?:time|t|sec|secs|seconds|timestamp|x|x\[s\]|'
                        r'time\s*\(s\)|time_?s?|elapsed|zeit|temps|tiempo)$', re.I)

FS_KEYS = re.compile(r'^(fs|f_s|fs_hz|srate|sr|sampling_?rate|sample_?rate|rate)$', re.I)
FS_IN_TEXT = re.compile(
    r'(?:^|\b)(?:fs|f_s|sampling\s*(?:rate|freq(?:uency)?)|sample\s*(?:rate|freq(?:uency)?))'
    r'\s*[:=]?\s*(\d+(?:\.\d+)?)', re.I)

FS_MIN, FS_MAX = 100.0, 20000.0


class IngestError(ValueError):
    """A recording that cannot be used, with a message meant for the athlete."""


@dataclass
class Ingest:
    channels: list[dict] = field(default_factory=list)   # {name, data, drop_reason}
    fs: float | None = None
    fs_source: str = 'none'      # 'given' | 'file' | 'time-column' | 'none'
    n_samples: int = 0
    notes: list[str] = field(default_factory=list)


def _norm_col(s) -> str:
    return re.sub(r'\s+', ' ', str(s).upper().strip()
                  .replace('_', ' ').replace('-', ' ').replace('.', ' ')
                  .replace('(', ' ').replace(')', ' ').replace(',', ' ')).strip()


def _col_base(name) -> str:
    """The name without pandas' duplicate suffix, so 'X[s].1' still reads as time."""
    return re.sub(r'\.\d+$', '', str(name).strip())


def _decode(content: bytes) -> str:
    if content[:2] in (b'\xff\xfe', b'\xfe\xff'):
        try:
            return content.decode('utf-16')
        except UnicodeDecodeError:
            pass
    for enc in ('utf-8-sig', 'utf-8', 'latin-1'):
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


def _candidate_seps(head: str) -> list[str]:
    counts = {sep: head.count(sep) for sep in (',', '\t', ';')}
    order = [sep for sep in sorted(counts, key=lambda s: -counts[s]) if counts[sep]]
    if order:
        return order
    # No comma, tab or semicolon: space separated (MATLAB save -ascii) or a
    # single column of numbers.
    return [r'\s+']


def _read_table(text: str) -> pd.DataFrame:
    """Pick the parse that yields the most numeric columns.

    Every separator is scored rather than the first one that parses, because a
    comma-decimal semicolon file also parses as one bad comma-separated column.
    Each separator is tried with the header in place and with up to two rows
    skipped under it, which is how a units row is read past.
    """
    head = text[:4096]
    order = _candidate_seps(head)

    best, best_score = None, (0, 0)
    for sep in order:
        whitespace = sep == r'\s+'
        for skip in (0, [1], [1, 2]):
            try:
                df = pd.read_csv(
                    io.StringIO(text), sep=sep, skiprows=skip,
                    on_bad_lines='skip', engine='python' if whitespace else 'c',
                    decimal=',' if sep == ';' else '.')
            except Exception:
                continue
            numeric = df.select_dtypes(include=[np.number])
            score = (len(numeric.columns), len(numeric))
            if score > best_score:
                best, best_score = df, score

    if best is not None and best_score[0] >= 1 and best_score[1] > 1:
        return best

    if len(text) <= MAX_FALLBACK_BYTES:
        # Ragged metadata or a quoted free-text column: let the sniffer decide,
        # then coerce. Bounded to a small file, because this parse is slow.
        for skip in range(0, 7):
            try:
                df = pd.read_csv(io.StringIO(text), sep=None, engine='python',
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
        if np.asarray(value).size <= 1:
            continue                      # a scalar carries no signal
        arr = np.asarray(value)
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

    # Keep the series that share the modal length. A short side array (gains,
    # event markers) used to truncate every channel to its own length.
    lengths = pd.Series([len(v) for v in series.values()])
    length = int(lengths.mode().iloc[0])
    kept, dropped = {}, []
    for key, value in series.items():
        if len(value) == length:
            kept[key] = value
        else:
            dropped.append(f'{key} ({len(value)} samples)')
    if dropped:
        notes.append('Skipped a series of a different length: ' + ', '.join(dropped))
    if not kept:
        raise IngestError('The arrays in the .mat file are of different lengths')
    df = pd.DataFrame({k: v for k, v in kept.items()})
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


def _fs_from_time_column(df: pd.DataFrame) -> tuple[float | None, object | None]:
    """Find a time column and read the rate off it.

    The rate is taken from the first and last sample rather than the median of
    the steps, because a time column written to four decimals rounds each step
    and biases the median (2048 Hz read as 2000 Hz).
    """
    for col in df.columns:
        numeric = pd.to_numeric(df[col], errors='coerce').to_numpy(dtype=float)
        finite = numeric[np.isfinite(numeric)]
        if len(finite) < 3:
            continue
        diffs = np.diff(finite)
        positive = diffs[diffs > 0]
        if len(positive) < 0.9 * len(diffs):
            continue
        median_step = float(np.median(positive))
        if median_step <= 0:
            continue
        uniform = np.allclose(positive, median_step, rtol=0.02)
        base = _col_base(col)
        if not (TIME_NAMES.match(base) or (uniform and median_step < 1.0)):
            continue
        span = float(finite[-1] - finite[0])
        fs = (len(finite) - 1) / span if uniform and span > 0 else 1.0 / median_step
        if FS_MIN <= fs <= FS_MAX:
            return fs, col
    return None, None


def _dedup_columns(cols) -> list[str]:
    seen, out = {}, []
    for col in cols:
        name = str(col).replace('"', '').strip() or 'channel'
        if name in seen:
            seen[name] += 1
            name = f'{name}.{seen[name]}'
        else:
            seen[name] = 1
        out.append(name)
    return out


def load(content: bytes, filename: str, fs: float | None = None) -> Ingest:
    """Parse a recording. Raises IngestError with a message fit for the athlete.

    `fs` is a rate the athlete supplied; it wins over anything in the file and is
    used for gap interpolation, which a file without its own rate cannot do.
    """
    name = (filename or '').lower()
    notes: list[str] = []
    fs_source = 'none'

    if fs is not None and not (FS_MIN <= fs <= FS_MAX):
        fs = None
    if fs is not None:
        fs_source = 'given'

    if name.endswith('.mat'):
        df, file_fs, mat_notes = _from_mat(content)
        notes += mat_notes
        if fs is None and file_fs is not None:
            fs, fs_source = file_fs, 'file'
    else:
        text = _decode(content)
        if fs is None:
            fs = _fs_from_text(text)
            if fs is not None:
                fs_source = 'file'
        df = _read_table(text)

    df.columns = _dedup_columns(df.columns)
    df = df.dropna(axis=1, how='all')
    if df.empty or len(df) < 10:
        raise IngestError('The file holds fewer than ten rows of data')

    if fs is None:
        fs, time_col = _fs_from_time_column(df)
        if fs is not None:
            fs_source = 'time-column'
            notes.append(f'Read {fs:.0f} Hz from the "{time_col}" column')
    else:
        time_col = None
        for col in df.columns:
            if TIME_NAMES.match(_col_base(col)):
                time_col = col
                break

    # Drop time and non-EMG columns now that the rate has been read. The column
    # the rate came from is dropped by identity, so a time column with an
    # unexpected name cannot be offered as a muscle.
    keep = []
    for col in df.columns:
        if col == time_col:
            continue
        base = _col_base(col)
        if TIME_NAMES.match(base) or NON_EMG.search(base) or UNIT_ONLY.match(base):
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
    """Known muscles in file order, then unmatched channels as plain channels.

    An unmatched channel takes its column name as its id, unless that would
    collide with a matched muscle or another channel: ids are what the client
    sends back, so two channels must never share one.
    """
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

    taken = set(used) | set(known)
    for name in names:
        if any(m['column'] == name for m in matched):
            continue
        base = str(name)[:64] or 'channel'
        candidate, n = base, 1
        while candidate in taken:
            n += 1
            candidate = f'{base[:60]}_{n}'
        taken.add(candidate)
        matched.append(dict(id=candidate, name='Unrecognised channel',
                            desc=str(name)[:120], column=name, recognised=False))
    return matched
