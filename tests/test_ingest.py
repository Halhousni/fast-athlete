"""Ingest tests: the rate is never invented, and non-EMG columns are not offered."""
import io
import os
import sys

import numpy as np
import pandas as pd
import pytest
import scipy.io

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

import ingest  # noqa: E402


def csv_bytes(frame: pd.DataFrame, sep: str = ',') -> bytes:
    buf = io.StringIO()
    frame.to_csv(buf, index=False, sep=sep)
    return buf.getvalue().encode()


def emg_frame(fs: float, seconds: float = 10.0, channels=('VL', 'VM')) -> pd.DataFrame:
    n = int(fs * seconds)
    t = np.arange(n) / fs
    data = {'Time': t}
    for i, ch in enumerate(channels):
        data[ch] = np.random.RandomState(i).randn(n) * 0.1
    return pd.DataFrame(data)


def test_rate_is_read_from_a_time_column():
    content = csv_bytes(emg_frame(2000.0))
    got = ingest.load(content, 'rec.csv')
    assert got.fs == pytest.approx(2000.0, rel=0.01)
    assert got.fs_source == 'time-column'
    assert 'Time' not in [c['name'] for c in got.channels]


def test_short_time_column_name_is_not_analysed_as_a_muscle():
    frame = emg_frame(1000.0).rename(columns={'Time': 't'})
    got = ingest.load(csv_bytes(frame), 'rec.csv')
    assert got.fs == pytest.approx(1000.0, rel=0.01)
    assert [c['name'] for c in got.channels] == ['VL', 'VM']


def test_rate_is_read_from_a_header_line():
    header = b'# Recording\n# Sampling rate: 2048 Hz\n'
    content = header + csv_bytes(emg_frame(2048.0))
    got = ingest.load(content, 'rec.csv')
    assert got.fs == pytest.approx(2048.0)
    assert got.fs_source == 'file'


def test_unknown_rate_is_reported_as_unknown():
    frame = emg_frame(1000.0).drop(columns=['Time'])
    got = ingest.load(csv_bytes(frame), 'rec.csv')
    assert got.fs is None
    assert got.fs_source == 'none'


def test_units_row_under_the_header_is_skipped():
    plain = csv_bytes(emg_frame(1000.0))
    lines = plain.split(b'\n')
    with_units = b'\n'.join([lines[0], b'mV,mV,mV'] + lines[1:])
    got = ingest.load(with_units, 'rec.csv')
    assert len(got.channels) == 2


def test_semicolon_and_decimal_comma():
    frame = emg_frame(1000.0)
    text = frame.to_csv(index=False, sep=';', decimal=',').encode()
    got = ingest.load(text, 'rec.csv')
    assert len(got.channels) == 2
    assert got.fs == pytest.approx(1000.0, rel=0.01)


def test_accelerometer_columns_are_dropped():
    frame = emg_frame(1000.0)
    frame['AccX'] = np.random.randn(len(frame)) * 0.01
    frame['Gyro_Z'] = np.random.randn(len(frame)) * 0.01
    got = ingest.load(csv_bytes(frame), 'rec.csv')
    assert [c['name'] for c in got.channels] == ['VL', 'VM']


def test_a_short_gap_is_interpolated():
    frame = emg_frame(1000.0)
    frame.loc[100:104, 'VL'] = np.nan          # 5 ms
    got = ingest.load(csv_bytes(frame), 'rec.csv')
    vl = [c for c in got.channels if c['name'] == 'VL'][0]
    assert vl['drop_reason'] is None
    assert np.isfinite(vl['data']).all()


def test_a_long_gap_drops_the_channel_with_a_reason():
    frame = emg_frame(1000.0)
    frame.loc[100:900, 'VL'] = np.nan          # 800 ms
    got = ingest.load(csv_bytes(frame), 'rec.csv')
    vl = [c for c in got.channels if c['name'] == 'VL'][0]
    assert vl['drop_reason'] is not None
    assert 'missing' in vl['drop_reason']


def test_mat_with_a_sampling_rate_keeps_its_channels():
    """The old loader counted the fs scalar as a channel and truncated every
    other channel to one sample."""
    n = 4000
    emg = np.random.RandomState(0).randn(n, 3) * 0.1
    buf = io.BytesIO()
    scipy.io.savemat(buf, dict(fs=1000, VL=emg[:, 0], VM=emg[:, 1], RF=emg[:, 2]))
    got = ingest.load(buf.getvalue(), 'rec.mat')
    assert got.fs == 1000.0
    assert len(got.channels) == 3
    assert all(len(c['data']) == n for c in got.channels)


def test_mat_matrix_is_split_into_channels():
    n = 4000
    emg = np.random.RandomState(1).randn(n, 4) * 0.1
    buf = io.BytesIO()
    scipy.io.savemat(buf, dict(emg=emg))
    got = ingest.load(buf.getvalue(), 'rec.mat')
    assert len(got.channels) == 4
    assert all(len(c['data']) == n for c in got.channels)


def test_a_single_nan_does_not_lose_the_channel():
    frame = emg_frame(1000.0)
    frame.loc[1234, 'VL'] = np.nan
    got = ingest.load(csv_bytes(frame), 'rec.csv')
    vl = [c for c in got.channels if c['name'] == 'VL'][0]
    assert vl['drop_reason'] is None


def test_empty_file_is_rejected():
    with pytest.raises(ingest.IngestError):
        ingest.load(b'', 'rec.csv')


def test_text_only_file_is_rejected():
    with pytest.raises(ingest.IngestError):
        ingest.load(b'hello\nworld\n', 'rec.csv')


def test_muscle_matching_prefers_full_names():
    matched = ingest.match_muscles(['VASTUS LATERALIS', 'ST', 'Channel 7'])
    ids = [m['id'] for m in matched]
    assert ids[0] == 'VL'
    assert ids[1] == 'ST'
    assert matched[2]['recognised'] is False


def test_muscle_matching_does_not_match_inside_a_word():
    matched = ingest.match_muscles(['STIM'])
    assert matched[0]['recognised'] is False

# ── regression tests for the second review ──────────────────────────────────

def test_metadata_lines_above_the_header_are_not_the_header():
    """A 'Sampling rate' line above the header must not become a channel."""
    body = csv_bytes(emg_frame(2000.0))
    for prefix, expect_fs in ((b'# Recording\n# Sampling rate: 2048 Hz\n', 2048.0),
                              (b'Sampling rate: 2000\n', 2000.0),
                              (b'Device: X\n\n', None)):
        got = ingest.load(prefix + body, 'rec.csv')
        names = [c['name'] for c in got.channels]
        assert names == ['VL', 'VM'], f'{prefix[:24]!r} gave {names}'
        if expect_fs:
            assert got.fs == pytest.approx(expect_fs, rel=0.01)


def test_channel_names_that_look_like_units_are_kept():
    """S1, A1, N1 and G1 are channel names, not unit cells."""
    for channels in (('S1', 'S2'), ('A1', 'A2'), ('N1', 'N2'), ('G1', 'G2')):
        got = ingest.load(csv_bytes(emg_frame(1000.0, channels=channels)), 'rec.csv')
        assert [c['name'] for c in got.channels] == list(channels)


def test_rounded_time_column_keeps_its_rate():
    """Four decimals of a 0.000488 s step must not read as 2000 Hz."""
    n = 6000
    frame = pd.DataFrame({'Time': np.round(np.arange(n) / 2048.0, 4),
                          'VL': np.random.RandomState(0).randn(n) * 0.1})
    got = ingest.load(csv_bytes(frame), 'rec.csv')
    assert got.fs == pytest.approx(2048.0, rel=0.01)


def test_mat_with_one_channel_and_a_short_side_array():
    """A markers or gains array must not truncate the channel."""
    buf = io.BytesIO()
    scipy.io.savemat(buf, {'fs': 2000.0,
                           'emg': np.random.RandomState(0).randn(5000),
                           'markers': np.arange(12)})
    got = ingest.load(buf.getvalue(), 'rec.mat')
    assert got.fs == 2000.0
    assert got.n_samples == 5000
    assert any('markers' in note for note in got.notes)


def test_column_names_that_collide_after_stripping():
    """'A', ' A' and 'A.2' must not all end up called 'A.2'."""
    n = 500
    frame = pd.DataFrame({'A': np.zeros(n), ' A': np.zeros(n), 'A.2': np.zeros(n)})
    got = ingest.load(csv_bytes(frame), 'rec.csv')
    names = [c['name'] for c in got.channels]
    assert len(names) == len(set(names)), names
