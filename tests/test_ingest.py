"""Ingest tests: the rate is never invented, and non-EMG columns are not offered."""
import io
import os
import sys

import numpy as np
import pandas as pd
import pytest
import scipy.io

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'current build'))

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
