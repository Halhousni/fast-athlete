"""Account tests: signup, sign in, throttling, deletion, and the migration.

These are the rules a person meets at the door, so they are tested the way the
door is used: over HTTP, with the cookie jar doing the work a browser would.
"""
import importlib
import os
import sqlite3
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'current build'))

from fastapi.testclient import TestClient  # noqa: E402

GOOD = 'quiet river lantern'


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv('FAST_DATA_DIR', str(tmp_path))
    import app as app_module
    importlib.reload(app_module)          # a store per test, in its own directory
    return TestClient(app_module.app)


def test_signup_signs_you_in_and_issues_a_cookie(client):
    response = client.post('/auth/signup',
                           json={'username': 'athlete', 'password': GOOD})
    assert response.status_code == 200
    assert response.json()['name'] == 'athlete'
    assert 'fast_token' in response.cookies

    me = client.get('/auth/me')
    assert me.status_code == 200
    assert me.json()['name'] == 'athlete'


def test_username_is_case_insensitive_and_unique(client):
    client.post('/auth/signup', json={'username': 'Athlete', 'password': GOOD})
    again = client.post('/auth/signup', json={'username': 'athlete', 'password': GOOD})
    assert again.status_code == 409


def test_short_password_is_refused(client):
    response = client.post('/auth/signup',
                           json={'username': 'athlete', 'password': 'short'})
    assert response.status_code == 400
    assert '10' in response.json()['error']


def test_password_cannot_be_the_username(client):
    response = client.post('/auth/signup',
                           json={'username': 'athlete', 'password': 'athlete'})
    assert response.status_code == 400


def test_username_shape_is_enforced(client):
    for bad in ('ab', 'has space', 'emoji🙂', '-leading'):
        response = client.post('/auth/signup', json={'username': bad, 'password': GOOD})
        assert response.status_code in (400, 422), bad


def test_wrong_password_and_unknown_user_answer_the_same(client):
    client.post('/auth/signup', json={'username': 'athlete', 'password': GOOD})
    client.post('/auth/logout')
    wrong = client.post('/auth/login', json={'username': 'athlete', 'password': 'nope nope nope'})
    unknown = client.post('/auth/login', json={'username': 'nobody', 'password': 'nope nope nope'})
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json()


def test_signing_in_works_and_signing_out_revokes(client):
    client.post('/auth/signup', json={'username': 'athlete', 'password': GOOD})
    client.post('/auth/logout')
    assert client.get('/auth/me').status_code == 401

    back = client.post('/auth/login', json={'username': 'athlete', 'password': GOOD})
    assert back.status_code == 200
    assert client.get('/auth/me').status_code == 200


def test_repeated_failures_are_throttled(client):
    client.post('/auth/signup', json={'username': 'athlete', 'password': GOOD})
    client.post('/auth/logout')
    codes = [client.post('/auth/login',
                         json={'username': 'athlete', 'password': 'wrong each time'}).status_code
             for _ in range(6)]
    assert codes == [401, 401, 401, 401, 401, 429]


def test_saved_results_belong_to_the_account(client):
    client.post('/auth/signup', json={'username': 'athlete', 'password': GOOD})
    assert client.get('/history').json() == {'history': []}

    saved = client.post('/save', json={
        'profile_id': 1,
        'filename': 'test.csv',
        'results': [{'id': 'VL', 'name': 'Vastus lateralis', 'status': 'ok',
                     'score': 82, 'nfi': 0.82, 'tip': '', 'level': 'low',
                     'delta_mdf': -2.0, 'delta_rms': 4.0}],
    })
    assert saved.status_code == 200
    history = client.get('/history').json()['history']
    assert len(history) == 1
    assert history[0]['filename'] == 'test.csv'
    # the server re-derives the colour from the score, whatever the client sent
    assert history[0]['status'] == 'green'


def test_history_needs_a_session(client):
    assert client.get('/history').status_code == 401


def test_deleting_an_account_needs_the_password(client):
    client.post('/auth/signup', json={'username': 'athlete', 'password': GOOD})
    assert client.request('DELETE', '/auth/account',
                          json={'password': 'wrong'}).status_code == 403
    assert client.get('/auth/me').status_code == 200

    gone = client.request('DELETE', '/auth/account', json={'password': GOOD})
    assert gone.status_code == 200
    assert client.get('/auth/me').status_code == 401
    # the name is free again, and the password no longer opens anything
    client.post('/auth/logout')
    assert client.post('/auth/login',
                       json={'username': 'athlete', 'password': GOOD}).status_code == 401


def test_an_older_database_keeps_its_rows_and_gains_password_columns(tmp_path):
    """The live server's database predates accounts. Opening it must not drop it."""
    db = tmp_path / 'fast.db'
    conn = sqlite3.connect(db)
    conn.executescript('''
        CREATE TABLE profiles(
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          name TEXT NOT NULL COLLATE NOCASE UNIQUE,
          pin_hash TEXT, pin_salt TEXT, created_at TEXT NOT NULL);
        CREATE TABLE results(
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          profile_id INTEGER NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
          ts TEXT NOT NULL, avg_score REAL, status TEXT, detail TEXT, engine TEXT);
        CREATE TABLE tokens(
          token_hash TEXT PRIMARY KEY,
          profile_id INTEGER NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
          expires_at REAL NOT NULL);
        INSERT INTO profiles(name, created_at) VALUES('hilal', '2026-11-01T00:00:00+00:00');
        INSERT INTO results(profile_id, ts, avg_score, status, detail, engine)
          VALUES(1, '2026-11-02T00:00:00+00:00', 61, 'warn', '{}', 'wsstpy-0.1.0');
    ''')
    conn.commit()
    conn.close()

    from store import Store
    store = Store(str(tmp_path))
    row = store.account_by_username('hilal')
    assert row is not None and row['password_hash'] is None
    assert store.history(1)[0]['avg_score'] == 61
    # and the column is now there to be filled
    with sqlite3.connect(db) as check:
        cols = {r[1] for r in check.execute('PRAGMA table_info(profiles)')}
    assert {'password_hash', 'password_salt'} <= cols


def test_set_password_adopts_an_account_without_one(tmp_path):
    from store import Store, verify_password
    store = Store(str(tmp_path))
    created = store.create_account('legacy', 'a temporary password')
    assert store.set_password(created['id'], 'a better password')
    row = store.account(created['id'])
    assert verify_password('a better password', row['password_hash'])
    assert not verify_password('a temporary password', row['password_hash'])
    # the cost travels inside the string, so it can be raised later
    assert row['password_hash'].startswith('scrypt$')


def test_a_write_from_another_origin_is_refused(client):
    client.post('/auth/signup', json={'username': 'athlete', 'password': GOOD})
    assert client.post('/auth/logout').status_code == 200
    crossed = client.post('/auth/login',
                          json={'username': 'athlete', 'password': GOOD},
                          headers={'origin': 'https://evil.example'})
    assert crossed.status_code == 403
    same = client.post('/auth/login',
                       json={'username': 'athlete', 'password': GOOD},
                       headers={'origin': 'http://testserver'})
    assert same.status_code == 200


def test_personal_responses_are_not_cached(client):
    client.post('/auth/signup', json={'username': 'athlete', 'password': GOOD})
    assert client.get('/history').headers.get('cache-control') == 'no-store'
    assert client.get('/auth/me').headers.get('cache-control') == 'no-store'
