import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.replay_health import calculate_health, register_replay
from app.storage import Store
from app.analyzer import parse_map
from test_app import chart


def setup_replay(tmp_path):
    store = Store(tmp_path / 'health.db')
    store.save_map(parse_map(chart(100)))
    store.save_play('Player', '100', 'NF', .8, False, play_id='nf-id', passed_known=False)
    path = tmp_path / 'replay.osr'
    path.write_bytes(b'header fixture')
    store.save_local_replay('nf-id', path, hashlib.sha256(path.read_bytes()).hexdigest())
    return store, path


def test_engine_result_cached_and_no_counterfactual_play_is_inserted(tmp_path, monkeypatch):
    import app.replay_health as health
    store, path = setup_replay(tmp_path)
    monkeypatch.setattr(health, '_engine', lambda: ('dotnet', Path('helper.dll')))
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        output = Path(command[-1])
        output.write_text(json.dumps({'status': 'verified', 'hp': [{'time_ms': 1000, 'hp': .2}],
                                     'without_nf_fail_time_ms': 2000, 'minimum_hp': 0}), encoding='utf-8')
        class Result: returncode = 0
        return Result()
    monkeypatch.setattr(health.subprocess, 'run', run)
    first = calculate_health(store, 'Player', 'nf-id')
    assert first['verified_counterfactual'] and not first['cached']
    assert calculate_health(store, 'Player', 'nf-id')['cached'] and len(calls) == 1
    assert store.play_count('Player') == 1 and not store.plays('Player')[0]['passed_known']
    assert path.read_bytes() == b'header fixture'


def test_modified_replay_and_wrong_user_are_rejected(tmp_path):
    store, path = setup_replay(tmp_path)
    with pytest.raises(ValueError):
        calculate_health(store, 'Other', 'nf-id')
    path.write_bytes(b'changed')
    with pytest.raises(ValueError):
        calculate_health(store, 'Player', 'nf-id')


def test_public_mode_cannot_read_or_run_local_nf_analysis(tmp_path):
    client = TestClient(create_app(tmp_path / 'public.db', public_mode=True))
    assert client.get('/api/local/nf-replays', params={'user': 'Player'}).status_code == 403
    assert client.post('/api/local/nf-health', json={'user': 'Player', 'play_id': 'id'}).status_code == 403
    assert client.post('/api/local/nf-replay', data={'user': 'Player'}, files={'file': ('r.osr', b'x')}).status_code == 403


@pytest.mark.parametrize('status,complete,mods', [('statistics_mismatch', True, ['NF']),
                                                ('verified', False, ['NF']),
                                                ('verified', True, ['NF', 'EZ'])])
def test_incomplete_or_mismatched_playback_never_confirms_counterfactual(tmp_path, monkeypatch, status, complete, mods):
    import app.replay_health as health
    store, _ = setup_replay(tmp_path)
    if not complete:
        with store._connection() as db:
            db.execute("UPDATE plays SET complete=0 WHERE id='nf-id'")
    monkeypatch.setattr(health, '_engine', lambda: ('dotnet', Path('helper.dll')))
    def run(command, **kwargs):
        Path(command[-1]).write_text(json.dumps({'status': status, 'mods': mods, 'hp': [], 'minimum_hp': 0,
                                                'without_nf_fail_time_ms': 1000}), encoding='utf-8')
        class Result: returncode = 0
        return Result()
    monkeypatch.setattr(health.subprocess, 'run', run)
    assert not calculate_health(store, 'Player', 'nf-id')['verified_counterfactual']
