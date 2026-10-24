from copy import deepcopy

from fastapi.testclient import TestClient

from app.analyzer import analyze_map, parse_map
from app.capabilities import coverage_warnings
from app.main import create_app
from app.predictor import predict, training_maps
from app.storage import Store
from test_app import chart


def records_for(analysis, count=5):
    return [{'analysis': deepcopy(analysis), 'accuracy': .98, 'passed': True,
             'passed_known': 1, 'weight': 1, 'source': 'manual', 'pp': None, 'map_id': analysis['id'],
             'mods': analysis['mods']} for _ in range(count)]


def test_tapping_demand_uses_notes_and_clock_rate_not_song_bpm():
    sparse = analyze_map(parse_map(chart(1, 300)))
    stream = analyze_map(parse_map(chart(2, 75)))
    faster = analyze_map(parse_map(chart(2, 75)), 'DT')
    assert sparse['bpm'] == stream['bpm']
    assert sparse['stream_bpm'] == 0
    assert stream['stream_bpm'] == 200
    assert faster['stream_bpm'] == 300


def test_saved_manual_limits_no_longer_override_ml(tmp_path):
    store = Store(tmp_path / 'store.db')
    target = analyze_map(parse_map(chart(2, 75)), 'HDDT')
    records = records_for(target)
    store.save_player_limits('Player', {'stream_bpm': 100, 'max_ar': 5, 'hd_min_ar': 11})
    from unittest.mock import patch
    with patch('app.predictor.predict_pass', return_value=(.83, {'status': 'trained'})), \
         patch('app.predictor.predict_accuracy', return_value=(None, {'status': 'insufficient_data'})):
        result = predict(store, 'Player', target, records=records)
        store.save_player_limits('Player', {})
        without = predict(store, 'Player', target, records=records)
    assert result['pass_probability'] == without['pass_probability']
    assert result['model'] == 'personal_history_posterior'
    assert not result['coverage_warnings']


def test_unseen_low_ar_hidden_is_unknown_not_an_automatic_failure():
    target = analyze_map(parse_map(chart(3)))
    records = records_for(target)
    assert not coverage_warnings(target, records)
    target = {**target, 'ar': 7, 'mods': 'HD'}
    warnings = coverage_warnings(target, records)
    assert any(w['skill'] == 'hd_reading' for w in warnings)
    assert all('pass_cap' not in w for w in warnings)


def test_unseen_skill_range_abstains_instead_of_inventing_low_probability(tmp_path):
    store = Store(tmp_path / 'store.db')
    normal = analyze_map(parse_map(chart(4)))
    target = {**normal, 'stars': normal['stars'] + 8, 'aim': normal['aim'] + 8}
    result = predict(store, 'Player', target, records=records_for(normal))
    assert result['pass_probability'] is None
    assert result['prediction_status'] == 'unknown_skill_range'
    assert result['suggest_training'] and not result['too_hard']


def test_public_mode_blocks_local_access_even_via_loopback(tmp_path):
    client = TestClient(create_app(tmp_path / 'public.db', public_mode=True))
    assert client.get('/api/health').json()['public_mode']
    assert client.post('/api/local/import', json={}).status_code == 403
    assert client.get('/api/local/clients').status_code == 403
    assert client.post('/api/session/cookies', json={'content': 'osu_session=never-save'}).status_code == 403


def test_training_maps_filter_accuracy_pass_and_similarity(tmp_path, monkeypatch):
    import app.predictor as predictor
    store = Store(tmp_path / 'store.db')
    for ident in range(10, 15):
        store.save_map(parse_map(chart(ident)))
    target = {**store.analysis('10'), 'stars': 8}
    monkeypatch.setattr(predictor, '_records', lambda *_: records_for(target))
    monkeypatch.setattr(predictor, 'model_context', lambda *_: (None, {}))
    monkeypatch.setattr(predictor, 'accuracy_model_context', lambda *_: (None, {}))
    def prediction(_store, _user, analysis, *args):
        return {'expected_accuracy': 94 if analysis['id'] == '11' else 97,
                'pass_probability': None if analysis['id'] == '13' else 1 if analysis['id'] == '12' else 80,
                'coverage_warnings': []}
    monkeypatch.setattr(predictor, 'predict', prediction)
    result = training_maps(store, 'Player', target)
    assert [item['id'] for item in result['maps']] == ['14']


def test_model_learns_speed_ar_and_hidden_from_attempts_without_probability_caps(tmp_path, monkeypatch):
    import numpy as np
    import json
    from app.ml import train, predict_pass, _artifact_path
    store = Store(tmp_path / 'learned.db')
    base = analyze_map(parse_map(chart(42)))
    rng = np.random.default_rng(42)
    rows = []
    # Synthetic learning fixture only. Stars/aim/speed stay constant so the new
    # pattern and reading features must explain the outcomes on unseen charts.
    for bpm in (120, 140, 160, 180, 200, 240):
        for ar in (6, 7, 8, 9, 10, 11):
            for hidden in (False, True):
                analysis = {**base, 'stream_bpm': bpm, 'burst_bpm': bpm, 'ar': ar,
                            'mods': 'HD' if hidden else 'NM'}
                mid = f'{bpm}-{ar}-{hidden}'
                logit = 6 - .1 * (bpm - 120) - 3 * max(0, ar - 9) ** 2 - hidden * 1.4 * max(0, 8 - ar) ** 2
                for attempt in range(4):
                    rows.append({'id': f'{mid}-{attempt}', 'map_id': mid, 'mods': analysis['mods'],
                                 'source': 'manual', 'passed_known': 1,
                                 'passed': int(logit + rng.normal(0, 1.5) > 0), 'analysis': analysis})
    monkeypatch.setattr(store, 'plays_with_analysis', lambda user: rows)
    status = train(store, 'FixturePlayer')
    assert status['status'] == 'trained'
    model = json.loads(_artifact_path(store, 'FixturePlayer').read_text(encoding='utf-8'))
    def score(bpm, ar, hidden=False):
        target = {**base, 'stream_bpm': bpm, 'burst_bpm': bpm, 'ar': ar, 'mods': 'HD' if hidden else 'NM'}
        return predict_pass(store, 'FixturePlayer', target, context=(model, status))[0]
    assert score(120, 9) > score(240, 9)
    assert score(120, 9) > score(120, 11)
    assert score(120, 6, True) < score(120, 9, True)
    assert score(240, 11) < .1
