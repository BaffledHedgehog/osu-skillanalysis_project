import hashlib
import json

import pytest

from app.analyzer import analyze_map, parse_map
from app.storage import Store
from app.lazer_client import import_lazer_scores
from app.ml import _dataset
from app.predictor import predict
from test_app import chart
from test_capabilities import records_for


def test_rank_restores_pass_and_partial_failure_without_duplicates(tmp_path, monkeypatch):
    from app import lazer_client
    root = tmp_path / 'lazer'
    root.mkdir()
    (root / 'client.realm').write_bytes(b'realm-fixture')
    raw = chart(500)
    digest = hashlib.sha256(raw).hexdigest()
    file = root / 'files' / digest[0] / digest[:2] / digest
    file.parent.mkdir(parents=True)
    file.write_bytes(raw)
    base = {'id': 'known-pass', 'user': 'Player', 'mode': 0, 'accuracy': .96,
            'map_file_hash': digest, 'mods': '[]', 'statistics': '{"great":24}', 'replay_files': []}
    rows = [base]
    monkeypatch.setattr(lazer_client, '_read_realm', lambda folder: rows)
    store = Store(tmp_path / 'scores.db')
    assert import_lazer_scores(store, root)['imported'] == 1
    assert not store.plays('Player')[0]['passed_known']
    before = store.cache_token('Player')
    rows[0]['rank'] = 0  # D is a pass, despite the enum value being zero.
    rows.extend([{**base, 'id': 'partial-fail', 'rank': -1, 'statistics': '{"great":10,"miss":2}'},
                 {**base, 'id': 'no-fail', 'rank': 3, 'mods': '[{"acronym":"NF"}]'}])
    result = import_lazer_scores(store, root, force=True)
    assert result['imported'] == 2 and store.play_count('Player') == 3
    plays = {p['id']: p for p in store.plays('Player')}
    assert plays['lazer-known-pass']['passed_known'] and plays['lazer-known-pass']['passed']
    assert plays['lazer-known-pass']['grade'] == 'D'
    assert plays['lazer-partial-fail']['passed_known'] and not plays['lazer-partial-fail']['passed']
    assert not plays['lazer-partial-fail']['complete']
    assert plays['lazer-partial-fail']['grade'] == 'F'
    assert plays['lazer-no-fail']['mods'] == 'NF' and not plays['lazer-no-fail']['passed_known']
    assert before != store.cache_token('Player')
    _, labels, _, _ = _dataset(store, 'Player')
    assert sorted(labels.tolist()) == [0, 1]
    token = store.cache_token('Player')
    assert import_lazer_scores(store, root, force=True)['imported'] == 0
    assert store.cache_token('Player') == token


def test_same_chart_failures_update_probability_without_a_hard_threshold(tmp_path, monkeypatch):
    import app.predictor as predictor
    target = analyze_map(parse_map(chart(501)))
    records = records_for(target, count=10)
    for r in records:
        r['passed'] = False
        r['accuracy'] = .83
    monkeypatch.setattr(predictor, 'predict_pass', lambda *_: (.8, {'status': 'trained'}))
    monkeypatch.setattr(predictor, 'predict_accuracy', lambda *_: (None, {'status': 'insufficient_data'}))
    result = predict(Store(tmp_path / 'db'), 'Player', target, records=records)
    assert 1 < result['pass_probability'] < 30
    assert result['same_chart_history'] == {'attempts': 10, 'passes': 0, 'fails': 10}
    # Changing chart version or mods must not transfer its failure count.
    other = predict(Store(tmp_path / 'db'), 'Player', {**target, 'checksum': 'different'}, records=records)
    assert other['pass_probability'] == 80 and other['same_chart_history']['attempts'] == 0


def test_short_ending_spike_has_its_own_model_features():
    regular = parse_map(chart(502, 200))
    lines = regular.content.split('[HitObjects]')[1].strip().splitlines()
    changed = []
    for i, line in enumerate(lines):
        parts = line.split(',')
        parts[2] = str(1000 + min(i, 15) * 200 + max(0, i - 15) * 50)
        changed.append(','.join(parts))
    spiky = parse_map((regular.content.split('[HitObjects]')[0] + '[HitObjects]\n' + '\n'.join(changed)).encode())
    a, b = analyze_map(regular), analyze_map(spiky)
    assert b['ending_peak_load'] > a['ending_peak_load']
    assert b['peak_load'] > a['peak_load']
    from app.ml import vector, FEATURES
    assert vector(b)[FEATURES.index('ending_peak_load')] == b['ending_peak_load']


def test_import_batch_rolls_back_on_error(tmp_path):
    store = Store(tmp_path / 'db')
    with pytest.raises(RuntimeError):
        with store.batch():
            store.save_map(parse_map(chart(503)))
            raise RuntimeError('abort batch')
    assert store.get_map('503') is None
