"""The course cases, including the sixth case required by testing homework.

Brier numbers in TC-02/04/05 are deliberately controlled test inputs, not
measurements of player prediction quality. The real training/acceptance/API
path runs against an isolated SQLite fixture.
"""
import io
import json
import os
from pathlib import Path
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import ml
from app.analyzer import parse_map
from app.main import create_app
from test_app import chart

ROOT = Path(__file__).resolve().parents[1]
MAP = ROOT / 'deliverables/test-kit/inputs/valid-map.osu'


def save_evidence(tmp_path, case, name, value):
    folder = Path(os.environ.get('PULSE_TEST_EVIDENCE_DIR', str(tmp_path))) / case
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def test_tc01_correct_osu_and_osz(tmp_path):
    client = TestClient(create_app(tmp_path / 'tc01.db', public_mode=False))
    raw = MAP.read_bytes() if MAP.is_file() else chart(4864861)
    parsed = parse_map(raw)
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('valid-map.osu', raw)
    for name, payload in [('valid-map.osu', raw), ('valid-map.osz', archive.getvalue())]:
        response = client.post('/api/maps/upload', files={'file': (name, payload)})
        assert response.status_code == 200, response.text
        result = response.json()
        assert len(result['maps']) == 1
        assert result['maps'][0]['id'] == parsed.map_id
        actual = client.get(f'/api/maps/{parsed.map_id}/analysis').json()
        assert actual['title'] == parsed.metadata['title']
        assert actual['object_count'] == parsed.metadata['object_count']
        assert actual['stars'] > 0 and actual['aim'] >= 0 and actual['speed'] >= 0
        save_evidence(tmp_path, 'TC-01', name + '-upload.json', result)
        save_evidence(tmp_path, 'TC-01', name + '-analysis.json', actual)
    # Loading the archive containing the same map does not duplicate its ID.
    assert [m['id'] for m in client.get('/api/maps').json()] == [parsed.map_id]


def test_tc03_corrupt_map_leaves_catalog_unchanged(tmp_path):
    client = TestClient(create_app(tmp_path / 'tc03.db', public_mode=False))
    client.post('/api/maps/upload', files={'file': ('existing.osu', chart(100))})
    before = sorted(m['id'] for m in client.get('/api/maps').json())
    response = client.post('/api/maps/upload', files={'file': ('broken-map.osu', b'This is not an osu beatmap.\n')})
    after = sorted(m['id'] for m in client.get('/api/maps').json())
    assert response.status_code == 400
    assert isinstance(response.json().get('detail'), str) and response.json()['detail']
    assert before == after
    assert client.get('/api/health').status_code == 200
    save_evidence(tmp_path, 'TC-03', 'error.json', {'http_status': response.status_code, **response.json()})
    save_evidence(tmp_path, 'TC-03', 'catalog.json', {'before': before, 'after': after})


@pytest.mark.parametrize('case,model_brier,baseline_brier,expected', [
    ('TC-02', .12, .20, 'trained'),
    ('TC-04', .28, .20, 'rejected'),
    ('TC-05', .20, .20, 'rejected'),
], ids=['TC-02-better', 'TC-04-worse', 'TC-05-equal'])
def test_controlled_pass_model_gate(tmp_path, monkeypatch, case, model_brier, baseline_brier, expected):
    app = create_app(tmp_path / f'{case}.db', public_mode=False)
    store = app.state.store
    for index in range(12):
        mid = str(800000 + index)
        store.save_map(parse_map(chart(int(mid), spacing=275-index*17)))
        for attempt in range(5):
            passed = index < 6
            store.save_play('QA_Control', mid, 'NM', .95 if passed else .72, passed,
                            play_id=f'{mid}-{attempt}', source='manual')
    # New checksum and ID, but load inside the training range. No same-chart
    # history exists, so a history posterior cannot hide the selected predictor.
    probe = parse_map(chart(899999, spacing=190))
    store.save_map(probe)
    x, _, groups, _ = ml._dataset(store, 'QA_Control')
    fit_rows, validation_rows = [], []
    original_fit, original_probabilities = ml._fit, ml._probabilities

    def fit(*args, **kwargs):
        fit_rows.append(args[0].copy())
        return original_fit(*args, **kwargs)

    def probabilities(model, features):
        validation_rows.append(features.copy())
        return original_probabilities(model, features)

    calls = []
    def controlled_brier(actual, predicted, **kwargs):
        value = (model_brier, baseline_brier)[len(calls) % 2]
        calls.append(value)
        return value

    monkeypatch.setattr(ml, '_fit', fit)
    monkeypatch.setattr(ml, '_probabilities', probabilities)
    monkeypatch.setattr(ml, 'brier_score_loss', controlled_brier)
    client = TestClient(app)
    response = client.post('/api/users/QA_Control/model/train')
    assert response.status_code == 200
    training = response.json()['pass']
    assert training['status'] == expected
    assert training['metrics']['brier'] == model_brier
    assert training['metrics']['baseline_brier'] == baseline_brier
    assert calls[:2] == [model_brier, baseline_brier]
    def maps_for(rows):
        return {groups[i] for row in rows for i in np.flatnonzero(np.all(x == row, axis=1))}
    fit_maps, test_maps = maps_for(fit_rows[0]), maps_for(validation_rows[0])
    assert fit_maps and test_maps and fit_maps.isdisjoint(test_maps)
    analysis = client.get('/api/maps/899999/analysis', params={'user': 'QA_Control'}).json()
    prediction = analysis['prediction']
    assert prediction['ml_status']['status'] == expected
    assert prediction['pass_probability'] is not None
    assert prediction['same_chart_history']['attempts'] == 0
    source = 'validated_personal_logistic' if expected == 'trained' else 'similarity_and_prior'
    assert prediction['model'] == source
    save_evidence(tmp_path, case, 'training.json', training)
    save_evidence(tmp_path, case, 'analysis.json', analysis)
    save_evidence(tmp_path, case, 'prediction.json', prediction)
    save_evidence(tmp_path, case, 'conditions.json', {
        'controlled_metrics': True, 'not_player_quality_measurement': True,
        'model_brier': model_brier, 'baseline_brier': baseline_brier,
        'training_maps': sorted(fit_maps), 'validation_maps': sorted(test_maps),
        'map_overlap': sorted(fit_maps & test_maps), 'probe_id': probe.map_id,
    })


def test_tc06_two_results_do_not_invent_a_personal_prediction(tmp_path):
    """The product requires three results; two must not produce fake numbers."""
    app = create_app(tmp_path / 'tc06.db', public_mode=False)
    store = app.state.store
    store.save_map(parse_map(chart(600001)))
    store.save_map(parse_map(chart(600002)))
    for i in range(2):
        store.save_play('QA_TwoResults', '600001', 'NM', .96, True,
                        play_id=f'tc06-{i}', source='manual')
    client = TestClient(app)
    response = client.get('/api/maps/600002/analysis', params={'user': 'QA_TwoResults'})
    assert response.status_code == 200
    analysis = response.json()
    p = analysis['prediction']
    assert p['observations'] == 2
    assert p['pass_probability'] is None and p['expected_accuracy'] is None
    assert p['accuracy_range'] is None and p['model'] == 'insufficient_data'
    assert p['message'] and analysis['stars'] > 0
    assert client.get('/api/health').status_code == 200
    save_evidence(tmp_path, 'TC-06', 'analysis.json', analysis)
    save_evidence(tmp_path, 'TC-06', 'conditions.json', {
        'synthetic_user': True, 'observations': 2, 'minimum_results': 3,
        'expected_pass_probability': None, 'expected_accuracy': None,
        'expected_model': 'insufficient_data', 'http_status': response.status_code,
    })
