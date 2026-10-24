from datetime import datetime, timedelta, timezone

import numpy as np

from app.recency import sample_weights, progress_summary
from app.ml import _fit, _probabilities


NOW = datetime(2026, 10, 24, tzinfo=timezone.utc)


def attempt(days):
    return {'played_at': (NOW - timedelta(days=days)).isoformat()}


def test_recency_half_life_and_large_old_batch_do_not_dominate():
    weights = sample_weights([attempt(0), attempt(60), attempt(180)], now=NOW)
    assert weights == [1, .5, .125]
    many = sample_weights([attempt(180)] * 1000 + [attempt(1)] * 10, now=NOW)
    assert sum(many[:1000]) < sum(many[1000:])


def test_recent_outcomes_change_the_fitted_model_not_only_ui():
    plays = [attempt(180)] * 400 + [attempt(1)] * 20
    weights = np.asarray(sample_weights(plays, now=NOW))
    x, y = np.zeros((420, 2)), np.asarray([0] * 400 + [1] * 20)
    old_p = _probabilities(_fit(x, y), np.zeros((1, 2)))[0]
    current_p = _probabilities(_fit(x, y, weights), np.zeros((1, 2)))[0]
    assert old_p < .1 and current_p > .6


def test_progress_compares_same_charts_and_mods():
    records = []
    for i in range(5):
        base = {'map_id': str(i), 'mods': 'NM', 'source': 'manual', 'passed_known': True,
                'passed': True, 'complete': True, 'analysis': {'checksum': str(i)}}
        records.extend([{**base, **attempt(90), 'accuracy': .80}, {**base, **attempt(1), 'accuracy': .96}])
    result = progress_summary(records, now=NOW)
    assert result['status'] == 'available' and result['matched_maps'] == 5
    assert result['accuracy_change_pp'] == 16 and result['pass_rate_change_pp'] == 0
    for r in records:
        if r['accuracy'] == .96:
            r['mods'] = 'HD'
    assert progress_summary(records, now=NOW)['status'] == 'insufficient_overlap'


def test_undated_and_naive_timestamps_are_handled_without_freshness_assumption():
    w = sample_weights([{}, {'played_at': 'not-a-date'}, {'played_at': '2026-10-24T00:00:00'}], now=NOW)
    assert w == [.25, .25, 1]
