"""Shared recency weighting and progress on repeated charts, without skill caps."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from statistics import mean, median

HALF_LIFE_DAYS = 60
WEEKLY_WEIGHT_BUDGET = 30
POLICY_VERSION = 'recency-60d-week30-v1'


def as_of_day():
    return datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)


def played_date(play):
    try:
        value = datetime.fromisoformat(play.get('played_at', '').replace('Z', '+00:00'))
        return value.replace(tzinfo=value.tzinfo or timezone.utc).astimezone(timezone.utc)
    except (ValueError, TypeError, AttributeError):
        return None


def sample_weights(plays, now=None):
    now = now or as_of_day()
    dates = [played_date(p) for p in plays]
    buckets = [(d.isocalendar().year, d.isocalendar().week) if d else ('undated', i) for i, d in enumerate(dates)]
    volumes = Counter(buckets)
    return [(max(1e-9, .5 ** (max(0, (now - d).total_seconds() / 86400) / HALF_LIFE_DAYS)) if d else .25)
            * min(1, WEEKLY_WEIGHT_BUDGET / volumes[bucket]) for d, bucket in zip(dates, buckets)]


def progress_summary(records, now=None):
    now = now or datetime.now(timezone.utc)
    periods = [defaultdict(list), defaultdict(list)]
    for p in records:
        date = played_date(p)
        if not date or 'NF' in p['mods'] or p['source'].endswith('_best') or p['source'] == 'osu_api':
            continue
        age = (now - date).total_seconds() / 86400
        if not 0 <= age <= 180:
            continue
        key = (p['analysis'].get('checksum', p['map_id']), p['mods'])
        periods[0 if age <= 30 else 1][key].append(p)
    shared = periods[0].keys() & periods[1].keys()
    accuracy, passes = [], []
    for key in shared:
        recent, previous = periods[0][key], periods[1][key]
        ac = [[p['accuracy'] for p in group if p.get('complete', 1) and p.get('passed_known', 1) and p['passed']]
              for group in (recent, previous)]
        outcomes = [[float(p['passed']) for p in group if p.get('passed_known', 1)] for group in (recent, previous)]
        if all(ac):
            accuracy.append((mean(ac[0]) - mean(ac[1])) * 100)
        if all(outcomes):
            passes.append((mean(outcomes[0]) - mean(outcomes[1])) * 100)
    return {'status': 'available' if max(len(accuracy), len(passes)) >= 5 else 'insufficient_overlap',
            'matched_maps': len(shared), 'accuracy_maps': len(accuracy), 'pass_maps': len(passes),
            'accuracy_change_pp': round(median(accuracy), 2) if len(accuracy) >= 5 else None,
            'pass_rate_change_pp': round(median(passes), 2) if len(passes) >= 5 else None,
            'recent_days': 30, 'comparison_days': [31, 180], 'half_life_days': HALF_LIFE_DAYS}
