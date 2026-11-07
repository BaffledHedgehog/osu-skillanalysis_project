"""Descriptive skill references, kept separate from learned pass probabilities."""
from __future__ import annotations

import math
from collections import Counter

from .analyzer import normalize_mods
from .ml import BEST_SOURCES

MIN_MAPS = 3
SKILLS = (('aim', 'Aim', 'Наведение на объекты'),
          ('speed', 'Speed', 'Скорость нажатий'))


def _number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) and result > 0 else None
    except (TypeError, ValueError):
        return None


def _key(row):
    a = row['analysis']
    return a.get('checksum') or str(row['map_id'])


def _quantile(values, q):
    ordered = sorted(values)
    threshold = sum(weight for _, weight in ordered) * q
    current = 0.0
    for value, weight in ordered:
        current += weight
        if current >= threshold:
            return value
    return ordered[-1][0]


def build_skill_profile(records, mods='NM', target=None):
    """Use complete known passes with matching mods, one vote per map version.

    Ordinary attempts take precedence once three distinct maps exist. Otherwise
    achieved bests may supplement the descriptive reference, clearly identified.
    Neither these references nor display ratios constrain ML probabilities.
    """
    mods, _ = normalize_mods(mods)
    target = target or {}
    eligible = []
    for row in records:
        a = row.get('analysis') or {}
        if not a or not row.get('passed') or not row.get('passed_known', 1) or not row.get('complete', 1):
            continue
        row_mods = a.get('mods', row.get('mods', 'NM'))
        if 'NF' in row_mods or row_mods != mods or not _number(row.get('weight', 1)):
            continue
        if ((target.get('checksum') and a.get('checksum') == target['checksum'])
                or (target.get('id') is not None and str(row['map_id']) == str(target['id']))):
            continue
        eligible.append(row)
    ordinary = [r for r in eligible if r.get('source') not in BEST_SOURCES]
    if len({_key(r) for r in ordinary}) >= MIN_MAPS:
        selected, source_mode = ordinary, 'ordinary_passes'
    else:
        selected = eligible
        curated = sum(r.get('source') in BEST_SOURCES for r in selected)
        source_mode = 'mixed_achievements' if curated and ordinary else 'best_scores' if curated else 'ordinary_passes'
    skills = []
    for key, label, explanation in SKILLS:
        useful = [r for r in selected if _number(r['analysis'].get(key))]
        by_map = {}
        for row in useful:
            value = float(row['analysis'][key])
            weight = float(row.get('weight', 1))
            group = _key(row)
            if group not in by_map or weight > by_map[group][1]:
                by_map[group] = (value, weight)
        reference = _quantile(list(by_map.values()), .7) if len(by_map) >= MIN_MAPS else None
        demand = _number(target.get(key))
        ratio = demand / reference if reference is not None and demand is not None else None
        skills.append({
            'id': key, 'label': label, 'description': explanation,
            'status': 'available' if reference is not None else 'insufficient_data',
            'level': round(reference, 3) if reference is not None else None,
            'target_load': round(demand, 3) if demand is not None else None,
            'relative_load': round(ratio, 3) if ratio is not None else None,
            'difference_percent': round((ratio - 1) * 100, 1) if ratio is not None else None,
            'attempts': len(useful), 'maps': len(by_map),
        })
    compared = [s for s in skills if s['relative_load'] is not None]
    dominant = max(compared, key=lambda s: s['relative_load']) if len(compared) == len(SKILLS) else None
    return {
        'status': 'available' if all(s['status'] == 'available' for s in skills) else 'insufficient_data',
        'mods': mods, 'skills': skills, 'source_mode': source_mode,
        'attempts': len(selected), 'maps': len({_key(r) for r in selected}),
        'sources': dict(Counter(r.get('source', 'unknown') for r in selected)),
        'minimum_maps': MIN_MAPS, 'quantile': .7,
        'target_excluded': bool(target),
        'dominant_demand': dominant['id'] if dominant and dominant['relative_load'] > 1 else None,
        'method': 'recency_weighted_unique_map_quantile',
        'note': 'Ориентир по пройденным картам, не физический предел и не вероятность. '
                'Моды сравниваются отдельно, давность учитывается, повторы одной карты не добавляют ей голосов.',
    }
