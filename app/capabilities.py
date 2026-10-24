"""Report missing experience without turning it into a probability cap."""


def coverage_warnings(target, records):
    results = []
    analyses = [record['analysis'] for record in records]
    if not analyses:
        return results
    for key, label in [('stars', 'Сложность'), ('ar', 'Эффективный AR'),
                       ('stream_bpm', 'Скорость серий'), ('burst_bpm', 'Скорость коротких серий')]:
        observed = max(a.get(key, 0) for a in analyses)
        if target.get(key, 0) > observed + 1e-6:
            results.append({'skill': key, 'value': target[key], 'observed_max': observed,
                            'reason': f'{label} выше диапазона твоей истории. Это нехватка данных, а не доказанный предел навыка.'})
    if 'HD' in target.get('mods', ''):
        hd = [a for a in analyses if 'HD' in a.get('mods', '')]
        if not hd or target['ar'] < min(a['ar'] for a in hd) - 1e-6:
            results.append({'skill': 'hd_reading', 'reason': 'В истории нет примеров HD с таким низким AR. Шанс прохождения пока неизвестен.'})
    return results


def practice_distance(target, candidate):
    def aim_share(a):
        return a['aim'] / max(.1, a['aim'] + a['speed'])
    return (4 * abs(aim_share(target) - aim_share(candidate))
            + abs(target.get('slider_fraction', 0) - candidate.get('slider_fraction', 0))
            + .4 * abs(target.get('rhythm_variation', 0) - candidate.get('rhythm_variation', 0))
            + .12 * abs(target['ar'] - candidate['ar']))
