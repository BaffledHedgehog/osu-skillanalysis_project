"""Replay-backed note labels. Final score grades never label an unplayed suffix."""
from __future__ import annotations

import hashlib
import json

import numpy as np

from .ml import BEST_SOURCES, vector
from .recency import sample_weights
from .replay_events import SCHEMA as EVENT_SCHEMA, event_key
from .sequence_data import Corpus, family_groups, split_families, profile_context

TARGETS = ('miss_or_combo_break', 'non_great_judgement')
LABEL_SCHEMA = 'replay-note-labels-v2'


def note_labels(result, sequence):
    """Two labels: miss/combo break, and non-Great circle/head judgement.

    A slider has multiple judgements: any missed head/tick/tail or combo break
    marks its root. Non-Great is observed only on a timed circle/head judgement,
    never inferred from final accuracy or a slider's aggregate grade. A miss
    alone cannot tell whether the cause was aim, reading, or timing.
    """
    engine = result.get('engine') or {}
    count = len(sequence.values)
    if (result.get('schema') != EVENT_SCHEMA or not result.get('verified') or not result.get('complete')
            or not engine.get('events_verified') or not engine.get('statistics_match')
            or engine.get('root_count') != count or engine.get('unmapped_events') != 0
            or engine.get('reverted_events')):
        raise ValueError('Replay events are not verified for this complete sequence.')
    if result['map_checksum'] != sequence.source_checksum or result['mods'] != sequence.mods:
        raise ValueError('Replay labels do not match the source map/mods.')
    y = np.zeros((count, len(TARGETS)), dtype=np.float32)
    mask = np.zeros_like(y, dtype=bool)
    observed = set()
    clock = 1.
    # Native events use beatmap time; feature sequences use real time after rate.
    if 'DT' in sequence.mods or 'NC' in sequence.mods: clock = 1.5
    elif 'HT' in sequence.mods: clock = .75
    for e in engine.get('events') or []:
        i = e['root_index']
        if not isinstance(i, int) or not 0 <= i < count:
            raise ValueError('Replay root index is outside the map.')
        if not all(np.isfinite(e[k]) for k in ('time_ms', 'object_time_ms', 'offset_ms', 'hp_before')):
            raise ValueError('Invalid replay event coordinates.')
        # Nested slider events may start later; they must belong to this root's
        # lifetime. Event *judgement* time can be early/late and is not clipped.
        obj_time = e['object_time_ms'] / clock
        if obj_time < sequence.times_ms[i] - 3 or obj_time > sequence.end_times_ms[i] + 3:
            raise ValueError('Replay object does not align with its source note.')
        observed.add(i)
        mask[i, 0] = True
        if not e['is_hit'] or e['combo_after'] < e['combo_before']:
            y[i, 0] = 1
        if e['object_type'] in ('HitCircle', 'SliderHeadCircle') and e['result'] in ('great', 'ok', 'meh', 'miss'):
            mask[i, 1] = True
            if e['result'] != 'great': y[i, 1] = 1
    if observed != set(range(count)):
        raise ValueError('Replay omits source objects; no full-map labels are allowed.')
    return y, mask


def build_event_corpus(store, user, cache):
    """All available verified replays, grouped before view generation.

    Stored event JSON files are looked up through this user's registered replay
    hashes. Other users' files in the same cache cannot enter the corpus.
    """
    collection_path = store.path.parent / 'replay-events' / ('collection-' + hashlib.sha256(user.strip().casefold().encode()).hexdigest()[:16] + '.json')
    if not collection_path.is_file():
        raise ValueError('Сначала соберите события реплеев: python -m app.replay_events USER')
    collection_raw = collection_path.read_bytes()
    collection = json.loads(collection_raw)
    if collection.get('status') != 'completed':
        raise ValueError('Сбор событий реплеев еще не завершен.')
    cohort = {p['play_id'] for p in collection['results'] if p.get('verified')}
    history = [p for p in store.plays_with_analysis(user) if p.get('analysis')]
    history.sort(key=lambda p: (p['played_at'], p['id']))
    all_maps = {mid: store.get_map(mid) for mid in sorted({p['map_id'] for p in history})}
    all_maps = {mid: p for mid, p in all_maps.items() if p is not None}
    families = family_groups(all_maps)
    assignment = split_families(families.values())
    allowed = {g for g, split in assignment.items() if split == 'train'}
    for p in history:
        p['pass_mask'] = bool(p.get('passed_known', 1)) and p['source'] not in BEST_SOURCES and 'NF' not in p['mods']
        p['accuracy_mask'] = p['pass_mask'] and bool(p['passed']) and bool(p.get('complete', 1))
    profiles = profile_context(history, allowed, families)
    records, context, labels, masks, hashes, rejected = [], [], [], [], [], []
    for i, p in enumerate(history):
        if p['id'] not in cohort: continue
        replay = store.replay_input(user, p['id'])
        if not replay or not p.get('complete', 1): continue
        parsed = all_maps.get(p['map_id'])
        if parsed is None: continue
        path = store.path.parent / 'replay-events' / (event_key(replay, parsed.checksum) + '.json')
        if not path.is_file(): continue
        try:
            raw = path.read_bytes()
            result = json.loads(raw)
            if result.get('replay_sha256') != replay['sha256']: raise ValueError('Replay checksum mismatch.')
            sequence = cache.get(parsed, p['analysis'])
            y, mask = note_labels(result, sequence)
        except (OSError, ValueError, KeyError) as exc:
            rejected.append({'id': p['id'], 'reason': str(exc)}); continue
        records.append({**p, 'event_key': path.stem})
        context.append([*vector(p['analysis']), *profiles[i]])
        labels.append(y); masks.append(mask); hashes.append(hashlib.sha256(raw).hexdigest())
    if len(records) < 30:
        raise ValueError('Для локальной модели нужны хотя бы 30 полностью проверенных реплеев.')
    weights = np.asarray(sample_weights(records), dtype=np.float32)
    manifest = {'schema': LABEL_SCHEMA, 'user': user, 'seed': 2026, 'targets': list(TARGETS),
                'protocol': 'unseen_song_families_full_verified_replays', 'attempts': len(records),
                'profile_reference': sorted({p.checksum for mid,p in all_maps.items() if families[mid] in allowed}),
                'cohort_policy': collection['policy'], 'requested_replays': collection['requested'],
                'collection_sha256': hashlib.sha256(collection_raw).hexdigest(),
                'rejected': rejected, 'event_sha256': hashes, 'splits': {}}
    for split in ('train', 'validation', 'test'):
        ids = [i for i, p in enumerate(records) if assignment[families[p['map_id']]] == split]
        positive = [int(sum(labels[i][:, k][masks[i][:, k]].sum() for i in ids)) for k in range(len(TARGETS))]
        observed = [int(sum(masks[i][:, k].sum() for i in ids)) for k in range(len(TARGETS))]
        manifest['splits'][split] = {'indices': ids, 'attempts': len(ids), 'positive_notes': positive,
                                     'observed_notes': observed,
                                     'families': sorted({families[records[i]['map_id']] for i in ids})}
    if any(manifest['splits'][s]['attempts'] < 5 for s in ('train', 'validation', 'test')):
        raise ValueError('Недостаточно проверенных реплеев в независимых частях выборки.')
    identity = [(p['id'], p['analysis']['checksum'], hashes[i]) for i, p in enumerate(records)]
    fingerprint = hashlib.sha256(json.dumps([LABEL_SCHEMA, identity, manifest['splits'], weights.tolist()]).encode()).hexdigest()
    manifest['fingerprint'] = fingerprint
    maps = {mid: all_maps[mid] for mid in {p['map_id'] for p in records}}
    corpus = Corpus(store, records, maps, families, assignment, np.asarray(context, dtype=np.float32), weights, fingerprint, manifest)
    return corpus, labels, masks
