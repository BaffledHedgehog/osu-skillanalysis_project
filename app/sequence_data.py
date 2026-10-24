"""Frozen, family-grouped corpus with prequential player context and mirror views."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import unicodedata

import numpy as np

from .analyzer import _sections, _pairs, ParsedMap
from .ml import BEST_SOURCES, vector, FEATURES as GLOBAL_FEATURES
from .recency import HALF_LIFE_DAYS, POLICY_VERSION, played_date, sample_weights
from .sequence import NoteSequence, VERSION, VIEWS, encode_map, mirror_sequence, windows
from .storage import Store

PROFILE_FEATURES = ('log_history', 'pass_rate', 'accuracy', 'stars', 'aim', 'speed', 'ar', 'hd_pass_rate')
CONTEXT_FEATURES = (*GLOBAL_FEATURES, *PROFILE_FEATURES)
CORPUS_VERSION = 'cnn-family-prequential-v1'


def validated_mods(records, families, test_indices):
    """Coverage evidence per exact mod combination, separate from model quality."""
    result = {'pass': [], 'accuracy': []}
    for mods in sorted({records[i]['mods'] for i in test_indices}):
        ids = [i for i in test_indices if records[i]['mods'] == mods]
        groups = {families[records[i]['map_id']] for i in ids}
        if len(groups) < 8: continue
        passed = sum(bool(records[i]['passed']) for i in ids)
        if passed >= 8 and len(ids)-passed >= 8: result['pass'].append(mods)
        accurate = [i for i in ids if records[i].get('accuracy_mask')]
        if len(accurate) >= 20 and len({families[records[i]['map_id']] for i in accurate}) >= 8:
            result['accuracy'].append(mods)
    return result


def _text(value):
    return re.sub(r'[^\w]+', '', unicodedata.normalize('NFKC', value).casefold())


def family_groups(maps: dict[str, ParsedMap]):
    """Union sets, same-song maps and exact note-pattern versions before splitting."""
    parent = {key: key for key in maps}
    def root(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key
    seen = {}
    for key, parsed in sorted(maps.items()):
        sections = _sections(parsed.content)
        meta = _pairs(sections.get('Metadata', []))
        signatures = [('checksum', parsed.checksum)]
        if meta.get('BeatmapSetID', '').isdigit() and int(meta['BeatmapSetID']) > 0:
            signatures.append(('set', meta['BeatmapSetID']))
        artist, title = _text(meta.get('Artist', '')), _text(meta.get('Title', ''))
        if artist and title:
            signatures.append(('song', artist, title))
        # Ignore cosmetic metadata in exact pattern duplicates.
        pattern = '\n'.join(sections.get('HitObjects', [])) + '\n' + '\n'.join(sections.get('TimingPoints', []))
        signatures.append(('pattern', hashlib.sha256(pattern.encode()).hexdigest()))
        for signature in signatures:
            if signature in seen:
                a, b = root(key), root(seen[signature])
                parent[max(a, b)] = min(a, b)
            else:
                seen[signature] = key
    members = defaultdict(list)
    for key in maps:
        members[root(key)].append(maps[key].checksum)
    names = {key: hashlib.sha256('|'.join(sorted(values)).encode()).hexdigest()[:24] for key, values in members.items()}
    return {key: names[root(key)] for key in maps}


def split_families(groups, seed=2026):
    ordered = sorted(set(groups), key=lambda g: hashlib.sha256(f'{seed}:{g}'.encode()).digest())
    if len(ordered) < 10:
        raise ValueError('Для CNN нужны хотя бы 10 независимых семейств карт.')
    n_train, n_val = max(1, int(len(ordered) * .6)), max(1, int(len(ordered) * .2))
    return {group: 'train' if i < n_train else 'validation' if i < n_train + n_val else 'test'
            for i, group in enumerate(ordered)}


class SequenceCache:
    def __init__(self, store: Store):
        self.store = store
        self.folder = store.path.parent / 'sequence-cache'
        self.folder.mkdir(parents=True, exist_ok=True)
        self.memory = {}

    def get(self, parsed, analysis, view='original'):
        key = hashlib.sha256(f'{VERSION}:{parsed.checksum}:{analysis["mods"]}'.encode()).hexdigest()
        if key not in self.memory:
            path = self.folder / f'{key}.npz'
            try:
                with np.load(path, allow_pickle=False) as data:
                    sequence = NoteSequence(data['values'], data['times'], data['ends'], parsed.checksum, analysis['mods'])
                    if sequence.values.shape[1] != len(__import__('app.sequence', fromlist=['FEATURES']).FEATURES) or not np.isfinite(sequence.values).all():
                        raise ValueError('Invalid sequence cache')
            except (OSError, ValueError, KeyError):
                sequence = encode_map(parsed, analysis)
                temporary = path.with_suffix('.tmp.npz')
                np.savez_compressed(temporary, values=sequence.values, times=sequence.times_ms, ends=sequence.end_times_ms)
                temporary.replace(path)
            # Bound memory; persisted cache remains reusable across training runs.
            if len(self.memory) >= 192:
                self.memory.pop(next(iter(self.memory)))
            self.memory[key] = sequence
        return mirror_sequence(self.memory[key], view)


@dataclass
class Corpus:
    store: Store
    records: list[dict]
    maps: dict[str, ParsedMap]
    families: dict[str, str]
    assignment: dict[str, str]
    contexts: np.ndarray
    weights: np.ndarray
    fingerprint: str
    manifest: dict


def profile_context(records, allowed_families, families):
    """Only strictly earlier training-family outcomes enter a row's player context."""
    result = np.zeros((len(records), len(PROFILE_FEATURES)), dtype=np.float32)
    sorted_rows = sorted(range(len(records)), key=lambda i: (played_date(records[i]) or datetime.min.replace(tzinfo=timezone.utc), records[i]['id']))
    state = np.zeros(9, dtype=float)  # count, passes, acc_count, acc_sum, stars, aim, speed, ar, hd_count
    hd_passes, last = 0., None
    buckets = defaultdict(list)
    for i in sorted_rows:
        buckets[played_date(records[i])].append(i)
    for date, indices in buckets.items():
        if date and last:
            decay = .5 ** (max(0, (date - last).total_seconds() / 86400) / HALF_LIFE_DAYS)
            state *= decay
            hd_passes *= decay
        count, passed, ac_count, ac_sum, stars, aim, speed, ar, hd_count = state
        values = [math.log1p(count), (passed + 1) / (count + 2), ac_sum / ac_count if ac_count else .9,
                  stars / passed if passed else 0, aim / passed if passed else 0,
                  speed / passed if passed else 0, ar / passed if passed else 0,
                  (hd_passes + 1) / (hd_count + 2)]
        for i in indices:
            result[i] = values
        # Equal timestamps form a block: no other outcome in that block leaks in.
        for i in indices:
            p = records[i]
            if date is None or families[p['map_id']] not in allowed_families or not p['pass_mask']:
                continue
            a, success = p['analysis'], float(p['passed'])
            state[0] += 1; state[1] += success
            if p['accuracy_mask']:
                state[2] += 1; state[3] += p['accuracy']
            state[4:8] += success * np.asarray([a['stars'], a['aim'], a['speed'], a['ar']])
            if 'HD' in p['mods']:
                state[8] += 1; hd_passes += success
        if date:
            last = date
    return result


def build_corpus(store: Store, user: str, *, seed=2026):
    records = []
    for play in store.plays_with_analysis(user):
        if play['analysis'] is None:
            continue
        pass_mask = bool(play.get('passed_known', 1)) and play['source'] not in BEST_SOURCES and 'NF' not in play['mods']
        accuracy_mask = pass_mask and bool(play['passed']) and bool(play.get('complete', 1))
        if pass_mask or accuracy_mask:
            records.append({**play, 'pass_mask': pass_mask, 'accuracy_mask': accuracy_mask})
    records.sort(key=lambda p: (p['played_at'], p['id']))
    if len(records) < 80:
        raise ValueError('Для эксперимента CNN нужны хотя бы 80 обычных попыток с известным исходом.')
    maps = {mid: store.get_map(mid) for mid in sorted({p['map_id'] for p in records})}
    maps = {mid: parsed for mid, parsed in maps.items() if parsed is not None}
    records = [p for p in records if p['map_id'] in maps]
    families = family_groups(maps)
    assignment = split_families(families.values(), seed)
    allowed = {group for group, split in assignment.items() if split == 'train'}
    profile = profile_context(records, allowed, families)
    contexts = np.asarray([vector(p['analysis']) for p in records], dtype=np.float32)
    contexts = np.concatenate([contexts, profile], axis=1)
    weights = np.asarray(sample_weights(records), dtype=np.float32)
    identity = [(p['id'], p['map_id'], p['mods'], p['played_at'], p['passed'], p['accuracy'], p.get('complete', 1), p['analysis']['checksum']) for p in records]
    fingerprint = hashlib.sha256(json.dumps([CORPUS_VERSION, VERSION, POLICY_VERSION, seed, identity, weights.tolist()], sort_keys=True).encode()).hexdigest()
    manifest = {'schema': CORPUS_VERSION, 'fingerprint': fingerprint, 'user': user, 'seed': seed,
                'protocol': 'unseen_song_families', 'notes': 'Not a prospective validation. Player context only uses earlier outcomes from training families.',
                'views': list(VIEWS), 'source_attempts': len(records), 'families': len(assignment), 'splits': {}}
    for split in ('train', 'validation', 'test'):
        indices = [i for i, p in enumerate(records) if assignment[families[p['map_id']]] == split]
        manifest['splits'][split] = {'attempts': len(indices), 'passes': sum(records[i]['passed'] for i in indices),
                                     'fails': sum(not records[i]['passed'] for i in indices),
                                     'maps': sorted({records[i]['map_id'] for i in indices}),
                                     'families': sorted({families[records[i]['map_id']] for i in indices}),
                                     'indices': indices}
    if any(manifest['splits'][s]['attempts'] < 16 for s in ('train', 'validation', 'test')):
        raise ValueError('Недостаточно попыток в независимых частях выборки.')
    return Corpus(store, records, maps, families, assignment, contexts, weights, fingerprint, manifest)
