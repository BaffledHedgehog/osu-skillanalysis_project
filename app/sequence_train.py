"""Reproducible CNN ablation. Never trains from a prediction request.

python -m app.sequence_train USER --steps 400 --pretrain-steps 300
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
from sklearn.linear_model import Ridge
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch.nn import functional as F

from .ml import FEATURES as GLOBAL_FEATURES, _fit, _relative_fit_weights
from .sequence import FEATURES, VIEWS, VERSION, windows
from .sequence_data import CONTEXT_FEATURES, Corpus, SequenceCache, build_corpus, validated_mods
from .sequence_net import PatternEncoder, PatternModel, save_weights, load_weights
from .storage import Store

VARIANTS = {'context_only': ('original',), 'original': ('original',), 'mirror_x': ('x',),
            'mirror_y': ('y',), 'mirror_xy': ('xy',), 'four_views': VIEWS}
SCHEMA = 'pattern-cnn-v1'


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def write_corpus_snapshot(corpus, output):
    """Persist the exact in-memory snapshot, so later account imports cannot change it."""
    folder = Path(output) / 'snapshot'
    folder.mkdir(parents=True, exist_ok=True)
    map_index = {}
    for mid, parsed in corpus.maps.items():
        name = hashlib.sha256(parsed.content.encode()).hexdigest() + '.osu'
        (folder / name).write_text(parsed.content, encoding='utf-8', newline='')
        map_index[mid] = {'file': name, 'source_checksum': parsed.checksum, 'family': corpus.families[mid]}
    samples = [{key: p.get(key) for key in ('id', 'map_id', 'mods', 'accuracy', 'passed', 'passed_known', 'complete', 'source', 'played_at', 'pass_mask', 'accuracy_mask', 'analysis')}
               for p in corpus.records]
    write_json(folder / 'samples.json', samples)
    write_json(folder / 'maps.json', map_index)
    np.savez_compressed(folder / 'inputs.npz', context=corpus.contexts, weights=corpus.weights)
    index_hash = hashlib.sha256((folder / 'samples.json').read_bytes() + (folder / 'maps.json').read_bytes() + (folder / 'inputs.npz').read_bytes()).hexdigest()
    write_json(folder / 'snapshot.json', {'fingerprint': corpus.fingerprint, 'index_sha256': index_hash,
                                         'attempts': len(samples), 'maps': len(map_index)})


def load_corpus_snapshot(output):
    """Restore and verify frozen inputs without reading the current account DB."""
    from .analyzer import parse_map
    output = Path(output)
    folder = output / 'snapshot'
    meta = json.loads((folder / 'snapshot.json').read_text(encoding='utf-8'))
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    files = [folder / name for name in ('samples.json', 'maps.json', 'inputs.npz')]
    digest = hashlib.sha256(b''.join(p.read_bytes() for p in files)).hexdigest()
    if digest != meta['index_sha256'] or meta['fingerprint'] != manifest['fingerprint']:
        raise ValueError('Frozen CNN inputs failed checksum verification.')
    records = json.loads(files[0].read_text(encoding='utf-8'))
    map_index = json.loads(files[1].read_text(encoding='utf-8'))
    maps, families = {}, {}
    for mid, item in map_index.items():
        name = item['file']
        if Path(name).name != name or not name.endswith('.osu'):
            raise ValueError('Invalid snapshot map path.')
        raw = (folder / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() + '.osu' != name:
            raise ValueError('Frozen map failed checksum verification.')
        parsed = parse_map(raw, forced_id=mid)
        maps[mid] = replace(parsed, checksum=item['source_checksum'])
        families[mid] = item['family']
    with np.load(files[2], allow_pickle=False) as data:
        contexts, weights = data['context'].copy(), data['weights'].copy()
    if contexts.shape != (len(records), len(CONTEXT_FEATURES)) or weights.shape != (len(records),):
        raise ValueError('Invalid snapshot dimensions.')
    if not np.isfinite(contexts).all() or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('Invalid snapshot values.')
    if any(not p.get('analysis') or p['map_id'] not in maps for p in records):
        raise ValueError('Snapshot lacks exact map analyses; run a new experiment.')
    assignment, all_indices = {}, []
    for split, values in manifest['splits'].items():
        for i in values['indices']:
            group = families[records[i]['map_id']]
            if group in assignment and assignment[group] != split:
                raise ValueError('Map family leaks between snapshot partitions.')
            assignment[group] = split
        all_indices.extend(values['indices'])
    if sorted(all_indices) != list(range(len(records))):
        raise ValueError('Invalid snapshot partition indices.')
    return Corpus(None, records, maps, families, assignment, contexts, weights, meta['fingerprint'], manifest)


def audit_experiment(output, *, device='cpu'):
    """Re-evaluate released weights on the immutable snapshot, never reselect."""
    output = Path(output)
    corpus = load_corpus_snapshot(output)
    artifact = json.loads((output / 'artifact.json').read_text(encoding='utf-8'))
    report = json.loads((output / 'report.json').read_text(encoding='utf-8'))
    if artifact['fingerprint'] != corpus.fingerprint or report['fingerprint'] != corpus.fingerprint:
        raise ValueError('Weights and corpus belong to different experiments.')
    context = ((corpus.contexts - artifact['context_mean']) / artifact['context_scale']).astype(np.float32)
    raw = corpus.contexts[:, :len(GLOBAL_FEATURES)]
    def linear(model):
        return ((raw - model['mean']) / model['scale'] @ np.asarray(model['coef']) + model['intercept']).astype(np.float32)
    base_p = linear(artifact['baseline']['pass'])
    base_a = np.clip(linear(artifact['baseline']['accuracy']), 0, 1)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    # The cache is disposable; all source maps come from the verified snapshot.
    cache = SequenceCache(type('SnapshotStore', (), {'path': output / 'audit-cache.db'})())
    checked = {}
    for name in VARIANTS:
        path = output / f'{name}.npz'
        if hashlib.sha256(path.read_bytes()).hexdigest() != artifact['weight_hashes'][name]:
            raise ValueError('CNN weights failed checksum verification.')
        model = load_weights(PatternModel(**artifact['model_options'], sequence=name != 'context_only'), path).to(device)
        checked[name] = {}
        for split in ('validation', 'test'):
            ids = corpus.manifest['splits'][split]['indices']
            batches = Batches(corpus, cache, context, base_p, base_a, device, ids)
            pp, aa = evaluate(model, batches)
            actual = metrics(corpus, ids, pp, aa)
            expected = report['variants'][name][split]
            for key in ('brier', 'mae_pp', 'log_loss', 'ece'):
                if not np.isclose(actual[key], expected[key], atol=2e-5, rtol=2e-5):
                    raise ValueError(f'CNN audit mismatch: {name}/{split}/{key}')
            checked[name][split] = actual
            checked[name][split]['subgroups'] = subgroup_metrics(corpus, ids, pp, aa)
    result = {'status': 'verified', 'fingerprint': corpus.fingerprint,
              'device': device, 'variants': checked, 'time': datetime.now(timezone.utc).isoformat()}
    write_json(output / 'audit.json', result)
    return result


def fitted_baselines(corpus, indices):
    x = corpus.contexts[:, :len(GLOBAL_FEATURES)].astype(float)
    y = np.asarray([p['passed'] for p in corpus.records], dtype=float)
    accuracy = np.asarray([p['accuracy'] for p in corpus.records], dtype=float)
    mask = np.asarray([p['accuracy_mask'] for p in corpus.records])
    w = corpus.weights
    train = np.asarray(indices)
    prevalence = float(np.average(y[train], weights=w[train]))
    if len(set(y[train])) == 2:
        scaler, classifier = _fit(x[train], y[train], w[train])
        pass_artifact = {'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(),
                         'coef': classifier.coef_[0].tolist(), 'intercept': float(classifier.intercept_[0])}
        linear = scaler.transform(x) @ classifier.coef_[0] + classifier.intercept_[0]
    else:
        logit = math.log(max(1e-5, prevalence) / max(1e-5, 1 - prevalence))
        pass_artifact = {'mean': [0.] * x.shape[1], 'scale': [1.] * x.shape[1], 'coef': [0.] * x.shape[1], 'intercept': logit}
        linear = np.full(len(x), logit)
    ac_train = train[mask[train]]
    if len(ac_train) < 20:
        raise ValueError('Недостаточно полных пройденных попыток в обучающей части.')
    from sklearn.preprocessing import StandardScaler
    aw = _relative_fit_weights(w[ac_train])
    scaler = StandardScaler().fit(x[ac_train], sample_weight=aw)
    ridge = Ridge(alpha=10).fit(scaler.transform(x[ac_train]), accuracy[ac_train], sample_weight=aw)
    ac_artifact = {'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(),
                   'coef': ridge.coef_.tolist(), 'intercept': float(ridge.intercept_)}
    predicted_accuracy = np.clip(scaler.transform(x) @ ridge.coef_ + ridge.intercept_, 0, 1)
    return {'pass': pass_artifact, 'accuracy': ac_artifact}, linear.astype(np.float32), predicted_accuracy.astype(np.float32)


def metrics(corpus, indices, pass_probabilities, accuracies):
    indices = np.asarray(indices)
    labels = np.asarray([corpus.records[i]['passed'] for i in indices], dtype=float)
    accuracy = np.asarray([corpus.records[i]['accuracy'] for i in indices], dtype=float)
    amask = np.asarray([corpus.records[i]['accuracy_mask'] for i in indices], dtype=bool)
    w = corpus.weights[indices].astype(float)
    p = np.clip(np.asarray(pass_probabilities), 1e-6, 1 - 1e-6)
    ac = np.asarray(accuracies)
    ece = 0.
    for low in np.linspace(0, .9, 10):
        selected = (p >= low) & (p < low + .1 + (1e-8 if low > .89 else 0))
        if selected.any():
            ece += w[selected].sum() / w.sum() * abs(np.average(p[selected] - labels[selected], weights=w[selected]))
    err = np.abs(ac[amask] - accuracy[amask])
    return {'brier': float(np.average((p - labels) ** 2, weights=w)),
            'log_loss': float(np.average(-labels * np.log(p) - (1 - labels) * np.log(1 - p), weights=w)),
            'ece': float(ece), 'mae_pp': float(np.average(err, weights=w[amask]) * 100) if amask.any() else None,
            'accuracy_error_q90': float(np.quantile(err, .9)) if len(err) else None,
            'attempts': len(indices), 'passes': int(labels.sum()), 'fails': int(len(labels) - labels.sum()),
            'accuracy_attempts': int(amask.sum())}


def subgroup_metrics(corpus, indices, probabilities, accuracies):
    """Fixed descriptive slices; never used to choose a variant or split seed."""
    selectors = {
        'hidden': lambda p: 'HD' in p['mods'],
        'without_hidden': lambda p: 'HD' not in p['mods'],
        'ar_below_8': lambda p: p['analysis']['ar'] < 8,
        'ar_8_to_10': lambda p: 8 <= p['analysis']['ar'] <= 10,
        'ar_above_10': lambda p: p['analysis']['ar'] > 10,
        'sharp_peak': lambda p: p['analysis'].get('peak_to_mean', 0) >= 1.8,
        'ending_peak': lambda p: p['analysis'].get('ending_peak_fraction', 0) >= .8,
    }
    result = {}
    indices = np.asarray(indices)
    for name, selector in selectors.items():
        mask = np.asarray([selector(corpus.records[i]) for i in indices])
        if not mask.any():
            result[name] = {'attempts': 0, 'status': 'no_data'}
            continue
        values = metrics(corpus, indices[mask], np.asarray(probabilities)[mask], np.asarray(accuracies)[mask])
        result[name] = {**values, 'status': 'descriptive_only',
                        'families': len({corpus.families[corpus.records[i]['map_id']] for i in indices[mask]})}
    return result


class Batches:
    def __init__(self, corpus, cache, context, base_p, base_a, device, indices, views=('original',), batch_size=8):
        self.corpus, self.cache, self.context = corpus, cache, context
        self.base_p, self.base_a, self.device = base_p, base_a, device
        self.indices, self.views, self.batch_size = list(indices), tuple(views), batch_size

    def batch(self, pairs):
        notes, masks, ids = [], [], []
        indices = []
        for bag, (index, view) in enumerate(pairs):
            p = self.corpus.records[index]
            sequence = self.cache.get(self.corpus.maps[p['map_id']], p['analysis'], view)
            chunks, mask, _ = windows(sequence)
            notes.append(chunks); masks.append(mask); ids.extend([bag] * len(chunks)); indices.append(index)
        rows = self.corpus.records
        arrays = {'notes': np.concatenate(notes), 'mask': np.concatenate(masks), 'bags': np.asarray(ids, dtype=np.int64),
                  'context': self.context[indices], 'base_p': self.base_p[indices], 'base_a': self.base_a[indices],
                  'pass_y': np.asarray([rows[i]['passed'] for i in indices], dtype=np.float32),
                  'accuracy_y': np.asarray([rows[i]['accuracy'] for i in indices], dtype=np.float32),
                  'accuracy_mask': np.asarray([rows[i]['accuracy_mask'] for i in indices], dtype=np.float32),
                  'weights': self.corpus.weights[indices]}
        return {name: torch.from_numpy(value).to(self.device) for name, value in arrays.items()}

    def epoch(self, rng=None):
        pairs = [(i, view) for i in self.indices for view in self.views]
        if rng is not None: rng.shuffle(pairs)
        for start in range(0, len(pairs), self.batch_size):
            yield pairs[start:start + self.batch_size]


@torch.no_grad()
def evaluate(model, batches):
    model.eval()
    probabilities, accuracies = [], []
    for pairs in batches.epoch():
        b = batches.batch(pairs)
        p, a, _ = model(b['notes'], b['mask'], b['bags'], b['context'], b['base_p'], b['base_a'])
        probabilities.extend(torch.sigmoid(p).float().cpu().tolist()); accuracies.extend(a.float().cpu().tolist())
    return np.asarray(probabilities), np.asarray(accuracies)


def pretrain(corpus, cache, indices, steps, device, progress):
    seed_all(2026)
    encoder = PatternEncoder().to(device)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-3, weight_decay=.01)
    rng = np.random.default_rng(2026)
    # Original views only: the no-augmentation control must not inherit mirrors.
    unique = list({(corpus.records[i]['map_id'], corpus.records[i]['mods']): i for i in indices}.values())
    history = []
    for step in range(steps):
        data, masks = [], []
        for i in rng.choice(unique, size=min(32, len(unique)), replace=False):
            p = corpus.records[i]
            sequence = cache.get(corpus.maps[p['map_id']], p['analysis'])
            chunks, chunk_masks, _ = windows(sequence)
            selected = int(rng.integers(len(chunks)))
            data.append(chunks[selected]); masks.append(chunk_masks[selected])
        x = torch.from_numpy(np.stack(data)).to(device)
        mask = torch.from_numpy(np.stack(masks)).to(device)
        hidden = (torch.rand(mask.shape, device=device) < .15) & mask
        if not hidden.any(): hidden[:, 0] = True
        corrupted = x.masked_fill(hidden[:, None], 0)
        prediction = encoder.reconstruct(corrupted, mask)
        loss = F.smooth_l1_loss(prediction.transpose(1, 2)[hidden], x.transpose(1, 2)[hidden])
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(encoder.parameters(), 2); optimizer.step()
        history.append(float(loss.detach()))
        if step % 20 == 0 or step == steps - 1:
            progress('pretraining', step + 1, steps, {'reconstruction_loss': float(np.mean(history[-20:]))})
    return {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}, history


def cluster_delta(corpus, indices, candidate, reference, head, repeats=500):
    indices = np.asarray(indices)
    mask = np.ones(len(indices), dtype=bool) if head == 'pass' else np.asarray([corpus.records[i]['accuracy_mask'] for i in indices])
    actual = np.asarray([corpus.records[i]['passed'] if head == 'pass' else corpus.records[i]['accuracy'] for i in indices])[mask]
    candidate, reference = np.asarray(candidate)[mask], np.asarray(reference)[mask]
    groups = [corpus.families[corpus.records[i]['map_id']] for i in indices[mask]]
    weight = corpus.weights[indices][mask]
    difference = ((candidate - actual) ** 2 - (reference - actual) ** 2) if head == 'pass' else (np.abs(candidate - actual) - np.abs(reference - actual))
    sums = {}
    for g in sorted(set(groups)):
        selected = np.asarray([group == g for group in groups])
        sums[g] = (float(np.sum(weight[selected] * difference[selected])), float(weight[selected].sum()))
    values = np.asarray(list(sums.values()))
    rng = np.random.default_rng(2026)
    samples = []
    for _ in range(repeats):
        picked = values[rng.integers(0, len(values), len(values))]
        samples.append(picked[:, 0].sum() / picked[:, 1].sum())
    return {'delta': float(np.average(difference, weights=weight)), 'upper95': float(np.quantile(samples, .95)),
            'families': len(values), 'bootstrap_repeats': repeats}


def train_experiment(store, user, output, *, steps=400, pretrain_steps=300, batch_size=8, device='auto'):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    def progress(phase, step=0, total=0, extra=None):
        write_json(output / 'progress.json', {'status': 'running', 'phase': phase, 'step': step, 'total': total,
                                             'elapsed_seconds': round(time.time() - started, 1), **(extra or {})})
    device = torch.device('cuda' if device == 'auto' and torch.cuda.is_available() else 'cpu' if device == 'auto' else device)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    seed_all(2026)
    progress('snapshot')
    corpus = build_corpus(store, user)
    write_json(output / 'manifest.json', corpus.manifest)
    write_corpus_snapshot(corpus, output)
    train_ids = corpus.manifest['splits']['train']['indices']
    val_ids = corpus.manifest['splits']['validation']['indices']
    test_ids = corpus.manifest['splits']['test']['indices']
    baseline, base_p, base_a = fitted_baselines(corpus, train_ids)
    mean = np.average(corpus.contexts[train_ids], axis=0, weights=corpus.weights[train_ids])
    variance = np.average((corpus.contexts[train_ids] - mean) ** 2, axis=0, weights=corpus.weights[train_ids])
    scale = np.maximum(np.sqrt(variance), .01)
    context = ((corpus.contexts - mean) / scale).astype(np.float32)
    cache = SequenceCache(store)
    # Persist all base sequences once and report progress during potentially long parsing.
    unique = list({(p['map_id'], p['mods']): p for p in corpus.records}.values())
    for i, p in enumerate(unique):
        cache.get(corpus.maps[p['map_id']], p['analysis'])
        if i % 30 == 0: progress('sequences', i + 1, len(unique))
    pretrained, pretraining_loss = pretrain(corpus, cache, train_ids, pretrain_steps, device, progress)
    predictions, summaries = {}, {}
    all_indices = {'validation': val_ids, 'test': test_ids}
    for name, views in VARIANTS.items():
        seed_all(2026)
        model = PatternModel(len(CONTEXT_FEATURES), sequence=name != 'context_only').to(device)
        model.encoder.load_state_dict(pretrained)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
        batches = Batches(corpus, cache, context, base_p, base_a, device, train_ids, views, batch_size)
        validation = Batches(corpus, cache, context, base_p, base_a, device, val_ids, batch_size=batch_size)
        rng = np.random.default_rng(2026)
        iterator = iter(batches.epoch(rng))
        best_state, best_metric, best_step = None, float('inf'), 0
        train_weight_mean = float(corpus.weights[train_ids].mean())
        accuracy_coverage = np.mean([corpus.records[i]['accuracy_mask'] for i in train_ids])
        baseline_metrics = metrics(corpus, val_ids, 1 / (1 + np.exp(-base_p[val_ids])), base_a[val_ids])
        history = []
        for step in range(steps):
            try: pairs = next(iterator)
            except StopIteration:
                iterator = iter(batches.epoch(rng)); pairs = next(iterator)
            b = batches.batch(pairs)
            model.train()
            p, a, _ = model(b['notes'], b['mask'], b['bags'], b['context'], b['base_p'], b['base_a'])
            loss_p = (F.binary_cross_entropy_with_logits(p, b['pass_y'], reduction='none') * b['weights']).mean() / train_weight_mean
            loss_a = (((a - b['accuracy_y']) ** 2) * b['weights'] * b['accuracy_mask']).mean() / max(1e-8, train_weight_mean * accuracy_coverage)
            loss = loss_p + 50 * loss_a
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2); optimizer.step()
            if (step + 1) % 25 == 0 or step == steps - 1:
                vp, va = evaluate(model, validation)
                score = metrics(corpus, val_ids, vp, va)
                objective = score['brier'] / max(1e-5, baseline_metrics['brier']) + score['mae_pp'] / max(.01, baseline_metrics['mae_pp'])
                history.append({'step': step + 1, 'objective': objective, **score})
                if objective < best_metric:
                    best_metric, best_step = objective, step + 1
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                progress(name, step + 1, steps, {'validation': score})
        model.load_state_dict(best_state)
        weight_path = output / f'{name}.npz'
        save_weights(model, weight_path)
        predictions[name], summaries[name] = {}, {'views': list(views), 'best_step': best_step, 'training_curve': history}
        # Keep test labels unused for selection below; predictions alone are harmless.
        for split, ids in all_indices.items():
            split_batches = Batches(corpus, cache, context, base_p, base_a, device, ids, batch_size=batch_size)
            pp, aa = evaluate(model, split_batches)
            predictions[name][split] = (pp, aa)
        np.savez_compressed(output / f'{name}-predictions.npz',
                            validation_pass=predictions[name]['validation'][0], validation_accuracy=predictions[name]['validation'][1],
                            test_pass=predictions[name]['test'][0], test_accuracy=predictions[name]['test'][1])
        del model
    # Lock both selections and references using validation ONLY, before test scores.
    val_scores = {name: metrics(corpus, val_ids, *values['validation']) for name, values in predictions.items()}
    structured_val = metrics(corpus, val_ids, 1 / (1 + np.exp(-base_p[val_ids])), base_a[val_ids])
    selected = {head: min((name for name in VARIANTS if name != 'context_only'), key=lambda n: val_scores[n][metric])
                for head, metric in [('pass', 'brier'), ('accuracy', 'mae_pp')]}
    reference = {head: 'context_only' if val_scores['context_only'][metric] < structured_val[metric] else 'structured'
                 for head, metric in [('pass', 'brier'), ('accuracy', 'mae_pp')]}
    report = {'schema': SCHEMA, 'status': 'completed', 'created_at': datetime.now(timezone.utc).isoformat(),
              'fingerprint': corpus.fingerprint, 'dataset': corpus.manifest, 'selected_on_validation': selected,
              'reference_on_validation': reference, 'steps_per_variant': steps, 'pretrain_steps': pretrain_steps,
              'pretraining_original_only': True, 'pretraining_loss': pretraining_loss,
              'device': str(device), 'torch': torch.__version__, 'independent_attempts': len(corpus.records),
              'augmentation_label_policy': 'original, x, y, xy preserve labels; no arbitrary rotations; no new plays',
              'context_features': list(CONTEXT_FEATURES), 'note_features': list(FEATURES),
              'variants': {}, 'release': {}, 'elapsed_seconds': round(time.time() - started, 1)}
    for name in VARIANTS:
        report['variants'][name] = {**summaries[name], 'validation': val_scores[name],
                                    'test': metrics(corpus, test_ids, *predictions[name]['test']),
                                    'test_subgroups': subgroup_metrics(corpus, test_ids, *predictions[name]['test'])}
    report['structured'] = {'validation': structured_val,
                             'test': metrics(corpus, test_ids, 1 / (1 + np.exp(-base_p[test_ids])), base_a[test_ids])}
    for head, metric, position in [('pass', 'brier', 0), ('accuracy', 'mae_pp', 1)]:
        candidate, ref_name = selected[head], reference[head]
        ref_values = predictions[ref_name]['test'][position] if ref_name != 'structured' else (1 / (1 + np.exp(-base_p[test_ids])) if head == 'pass' else base_a[test_ids])
        candidate_score = report['variants'][candidate]['test']
        ref_score = report['variants'][ref_name]['test'] if ref_name != 'structured' else report['structured']['test']
        interval = cluster_delta(corpus, test_ids, predictions[candidate]['test'][position], ref_values, head)
        validation_ref = val_scores[ref_name][metric] if ref_name != 'structured' else structured_val[metric]
        enough = candidate_score['fails'] >= 8 and candidate_score['passes'] >= 8 if head == 'pass' else candidate_score['accuracy_attempts'] >= 40
        improves = val_scores[candidate][metric] < validation_ref and candidate_score[metric] < ref_score[metric] and interval['upper95'] < 0
        calibration = head != 'pass' or candidate_score['ece'] <= ref_score['ece'] + .02
        accepted = bool(enough and improves and calibration)
        report['release'][head] = {'accepted': accepted, 'variant': candidate, 'reference': ref_name,
                                   'reason': 'accepted' if accepted else 'insufficient_validation' if not enough else 'no_verified_improvement',
                                   'paired_family_bootstrap': interval, 'test_metric': candidate_score[metric], 'reference_metric': ref_score[metric]}
    artifact = {'schema': SCHEMA, 'sequence_version': VERSION, 'fingerprint': corpus.fingerprint, 'user': user,
                'baseline': baseline, 'context_mean': mean.tolist(), 'context_scale': scale.tolist(),
                'release': report['release'], 'selected': selected,
                'features': list(FEATURES), 'context_features': list(CONTEXT_FEATURES),
                'support': {'mods': sorted({corpus.records[i]['mods'] for i in train_ids}),
                            'validated_mods': validated_mods(corpus.records, corpus.families, test_ids),
                            'stars': [min(corpus.records[i]['analysis']['stars'] for i in train_ids), max(corpus.records[i]['analysis']['stars'] for i in train_ids)]},
                'weight_hashes': {name: hashlib.sha256((output / f'{name}.npz').read_bytes()).hexdigest() for name in VARIANTS},
                'model_options': {'channels': 32, 'context_size': len(CONTEXT_FEATURES)}}
    write_json(output / 'artifact.json', artifact)
    write_json(output / 'report.json', report)
    progress('completed', 1, 1, {'status': 'completed', 'release': report['release']})
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('user'); parser.add_argument('--database', default='.data/osu_skill.db')
    parser.add_argument('--output', required=True); parser.add_argument('--steps', type=int, default=400)
    parser.add_argument('--audit', action='store_true', help='Verify saved weights against the frozen snapshot; no training.')
    parser.add_argument('--pretrain-steps', type=int, default=300); parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    args = parser.parse_args()
    if args.audit:
        audit = audit_experiment(args.output, device='cpu' if args.device == 'auto' else args.device)
        print(json.dumps({'status': audit['status'], 'fingerprint': audit['fingerprint']}))
        raise SystemExit(0)
    try:
        if not 1 <= args.steps <= 5000 or not 0 <= args.pretrain_steps <= 5000 or not 1 <= args.batch_size <= 32:
            raise ValueError('Invalid training budget.')
        report = train_experiment(Store(args.database), args.user, args.output, steps=args.steps,
                                  pretrain_steps=args.pretrain_steps, batch_size=args.batch_size, device=args.device)
        print(json.dumps({'status': report['status'], 'release': report['release']}, ensure_ascii=True))
    except Exception as exc:
        write_json(Path(args.output) / 'progress.json', {'status': 'failed', 'error': str(exc)})
        raise
