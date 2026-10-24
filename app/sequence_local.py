"""Train and audit replay-supervised local note-error CNNs.

python -m app.sequence_local USER --output .data/local-experiment --collect 120
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch.nn import functional as F

from .ml import _fit
from .sequence import FEATURES, VIEWS, VERSION, windows
from .sequence_data import CONTEXT_FEATURES, SequenceCache
from .sequence_labels import TARGETS, build_event_corpus
from .sequence_net import LocalPatternModel, save_weights, load_weights
from .sequence_train import seed_all, pretrain, write_json
from .storage import Store

SCHEMA = 'local-pattern-cnn-v1'
VARIANTS = {'original': ('original',), 'four_views': VIEWS}


def _note_features(sequence, context):
    return np.concatenate([sequence.values, np.broadcast_to(context, (len(sequence.values), len(context)))], axis=1)


def fit_baselines(corpus, labels, masks, cache, train_ids):
    features = [_note_features(cache.get(corpus.maps[corpus.records[i]['map_id']], corpus.records[i]['analysis']), corpus.contexts[i]) for i in train_ids]
    x = np.concatenate(features).astype(float)
    models = []
    for k in range(len(TARGETS)):
        mask = np.concatenate([masks[i][:, k] for i in train_ids])
        y = np.concatenate([labels[i][:, k] for i in train_ids])[mask]
        w = np.concatenate([np.full(len(labels[i]), corpus.weights[i] / max(1, masks[i][:, k].sum())) for i in train_ids])[mask]
        if not len(y): raise ValueError('Нет проверенных меток одного из локальных выходов.')
        if len(set(y)) == 2:
            scaler, classifier = _fit(x[mask], y, w)
            models.append({'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(),
                           'coef': classifier.coef_[0].tolist(), 'intercept': float(classifier.intercept_[0])})
        else:
            rate = float((np.sum(w * y) + 1) / (w.sum() + 2))
            models.append({'mean': [0.] * x.shape[1], 'scale': [1.] * x.shape[1],
                           'coef': [0.] * x.shape[1], 'intercept': float(np.log(rate / (1 - rate)))})
    return models


def baseline_logits(sequence, context, models):
    x = _note_features(sequence, context)
    return np.asarray([((x - m['mean']) / m['scale']) @ np.asarray(m['coef']) + m['intercept'] for m in models], dtype=np.float32).T


class LocalBatches:
    def __init__(self, corpus, labels, masks, cache, context, models, device):
        self.corpus, self.labels, self.masks, self.cache = corpus, labels, masks, cache
        self.context, self.models, self.device = context, models, device
        self.original_logits = {}

    def sequence(self, index, view='original'):
        p = self.corpus.records[index]
        return self.cache.get(self.corpus.maps[p['map_id']], p['analysis'], view)

    def logits(self, i, sequence, view):
        # A reflected input must receive its reflected linear baseline as well.
        if view == 'original' and i in self.original_logits: return self.original_logits[i]
        result = baseline_logits(sequence, self.corpus.contexts[i], self.models)
        if view == 'original': self.original_logits[i] = result
        return result

    def batch(self, pairs):
        data = {key: [] for key in ('notes', 'mask', 'context', 'baseline', 'labels', 'weights')}
        for i, view, window in pairs:
            sequence = self.sequence(i, view)
            chunks, note_mask, starts = windows(sequence)
            start, n = int(starts[window]), int(note_mask[window].sum())
            size = chunks.shape[-1]
            labels = np.zeros((len(TARGETS), size), dtype=np.float32)
            labels[:, :n] = self.labels[i][start:start+n].T
            weights = np.zeros_like(labels)
            for k in range(len(TARGETS)):
                # One uniformly sampled window per selected attempt. Correct for
                # overlaps, so the same root is not counted repeatedly.
                coverage = np.zeros(len(sequence.values), dtype=np.float32)
                for pos in starts: coverage[pos:min(pos+size, len(coverage))] += 1
                weights[k, :n] = (self.masks[i][start:start+n, k] / coverage[start:start+n]
                    * len(starts) / max(1, self.masks[i][:, k].sum()) * self.corpus.weights[i])
            baseline = np.zeros_like(labels)
            baseline[:, :n] = self.logits(i, sequence, view)[start:start+n].T
            data['notes'].append(chunks[window]); data['mask'].append(note_mask[window])
            data['context'].append(self.context[i]); data['baseline'].append(baseline)
            data['labels'].append(labels); data['weights'].append(weights)
        return {k: torch.from_numpy(np.stack(v)).to(self.device) for k, v in data.items()}


@torch.no_grad()
def predict_notes(model, sequence, context, raw_context, models, device='cpu'):
    """Average overlapping window probabilities to one prediction per root."""
    model.eval()
    chunks, masks, starts = windows(sequence)
    logits = baseline_logits(sequence, raw_context, models)
    total = np.zeros((len(sequence.values), len(TARGETS)), dtype=float)
    count = np.zeros(len(sequence.values), dtype=float)
    for begin in range(0, len(chunks), 32):
        part, valid = chunks[begin:begin+32], masks[begin:begin+32]
        base = np.zeros((len(part), len(TARGETS), chunks.shape[-1]), dtype=np.float32)
        for j in range(len(part)):
            start, n = int(starts[begin+j]), int(valid[j].sum())
            base[j, :, :n] = logits[start:start+n].T
        ctx = np.broadcast_to(np.asarray(context, dtype=np.float32), (len(part), len(context))).copy()
        prediction = torch.sigmoid(model(torch.from_numpy(part).to(device), torch.from_numpy(valid).to(device),
            torch.from_numpy(ctx).to(device), torch.from_numpy(base).to(device))).cpu().numpy()
        for j in range(len(part)):
            start, n = int(starts[begin+j]), int(valid[j].sum())
            total[start:start+n] += prediction[j, :, :n].T
            count[start:start+n] += 1
    return total / count[:, None]


def evaluate_notes(model, corpus, batches, indices):
    return [predict_notes(model, batches.sequence(i), batches.context[i], corpus.contexts[i], batches.models, batches.device) for i in indices]


def local_metrics(corpus, labels, masks, indices, predictions):
    result = {}
    for k, target in enumerate(TARGETS):
        ys, ps, ws = [], [], []
        for i, prediction in zip(indices, predictions):
            mask = masks[i][:, k]
            ys.extend(labels[i][mask, k]); ps.extend(prediction[mask, k])
            ws.extend([float(corpus.weights[i]) / max(1, mask.sum())] * int(mask.sum()))
        y, p, w = np.asarray(ys), np.clip(ps, 1e-6, 1-1e-6), np.asarray(ws)
        if not len(y): raise ValueError('Нет меток в проверочной части.')
        ece = 0.
        for lo in np.linspace(0, .9, 10):
            sel = (p >= lo) & (p < lo + .1 + (1e-8 if lo > .89 else 0))
            if sel.any(): ece += w[sel].sum()/w.sum()*abs(np.average(p[sel]-y[sel], weights=w[sel]))
        result[target] = {'brier': float(np.average((p-y)**2, weights=w)), 'ece': float(ece),
            'log_loss': float(np.average(-y*np.log(p)-(1-y)*np.log(1-p), weights=w)),
            'notes': len(y), 'positive_notes': int(y.sum()), 'attempts': len(indices)}
    return result


def paired_delta(corpus, labels, masks, indices, candidate, reference, k):
    grouped = {}
    for i, p, ref in zip(indices, candidate, reference):
        mask = masks[i][:, k]
        if not mask.any(): continue
        difference = float(np.mean((p[mask, k]-labels[i][mask, k])**2 - (ref[mask, k]-labels[i][mask, k])**2))
        family = corpus.families[corpus.records[i]['map_id']]
        total, mass = grouped.get(family, (0., 0.))
        grouped[family] = total+difference*corpus.weights[i], mass+corpus.weights[i]
    values = np.asarray([grouped[g] for g in sorted(grouped)])
    rng = np.random.default_rng(2026)
    samples = []
    for _ in range(500):
        selected = values[rng.integers(len(values), size=len(values))]
        samples.append(selected[:, 0].sum()/selected[:, 1].sum())
    return {'delta': float(values[:, 0].sum()/values[:, 1].sum()), 'upper95': float(np.quantile(samples, .95)),
            'families': len(values), 'bootstrap_repeats': 500}


def write_snapshot(output, corpus, labels, masks):
    from .sequence_train import write_corpus_snapshot
    write_corpus_snapshot(corpus, output)
    folder = Path(output) / 'snapshot'
    arrays = {f'y_{i}': y for i, y in enumerate(labels)}
    arrays.update({f'mask_{i}': mask for i, mask in enumerate(masks)})
    path = folder/'labels.npz'
    np.savez_compressed(path, **arrays)
    meta = json.loads((folder/'snapshot.json').read_text(encoding='utf-8'))
    meta['labels_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(folder/'snapshot.json', meta)


def load_snapshot(output):
    from .sequence_train import load_corpus_snapshot
    output = Path(output)
    corpus = load_corpus_snapshot(output)
    meta = json.loads((output/'snapshot/snapshot.json').read_text(encoding='utf-8'))
    path = output/'snapshot/labels.npz'
    if hashlib.sha256(path.read_bytes()).hexdigest() != meta['labels_sha256']:
        raise ValueError('Local labels failed checksum verification.')
    labels, masks = [], []
    with np.load(path, allow_pickle=False) as data:
        for i in range(len(corpus.records)):
            y, mask = data[f'y_{i}'].copy(), data[f'mask_{i}'].copy()
            if y.ndim != 2 or y.shape[1] != len(TARGETS) or y.shape != mask.shape or mask.dtype != bool:
                raise ValueError('Invalid frozen local labels.')
            if not np.isfinite(y).all() or not np.isin(y, [0, 1]).all():
                raise ValueError('Invalid frozen label values.')
            labels.append(y); masks.append(mask)
    return corpus, labels, masks


def audit_local(output, device='cpu'):
    output = Path(output)
    corpus, labels, masks = load_snapshot(output)
    artifact=json.loads((output/'artifact.json').read_text(encoding='utf-8'))
    report=json.loads((output/'report.json').read_text(encoding='utf-8'))
    if (artifact['fingerprint'] != corpus.fingerprint or report['fingerprint'] != corpus.fingerprint
            or artifact['targets'] != list(TARGETS)):
        raise ValueError('Local weights and frozen labels do not match.')
    torch.set_num_threads(min(4,torch.get_num_threads()))
    context=((corpus.contexts-artifact['context_mean'])/artifact['context_scale']).astype(np.float32)
    cache=SequenceCache(type('SnapshotStore',(),{'path':output/'audit-cache.db'})())
    batches=LocalBatches(corpus,labels,masks,cache,context,artifact['baseline'],device)
    result={'status':'verified','fingerprint':corpus.fingerprint,'variants':{},'device':device}
    for v in VARIANTS:
        path=output/f'{v}.npz'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=artifact['weight_hashes'][v]:
            raise ValueError('Local weights failed checksum verification.')
        model=load_weights(LocalPatternModel(**artifact['model_options']),path).to(device)
        result['variants'][v]={}
        for split in ('validation','test'):
            ids=corpus.manifest['splits'][split]['indices']
            predictions=evaluate_notes(model,corpus,batches,ids)
            actual=local_metrics(corpus,labels,masks,ids,predictions)
            for t in TARGETS:
                for key in ('brier','log_loss','ece'):
                    if not np.isclose(actual[t][key],report['variants'][v][split][t][key],atol=2e-5,rtol=2e-5):
                        raise ValueError(f'Local model audit mismatch: {v}/{split}/{t}/{key}')
            result['variants'][v][split]=actual
    write_json(output/'audit.json',result)
    return result


def train_local(store, user, output, *, steps=400, pretrain_steps=200, device='auto'):
    started = time.time(); output = Path(output); output.mkdir(parents=True, exist_ok=True)
    def progress(phase, step=0, total=0, extra=None):
        write_json(output/'progress.json', {'status':'running','phase':phase,'step':step,'total':total,
            'elapsed_seconds':round(time.time()-started, 1), **(extra or {})})
    progress('snapshot')
    device = torch.device('cuda' if device == 'auto' and torch.cuda.is_available() else 'cpu' if device == 'auto' else device)
    torch.set_num_threads(min(4, torch.get_num_threads())); seed_all(2026)
    cache = SequenceCache(store)
    corpus, labels, masks = build_event_corpus(store, user, cache)
    write_json(output/'manifest.json', corpus.manifest)
    write_snapshot(output, corpus, labels, masks)
    train, val, test = [corpus.manifest['splits'][s]['indices'] for s in ('train','validation','test')]
    models = fit_baselines(corpus, labels, masks, cache, train)
    mean = np.average(corpus.contexts[train], axis=0, weights=corpus.weights[train])
    scale = np.maximum(.01, np.sqrt(np.average((corpus.contexts[train]-mean)**2, axis=0, weights=corpus.weights[train])))
    context = ((corpus.contexts-mean)/scale).astype(np.float32)
    batches = LocalBatches(corpus, labels, masks, cache, context, models, device)
    baseline = {s: [1/(1+np.exp(-np.clip(batches.logits(i, batches.sequence(i), 'original'), -30, 30))) for i in ids]
                for s, ids in [('validation',val),('test',test)]}
    pretrained, loss = pretrain(corpus, cache, train, pretrain_steps, device, progress)
    predictions, report_variants = {}, {}
    train_mean = float(corpus.weights[train].mean())
    for variant, views in VARIANTS.items():
        seed_all(2026); rng = np.random.default_rng(2026)
        model = LocalPatternModel(len(CONTEXT_FEATURES)).to(device)
        model.encoder.load_state_dict(pretrained)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
        best_state, best_score, best_step = None, float('inf'), 0
        history = []
        for step in range(steps):
            pairs = []
            for i in rng.choice(train, size=min(16, len(train)), replace=False):
                view = views[int(rng.integers(len(views)))]; sequence = batches.sequence(i, view)
                pairs.append((i, view, int(rng.integers(len(windows(sequence)[0])))))
            b = batches.batch(pairs); model.train()
            logits = model(b['notes'], b['mask'], b['context'], b['baseline'])
            loss_value = (F.binary_cross_entropy_with_logits(logits, b['labels'], reduction='none')*b['weights']).sum() / (len(pairs)*len(TARGETS)*train_mean)
            optimizer.zero_grad(); loss_value.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2); optimizer.step()
            if (step+1)%25==0 or step == steps-1:
                pred = evaluate_notes(model, corpus, batches, val)
                scores = local_metrics(corpus, labels, masks, val, pred)
                objective = sum(scores[t]['brier'] for t in TARGETS)
                history.append({'step':step+1, 'objective':objective})
                if objective < best_score:
                    best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                    best_score,best_step=objective,step+1
                progress(variant,step+1,steps,{'validation':scores})
        model.load_state_dict(best_state); save_weights(model, output/f'{variant}.npz')
        predictions[variant] = {s:evaluate_notes(model,corpus,batches,ids) for s,ids in [('validation',val),('test',test)]}
        report_variants[variant] = {'views':list(views),'best_step':best_step,'training_curve':history,
            'validation':local_metrics(corpus,labels,masks,val,predictions[variant]['validation'])}
    # Lock candidate selection before reading any test scores.
    selected = {t:min(VARIANTS, key=lambda v:report_variants[v]['validation'][t]['brier']) for t in TARGETS}
    report = {'schema':SCHEMA,'status':'completed','created_at':datetime.now(timezone.utc).isoformat(),
        'fingerprint':corpus.fingerprint,'dataset':corpus.manifest,'independent_attempts':len(corpus.records),
        'population':'Full locally stored replays only; this is not a random sample of all attempts.',
        'steps_per_variant':steps,'pretrain_steps':pretrain_steps,'pretraining_original_only':True,
        'selected_on_validation':selected,'variants':report_variants,'baseline':{},'release':{},
        'device':str(device),'elapsed_seconds':round(time.time()-started,1)}
    for s,ids in [('validation',val),('test',test)]:report['baseline'][s]=local_metrics(corpus,labels,masks,ids,baseline[s])
    for v in VARIANTS:
        report['variants'][v]['test']=local_metrics(corpus,labels,masks,test,predictions[v]['test'])
        np.savez_compressed(output/f'{v}-predictions.npz', **{f'{s}_{i}':p for s in ('validation','test') for i,p in enumerate(predictions[v][s])})
    for k,t in enumerate(TARGETS):
        v=selected[t]; candidate=report['variants'][v]['test'][t]; ref=report['baseline']['test'][t]
        interval=paired_delta(corpus,labels,masks,test,predictions[v]['test'],baseline['test'],k)
        enough = (len(test)>=10 and interval['families']>=10 and candidate['positive_notes']>=40
                  and candidate['notes']-candidate['positive_notes']>=40)
        improves=(report['variants'][v]['validation'][t]['brier']<report['baseline']['validation'][t]['brier']
                  and candidate['brier']<ref['brier'] and interval['upper95']<0 and candidate['ece']<=ref['ece']+.02)
        accepted=bool(enough and improves)
        report['release'][t]={'accepted':accepted,'variant':v,'test_brier':candidate['brier'],'reference_brier':ref['brier'],
            'reason':'accepted' if accepted else 'insufficient_validation' if not enough else 'no_verified_improvement',
            'paired_family_bootstrap':interval}
    mod_support = {t: [] for t in TARGETS}
    report['mod_validation'] = {}
    for k,t in enumerate(TARGETS):
        v = selected[t]
        report['mod_validation'][t] = {}
        for mods in sorted({corpus.records[i]['mods'] for i in train}):
            selected_val = [j for j,i in enumerate(val) if corpus.records[i]['mods'] == mods]
            selected_test = [j for j,i in enumerate(test) if corpus.records[i]['mods'] == mods]
            ids = [test[j] for j in selected_test]
            groups = {corpus.families[corpus.records[i]['map_id']] for i in ids}
            coverage = {'attempts':len(ids),'families':len(groups),'accepted':False,'reason':'insufficient_validation'}
            if len(ids) >= 5 and len(groups) >= 5 and len(selected_val) >= 3:
                cp = [predictions[v]['test'][j] for j in selected_test]
                rp = [baseline['test'][j] for j in selected_test]
                cm = local_metrics(corpus,labels,masks,ids,cp)[t]
                rm = local_metrics(corpus,labels,masks,ids,rp)[t]
                vi = [val[j] for j in selected_val]
                cv = local_metrics(corpus,labels,masks,vi,[predictions[v]['validation'][j] for j in selected_val])[t]
                rv = local_metrics(corpus,labels,masks,vi,[baseline['validation'][j] for j in selected_val])[t]
                interval = paired_delta(corpus,labels,masks,ids,cp,rp,k)
                enough = cm['positive_notes'] >= 20 and cm['notes']-cm['positive_notes'] >= 20
                better = cm['brier'] < rm['brier'] and cv['brier'] < rv['brier'] and interval['upper95'] < 0
                valid = bool(enough and better and cm['ece'] <= rm['ece']+.02)
                coverage.update(accepted=valid,reason='accepted' if valid else 'no_verified_improvement',
                                candidate=cm,reference=rm,paired_family_bootstrap=interval)
                if valid: mod_support[t].append(mods)
            report['mod_validation'][t][mods] = coverage
        if report['release'][t]['accepted'] and not mod_support[t]:
            report['release'][t].update(accepted=False,reason='insufficient_mod_validation')
    artifact={'schema':SCHEMA,'sequence_version':VERSION,'fingerprint':corpus.fingerprint,'context_mean':mean.tolist(),'context_scale':scale.tolist(),
        'baseline':models,'features':list(FEATURES),'context_features':list(CONTEXT_FEATURES),'targets':list(TARGETS),
        'release':report['release'],'model_options':{'context_size':len(CONTEXT_FEATURES),'channels':32,'targets':len(TARGETS)},
        'support':{'mods':sorted({corpus.records[i]['mods'] for i in train}),'validated_mods':mod_support,
            'stars':[min(corpus.records[i]['analysis']['stars'] for i in train),max(corpus.records[i]['analysis']['stars'] for i in train)]},
        'weight_hashes':{v:hashlib.sha256((output/f'{v}.npz').read_bytes()).hexdigest() for v in VARIANTS}}
    write_json(output/'artifact.json',artifact);write_json(output/'report.json',report)
    progress('completed',1,1,{'status':'completed','release':report['release']})
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('user');parser.add_argument('--database',default='.data/osu_skill.db');parser.add_argument('--output',required=True)
    parser.add_argument('--steps',type=int,default=400);parser.add_argument('--pretrain-steps',type=int,default=200)
    parser.add_argument('--collect',type=int,default=0);parser.add_argument('--device',choices=['auto','cuda','cpu'],default='auto')
    parser.add_argument('--audit',action='store_true')
    args=parser.parse_args()
    if args.audit:
        result=audit_local(args.output,device='cpu' if args.device=='auto' else args.device)
        print(json.dumps({'status':result['status'],'fingerprint':result['fingerprint']}));raise SystemExit(0)
    if not 1<=args.steps<=5000 or not 0<=args.pretrain_steps<=5000 or not 0<=args.collect<=2000:parser.error('Invalid training budget.')
    try:
        store=Store(args.database)
        if args.collect:
            from .replay_events import collect_events
            write_json(Path(args.output)/'progress.json',{'status':'running','phase':'replays','step':0,'total':args.collect})
            collect_events(store,args.user,limit=args.collect,workers=1)
        report=train_local(store,args.user,args.output,steps=args.steps,pretrain_steps=args.pretrain_steps,device=args.device)
        print(json.dumps({'status':report['status'],'release':report['release']},ensure_ascii=True))
    except Exception as exc:
        write_json(Path(args.output)/'progress.json',{'status':'failed','error':str(exc)})
        raise
