"""Evaluate immutable CNN checkpoints on plays made after model creation.

python -m app.sequence_forward USER --run PATH
No training, checkpoint selection, or automatic release changes happen here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from .ml import BEST_SOURCES, vector
from .recency import played_date, sample_weights
from .sequence_data import Corpus, SequenceCache, family_groups
from .storage import Store


def forward_rows(history, cutoff, known_ids, reference):
    """Outcome of the current or a later attempt never enters its context."""
    from .sequence_service import current_profile
    rows, contexts = [], []
    for p in sorted(history, key=lambda p: (played_date(p) or datetime.min.replace(tzinfo=timezone.utc),p['id'])):
        date = played_date(p)
        if (not date or date <= cutoff or p['id'] in known_ids or p['source'] in BEST_SOURCES
                or not p.get('passed_known',1) or 'NF' in p['mods'] or not p.get('analysis')):
            continue
        prior = [r for r in history if r.get('analysis') and r['analysis']['checksum'] in reference]
        profile = current_profile(prior,now=date)
        rows.append({**p,'pass_mask':True,'accuracy_mask':bool(p['passed']) and bool(p.get('complete',1))})
        contexts.append([*vector(p['analysis']),*profile])
    return rows,contexts


def evaluate_forward(store,user,run):
    from .sequence_service import _load
    from .sequence_train import Batches, evaluate, metrics, write_json, write_corpus_snapshot, load_corpus_snapshot
    run=Path(run)
    report=json.loads((run/'report.json').read_text(encoding='utf-8'))
    artifact=json.loads((run/'artifact.json').read_text(encoding='utf-8'))
    frozen_corpus=load_corpus_snapshot(run)
    frozen=frozen_corpus.records
    map_index=json.loads((run/'snapshot/maps.json').read_text(encoding='utf-8'))
    cutoff=datetime.fromisoformat(report['created_at'].replace('Z','+00:00'))
    train_families=set(report['dataset']['splits']['train']['families'])
    reference={p['source_checksum'] for p in map_index.values() if p['family'] in train_families}
    if report['fingerprint']!=artifact['fingerprint']:
        raise ValueError('Checkpoint fingerprint mismatch.')
    if artifact['user'].strip().casefold()!=user.strip().casefold():
        raise ValueError('Checkpoint belongs to a different player.')
    rows,contexts=forward_rows(store.plays_with_analysis(user),cutoff,{p['id'] for p in frozen},reference)
    result={'status':'awaiting_new_plays' if not rows else 'evaluated', 'created_at':datetime.now(timezone.utc).isoformat(),
        'model_created_at':report['created_at'],'fingerprint':report['fingerprint'],'attempts':len(rows),
        'notes':'Only plays after checkpoint creation; context is strictly earlier and restricted to training families. No model or release changes.'}
    if not rows:
        write_json(run/'forward.json',result);return result
    maps={p['map_id']:store.get_map(p['map_id']) for p in rows}
    if any(p is None for p in maps.values()):raise ValueError('Missing map for a future attempt.')
    families=family_groups(maps);raw=np.asarray(contexts,dtype=np.float32)
    ids=list(range(len(rows)))
    identity=[(p['id'],p['played_at'],p['analysis']['checksum'],p['passed'],p['accuracy']) for p in rows]
    fingerprint=hashlib.sha256(json.dumps([report['fingerprint'],identity,contexts]).encode()).hexdigest()
    manifest={'schema':'cnn-forward-v1','fingerprint':fingerprint,'user':user,
        'splits':{'test':{'indices':ids}},'protocol':'strictly_after_checkpoint_creation'}
    corpus=Corpus(store,rows,maps,families,{g:'test' for g in families.values()},raw,
        np.asarray(sample_weights(rows),dtype=np.float32),fingerprint,manifest)
    x=raw[:,:len(artifact['baseline']['pass']['coef'])]
    def linear(m):return (((x-m['mean'])/m['scale'])@np.asarray(m['coef'])+m['intercept']).astype(np.float32)
    bp=linear(artifact['baseline']['pass']);ba=np.clip(linear(artifact['baseline']['accuracy']),0,1)
    context=((raw-artifact['context_mean'])/artifact['context_scale']).astype(np.float32)
    import torch
    torch.set_num_threads(min(4,torch.get_num_threads()))
    cache=SequenceCache(store);batches=Batches(corpus,cache,context,bp,ba,'cpu',ids)
    variants={**report['selected_on_validation'],**{f'reference_{h}':v for h,v in report['reference_on_validation'].items()}}
    checked={}
    for name in sorted(set(variants.values())-{'structured'}):
        if name=='context_only':
            from .sequence_net import PatternModel,load_weights
            path=run/f'{name}.npz'
            if hashlib.sha256(path.read_bytes()).hexdigest()!=artifact['weight_hashes'][name]:raise ValueError('Weight checksum mismatch.')
            model=load_weights(PatternModel(**artifact['model_options'],sequence=False),path).eval()
        else:_,model=_load(str(run.resolve()),name,(run/'artifact.json').stat().st_mtime_ns)
        pp,aa=evaluate(model,batches)
        checked[name]=metrics(corpus,ids,pp,aa)
    checked['structured']=metrics(corpus,ids,1/(1+np.exp(-np.clip(bp,-30,30))),ba)
    result.update(metrics=checked,selected_before_plays=variants,source_fingerprint=fingerprint,
        pass_validation_sufficient=sum(not p['passed'] for p in rows)>=8 and sum(p['passed'] for p in rows)>=8,
        accuracy_validation_sufficient=sum(p['accuracy_mask'] for p in rows)>=40)
    folder=run/'forward-evaluations'/fingerprint[:24]
    write_json(folder/'manifest.json',manifest);write_corpus_snapshot(corpus,folder)
    write_json(folder/'report.json',result);write_json(run/'forward.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('user');parser.add_argument('--run',required=True);parser.add_argument('--database',default='.data/osu_skill.db')
    args=parser.parse_args()
    print(json.dumps(evaluate_forward(Store(args.database),args.user,args.run),ensure_ascii=True))
