"""Shared models with causal player context and disjoint player/song holdouts.

Training is explicit. Prediction reads plain JSON weights and never starts training.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import math
import os
import sys
import importlib.metadata
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from .ml import BEST_SOURCES, FEATURES, vector
from .population_data import DEFAULT_FOLDER, confirmed_candidates, read_json, write_json, ROOT
from .recency import played_date, sample_weights
from .sequence_data import family_groups
from .storage import Store

SCHEMA = 'population-context-v1'
PROFILE_FEATURES = ('log_history', 'ordinary_pass_rate', 'ordinary_accuracy', 'curated_accuracy',
                    'stars_level','aim_level','speed_level','ar_level','stream_level','burst_level',
                    'rhythm_level','peak_level','hidden_history','hidden_ar_level','ordinary_log_count','curated_log_count')
DELTA_KEYS = ('stars','aim','speed','ar','stream_bpm','burst_bpm','rhythm_variation','peak_load')
ALL_FEATURES = (*FEATURES, *PROFILE_FEATURES, *(f'delta_{k}' for k in DELTA_KEYS),
                *(f'delta_squared_{k}' for k in DELTA_KEYS))


def quantile(values, weights, q):
    if not values: return 0.
    order = np.argsort(values)
    v, w = np.asarray(values)[order], np.asarray(weights)[order]
    return float(v[min(len(v)-1, np.searchsorted(np.cumsum(w), q*w.sum()))])


def context_features(history, now=None):
    history = [p for p in history if 'NF' not in p['mods'] and p.get('analysis')]
    ordinary = sorted((p for p in history if p['source'] not in BEST_SOURCES),
                      key=lambda p:p['played_at'],reverse=True)[:100]
    curated = sorted((p for p in history if p['source'] in BEST_SOURCES),
                     key=lambda p:hashlib.sha256(p['id'].encode()).digest())[:30]
    history = [*ordinary,*curated]
    if len(history) < 3: return None
    weights = np.asarray(sample_weights(history, now=now), dtype=float)
    ordinary = np.asarray([p['source'] not in BEST_SOURCES and bool(p.get('passed_known',1)) for p in history], dtype=bool)
    passed = np.asarray([bool(p['passed']) and bool(p.get('passed_known',1)) for p in history], dtype=bool)
    curated = np.asarray([p['source'] in BEST_SOURCES for p in history])
    def average(mask, key, default):
        return float(np.average([history[i][key] for i in np.flatnonzero(mask)], weights=weights[mask])) if np.any(mask) else default
    skills = []
    for key, q in zip(DELTA_KEYS, (.7,.7,.7,.9,.9,.9,.7,.9)):
        skills.append(quantile([p['analysis'].get(key,0) for p, ok in zip(history,passed) if ok], weights[passed].tolist(), q))
    hd = np.asarray(['HD' in p['mods'] and ok for p,ok in zip(history,passed)])
    pass_rate = (float(sum(w*p['passed'] for p,w,ok in zip(history,weights,ordinary) if ok))+1)/(weights[ordinary].sum()+2)
    values = [math.log1p(weights.sum()), pass_rate, average(ordinary & passed, 'accuracy', .9),
              average(curated & passed, 'accuracy', .9), *skills,
              math.log1p(weights[hd].sum()),
              quantile([p['analysis']['ar'] for p,ok in zip(history,hd) if ok],weights[hd].tolist(),.1),
              math.log1p(weights[ordinary].sum()), math.log1p(weights[curated].sum())]
    return np.asarray(values,dtype=float)


def input_vector(target, profile):
    # Same feature order for training and live inference. No username, rank or current PP.
    delta = np.asarray([float(target.get(k,0)) for k in DELTA_KEYS]) - profile[4:12]
    return np.concatenate([vector(target), profile, delta, np.sign(delta)*np.square(delta)])


def player_assignment(players, seed=20261104):
    bins = defaultdict(list)
    for uid,p in players.items(): bins[p['pp_bin']].append(uid)
    assignment = {}
    for uids in bins.values():
        ordered=sorted(uids,key=lambda uid:hashlib.sha256(f'{seed}:{uid}'.encode()).digest())
        # Small top bands are round-robin, not automatically all put into training.
        for i,uid in enumerate(ordered):
            assignment[uid] = ('train','train','train','validation','test')[i%5]
    return assignment


def build_samples(folder=DEFAULT_FOLDER):
    folder=Path(folder); store=Store(folder/'population.db')
    permitted=confirmed_candidates(ROOT/'deliverables/dataset-candidates.csv',folder/'consents.json')
    snapshots=read_json(folder/'players.json',{})
    players={p['user_id']:{**p, 'username':snapshots.get(p['user_id'],{}).get('username',p['username'])} for p in permitted}
    player_split=player_assignment(players)
    with store.batch():
        records={uid:store.plays_with_analysis(p['username']) for uid,p in players.items()}
        mids={p['map_id'] for rows in records.values() for p in rows if p.get('analysis')}
        maps={mid:store.get_map(mid) for mid in mids}
    families=family_groups(maps)
    family_split={f:('train' if (int(f[:8],16)%100)<60 else 'validation' if (int(f[:8],16)%100)<80 else 'test') for f in set(families.values())}
    samples=[]; skipped=Counter()
    for uid,rows in records.items():
        for p in rows:
            if p['source'] in BEST_SOURCES or not p.get('passed_known',1) or 'NF' in p['mods']:
                continue
            date=played_date(p)
            if not date or p.get('analysis') is None:
                skipped['missing_date_or_analysis']+=1;continue
            family=families[p['map_id']]
            split=player_split[uid]
            if family_split[family] != split:
                skipped['player_family_split_mismatch']+=1;continue
            history=[r for r in rows if r['id']!=p['id'] and played_date(r) is not None and played_date(r)<date
                     and r.get('analysis') is not None and families[r['map_id']] != family
                     and family_split[families[r['map_id']]]=='train']
            profile=context_features(history, now=date)
            if profile is None:
                skipped['insufficient_earlier_history']+=1;continue
            samples.append({'x':input_vector(p['analysis'],profile).tolist(),'passed':int(p['passed']),
                            'accuracy':p['accuracy'] if p['passed'] and p.get('complete',1) else None,
                            'user_id':uid,'pp_bin':players[uid]['pp_bin'],'family':family,'map_id':p['map_id'],
                            'mods':p['mods'],'played_at':p['played_at'],'score_id':p['id'],'split':split,
                            'baseline_pass':float(profile[1]),'baseline_accuracy':float(profile[2]),
                            'history_ids':[r['id'] for r in history]})
    manifest={'schema':SCHEMA,'protocol':'disjoint_players_and_song_families_with_strictly_earlier_train_family_context',
              'seed':20261104,'players':len(players),'samples':len(samples),'skipped':dict(skipped),
              'player_split':player_split,'family_split':family_split,
              'no_current_pp_feature':True,'prospective_validation':False,'features':list(ALL_FEATURES)}
    manifest['software']={'python':sys.version.split()[0],**{p:importlib.metadata.version(p) for p in ('numpy','scikit-learn','rosu-pp-py')}}
    manifest['code_sha256']={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                             for name in ('population_ml.py','population_data.py','ml.py','analyzer.py','recency.py')}
    write_json(folder/'training/samples.json',samples)
    write_json(folder/'training/manifest.json',manifest)
    return samples,manifest


def equal_player_weights(rows):
    counts=Counter(r['user_id'] for r in rows)
    return np.asarray([1/counts[r['user_id']] for r in rows],dtype=float)


def balanced_training_weights(rows):
    users={r['user_id']:r['pp_bin'] for r in rows}
    bins=Counter(users.values())
    return equal_player_weights(rows)/np.asarray([bins[r['pp_bin']] for r in rows],dtype=float)


def summary(rows):
    return {'samples':len(rows),'players':len({r['user_id'] for r in rows}),
            'families':len({r['family'] for r in rows}),
            'passes':sum(r['passed'] for r in rows),'fails':sum(not r['passed'] for r in rows),
            'pp_bins':dict(Counter(r['pp_bin'] for r in rows)), 'mods':dict(Counter(r['mods'] for r in rows))}


def release_gate(metrics, task):
    for split in ('validation','test'):
        part=metrics.get(split,{})
        if part.get('players',0)<5 or part.get('samples',0)<30: return False
        if task=='pass' and min(part.get('passes',0),part.get('fails',0))<8: return False
        if not all(math.isfinite(part.get(k,math.inf)) for k in ('model_error','baseline_error')): return False
        if part.get('model_error',math.inf)>=part.get('baseline_error',-math.inf)-1e-8: return False
    return True


def train_population(folder=DEFAULT_FOLDER):
    folder=Path(folder)
    samples,manifest=build_samples(folder)
    report={'schema':SCHEMA,'trained_at':datetime.now(timezone.utc).isoformat(),
            'protocol':manifest['protocol'],'prospective_validation':False,'tasks':{},
            'splits':{s:summary([r for r in samples if r['split']==s]) for s in ('train','validation','test')},
            'notes':['PP is used for recruitment strata only, not as an outcome or historical feature.',
                     'All targets are ordinary recent attempts; curated scores enter only earlier player context.',
                     'This snapshot measures cross-player and cross-song transfer, not future live performance.']}
    artifact={'schema':SCHEMA,'features':list(ALL_FEATURES),'trained_at':report['trained_at'],'tasks':{}}
    for task in ('pass','accuracy'):
        rows=[r for r in samples if task=='pass' or r['accuracy'] is not None]
        splits={s:[r for r in rows if r['split']==s] for s in ('train','validation','test')}
        training=splits['train']
        if len(training)<80 or len({r['user_id'] for r in training})<8 or (task=='pass' and len({r['passed'] for r in training})<2):
            report['tasks'][task]={'status':'insufficient_data','splits':{s:summary(r) for s,r in splits.items()}}
            continue
        x=np.asarray([r['x'] for r in training],dtype=float)
        y=np.asarray([r['passed'] if task=='pass' else r['accuracy'] for r in training],dtype=float)
        weights=balanced_training_weights(training)
        # Each represented PP band has equal total weight; players within a band are equal.
        weights*=len(weights)/weights.sum()
        scaler=StandardScaler().fit(x,sample_weight=weights)
        estimator=(LogisticRegression(C=.35,max_iter=1500) if task=='pass' else Ridge(alpha=10))
        estimator.fit(scaler.transform(x),y,sample_weight=weights)
        metrics={}; test_errors=[]
        for split,group in splits.items():
            if not group:
                metrics[split]=summary(group);continue
            xx=scaler.transform(np.asarray([r['x'] for r in group],dtype=float))
            yy=np.asarray([r['passed'] if task=='pass' else r['accuracy'] for r in group],dtype=float)
            prediction=estimator.predict_proba(xx)[:,1] if task=='pass' else np.clip(estimator.predict(xx),0,1)
            w=equal_player_weights(group)
            constant=np.full(len(group),np.average(y,weights=weights))
            context=np.asarray([r['baseline_pass'] if task=='pass' else r['baseline_accuracy'] for r in group])
            loss=(lambda a,b:np.square(a-b)) if task=='pass' else (lambda a,b:np.abs(a-b))
            error=float(np.average(loss(yy,prediction),weights=w))
            errors={'constant':float(np.average(loss(yy,constant),weights=w)),
                    'player_history':float(np.average(loss(yy,context),weights=w))}
            by_bin={}
            for band in sorted({r['pp_bin'] for r in group}):
                ix=np.asarray([i for i,r in enumerate(group) if r['pp_bin']==band])
                by_bin[band]={'samples':len(ix),'players':len({group[i]['user_id'] for i in ix}),
                              'model_error':float(np.average(loss(yy[ix],prediction[ix]),weights=w[ix]))}
            metrics[split]={**summary(group),'model_error':error,'baseline_error':min(errors.values()),
                            'baselines':errors,'metric':'brier' if task=='pass' else 'mae_fraction','by_pp_bin':by_bin}
            if task=='accuracy' and split=='validation': test_errors=np.abs(yy-prediction).tolist()
        accepted=release_gate(metrics,task)
        supported=[]
        for mods in sorted({r['mods'] for r in training}):
            evidence=[[r for r in splits[s] if r['mods']==mods] for s in ('validation','test')]
            if all(len(g)>=20 and len({r['user_id'] for r in g})>=3 and (task=='accuracy' or min(sum(r['passed'] for r in g),sum(not r['passed'] for r in g))>=5) for g in evidence):
                supported.append(mods)
        coefficients=estimator.coef_.reshape(-1).tolist()
        values={'status':'accepted' if accepted and supported else 'candidate', 'accepted':accepted and bool(supported),
                'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),'coef':coefficients,
                'intercept':float(np.asarray(estimator.intercept_).reshape(-1)[0]),'metrics':metrics,'supported_mods':supported,
                'stars_range':[float(x[:,0].min()),float(x[:,0].max())],
                'profile_stars_range':[float(x[:,len(FEATURES)+4].min()),float(x[:,len(FEATURES)+4].max())],
                'accuracy_error_q90':float(np.quantile(test_errors,.9)) if test_errors else None}
        reasons=[]
        for split in ('validation','test'):
            part=metrics[split]
            if part.get('players',0)<5:reasons.append(f'{split}_too_few_players')
            if part.get('samples',0)<30:reasons.append(f'{split}_too_few_samples')
            if task=='pass' and min(part.get('passes',0),part.get('fails',0))<8:reasons.append(f'{split}_too_few_outcomes')
            if part.get('model_error',math.inf)>=part.get('baseline_error',-math.inf):reasons.append(f'{split}_not_better_than_baseline')
        if not supported:reasons.append('no_validated_mods')
        values['release_reasons']=reasons
        # Deployment stays inside the skill range actually represented in both holdouts.
        for field,column in [('stars_range',0),('profile_stars_range',len(FEATURES)+4)]:
            evidence=[np.asarray([r['x'][column] for r in splits[s]]) for s in ('validation','test')]
            if all(len(v) for v in evidence):
                values[field]=[max(values[field][0],*(float(v.min()) for v in evidence)),
                               min(values[field][1],*(float(v.max()) for v in evidence))]
        artifact['tasks'][task]=values
        report['tasks'][task]={k:v for k,v in values.items() if k not in ('mean','scale','coef','intercept')}
    fingerprint=hashlib.sha256((folder/'training/samples.json').read_bytes()).hexdigest()
    artifact['fingerprint']=fingerprint;report['fingerprint']=fingerprint
    write_json(folder/'model.json',artifact)
    write_json(folder/'report.json',report)
    snapshot=folder/'runs'/fingerprint[:16]
    snapshot.mkdir(parents=True,exist_ok=True)
    write_json(snapshot/'model.json',artifact);write_json(snapshot/'report.json',report)
    write_json(snapshot/'manifest.json',manifest);write_json(snapshot/'samples.json',samples)
    print({k:v['status'] for k,v in report['tasks'].items()},flush=True)
    return report


def predict_population(target, records, path=None):
    path=Path(path or os.getenv('PULSE_POPULATION_MODEL',str(DEFAULT_FOLDER/'model.json')))
    model=read_json(path,{})
    if model.get('schema')!=SCHEMA or model.get('features')!=list(ALL_FEATURES):
        return {'status':'not_trained'}
    def song(a):
        return tuple(''.join(c for c in a.get(k,'').casefold() if c.isalnum()) for k in ('artist','title'))
    target_song=song(target)
    history=[r for r in records if r['map_id']!=target.get('id')
             and (not all(target_song) or song(r.get('analysis') or {})!=target_song)]
    profile=context_features(history)
    if profile is None: return {'status':'insufficient_history'}
    result={'status':'candidate','pass_probability':None,'accuracy':None,'tasks':{},'metrics':{},'trained_at':model['trained_at']}
    values=input_vector(target,profile)
    for task,m in model.get('tasks',{}).items():
        if not m.get('accepted') or target.get('mods','NM') not in m['supported_mods']:
            result['tasks'][task]='not_validated_for_mods';continue
        if not m['stars_range'][0]<=target['stars']<=m['stars_range'][1] or not m['profile_stars_range'][0]<=profile[4]<=m['profile_stars_range'][1]:
            result['tasks'][task]='out_of_range';continue
        linear=float(m['intercept']+np.dot((values-np.asarray(m['mean']))/np.asarray(m['scale']),m['coef']))
        result['pass_probability' if task=='pass' else 'accuracy']=1/(1+math.exp(-max(-30,min(30,linear)))) if task=='pass' else float(np.clip(linear,0,1))
        result['tasks'][task]='accepted'
        result['metrics'][task]={s:{k:v for k,v in m['metrics'][s].items() if k not in ('by_pp_bin','mods','pp_bins')}
                                 for s in ('validation','test')}
        if task=='accuracy': result['accuracy_error_q90']=m['accuracy_error_q90']
        result['status']='trained'
    return result


def population_status():
    path=Path(os.getenv('PULSE_POPULATION_MODEL',str(DEFAULT_FOLDER/'model.json')))
    report=read_json(path.with_name('report.json'),{})
    if report.get('schema')!=SCHEMA: return {'status':'not_trained'}
    return {'status':'trained' if any(t.get('accepted') for t in report.get('tasks',{}).values()) else 'candidate',
            'trained_at':report['trained_at'],'protocol':report['protocol'],
            'prospective_validation':False,'splits':report['splits'],'tasks':report['tasks']}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder',type=Path,default=DEFAULT_FOLDER)
    train_population(p.parse_args().folder)
