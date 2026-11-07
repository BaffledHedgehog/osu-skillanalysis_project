"""Evaluate effective-map accuracy against a frozen previous benchmark."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import sqlite3

import numpy as np
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from .accuracy_transfer import (TransferContext,ALL_FEATURES,SCHEMA,artifact_path,effective_analysis,
                                HIDDEN,READING,reading_head_vector,predict_effective)
from .accuracy_models import fit_model,predict_model,DEFAULT_HGB_PARAMS
from .accuracy_lab import load_records,eligible
from .accuracy_context import CONTEXT_FEATURES as OLD_FEATURES
from .ml import BEST_SOURCES,FEATURES
from .population_data import read_json,write_json
from .recency import sample_weights
from .sequence_data import family_groups
from .storage import Store

REFERENCE=Path('.data/accuracy-improvement')
FOLDER=Path('.data/accuracy-easy-map-2026-11-07')
CANDIDATES=('hgb_absolute_mae','hgb_residual_mae','hgb_balanced_absolute_mae','ridge_residual_100')


def make_rows(targets,context,allowed=None):
    rows=[]
    for p in targets:
        a=context.input(p['analysis'],p['_date'],allowed)
        if a is None:continue
        rows.append({**a,'id':p['id'],'y':p['accuracy'],'map_id':p['map_id'],'family':context.families[p['map_id']],
                     'date':p['_date'],'mods':p['mods'],'stars':effective_analysis(p['analysis'])['stars']})
    return rows


def fit(name,rows):
    x=np.asarray([r['x'] for r in rows]);y=np.asarray([r['y'] for r in rows]);b=np.asarray([r['neighbor'] for r in rows])
    counts=Counter(r['family'] for r in rows)
    weights=np.asarray(sample_weights([{'played_at':r['date'].isoformat()} for r in rows],now=max(r['date'] for r in rows)))
    weights/=np.asarray([counts[r['family']] for r in rows])
    if 'balanced' in name:
        buckets=Counter(int(r['stars']) for r in rows)
        weights/=np.asarray([buckets[int(r['stars'])] for r in rows])
    hd=np.asarray(['HD' in r['mods'] for r in rows])
    base_x=x.copy();start=len(FEATURES)+len(READING)
    base_x[:,start:start+len(HIDDEN)]=0
    base_model=fit_model(name,base_x[~hd],y[~hd],b[~hd],weights[~hd].copy(),DEFAULT_HGB_PARAMS)
    coef=np.zeros(len(HIDDEN))
    if hd.sum()>=3:
        z=reading_head_vector(x[hd]);w=weights[hd]
        residual=y[hd]-predict_model(base_model,base_x[hd],b[hd])
        coef=np.linalg.solve(z.T@(w[:,None]*z)+100*np.eye(len(HIDDEN)),z.T@(w*residual))
    return {'algorithm':'effective_reading_model','base':base_model,
            'reading_head':{'coef':coef.tolist(),'samples':int(hd.sum()),'regularization':100}}


def infer(model,rows):
    return predict_effective(model,np.asarray([r['x'] for r in rows]),np.asarray([r['neighbor'] for r in rows]))


def metric(rows,pred):
    if not rows:return {'samples':0,'families':0}
    groups=Counter(r['family'] for r in rows);w=np.asarray([1/groups[r['family']] for r in rows])
    y=np.asarray([r['y'] for r in rows]);pred=np.asarray(pred)
    result={'samples':len(rows),'families':len(groups),'mae_pp':float(np.average(abs(y-pred),weights=w)*100),
            'bias_pp':float(np.average(pred-y,weights=w)*100),'error_q90':float(np.quantile(abs(y-pred),.9))}
    return result


def selection_score(rows,pred):
    global_score=metric(rows,pred)['mae_pp']
    buckets=[]
    for low,high in [(0,4),(4,5),(5,6),(6,100)]:
        ix=[i for i,r in enumerate(rows) if r['mods']=='NM' and low<=r['stars']<high]
        if len(ix)>=5:buckets.append(metric([rows[i] for i in ix],pred[ix])['mae_pp'])
    return .5*global_score+.5*np.mean(buckets) if buckets else global_score


def train(database,user,folder=FOLDER,publish=False):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    plan={'candidates':list(CANDIDATES),'selection':'3-fold family-purged CV on pre-cutoff history; half overall MAE and half difficulty-bin MAE',
          'cutoff':read_json(REFERENCE/'personal-plan.json')['cutoff'],'external_map':'4934415',
          'external_result_not_a_training_label':True,'reference':'frozen previous test predictions, same score ids',
          'data_policy':'ordinary completed known passes, causal references, current family excluded',
          'context':'100 recent plus 60 similar older passes and 30 curated context scores',
          'hidden':'reading interactions; physical demand shared','parameter_mods':'effective map attributes without HR/DT/EZ/HT flags'}
    write_json(folder/'transfer-plan.json',plan)
    old=Store(REFERENCE/'personal-history.db')
    records,maps=load_records(old,{user:{'username':user}});records=records[user];families=family_groups(maps)
    cutoff=datetime.fromisoformat(plan['cutoff'])
    saved=read_json(REFERENCE/'personal-test-predictions.json');test_ids={r['id'] for r in saved}
    test_families={families[p['map_id']] for p in records if p['id'] in test_ids}
    allowed=set(families.values())-test_families
    targets=sorted([p for p in records if eligible(p) and p['source'] not in BEST_SOURCES and 'FL' not in p['mods']],key=lambda p:(p['_date'],p['id']))
    training=[p for p in targets if p['_date']<cutoff and families[p['map_id']] in allowed]
    context=TransferContext(records,families)
    groups=[families[p['map_id']] for p in training]
    scores={name:([],[]) for name in CANDIDATES}
    for fold,(tr,va) in enumerate(GroupKFold(n_splits=3).split(np.zeros(len(groups)),groups=groups)):
        validation=[training[i] for i in va];va_families={families[p['map_id']] for p in validation}
        fit_allowed=allowed-va_families
        tr_rows=make_rows([training[i] for i in tr],context,fit_allowed)
        va_rows=make_rows(validation,context,fit_allowed)
        for name in CANDIDATES:
            model=fit(name,tr_rows);pred=infer(model,va_rows)
            scores[name][0].extend(va_rows);scores[name][1].extend(pred.tolist())
        print({'stage':'transfer_cv','fold':fold,'train':len(tr_rows),'validation':len(va_rows)},flush=True)
    cv={name:{**metric(rows,np.asarray(pred)),'selection_score':float(selection_score(rows,np.asarray(pred)))} for name,(rows,pred) in scores.items()}
    selected=min(CANDIDATES,key=lambda name:(cv[name]['selection_score'],name))
    write_json(folder/'transfer-selected-before-test.json',{'selected':selected,'cv':cv})
    tr_rows=make_rows(training,context,allowed)
    te_rows=make_rows([p for p in targets if p['id'] in test_ids],context)
    model=fit(selected,tr_rows);pred=infer(model,te_rows)
    write_json(folder/'transfer-benchmark-model.json',{'model':model,'features':list(ALL_FEATURES),
                 'cutoff':cutoff.isoformat(),'test_ids':[r['id'] for r in te_rows]})
    np.savez_compressed(folder/'transfer-benchmark-inputs.npz',x=np.asarray([r['x'] for r in te_rows]),
                        y=np.asarray([r['y'] for r in te_rows]),neighbor=np.asarray([r['neighbor'] for r in te_rows]))
    old_by_id={r['id']:r['prediction'] for r in saved}
    reference=np.asarray([old_by_id[r['id']] for r in te_rows])
    def subset(indices):
        rr=[te_rows[i] for i in indices]
        return {**metric(rr,pred[indices]),'reference_mae_pp':metric(rr,reference[indices]).get('mae_pp'),
                'reference_bias_pp':metric(rr,reference[indices]).get('bias_pp')}
    test={**subset(list(range(len(te_rows)))),'by_mod':{},'difficulty_bins':{}}
    for mod in sorted({r['mods'] for r in te_rows}):test['by_mod'][mod]=subset([i for i,r in enumerate(te_rows) if r['mods']==mod])
    for lo,hi in [(0,4),(4,5),(5,6),(6,100)]:
        test['difficulty_bins'][f'{lo}-{hi}']=subset([i for i,r in enumerate(te_rows) if r['mods']=='NM' and lo<=r['stars']<hi])
    nm=test['by_mod']['NM'];easy=test['difficulty_bins']['4-5']
    accepted=nm['mae_pp']<nm['reference_mae_pp'] and easy['mae_pp']<easy['reference_mae_pp']
    report={'schema':SCHEMA,'selected':selected,'cv':cv,'test':test,'accepted':bool(accepted),
            'protocol':'chronological_purged_families_with_recent_and_similar_past_context',
            'cutoff':cutoff.isoformat(),'external_result_used':False,'retrospective_recheck':True,
            'baseline':'frozen previous HGB predictions on identical historical score ids',
            'context_features':len(ALL_FEATURES)}
    write_json(folder/'transfer-report.json',report)
    write_json(folder/'transfer-test-predictions.json',[{k:r[k] for k in ('id','map_id','mods','family','stars','y','history_ids','history_dates')}|
               {'prediction':float(p),'reference':float(q)} for r,p,q in zip(te_rows,pred,reference)])
    print({'stage':'transfer_test','selected':selected,'test':test,'accepted':bool(accepted)},flush=True)
    if not accepted:return report
    snapshot=folder/'transfer-refit-history.db'
    with sqlite3.connect(database) as src,sqlite3.connect(snapshot) as dst:src.backup(dst)
    all_records,all_maps=load_records(Store(snapshot),{user:{'username':user}});all_records=all_records[user]
    external=Store(snapshot).get_map('4934415')
    if external is not None:all_maps['4934415']=external
    all_families=family_groups(all_maps);all_context=TransferContext(all_records,all_families)
    excluded={f for mid,f in all_families.items() if mid=='4934415'}
    all_targets=sorted([p for p in all_records if eligible(p) and p['source'] not in BEST_SOURCES
                        and 'FL' not in p['mods'] and p['map_id']!='4934415'
                        and all_families[p['map_id']] not in excluded],key=lambda p:(p['_date'],p['id']))
    final_rows=make_rows(all_targets,all_context)
    final=fit(selected,final_rows)
    artifact={'schema':SCHEMA,'features':list(ALL_FEATURES),'accepted':True,'user':user.casefold(),
              'deployment_database':str(Path(database).resolve()),'source_database':str(snapshot.resolve()),
              'trained_at':datetime.now(timezone.utc).isoformat(),'model':final,'report':report,
              'accuracy_error_q90':cv[selected]['error_q90'],'validated_mods':['NM'],
              'stars_range':[min(r['stars'] for r in final_rows),max(r['stars'] for r in final_rows)]}
    write_json(folder/'transfer-model.json',artifact)
    if publish:
        dest=artifact_path(database,user);dest.parent.mkdir(exist_ok=True,parents=True)
        write_json(dest,artifact);print({'deployed':str(dest)},flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--database',default='.data/test-stand-updated-2026-11-04/test.db')
    parser.add_argument('--user',default='BaffledHedgehog');parser.add_argument('--folder',type=Path,default=FOLDER)
    parser.add_argument('--publish',action='store_true');args=parser.parse_args()
    with threadpool_limits(limits=1):train(args.database,args.user,args.folder,args.publish)
