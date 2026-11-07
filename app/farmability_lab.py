"""Offline farm-pattern experiment. Never deploys or restarts the demo server.

Run in a fresh folder. Historical holdout is reused and explicitly labelled as
retrospective; reported external 98.2% is never a fit or selection label.
"""
import argparse
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import sqlite3

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from .accuracy_lab import load_records,eligible
from .accuracy_transfer import TransferContext,HIDDEN,READING,reading_head_vector
from .accuracy_transfer_train import make_rows,metric,selection_score
from .accuracy_models import fit_model,predict_model,DEFAULT_HGB_PARAMS
from .analyzer import analyze_map
from .farmability import descriptors,FEATURES as PATTERNS
from .ml import BEST_SOURCES,FEATURES
from .population_data import read_json,write_json
from .recency import sample_weights
from .sequence_data import family_groups
from .storage import Store


CANDIDATES = ('control_v3', 'patterns_ridge_residual_100', 'patterns_ridge_absolute_100',
              'patterns_hgb_absolute', 'patterns_hgb_residual',
              'patterns_log_error_10', 'patterns_log_error_100', 'control_log_error_100')
OLD = Path('.data/accuracy-improvement')
V3 = Path('.data/accuracy-easy-map-2026-11-07/final')


def predict_base(model,x,base):
    if model['algorithm']!='log_error_ridge':return predict_model(model,x,base)
    z=(x-np.asarray(model['mean']))/np.asarray(model['scale'])@np.asarray(model['coef'])+model['intercept']
    z+=np.log(np.maximum(1-np.asarray(base),.0005))
    return np.clip(1-np.exp(np.clip(z,-15,0)),0,1)


def fit_candidate(name,rows):
    use_patterns=name.startswith('patterns_')
    x=np.asarray([r['extended'] if use_patterns else r['x'] for r in rows])
    y=np.asarray([r['y'] for r in rows]);b=np.asarray([r['neighbor'] for r in rows])
    counts=Counter(r['family'] for r in rows)
    w=np.asarray(sample_weights([{'played_at':r['date'].isoformat()} for r in rows],now=max(r['date'] for r in rows)))
    w/=np.asarray([counts[r['family']] for r in rows])
    hd=np.asarray(['HD' in r['mods'] for r in rows]);base_x=x.copy()
    start=len(FEATURES)+len(READING);base_x[:,start:start+len(HIDDEN)]=0
    if 'log_error' in name:
        xx=base_x[~hd];ww=w[~hd].copy();ww*=len(ww)/ww.sum()
        z=np.log(np.maximum(1-y[~hd],.0005))-np.log(np.maximum(1-b[~hd],.0005))
        scaler=StandardScaler().fit(xx,sample_weight=ww)
        estimator=Ridge(alpha=float(name.rsplit('_',1)[-1])).fit(scaler.transform(xx),z,sample_weight=ww)
        model={'algorithm':'log_error_ridge','mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),
               'coef':estimator.coef_.tolist(),'intercept':float(estimator.intercept_)}
        direct=1-np.exp(np.clip(estimator.predict(scaler.transform(xx))+np.log(np.maximum(1-b[~hd],.0005)),-15,0))
        assert np.allclose(direct,predict_base(model,xx,b[~hd]),atol=1e-10)
    else:
        underlying='ridge_residual_100' if name=='control_v3' else name.removeprefix('patterns_')
        model=fit_model(underlying,base_x[~hd],y[~hd],b[~hd],w[~hd].copy(),DEFAULT_HGB_PARAMS)
    coef=np.zeros(len(HIDDEN))
    if hd.sum()>=3:
        z=reading_head_vector(x[hd]);ww=w[hd]
        residual=y[hd]-predict_base(model,base_x[hd],b[hd])
        coef=np.linalg.solve(z.T@(ww[:,None]*z)+100*np.eye(len(HIDDEN)),z.T@(ww*residual))
    return {'candidate':name,'patterns':use_patterns,'base':model,'reading_coef':coef.tolist()}


def infer(model,rows):
    x=np.asarray([r['extended'] if model['patterns'] else r['x'] for r in rows])
    start=len(FEATURES)+len(READING);base_x=x.copy();base_x[:,start:start+len(HIDDEN)]=0
    b=np.asarray([r['neighbor'] for r in rows])
    return np.clip(predict_base(model['base'],base_x,b)+reading_head_vector(x)@np.asarray(model['reading_coef']),0,1)


def run(folder,user='BaffledHedgehog'):
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=False)
    plan={'date':datetime.now(timezone.utc).isoformat(),'candidates':list(CANDIDATES),
          'selection':'3-fold early family-purged CV; half overall MAE and half NM difficulty-bin MAE',
          'retrospective_recheck':True,'external_result_is_label':False,
          'features':list(PATTERNS),'deployment':False,
          'source_history':str((OLD/'personal-history.db').resolve())}
    write_json(folder/'plan.json',plan)
    # All potential analysis caching is confined to this private copy.
    snapshot=folder/'history.db'
    with sqlite3.connect(OLD/'personal-history.db') as src,sqlite3.connect(snapshot) as dst:src.backup(dst)
    store=Store(snapshot)
    records,maps=load_records(store,{user:{'username':user}});records=records[user]
    families=family_groups(maps);context=TransferContext(records,families)
    cutoff=datetime.fromisoformat(read_json(OLD/'personal-plan.json')['cutoff'])
    saved=read_json(V3/'transfer-test-predictions.json');test_ids={p['id'] for p in saved}
    test_groups={families[p['map_id']] for p in records if p['id'] in test_ids}
    allowed=set(families.values())-test_groups
    targets=sorted([p for p in records if eligible(p) and p['source'] not in BEST_SOURCES and 'FL' not in p['mods']],key=lambda p:(p['_date'],p['id']))
    train=[p for p in targets if p['_date']<cutoff and families[p['map_id']] in allowed]
    analyses={(p['map_id'],p['mods']):p['analysis'] for p in records if p.get('analysis')}
    pattern_cache={}
    def extend(rows):
        for r in rows:
            key=r['map_id'],r['mods']
            if key not in pattern_cache:
                a=analyses[key]
                pattern_cache[key]=list(descriptors(maps[r['map_id']].content,a).values())
            r['extended']=np.concatenate([r['x'],pattern_cache[key]])
        return rows
    accumulated={name:([],[]) for name in CANDIDATES}
    group_ids=[families[p['map_id']] for p in train]
    for fold,(tr,va) in enumerate(GroupKFold(n_splits=3).split(np.zeros(len(train)),groups=group_ids)):
        validation=[train[i] for i in va];group_allowed=allowed-{families[p['map_id']] for p in validation}
        training=extend(make_rows([train[i] for i in tr],context,group_allowed))
        validation=extend(make_rows(validation,context,group_allowed))
        for name in CANDIDATES:
            model=fit_candidate(name,training);prediction=infer(model,validation)
            accumulated[name][0].extend(validation);accumulated[name][1].extend(prediction.tolist())
        print({'fold':fold,'train':len(training),'validation':len(validation)},flush=True)
    cv={name:{**metric(rows,np.asarray(pred)),'selection_score':float(selection_score(rows,np.asarray(pred)))}
        for name,(rows,pred) in accumulated.items()}
    selected=min(CANDIDATES,key=lambda name:(cv[name]['selection_score'],name))
    write_json(folder/'selection-before-test.json',{'selected':selected,'cv':cv})
    training=extend(make_rows(train,context,allowed))
    test=extend(make_rows([p for p in targets if p['id'] in test_ids],context))
    model=fit_candidate(selected,training);pred=infer(model,test)
    control=fit_candidate('control_v3',training);control_pred=infer(control,test)
    previous={p['id']:p['prediction'] for p in saved}
    assert np.allclose(control_pred,[previous[r['id']] for r in test],atol=1e-10),'Control V3 must reproduce frozen predictions'
    def compare(indices):
        rr=[test[i] for i in indices]
        return {'candidate':metric(rr,pred[indices]),'deployed_v3':metric(rr,control_pred[indices])}
    nm=[i for i,r in enumerate(test) if r['mods']=='NM']
    easy=[i for i in nm if 4<=test[i]['stars']<5]
    results={'all':compare(list(range(len(test)))),'NM':compare(nm),'NM_4_5_stars':compare(easy)}
    accepted=selected!='control_v3' and all(results[k]['candidate']['mae_pp']<results[k]['deployed_v3']['mae_pp']
                                         for k in ('NM','NM_4_5_stars'))
    report={**plan,'selected':selected,'cv':cv,'test':results,'improves_release_criteria':bool(accepted),
            'notes':'Candidate selected before reading historical test; known historical test reused, no prospective claim. No deployment.'}
    write_json(folder/'report.json',report)
    write_json(folder/'benchmark-model.json',model)
    write_json(folder/'predictions.json',[{'id':r['id'],'map_id':r['map_id'],'mods':r['mods'],'stars':r['stars'],
                  'accuracy':r['y'],'prediction':float(p),'v3':float(q),'history_dates':r['history_dates']}
                  for r,p,q in zip(test,pred,control_pred)])
    np.savez_compressed(folder/'benchmark-inputs.npz',x=np.asarray([r['x'] for r in test]),
        extended=np.asarray([r['extended'] for r in test]),base=np.asarray([r['neighbor'] for r in test]))
    print({'selected':selected,'results':results,'improves_release_criteria':bool(accepted)},flush=True)
    # Diagnostic query only after selection and evaluation, never a fit label.
    current=Store(V3/'transfer-refit-history.db')
    current_rows,current_maps=load_records(current,{user:{'username':user}});current_rows=current_rows[user]
    target=current.get_map('4934415')
    if target:
        a=analyze_map(target,'NM')
        current_maps['4934415']=target
        final_families=family_groups(current_maps)
        ctx=TransferContext(current_rows,final_families)
        group=final_families.get('4934415')
        final_targets=[p for p in current_rows if eligible(p) and p['source'] not in BEST_SOURCES
                       and 'FL' not in p['mods'] and p['map_id']!='4934415'
                       and (group is None or final_families[p['map_id']]!=group)]
        # Use the same descriptor cache and causal-context function for refit.
        maps.update(current_maps);analyses.update({(p['map_id'],p['mods']):p['analysis'] for p in current_rows if p.get('analysis')})
        refit=fit_candidate(selected,extend(make_rows(final_targets,ctx)))
        probe=ctx.input(a)
        probe['extended']=np.concatenate([probe['x'],list(descriptors(target.content,a).values())])
        estimate=float(infer(refit,[probe])[0])
        write_json(folder/'external-probe.json',{'map_id':'4934415','mods':'NM','prediction':estimate,
                    'reported_accuracy_not_used_for_fit':.982,'deployed':False,'descriptors':descriptors(target.content,a)})
        write_json(folder/'refit-model.json',refit)
        print({'external_probe':estimate,'deployed':False},flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--folder',type=Path,required=True);args=parser.parse_args()
    with threadpool_limits(limits=1):run(args.folder)
