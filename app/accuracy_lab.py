"""Accuracy experiments: fixed candidates, causal context, grouped selection, sealed fresh test."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
from pathlib import Path
import sqlite3

import numpy as np
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from .accuracy_context import accuracy_input,CONTEXT_FEATURES,SCHEMA
from .accuracy_models import fit_model,predict_model,DEFAULT_HGB_PARAMS
from .accuracy_refresh import FOLDER,freeze
from .ml import BEST_SOURCES,vector,FEATURES
from .population_data import read_json,write_json,confirmed_candidates,ROOT
from .population_ml import balanced_training_weights
from .recency import played_date,sample_weights
from .sequence_data import family_groups
from .storage import Store


def load_records(store,people):
    with store.batch():
        records={uid:store.plays_with_analysis(p['username']) for uid,p in people.items()}
        mids={p['map_id'] for rows in records.values() for p in rows if p.get('analysis')}
        maps={mid:store.get_map(mid) for mid in mids}
    for uid,rows in records.items():
        for p in rows:
            p['_date']=played_date(p);p['user_id']=uid;p['pp_bin']=people[uid].get('pp_bin','personal')
    return records,maps


def eligible(p):
    return p.get('analysis') is not None and p['_date'] is not None and p.get('passed_known',1) and p['passed'] and p.get('complete',1) and 'NF' not in p['mods']


def features(targets,records,families,allowed):
    out=[]
    for p in targets:
        a=accuracy_input(records[p['user_id']],p['analysis'],p['_date'],families,allowed,p['source'] in BEST_SOURCES)
        if a is None:continue
        out.append({**a,'y':float(p['accuracy']),'user_id':p['user_id'],'pp_bin':p['pp_bin'],'id':p['id'],
                    'map_id':p['map_id'],'family':families[p['map_id']],'mods':p['mods'],'played_at':p['played_at']})
        if '_fit_weight' in p:out[-1]['fit_weight']=p['_fit_weight']
        if p.get('_personal'):out[-1]['evaluation_group']=families[p['map_id']]
    return out


def arrays(rows):
    return np.asarray([r['x'] for r in rows]),np.asarray([r['y'] for r in rows]),np.asarray([r['neighbor'] for r in rows])


def fit(name,training,auxiliary,params):
    if name=='neighbors':return {'algorithm':'neighbors'}
    x,y,b=arrays(training);w=balanced_training_weights(training)
    w*=np.asarray([r.get('fit_weight',1.) for r in training])
    effective=w.sum()**2/np.square(w).sum();w*=effective/w.sum()
    if 'auxiliary' in name and auxiliary:
        xx,yy,bb=arrays(auxiliary);ww=balanced_training_weights(auxiliary)
        ww*=.15*w.sum()/ww.sum()
        x=np.concatenate([x,xx]);y=np.concatenate([y,yy]);b=np.concatenate([b,bb]);w=np.concatenate([w,ww])
    return fit_model(name,x,y,b,w,params)


def infer(model,rows):
    x,y,b=arrays(rows)
    return b if model['algorithm']=='neighbors' else predict_model(model,x,b)


def metrics(rows,pred):
    if not rows:return {'samples':0,'players':0,'families':0}
    count=Counter(r.get('evaluation_group',r['user_id']) for r in rows)
    w=np.asarray([1/count[r.get('evaluation_group',r['user_id'])] for r in rows])
    y=np.asarray([r['y'] for r in rows]);mean=np.asarray([r['mean'] for r in rows]);near=np.asarray([r['neighbor'] for r in rows])
    error=lambda a:float(np.average(np.abs(y-a),weights=w))
    return {'samples':len(rows),'players':len({r['user_id'] for r in rows}),'families':len({r['family'] for r in rows}),
            'mae_pp':100*error(pred),'mean_mae_pp':100*error(mean),'neighbors_mae_pp':100*error(near),
            'error_q90':float(np.quantile(np.abs(y-pred),.9)),
            'by_mod':{m:metrics([r for r in rows if r['mods']==m],np.asarray([v for r,v in zip(rows,pred) if r['mods']==m]))
                      for m in sorted({r['mods'] for r in rows})} if len({r['mods'] for r in rows})>1 else {}}


def select(targets,aux,records,families,base_allowed,candidates,params,personal=False):
    groups=np.asarray([families[p['map_id']] if personal else p['user_id'] for p in targets])
    if len(set(groups))<3:raise ValueError('Not enough independent groups for selection')
    accumulated={name:([],[]) for name in candidates};audit=[]
    controls=([],[])
    for fold,(train_ids,val_ids) in enumerate(GroupKFold(n_splits=3).split(np.zeros(len(groups)),groups=groups)):
        validation=[targets[i] for i in val_ids]
        val_families={families[p['map_id']] for p in validation}
        allowed=base_allowed-val_families
        training=[targets[i] for i in train_ids if families[targets[i]['map_id']] in allowed]
        train_users={p['user_id'] for p in training}
        aa=[p for p in aux if p['user_id'] in train_users and families[p['map_id']] in allowed]
        tr=features(training,records,families,allowed);va=features(validation,records,families,allowed)
        ax=features(aa[:2000],records,families,allowed) if any('auxiliary' in n for n in candidates) else []
        audit.append({'fold':fold,'train_ids':[p['id'] for p in tr],'validation_ids':[p['id'] for p in va],
                      'train_users':sorted({p['user_id'] for p in tr}),'validation_users':sorted({p['user_id'] for p in va}),
                      'train_families':sorted({p['family'] for p in tr}),'validation_families':sorted({p['family'] for p in va})})
        if len(tr)<30 or len(va)<5:raise ValueError('Insufficient fold coverage')
        for name in candidates:
            model=fit(name,tr,ax,params);pred=infer(model,va)
            accumulated[name][0].extend(va);accumulated[name][1].extend(pred.tolist())
        if personal:
            control=fit_control(tr)
            controls[0].extend(va);controls[1].extend(infer_control(control,va).tolist())
        print({'stage':'selection','fold':fold,'train':len(tr),'validation':len(va)},flush=True)
    result={n:metrics(rows,np.asarray(pred)) for n,(rows,pred) in accumulated.items()}
    if personal:result['reference_current_ridge']=metrics(controls[0],np.asarray(controls[1]))
    chosen=min(candidates,key=lambda n:(result[n]['mae_pp'],n))
    return chosen,result,audit


def fit_control(rows):
    x,y,b=arrays(rows);w=np.asarray([r.get('fit_weight',1.) for r in rows])
    return fit_model('ridge_absolute_10',x[:,:len(FEATURES)],y,b,w)


def infer_control(model,rows):
    x,y,b=arrays(rows)
    return predict_model(model,x[:,:len(FEATURES)],b)


def personal_experiment(database,user,folder=FOLDER):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    plan=read_json(folder/'plan.json',read_json(FOLDER/'plan.json',{'hgb_parameters':DEFAULT_HGB_PARAMS}))
    source_info={'database':str(Path(database).resolve()),'user':user.strip().casefold()}
    existing=read_json(folder/'personal-snapshot-source.json')
    if existing is not None and existing!=source_info:raise ValueError('Use a separate --folder for this player/database')
    snapshot=folder/'personal-history.db'
    if not snapshot.exists():
        with sqlite3.connect(database) as src,sqlite3.connect(snapshot) as dst:src.backup(dst)
    write_json(folder/'personal-snapshot-source.json',source_info)
    store=Store(snapshot);uid=user.strip().casefold()
    records,maps=load_records(store,{uid:{'username':user}})
    families=family_groups(maps)
    targets=sorted([p for p in records[uid] if eligible(p) and p['source'] not in BEST_SOURCES],key=lambda p:(p['_date'],p['id']))
    if len(targets)<400:return {'status':'insufficient_data','samples':len(targets)}
    cutoff=targets[int(len(targets)*.75)]['_date']
    testing=[p for p in targets if p['_date']>=cutoff]
    test_families={families[p['map_id']] for p in testing}
    allowed=set(families.values())-test_families
    training=[p for p in targets if p['_date']<cutoff and families[p['map_id']] in allowed]
    for p in targets:p['_personal']=True
    fit_counts=Counter(families[p['map_id']] for p in training)
    for p,w in zip(training,sample_weights(training,now=cutoff)):
        p['_fit_weight']=w/fit_counts[families[p['map_id']]]
    candidates=['neighbors','ridge_absolute_10','ridge_residual_10','ridge_residual_100','ridge_residual_1000','hgb_absolute_mae','hgb_residual_mae']
    personal_plan={'protocol':'chronological_last_quarter_with_purged_target_families_and_prequential_context',
                   'cutoff':cutoff.isoformat(),'candidates':candidates,'selection':'3-fold family-purged CV on earlier targets',
                   'metric':'equal_song_family_MAE','minimum_test_samples':30,'minimum_test_families':10,
                   'require_negative_upper_cluster_bootstrap_difference':True,'known_past_data_not_a_new_live_trial':True}
    write_json(folder/'personal-plan.json',personal_plan)
    chosen,cv,audit=select(training,[],records,families,allowed,candidates,plan['hgb_parameters'],personal=True)
    write_json(folder/'personal-selected-before-test.json',{'selected':chosen,'cv':cv,'folds':audit,'test_unsealed':False})
    tr=features(training,records,families,allowed)
    te=features(testing,records,families,None) # Only earlier games update a later test profile; model weights stay fixed.
    model=fit(chosen,tr,[],plan['hgb_parameters']);control=fit_control(tr)
    write_json(folder/'personal-benchmark-model.json',{'model':model,'control':control,'features':list(CONTEXT_FEATURES),
                                                      'selected':chosen,'cutoff':cutoff.isoformat()})
    pred=infer(model,te);reference=infer_control(control,te)
    evaluated=metrics(te,pred);evaluated['ridge_mae_pp']=metrics(te,reference)['mae_pp']
    family_errors={}
    for family in sorted({r['family'] for r in te}):
        ix=[i for i,r in enumerate(te) if r['family']==family]
        yy=np.asarray([te[i]['y'] for i in ix])
        family_errors[family]=float(np.mean(np.abs(yy-pred[ix])-np.abs(yy-reference[ix])))
    diffs=np.asarray(list(family_errors.values()));rng=np.random.default_rng(20261104)
    boot=np.mean(rng.choice(diffs,size=(500,len(diffs)),replace=True),axis=1)
    interval=(np.quantile(boot,[.025,.975])*100).tolist()
    cv_good=cv[chosen]['mae_pp']<min(cv['reference_current_ridge']['mae_pp'],cv[chosen]['mean_mae_pp'])
    accepted=cv_good and len(te)>=30 and len(family_errors)>=10 and evaluated['mae_pp']<min(evaluated['ridge_mae_pp'],evaluated['mean_mae_pp']) and interval[1]<0
    supported=[];per_mod_comparison={}
    for mod in sorted({r['mods'] for r in te}):
        ids=[i for i,r in enumerate(te) if r['mods']==mod]
        subset=[te[i] for i in ids]
        if len(ids)>=20 and len({r['family'] for r in subset})>=10:
            mm=metrics(subset,pred[ids]);rr=metrics(subset,reference[ids])
            per_mod_comparison[mod]={'samples':len(ids),'families':mm['families'],
                                     'mae_pp':mm['mae_pp'],'ridge_mae_pp':rr['mae_pp']}
            if mm['mae_pp']<min(rr['mae_pp'],mm['mean_mae_pp']):supported.append(mod)
    accepted=bool(accepted and supported)
    report={'schema':SCHEMA,'user':user,'selected':chosen,'protocol':personal_plan['protocol'],'cutoff':cutoff.isoformat(),
            'cv':cv,'test':evaluated,'difference_ci95_pp':interval,'supported_mods':supported,'accepted':accepted,
            'training_samples':len(tr),'test_prequential':True,'baseline':'same past-only Ridge algorithm and past-pass mean',
            'prospective_validation':False,'per_mod_ridge_comparison':per_mod_comparison}
    # Refit after the benchmark; the frozen test report is retained separately.
    all_counts=Counter(families[p['map_id']] for p in targets)
    now=datetime.now(timezone.utc)
    for p,w in zip(targets,sample_weights(targets,now=now)):p['_fit_weight']=w/all_counts[families[p['map_id']]]
    all_features=features(targets,records,families,None)
    final=fit(chosen,all_features,[],plan['hgb_parameters'])
    artifact={'schema':SCHEMA,'features':list(CONTEXT_FEATURES),'model':final,'report':report,'accepted':accepted,
              'user':uid,'source_database':str(Path(database).resolve()),'trained_at':now.isoformat(),
              'accuracy_error_q90':cv[chosen]['error_q90'],'supported_mods':supported,
              'stars_range':[min(r['x'][0] for r in all_features),max(r['x'][0] for r in all_features)]}
    write_json(folder/'personal-report.json',report);write_json(folder/'personal-model.json',artifact)
    write_json(folder/'personal-test-predictions.json',[{k:r[k] for k in ('id','map_id','mods','family','played_at','y','history_ids','history_dates')}|
               {'prediction':float(p),'reference_ridge':float(q)} for r,p,q in zip(te,pred,reference)])
    print({'personal_selected':chosen,'test':evaluated,'accepted':accepted,'ci95':interval},flush=True)
    return report


def verify_personal_benchmark(folder=FOLDER):
    """Reconstruct the already selected model, without selecting again or changing its test."""
    folder=Path(folder);source=read_json(folder/'personal-snapshot-source.json')
    plan=read_json(folder/'personal-plan.json');selection=read_json(folder/'personal-selected-before-test.json')
    params=read_json(folder/'plan.json',read_json(FOLDER/'plan.json'))['hgb_parameters']
    uid=source['user'];records,maps=load_records(Store(folder/'personal-history.db'),{uid:{'username':uid}})
    families=family_groups(maps);cutoff=datetime.fromisoformat(plan['cutoff'])
    targets=[p for p in records[uid] if eligible(p) and p['source'] not in BEST_SOURCES]
    testing=sorted([p for p in targets if p['_date']>=cutoff],key=lambda p:(p['_date'],p['id']))
    test_families={families[p['map_id']] for p in testing};allowed=set(families.values())-test_families
    training=sorted([p for p in targets if p['_date']<cutoff and families[p['map_id']] in allowed],key=lambda p:(p['_date'],p['id']))
    counts=Counter(families[p['map_id']] for p in training)
    for p in targets:p['_personal']=True
    for p,w in zip(training,sample_weights(training,now=cutoff)):p['_fit_weight']=w/counts[families[p['map_id']]]
    tr=features(training,records,families,allowed);te=features(testing,records,families,None)
    chosen=selection['selected'];model=fit(chosen,tr,[],params);control=fit_control(tr)
    prediction=infer(model,te);reference=infer_control(control,te)
    saved={p['id']:p for p in read_json(folder/'personal-test-predictions.json')}
    assert len(saved)==len(te)
    difference=max(abs(float(p)-saved[r['id']]['prediction']) for r,p in zip(te,prediction))
    control_difference=max(abs(float(p)-saved[r['id']]['reference_ridge']) for r,p in zip(te,reference))
    if max(difference,control_difference)>1e-9:raise ValueError('Frozen benchmark did not reproduce')
    for row in te:
        assert all(datetime.fromisoformat(d.replace('Z','+00:00'))<datetime.fromisoformat(row['played_at'].replace('Z','+00:00')) for d in row['history_dates'])
    write_json(folder/'personal-benchmark-model.json',{'model':model,'control':control,'features':list(CONTEXT_FEATURES),
                                                      'selected':chosen,'cutoff':cutoff.isoformat()})
    x,y,b=arrays(te)
    np.savez_compressed(folder/'personal-benchmark-inputs.npz',x=x,y=y,baseline=b)
    result={'reproduced':True,'samples':len(te),'max_prediction_difference':difference,
            'max_reference_difference':control_difference,'causal_history_verified':True}
    write_json(folder/'benchmark-reproduction.json',result)
    return result


def global_experiment(folder=FOLDER):
    folder=Path(folder);plan=read_json(folder/'plan.json')
    people=confirmed_candidates(ROOT/'deliverables/dataset-candidates.csv',folder/'consents.json')
    metadata=read_json(folder/'players.json')
    people={p['user_id']:{**p,'username':metadata.get(p['user_id'],{}).get('username',p['username'])} for p in people}
    old,maps=load_records(Store(folder/'historical.db'),people)
    current,new_maps=load_records(Store(folder/'population.db'),people);maps.update(new_maps)
    families=family_groups(maps);cutoff=datetime.fromisoformat(plan['time_cutoff'])
    roles=plan['player_split']
    old_targets=[p for uid,rows in old.items() for p in rows if eligible(p) and p['source'] not in BEST_SOURCES and p['_date']<=cutoff]
    # Seal the final targets. This metadata determines purging, never model choice.
    fresh=[p for uid,rows in current.items() if roles[uid]=='test' for p in rows
           if eligible(p) and p['source'] not in BEST_SOURCES and p['_date']>cutoff]
    test_families={families[p['map_id']] for p in fresh}
    validation=[p for p in old_targets if roles[p['user_id']]=='validation' and families[p['map_id']] not in test_families]
    val_families={families[p['map_id']] for p in validation}
    allowed=set(families.values())-test_families-val_families
    training=[p for p in old_targets if roles[p['user_id']]=='train' and families[p['map_id']] in allowed]
    aux=[p for uid,rows in old.items() if roles[uid]=='train' for p in rows
         if eligible(p) and p['source'] in BEST_SOURCES and p['_date']<=cutoff and families[p['map_id']] in allowed]
    aux.sort(key=lambda p:hashlib.sha256(p['id'].encode()).digest())
    chosen,cv,audit=select(training,aux,old,families,allowed,plan['candidates'],plan['hgb_parameters'])
    write_json(folder/'selected-before-test.json',{'selected':chosen,'cv':cv,'selected_at':datetime.now(timezone.utc).isoformat(),
                                                 'test_unsealed':False,'folds':audit})
    tr=features(training,old,families,allowed);ax=features(aux[:2000],old,families,allowed)
    model=fit(chosen,tr,ax,plan['hgb_parameters'])
    va=features(validation,old,families,allowed)
    # The selected model is frozen before any fresh accuracy is scored or reported.
    write_json(folder/'selected-model-before-test.json',model)
    te=features(fresh,current,families,allowed)
    val_metrics=metrics(va,infer(model,va)) if va else metrics([],[])
    test_metrics=metrics(te,infer(model,te)) if te else metrics([],[])
    def good(m):return m.get('players',0)>=5 and m.get('samples',0)>=30 and m['mae_pp']<min(m['mean_mae_pp'],m['neighbors_mae_pp'])-1e-8
    accepted=good(val_metrics) and good(test_metrics)
    supported=[]
    for mod in sorted({r['mods'] for r in te}):
        v=[r for r in va if r['mods']==mod];t=[r for r in te if r['mods']==mod]
        if len(v)>=20 and len(t)>=20 and len({r['user_id'] for r in v})>=3 and len({r['user_id'] for r in t})>=3:
            vm=metrics(v,infer(model,v));tm=metrics(t,infer(model,t))
            if vm['mae_pp']<min(vm['mean_mae_pp'],vm['neighbors_mae_pp']) and tm['mae_pp']<min(tm['mean_mae_pp'],tm['neighbors_mae_pp']):supported.append(mod)
    accepted=accepted and bool(supported)
    report={'schema':SCHEMA,'selected':chosen,'cv':cv,'validation':val_metrics,'fresh_test':test_metrics,
            'accepted':accepted,'supported_mods':supported,'cutoff':plan['time_cutoff'],
            'evaluation':'unseen_test_players_and_songs_after_previous_snapshot; retrospective temporal holdout',
            'fit_samples':len(tr),'auxiliary_samples':len(ax) if 'auxiliary' in chosen else 0}
    write_json(folder/'global-report.json',report)
    write_json(folder/'global-model.json',{'schema':SCHEMA,'features':list(CONTEXT_FEATURES),'model':model,'report':report,
                                         'accepted':accepted,'trained_at':datetime.now(timezone.utc).isoformat()})
    print({'selected':chosen,'validation':val_metrics,'fresh_test':test_metrics,'accepted':accepted},flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--personal-db',type=Path)
    parser.add_argument('--personal-user',default='BaffledHedgehog')
    parser.add_argument('--personal-only',action='store_true')
    parser.add_argument('--folder',type=Path,default=FOLDER)
    parser.add_argument('--publish-personal',action='store_true')
    parser.add_argument('--deploy-db',type=Path)
    args=parser.parse_args()
    with threadpool_limits(limits=2):
        if not args.personal_only:global_experiment(args.folder)
        if args.personal_db:personal_experiment(args.personal_db,args.personal_user,args.folder)
        if args.publish_personal:
            from .accuracy_runtime import deploy_personal
            print(deploy_personal(args.folder/'personal-model.json',args.deploy_db))
