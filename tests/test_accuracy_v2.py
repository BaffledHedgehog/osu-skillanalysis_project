from datetime import datetime,timezone
import json

import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from app.accuracy_context import accuracy_input,CONTEXT_FEATURES,SCHEMA
from app.accuracy_models import fit_model,predict_model
from app.accuracy_runtime import artifact_path,deploy_personal,predict_personal_accuracy
from app.analyzer import parse_map
from app.storage import Store
from test_app import chart


def records(tmp_path):
    s=Store(tmp_path/'test.db');rows=[]
    for mid in range(101,105):
        s.save_map(parse_map(chart(mid).replace(b'Title: Test Song',f'Title: Song {mid}'.encode())))
        if mid<104:
            rows.append({'id':str(mid),'map_id':str(mid),'mods':'NM','source':'hinai_recent','passed':True,
                         'passed_known':1,'complete':1,'accuracy':.97,'played_at':'2026-10-01T00:00:00Z',
                         'analysis':s.analysis(str(mid),'NM')})
    return s,rows,s.analysis('104','NM')


def test_reference_uses_only_earlier_complete_passes_from_other_families(tmp_path):
    s,rows,target=records(tmp_path);now=datetime(2026,11,4,tzinfo=timezone.utc)
    altered=[*rows,{**rows[0],'id':'future','accuracy':0,'played_at':'2026-11-05T00:00:00Z'},
             {**rows[0],'id':'equal','accuracy':0,'played_at':'2026-11-04T00:00:00Z'},
             {**rows[0],'id':'failed','accuracy':0,'passed':False},
             {**rows[0],'id':'partial','accuracy':0,'complete':0},
             {**rows[0],'id':'nf','accuracy':0,'mods':'NF'},
             {**rows[0],'id':'own','map_id':'104','analysis':target,'accuracy':0}]
    a=accuracy_input(rows,target,now);b=accuracy_input(altered,target,now)
    assert a['history_ids']==b['history_ids'] and np.array_equal(a['x'],b['x'])
    assert len(a['x'])==len(CONTEXT_FEATURES) and a['mean']==pytest.approx(.97)


@pytest.mark.parametrize('residual',[False,True])
def test_exported_boosting_matches_sklearn_on_unseen_inputs(residual):
    rng=np.random.default_rng(42)
    x=rng.normal(size=(180,4));base=np.full(180,.93)
    y=np.clip(.93-.1*np.square(x[:,0])+.02*x[:,1],.1,1)
    weights=rng.uniform(.2,2,len(y));weights*=len(weights)/weights.sum()
    params={'loss':'absolute_error','max_iter':25,'max_leaf_nodes':5,'min_samples_leaf':8,
            'learning_rate':.05,'early_stopping':False,'random_state':42}
    model=fit_model('hgb_residual_mae' if residual else 'hgb_absolute_mae',x,y,base,weights.copy(),params)
    reference=HistGradientBoostingRegressor(**params).fit(x,y-base if residual else y,sample_weight=weights)
    test=rng.normal(size=(50,4));bb=np.full(50,.93)
    assert np.allclose(predict_model(model,test,bb),np.clip(reference.predict(test)+(bb if residual else 0),0,1),rtol=1e-9,atol=1e-10)
    with pytest.raises(ValueError):predict_model(model,np.array([[np.nan]*4]),[.93])


def test_personal_artifact_is_bound_to_player_database_and_validated_mods(tmp_path):
    store,rows,target=records(tmp_path)
    metrics={'samples':100,'families':50,'mae_pp':2.,'ridge_mae_pp':3.,'mean_mae_pp':4.}
    m={'schema':SCHEMA,'features':list(CONTEXT_FEATURES),'accepted':True,'user':'player',
       'source_database':str(store.path.resolve()),'supported_mods':['NM'],'stars_range':[0,20],
       'trained_at':'2026-11-04T00:00:00Z','accuracy_error_q90':.05,'model':{'algorithm':'neighbors'},
       'report':{'test':metrics,'cutoff':'2026-10-01T00:00:00Z','protocol':'controlled_test'}}
    source=tmp_path/'model.json';source.write_text(json.dumps(m));deploy_personal(source)
    assert predict_personal_accuracy(store,'Player',target,rows)['accuracy']==pytest.approx(.97)
    assert predict_personal_accuracy(store,'Other',target,rows)['status']=='not_trained'
    assert predict_personal_accuracy(store,'Player',{**target,'mods':'HD'},rows)['status']=='insufficient_mod_data'
    assert predict_personal_accuracy(Store(tmp_path/'other.db'),'Player',target,rows)['status']=='not_trained'
    m['accepted']=False;source.write_text(json.dumps(m))
    with pytest.raises(ValueError):deploy_personal(source)


def test_deployment_to_default_store_requires_matching_history_and_keeps_provenance(tmp_path):
    source_store,rows,target=records(tmp_path)
    destination=Store(tmp_path/'default.db')
    destination.save_map(parse_map(chart(101)))
    for s in (source_store,destination):
        with s.batch():
            for i in range(210):s.save_play('Player','101','NM',.97,True,0,play_id=f'common-{i}')
    m={'schema':SCHEMA,'features':list(CONTEXT_FEATURES),'accepted':True,'user':'player',
       'source_database':str(source_store.path.resolve()),'model':{'algorithm':'neighbors'}}
    file=tmp_path/'source.json';file.write_text(json.dumps(m))
    out=deploy_personal(file,destination.path)
    deployed=json.loads(out.read_text())
    assert deployed['source_database']==str(source_store.path.resolve())
    assert deployed['deployment_database']==str(destination.path.resolve())
    unrelated=Store(tmp_path/'unrelated.db')
    with pytest.raises(ValueError):deploy_personal(file,unrelated.path)
    assert not artifact_path(unrelated.path,'Player').exists()
