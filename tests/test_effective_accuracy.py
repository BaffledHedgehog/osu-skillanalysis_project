from datetime import datetime,timezone,timedelta
import json

import numpy as np
import pytest

from app.analyzer import parse_map,analyze_map
from app.accuracy_transfer import (TransferContext,effective_vector,reading_vector,predict_effective,
                                   ALL_FEATURES,HIDDEN,READING)
from app.ml import FEATURES,model_vector
from app.storage import Store
from app.accuracy_transfer import SCHEMA,artifact_path,predict_transfer
from test_app import chart


def analysis():return analyze_map(parse_map(chart(990)))


def test_hidden_keeps_physical_demand_and_exposes_reading_separately():
    parsed=parse_map(chart(990))
    nm=analyze_map(parsed,'NM');hd=analyze_map(parsed,'HD')
    for key in ('mechanical_stars','mechanical_aim','mechanical_speed','windows','stream_bpm','burst_bpm'):
        assert nm[key]==hd[key]
    assert effective_vector(nm)==effective_vector(hd)
    assert reading_vector(nm)!=reading_vector(hd)


def test_low_ar_and_overlaps_raise_reading_descriptors_not_physical_notes():
    raw=chart(990)
    high=analyze_map(parse_map(raw),'HD')
    low=analyze_map(parse_map(raw.replace(b'ApproachRate: 9',b'ApproachRate: 4')),'HD')
    assert low['reading_density']>high['reading_density']
    assert low['preempt_ms']>high['preempt_ms']
    text=raw.decode();section=False;lines=[]
    for line in text.splitlines():
        if line=='[HitObjects]':section=True
        elif section and line:
            fields=line.split(',');fields[:2]=['256','192'];line=','.join(fields)
        lines.append(line)
    overlap=analyze_map(parse_map('\n'.join(lines).encode()),'HD')
    assert overlap['overlap_fraction']>high['overlap_fraction']


def test_real_tempo_changes_are_distinct_from_slider_velocity_changes():
    raw=chart(990)
    original=b'0,500,4,1,0,100,1,0'
    bpm=analyze_map(parse_map(raw.replace(original,original+b'\n2000,250,4,1,0,100,1,0')))
    sv=analyze_map(parse_map(raw.replace(original,original+b'\n2000,-50,4,1,0,100,0,0')))
    assert bpm['tempo_change_rate']>0
    assert sv['tempo_change_rate']==0


def test_parameter_mods_share_effective_input_but_hidden_does_not_share_visibility():
    a=analysis()
    for mods in ('HR','DT','HT','EZ','HDDT','HDHR'):
        b={**a,'mods':mods}
        assert effective_vector(b)==effective_vector(a)
        assert model_vector(b)[FEATURES.index('double_time')]==0
    assert model_vector({**a,'mods':'HD'})[FEATURES.index('hidden')]==1
    hr=analyze_map(parse_map(chart(990)),'HR')
    dt=analyze_map(parse_map(chart(990)),'DT')
    assert hr['hp']==pytest.approx(7.)
    assert hr['cs']>a['cs'] and dt['tap_rate_median']>a['tap_rate_median']


def rows(n=130):
    a=analysis();now=datetime(2026,11,7,tzinfo=timezone.utc);result=[]
    for i in range(n):
        mid=str(1000+i)
        result.append({'id':mid,'map_id':mid,'mods':'NM','source':'manual','passed':True,'complete':True,
                       'passed_known':True,'accuracy':.9,'played_at':(now-timedelta(days=i+1)).isoformat(),
                       'analysis':{**a,'id':mid,'checksum':mid,'title':'Song '+mid}})
    return result,a,now


def test_old_similar_passes_survive_recent_challenging_map_history():
    history,target,now=rows()
    for p in history[:100]:
        p['analysis']={**p['analysis'],'stars':10.,'mechanical_stars':10.,'aim':6.,'mechanical_aim':6.}
    p=TransferContext(history).input(target,now)
    assert p is not None
    assert history[-1]['id'] in p['history_ids']
    assert len(p['history_ids'])>100
    changed=history+[dict(history[-1],id='future',accuracy=0,played_at=(now+timedelta(days=1)).isoformat()),
                     dict(history[-1],id='nf',mods='NF',accuracy=0),dict(history[-1],id='partial',complete=False),
                     dict(history[-1],id='unknown',passed_known=False)]
    q=TransferContext(changed).input(target,now)
    assert p['history_ids']==q['history_ids']
    assert np.array_equal(p['x'],q['x'])
    assert len(p['x'])==len(ALL_FEATURES)
    assert TransferContext([]).input(target,now) is None


def test_effective_model_has_a_separate_learned_hidden_head_and_no_fixed_penalty():
    history,target,now=rows(5);c=TransferContext(history)
    x=c.input(target,now)['x'];hd=c.input({**target,'mods':'HD'},now)['x']
    base={'algorithm':'ridge','mean':[0.]*len(ALL_FEATURES),'scale':[1.]*len(ALL_FEATURES),
          'coef':[0.]*len(ALL_FEATURES),'intercept':.96,'residual':False}
    model={'base':base,'reading_head':{'coef':[0.]*len(HIDDEN),'samples':0}}
    assert predict_effective(model,x,0)[0]==pytest.approx(.96)
    assert predict_effective(model,hd,0)[0]==pytest.approx(.96)
    model['reading_head']['coef'][0]=-.02
    normal=predict_effective(model,hd,0)[0]
    harder=hd.copy();harder[len(FEATURES)+len(READING)]*=2
    assert predict_effective(model,harder,0)[0]<normal
    assert predict_effective(model,x,0)[0]==pytest.approx(.96)
    with pytest.raises(ValueError):predict_effective(model,np.full(len(ALL_FEATURES),np.nan),0)


def test_transfer_runtime_validates_artifact_and_shares_parameter_mod_history(tmp_path):
    store=Store(tmp_path/'test.db')
    for i in range(4):
        parsed=parse_map(chart(100+i).replace(b'Title: Test Song',f'Title: Song {i}'.encode()))
        store.save_map(parsed)
        if i<3:store.save_play('Player',str(100+i),'NM',.96,True,play_id=str(i),played_at='2026-10-01T00:00:00Z')
    target=store.analysis('103','NM')
    metrics={'samples':30,'families':20,'mae_pp':3.,'reference_mae_pp':4.,'error_q90':.06}
    base={'algorithm':'ridge','mean':[0.]*len(ALL_FEATURES),'scale':[1.]*len(ALL_FEATURES),
          'coef':[0.]*len(ALL_FEATURES),'intercept':.96,'residual':False}
    artifact={'schema':SCHEMA,'features':list(ALL_FEATURES),'accepted':True,'user':'player',
              'deployment_database':str(store.path.resolve()),'validated_mods':['NM'],'stars_range':[0,20],
              'accuracy_error_q90':.06,'trained_at':'2026-11-01T00:00:00Z',
              'model':{'algorithm':'effective_reading_model','base':base,'reading_head':{'coef':[0.]*len(HIDDEN),'samples':0}},
              'report':{'protocol':'controlled_test','test':{**metrics,'by_mod':{'NM':metrics}}}}
    path=artifact_path(store.path,'Player');path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(artifact))
    assert predict_transfer(store,'Player',target)['status']=='trained'
    for mod in ('HR','DT','HD'):
        p=predict_transfer(store,'Player',{**target,'mods':mod})
        assert p['status']=='transferred' and p['accuracy']==pytest.approx(.96)
    assert predict_transfer(store,'Player',{**target,'mods':'FL'})['status']=='unsupported_conditions'
    assert predict_transfer(store,'Other',target)['status']=='not_trained'
    path.write_text(json.dumps({**artifact,'accepted':False}))
    assert predict_transfer(store,'Player',target)['status']=='candidate'


def test_parametric_mods_no_longer_require_a_binary_mod_flag_in_legacy_models():
    from app.ml import predict_pass
    a=analysis()
    model={'mod_bounds':[[0,0]]*8,'intercept':0.,'coef':[0.]*len(FEATURES),
           'mean':[0.]*len(FEATURES),'scale':[1.]*len(FEATURES)}
    context=(model,{'status':'trained'})
    assert predict_pass(None,'Player',{**a,'mods':'DT'},context)[0]==pytest.approx(.5)
    assert predict_pass(None,'Player',{**a,'mods':'HR'},context)[0]==pytest.approx(.5)
    assert predict_pass(None,'Player',{**a,'mods':'HD'},context)[1]['status']=='insufficient_mod_data'
