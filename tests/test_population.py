import asyncio
import csv
from datetime import datetime, timezone
import json

import httpx
import numpy as np
import pytest

from app.analyzer import parse_map
from app.population_data import confirmed_candidates, normalize_score, collect_scores, collect_maps, normalize_population
from app.population_ml import (ALL_FEATURES, SCHEMA, build_samples, context_features, input_vector, release_gate,
                               player_assignment, predict_population, balanced_training_weights)
from app.sequence_data import family_groups
from app.storage import Store
from test_app import chart


def candidate_files(tmp_path, granted=True):
    path=tmp_path/'candidates.csv'
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['user_id','username','pp_bin','pp','country'],delimiter=';')
        writer.writeheader();writer.writerow({'user_id':'99','username':'Player','pp_bin':'0-2000','pp':'1000','country':'GB'})
    consent=tmp_path/'consents.json'
    consent.write_text(json.dumps({'players':[{'user_id':'99','status':'granted' if granted else 'revoked',
                                              'scopes':['public_scores_training_evaluation']}]}),encoding='utf-8')
    return path,consent


def test_collection_fails_closed_without_matching_scoped_consent(tmp_path):
    path,consent=candidate_files(tmp_path,False)
    assert confirmed_candidates(path,consent)==[]
    assert confirmed_candidates(path,tmp_path/'missing.json')==[]
    consent.write_text(json.dumps({'players':[{'user_id':'99','status':'granted','scopes':['replay_files']}]}))
    assert confirmed_candidates(path,consent)==[]


@pytest.mark.parametrize('patch,reason', [({'mods':['NF']},'no_fail'), ({'passed':'false'},'unknown_outcome'),
        ({'accuracy':float('nan')},'invalid_accuracy'), ({'created_at':None},'missing_date'),
        ({'mods':[{'acronym':'DT','settings':{'speed_change':1.2}}]},'custom_mod_settings'),
        ({'mode':'mania'},'unsupported_mode')])
def test_population_does_not_turn_unsupported_results_into_labels(patch,reason):
    score={'mods':[],'passed':False,'accuracy':.9,'created_at':'2026-11-04T18:00:00Z',**patch}
    with pytest.raises(ValueError,match=reason): normalize_score(score)


def test_public_collector_retains_sources_and_restarts_without_refetch(tmp_path,monkeypatch):
    candidates,_=candidate_files(tmp_path)
    requests=[]
    original=httpx.AsyncClient
    def handler(request):
        requests.append(str(request.url))
        if not request.url.path.endswith('/scores'):
            return httpx.Response(200,json={'id':99,'username':'Player'})
        source=request.url.params['type']
        score={'id':1,'user_id':99,'beatmap_id':101,'mods':[],'passed':True,'accuracy':.95,
               'max_combo':24,'created_at':'2026-11-04T18:00:00Z'}
        if source=='recent':
            assert request.url.params['include_fails']=='1'
            rows=[score,{**score,'id':2,'passed':False}, {**score,'id':3,'mods':['NF']}, {**score,'id':4,'user_id':999}]
        else: rows=[score]
        return httpx.Response(200,json={'scores':rows})
    monkeypatch.setattr('app.population_data.httpx.AsyncClient',lambda **kw:original(transport=httpx.MockTransport(handler),**kw))
    result=asyncio.run(collect_scores(tmp_path,candidates))
    assert result['99']['status']=='complete'
    store=Store(tmp_path/'population.db')
    rows=store.score_payloads('Player')
    assert len(rows)==3 and next(r for r in rows if r['score_key']=='1')['sources']==['best','firsts','recent']
    assert next(r for r in rows if r['score_key']=='3')['reason']=='no_fail'
    assert store.play_count('Player')==0  # no chart, no invented training row
    asyncio.run(collect_scores(tmp_path,candidates))
    assert len(requests)==4


def history_rows(tmp_path):
    store=Store(tmp_path/'test.db')
    rows=[]
    for i in range(3):
        store.save_map(parse_map(chart(100+i)))
        a=store.analysis(str(100+i),'NM')
        rows.append({'id':str(i),'map_id':str(100+i),'mods':'NM','analysis':a,'source':'hinai_best',
                     'passed':True,'passed_known':True,'accuracy':.95,'played_at':'2026-11-01T12:00:00Z'})
    return rows


def test_shared_features_have_no_player_id_or_current_pp_and_ignore_nf(tmp_path):
    rows=history_rows(tmp_path)
    date=datetime(2026,11,4,tzinfo=timezone.utc)
    p=context_features(rows,now=date)
    values=input_vector(rows[0]['analysis'],p)
    assert len(values)==len(ALL_FEATURES) and np.isfinite(values).all()
    assert all('pp' not in k and 'user' not in k for k in ALL_FEATURES)
    assert np.array_equal(p,context_features([*rows,{**rows[0],'mods':'NF','accuracy':0,'passed':False}],now=date))


def test_split_is_deterministic_and_no_user_belongs_to_multiple_splits():
    players={str(i):{'pp_bin':f'{(i//12)*2000}-{(i//12+1)*2000}'} for i in range(48)}
    a=player_assignment(players)
    assert a==player_assignment(dict(reversed(list(players.items()))))
    assert set(a.values())=={'train','validation','test'} and len(a)==len(players)


def test_training_weights_balance_pp_bands_and_players_without_duplicating_attempts():
    rows=[{'user_id':'a','pp_bin':'0-2000'}]*10+[{'user_id':'b','pp_bin':'0-2000'}]*2+[{'user_id':'c','pp_bin':'2000-4000'}]*3
    w=balanced_training_weights(rows)
    assert np.isclose(w[:12].sum(),w[12:].sum())
    assert np.isclose(w[:10].sum(),w[10:12].sum())


def test_release_requires_both_holdouts_and_enough_players_and_both_pass_classes():
    good={'players':5,'samples':40,'passes':20,'fails':20,'model_error':.1,'baseline_error':.2}
    assert release_gate({'validation':good,'test':good},'pass')
    for patch in ({'players':1},{'fails':0},{'model_error':.2},{'model_error':.3},{'model_error':float('nan')}):
        assert not release_gate({'validation':good,'test':{**good,**patch}},'pass')


def test_prediction_never_enables_candidate_or_invalid_artifact(tmp_path):
    path=tmp_path/'model.json'
    path.write_text('{}')
    assert predict_population({},[],path)['status']=='not_trained'


def test_shared_training_never_reads_future_equal_time_or_own_target_outcomes(tmp_path,monkeypatch):
    players=[{'user_id':str(i),'username':f'Player{i}','pp_bin':'0-2000'} for i in range(12)]
    monkeypatch.setattr('app.population_ml.confirmed_candidates',lambda *args:players)
    (tmp_path/'players.json').write_text('{}')
    store=Store(tmp_path/'population.db')
    grouped={'train':[],'validation':[],'test':[]}
    for mid in range(500,700):
        parsed=parse_map(chart(mid,mid).replace(b'Title: Test Song',f'Title: Song {mid}'.encode()))
        family=family_groups({str(mid):parsed})[str(mid)]
        value=int(family[:8],16)%100
        key='train' if value<60 else 'validation' if value<80 else 'test'
        grouped[key].append(parsed)
        if len(grouped['train'])>=6 and all(len(grouped[k])>=1 for k in ('validation','test')):break
    for group in grouped.values():
        for parsed in group:store.save_map(parsed)
    assignment=player_assignment({p['user_id']:p for p in players})
    for p in players:
        user=p['username'];uid=p['user_id']
        for i,parsed in enumerate(grouped['train'][:3]):
            store.save_play(user,parsed.map_id,'NM',.95,True,0,'2026-11-01T00:00:00Z','hinai_best',f'hist-{uid}-{i}')
        target=grouped[assignment[uid]][-1]
        store.save_play(user,target.map_id,'NM',.9,True,0,'2026-11-04T00:00:00Z','hinai_recent',f'target-{uid}')
        store.save_play(user,grouped['train'][3].map_id,'NM',.1,False,0,'2026-11-05T00:00:00Z','hinai_recent',f'future-{uid}')
        store.save_play(user,grouped['train'][4].map_id,'NM',.1,False,0,'2026-11-04T00:00:00Z','hinai_best',f'equal-{uid}')
    samples,manifest=build_samples(tmp_path)
    targets=[r for r in samples if r['score_id'].startswith('target-')]
    assert len(targets)==len(players)
    for row in targets:
        assert len(row['history_ids'])==3 and all(k.startswith('hist-') for k in row['history_ids'])
    for kind in ('user_id','family'):
        sets={s:{r[kind] for r in samples if r['split']==s} for s in ('train','validation','test')}
        assert not sets['train']&sets['validation'] and not sets['train']&sets['test'] and not sets['test']&sets['validation']


def test_candidate_remains_inactive_and_accepted_shared_model_is_mod_scoped(tmp_path):
    rows=history_rows(tmp_path)
    target={**rows[0]['analysis'],'id':'999','title':'Different song','mods':'NM'}
    path=tmp_path/'model.json'
    task={'accepted':False,'supported_mods':['NM'],'stars_range':[0,20],'profile_stars_range':[0,20],
          'intercept':0,'coef':[0]*len(ALL_FEATURES),'mean':[0]*len(ALL_FEATURES),'scale':[1]*len(ALL_FEATURES),
          'metrics':{'validation':{},'test':{}}}
    model={'schema':SCHEMA,'features':list(ALL_FEATURES),'trained_at':'2026-11-04T00:00:00Z','tasks':{'pass':task}}
    path.write_text(json.dumps(model))
    assert predict_population(target,rows,path)['pass_probability'] is None
    task['accepted']=True
    path.write_text(json.dumps(model))
    assert predict_population(target,rows,path)['pass_probability']==.5
    assert predict_population({**target,'mods':'HD'},rows,path)['pass_probability'] is None


def test_chart_identity_fallback_checksum_audit_and_raw_retention(tmp_path,monkeypatch):
    import hashlib
    player={'user_id':'99','username':'Player'}
    monkeypatch.setattr('app.population_data.confirmed_candidates',lambda *args:[player])
    (tmp_path/'players.json').write_text('{}')
    store=Store(tmp_path/'population.db')
    good={'id':1,'user_id':99,'beatmap':{'id':101,'checksum':hashlib.md5(chart(101)).hexdigest()},
          'mods':[],'passed':True,'accuracy':.95,'created_at':'2026-11-04T00:00:00Z'}
    store.save_score_payload('Player','hinai','recent','1','101',good)
    original=httpx.AsyncClient
    urls=[]
    def handler(request):
        urls.append(str(request.url))
        return httpx.Response(200,content=chart(999) if request.url.host=='mirror.hinamizawa.ai' else chart(101))
    monkeypatch.setattr('app.population_data.httpx.AsyncClient',lambda **kw:original(transport=httpx.MockTransport(handler),**kw))
    result=asyncio.run(collect_maps(tmp_path))
    assert result['normalized_by_source']=={'recent':1}
    assert len(urls)==2 and store.get_map('101').checksum==hashlib.md5(chart(101)).hexdigest()
    store.save_score_payload('Player','hinai','recent','2','101',{**good,'id':2,'passed':False,'beatmap':{'id':101,'checksum':'0'*32}})
    store.save_score_payload('Player','hinai','recent','3','101',{**good,'id':3,'mods':['NF']})
    audit=normalize_population(tmp_path)
    assert audit['normalized']==1 and audit['ordinary_passes']==1 and audit['ordinary_fails']==0
    assert audit['rejected']['chart_checksum_mismatch']==1 and len(store.score_payloads('Player'))==3


def test_api_uses_only_a_released_shared_model_when_personal_training_is_insufficient(tmp_path,monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import create_app
    path=tmp_path/'released.json'
    metric={'players':5,'samples':40,'model_error':.1,'baseline_error':.2}
    task={'accepted':True,'supported_mods':['NM'],'stars_range':[0,20],'profile_stars_range':[0,20],
          'intercept':0,'coef':[0]*len(ALL_FEATURES),'mean':[0]*len(ALL_FEATURES),'scale':[1]*len(ALL_FEATURES),
          'metrics':{'validation':metric,'test':metric}}
    model={'schema':SCHEMA,'features':list(ALL_FEATURES),'trained_at':'2026-11-04T00:00:00Z','tasks':{'pass':task}}
    path.write_text(json.dumps(model));monkeypatch.setenv('PULSE_POPULATION_MODEL',str(path))
    client=TestClient(create_app(tmp_path/'api.db'))
    store=client.app.state.store
    for mid in range(810,814):
        store.save_map(parse_map(chart(mid).replace(b'Title: Test Song',f'Title: Song {mid}'.encode())))
        if mid!=813:store.save_play('Player',str(mid),'NM',.95,True,0)
    p=client.get('/api/maps/813/analysis',params={'user':'Player'}).json()['prediction']
    assert p['model']=='validated_population_logistic' and p['pass_probability']==50.0
    task['accepted']=False;path.write_text(json.dumps(model))
    p=client.get('/api/maps/813/analysis',params={'user':'Player'}).json()['prediction']
    assert p['model']=='similarity_and_prior'
