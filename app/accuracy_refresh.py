"""Freeze a development copy, then quarantine newly observed recent scores.

Only metadata counts are printed. Fresh outcomes are evaluated after model selection.
"""
import asyncio
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3

import httpx

from .population_data import DEFAULT_FOLDER, ROOT, PublicFetcher, confirmed_candidates, read_json, write_json
from .osu_api import PUBLIC_API, USER_AGENT
from .storage import Store

FOLDER=ROOT/'.data/accuracy-improvement'


def freeze(folder=FOLDER,source=DEFAULT_FOLDER):
    folder=Path(folder);source=Path(source)
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'plan.json').exists():return read_json(folder/'plan.json')
    report=read_json(source/'report.json');manifest=read_json(source/'training/manifest.json')
    plan={'schema':'accuracy-improvement-v1','frozen_at':datetime.now(timezone.utc).isoformat(),
          'time_cutoff':report['trained_at'],'source_fingerprint':report['fingerprint'],
          'player_split':manifest['player_split'],
          'selection':'inner grouped CV on historical train users; historical validation users are a separate gate',
          'final_test':'Only fresh scores after cutoff from original test users. Purge their song families from fitting and validation targets.',
          'fresh_test_unsealed':False,'metric':'equal-player MAE on ordinary successful complete attempts',
          'candidates':['ridge_absolute_10','ridge_residual_10','ridge_residual_100','ridge_residual_1000',
                        'hgb_absolute_mae','hgb_residual_mae','ridge_residual_auxiliary_100'],
          'hgb_parameters':{'loss':'absolute_error','max_iter':160,'max_leaf_nodes':7,'min_samples_leaf':15,
                            'learning_rate':.05,'l2_regularization':10,'early_stopping':False,'random_state':20261104},
          'baseline_candidates':['past_pass_mean','similar_successful_maps'],
          'no_new_users_without_consent':True,'no_curated_scores_as_ordinary_targets':True,
          'release_requirements':{'validation_players':5,'validation_samples':30,'test_players':5,'test_samples':30,
                                  'improve_both_baselines':True,'exact_mod_evidence_required':True},
          'notes':'Retrospective temporal holdout collected after the previous snapshot; not a prospective live trial of this new model.'}
    with sqlite3.connect(source/'population.db') as src,sqlite3.connect(folder/'historical.db') as out:
        src.backup(out)
    shutil.copy2(folder/'historical.db',folder/'population.db')
    for name in ('players.json','consents.json'):shutil.copy2(source/name,folder/name)
    write_json(folder/'plan.json',plan)
    return plan


async def refresh(folder=FOLDER):
    folder=Path(folder);plan=freeze(folder)
    players=confirmed_candidates(ROOT/'deliverables/dataset-candidates.csv',folder/'consents.json')
    metadata=read_json(folder/'players.json');store=Store(folder/'population.db')
    completed=0;new_ids=0;future_ids=[]
    with sqlite3.connect(folder/'historical.db') as db:
        old_ids={r[0] for r in db.execute('SELECT score_key FROM score_payloads')}
    cutoff=datetime.fromisoformat(plan['time_cutoff'])
    sem=asyncio.Semaphore(2)
    async with httpx.AsyncClient(timeout=25,headers={'User-Agent':USER_AGENT}) as client:
        fetch=PublicFetcher(client)
        async def player(row):
            nonlocal completed,new_ids
            async with sem:
                uid=row['user_id'];file=folder/'fresh-recent'/f'{uid}.json'
                payload=read_json(file)
                try:
                    if payload is None:
                        r=await fetch.get(f'{PUBLIC_API}/{uid}/scores',params={'type':'recent','mode':'osu','limit':100,'include_fails':1})
                        payload=r.json();write_json(file,payload)
                    scores=payload.get('scores',[]) if isinstance(payload,dict) else payload
                    if not isinstance(scores,list):raise ValueError('invalid scores')
                    user=metadata.get(uid,{}).get('username',row['username'])
                    with store.batch():
                        for score in scores:
                            if not isinstance(score,dict):continue
                            if score.get('user_id') is not None and str(score['user_id'])!=uid:continue
                            key=str(score.get('legacy_score_id') or score.get('id') or hashlib.sha256(json.dumps(score,sort_keys=True).encode()).hexdigest())
                            mid=str((score.get('beatmap') or {}).get('id') or score.get('beatmap_id') or '')
                            # Old payloads remain frozen. Only new score identities enter this refresh.
                            if key not in old_ids:
                                store.save_score_payload(user,'hinai','recent',key,mid,score,'fresh_quarantine')
                                new_ids+=1
                            try:date=datetime.fromisoformat((score.get('ended_at') or score.get('created_at')).replace('Z','+00:00'))
                            except (TypeError,AttributeError,ValueError):continue
                            if date.tzinfo is None:date=date.replace(tzinfo=timezone.utc)
                            if date>cutoff:
                                future_ids.append({'score_key':key,'user_id':uid,'map_id':mid,'played_at':date.isoformat(),
                                                   'player_split':plan['player_split'][uid]})
                except (httpx.HTTPError,ValueError):
                    pass
                completed+=1
                if completed%10==0 or completed==len(players):
                    status={'stage':'fresh_recent','completed':completed,'players':len(players),'new_score_ids':new_ids,
                            'newer_than_cutoff':len(future_ids),'cutoff':plan['time_cutoff']}
                    write_json(folder/'refresh-status.json',status);print(status,flush=True)
        await asyncio.gather(*(player(p) for p in players))
    write_json(folder/'fresh-index.json',future_ids)


if __name__=='__main__':asyncio.run(refresh())
