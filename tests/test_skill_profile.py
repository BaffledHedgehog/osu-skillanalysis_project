from copy import deepcopy

from fastapi.testclient import TestClient

from app.analyzer import parse_map
from app.main import create_app
from app.skill_profile import build_skill_profile
from test_app import chart


def rows(aim=3., speed=2., mods='NM', count=3):
    return [{'map_id': str(i), 'mods': mods, 'passed': True, 'passed_known': True,
             'complete': True, 'source': 'manual', 'weight': 1.,
             'analysis': {'id': str(i), 'checksum': f'c{i}', 'mods': mods,
                          'aim': aim, 'speed': speed}} for i in range(count)]


def test_same_chart_reveals_opposite_personal_skill_demands():
    target={'id':'target', 'mods':'NM', 'aim':3., 'speed':3.}
    aim_player=build_skill_profile(rows(4.,2.),target=target)
    speed_player=build_skill_profile(rows(2.,4.),target=target)
    assert aim_player['dominant_demand']=='speed'
    assert speed_player['dominant_demand']=='aim'
    assert aim_player['skills'][0]['difference_percent']==-25
    assert aim_player['skills'][1]['difference_percent']==50
    assert all(s['maps']==3 and s['attempts']==3 for s in aim_player['skills'])


def test_mods_failures_nf_unknown_and_partial_do_not_create_skill_levels():
    eligible=rows()
    invalid=[]
    for patch in ({'mods':'CL'}, {'mods':'NF'}, {'passed':False}, {'passed_known':False}, {'complete':False}):
        group=rows(99,99)
        for r in group:
            r.update(patch);r['analysis']['mods']=r['mods']
        invalid+=group
    profile=build_skill_profile(eligible+invalid)
    assert profile['attempts']==3 and profile['maps']==3
    assert profile['skills'][0]['level']==3.
    classic=build_skill_profile(eligible+invalid,mods='CL')
    assert classic['skills'][0]['level']==99
    assert build_skill_profile(rows(mods='HD'),mods='NM')['status']=='available'


def test_target_is_excluded_and_repeating_one_map_does_not_manufacture_evidence():
    history=rows(count=2)
    history += [deepcopy(history[0]) for _ in range(50)]
    profile=build_skill_profile(history)
    assert profile['maps']==2 and profile['skills'][0]['level'] is None
    history=rows()
    assert build_skill_profile(history,target={'id':'0','mods':'NM'})['maps']==2
    assert build_skill_profile(history,target={'id':'other','checksum':'c0','mods':'NM'})['maps']==2


def test_recency_and_distinct_map_votes_resist_large_old_repeat_series():
    history=rows(4,4)
    old=rows(1,1,count=1)[0];old['map_id']='old';old['analysis']['checksum']='old';old['weight']=.01
    profile=build_skill_profile(history+[deepcopy(old) for _ in range(1000)])
    assert profile['skills'][0]['level']==4.
    assert profile['skills'][0]['maps']==4


def test_best_is_identified_and_does_not_override_sufficient_ordinary_attempts():
    best=rows(99,99)
    for r in best:r['source']='hinai_best';r['analysis']['checksum']='best'+r['analysis']['checksum']
    p=build_skill_profile(best)
    assert p['source_mode']=='best_scores' and p['status']=='available'
    p=build_skill_profile(rows()+best)
    assert p['source_mode']=='ordinary_passes' and p['skills'][0]['level']==3


def test_api_exposes_profile_counts_and_target_relative_load(tmp_path):
    app=create_app(tmp_path/'test.db');store=app.state.store
    for i in range(4):
        mid=str(500+i);store.save_map(parse_map(chart(int(mid),spacing=250-i*30)))
        store.save_play('Player',mid,'NM',.96,True,play_id=f'p{i}')
    client=TestClient(app)
    summary=client.get('/api/users/Player/summary').json()
    assert summary['skill_profile']['maps']==4
    analysis=client.get('/api/maps/503/analysis',params={'user':'Player'}).json()
    p=analysis['personal_skills']
    assert p['status']=='available' and p['maps']==3 and p['target_excluded']
    assert p['skills'][0]['target_load']==round(analysis['aim'],3)
    assert p['skills'][0]['relative_load'] is not None
    hd=client.get('/api/maps/503/analysis',params={'user':'Player','mods':'HD'}).json()
    assert hd['personal_skills']['status']=='available'
    assert [s['level'] for s in hd['personal_skills']['skills']]==[s['level'] for s in p['skills']]
    empty=client.get('/api/maps/503/analysis',params={'user':'Unknown'}).json()
    assert empty['personal_skills']['attempts']==0
    assert client.get('/api/maps/503/analysis').json()['personal_skills'] is None
