from copy import deepcopy
from datetime import datetime,timezone,timedelta
import json

import pytest

from app.accuracy_feedback import fit_feedback,infer,predict_feedback,artifact_path
from app.storage import Store
from app.main import create_app
from app.analyzer import parse_map
from fastapi.testclient import TestClient
from test_app import chart
from test_effective_accuracy import analysis


def model(path):
    a=analysis()
    return fit_feedback(path,'Player',[{'analysis':a,'predicted_accuracy':.90,'actual_accuracy':.982,'source':'user_chat'}]),a


def test_feedback_learns_an_error_ratio_not_an_accuracy_bonus(tmp_path):
    m,a=model(tmp_path/'db.db')
    p=infer(m,a,.90)
    assert .97<p['accuracy']<.982
    q=infer(m,a,.95)
    assert q['accuracy']>.98 and q['adjustment_pp']<p['adjustment_pp']
    assert p['target_in_feedback'] and p['unverified_feedback']
    assert not p['prospective_validation']


def test_feedback_does_not_apply_a_fixed_bonus_to_low_ar_or_nf(tmp_path):
    m,a=model(tmp_path/'db.db')
    low={**a,'ar':3.,'reading_density':50.,'checksum':'other'}
    assert abs(infer(m,low,.90)['adjustment_pp'])<.1
    assert infer(m,{**a,'mods':'NF'},.9) is None
    assert infer(m,{**a,'mods':'CL'},.9) is None
    old=datetime.fromisoformat(m['created_at'])+timedelta(days=91)
    assert infer(m,a,.90,now=old) is None


def test_hidden_reading_adjustment_remains_separate(tmp_path):
    m,a=model(tmp_path/'db.db')
    nm=infer(m,a,.90)
    hd=infer(m,{**a,'mods':'HD'},.88,reading_adjustment=-.02)
    assert hd['accuracy']==pytest.approx(nm['accuracy']-.02)


def test_feedback_is_bound_to_player_database_and_cannot_invent_a_base(tmp_path):
    store=Store(tmp_path/'db.db');m,a=model(store.path)
    path=artifact_path(store.path,'Player');path.parent.mkdir()
    path.write_text(json.dumps(m),encoding='utf-8')
    assert predict_feedback(store,'Player',a,.9)['status']=='preview_feedback'
    assert predict_feedback(store,'Other',a,.9)['status']=='disabled'
    assert predict_feedback(store,'Player',a,None)['status']=='no_base_prediction'
    broken=deepcopy(m);broken['database']=str(tmp_path/'different.db')
    path.write_text(json.dumps(broken),encoding='utf-8')
    assert predict_feedback(store,'Player',a,.9)['status']=='wrong_player_or_database'


def test_opt_in_feedback_reaches_api_and_preserves_base_and_uncertainty(tmp_path):
    app=create_app(tmp_path/'api.db');store=app.state.store
    for i in range(5):
        mid=str(700+i);store.save_map(parse_map(chart(int(mid),spacing=240-i*10)))
        if i<4:store.save_play('Player',mid,'NM',.94,True,play_id=f'p{i}')
    c=TestClient(app)
    before=c.get('/api/maps/704/analysis',params={'user':'Player'}).json()['prediction']
    a=store.analysis('704','NM')
    m=fit_feedback(store.path,'Player',[{'analysis':a,'predicted_accuracy':before['expected_accuracy']/100,'actual_accuracy':.982}])
    path=artifact_path(store.path,'Player');path.parent.mkdir(exist_ok=True);path.write_text(json.dumps(m),encoding='utf-8')
    after=c.get('/api/maps/704/analysis',params={'user':'Player'}).json()['prediction']
    assert after['accuracy_model']=='preview_feedback_accuracy'
    assert after['expected_accuracy']>before['expected_accuracy']
    assert after['uncalibrated_expected_accuracy']==before['expected_accuracy']
    assert after['accuracy_range'][0]<=before['accuracy_range'][0]
    assert after['accuracy_range'][1]>=before['accuracy_range'][1]
    assert after['pass_probability']==before['pass_probability']


@pytest.mark.parametrize('regularization',[float('nan'),float('inf'),0.,-1.])
def test_corrupt_feedback_falls_back_instead_of_returning_invalid_json(tmp_path,regularization):
    store=Store(tmp_path/'corrupt.db');m,a=model(store.path)
    m['regularization']=regularization
    path=artifact_path(store.path,'Player');path.parent.mkdir()
    path.write_text(json.dumps(m),encoding='utf-8')
    assert predict_feedback(store,'Player',a,.9)['status']=='disabled'
