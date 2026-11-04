from app.analyzer import parse_map
from app.predictor import predict
from app.storage import Store
from test_app import chart


def setup(tmp_path):
    store=Store(tmp_path/'test.db')
    for mid in (101,102):store.save_map(parse_map(chart(mid)))
    return store,store.analysis('102','NM')


def test_failed_fragment_accuracy_never_lowers_completed_pass_accuracy(tmp_path):
    store,target=setup(tmp_path)
    for i in range(3):store.save_play('Player','101','NM',.98,True,0,play_id=f'pass{i}')
    for i in range(20):store.save_play('Player','101','NM',.2,False,10,play_id=f'fail{i}',complete=False)
    p=predict(store,'Player',target)
    assert p['expected_accuracy']>94
    assert p['accuracy_reference_mode']=='attempts'


def test_only_failed_fragments_produce_unknown_accuracy_not_invented_full_accuracy(tmp_path):
    store,target=setup(tmp_path)
    for i in range(3):store.save_play('Player','101','NM',.2,False,10,play_id=f'fail{i}',complete=False)
    p=predict(store,'Player',target)
    assert p['expected_accuracy'] is None and p['accuracy_range'] is None


def test_curated_successes_describe_skills_but_are_not_empirical_passes(tmp_path):
    store,target=setup(tmp_path)
    for i in range(3):store.save_play('Player','101','NM',.9,False,1,source='hinai_recent',play_id=f'fail{i}')
    for i in range(100):store.save_play('Player','101','NM',.9,True,0,source='hinai_best',play_id=f'best{i}')
    p=predict(store,'Player',target)
    assert p['pass_observations']==3 and p['pass_probability']<60
    assert p['accuracy_reference_mode']=='best_scores'
