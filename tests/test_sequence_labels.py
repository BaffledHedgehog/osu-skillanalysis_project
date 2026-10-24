import numpy as np
import pytest
import json

from app.analyzer import parse_map, analyze_map
from app.sequence import encode_map, mirror_sequence
from app.sequence_labels import TARGETS, note_labels
from app.replay_events import SCHEMA
from test_app import chart


def labelled_chart():
    parsed = parse_map(chart(800))
    sequence = encode_map(parsed, analyze_map(parsed))
    events = [{'root_index': i, 'object_type': 'HitCircle', 'object_time_ms': float(t),
               'time_ms': float(t)+20, 'offset_ms': 20., 'hp_before': .9,
               'result': 'great', 'is_hit': True, 'combo_before': i, 'combo_after': i+1}
              for i, t in enumerate(sequence.times_ms)]
    result = {'schema': SCHEMA, 'verified': True, 'complete': True, 'mods': 'NM',
              'map_checksum': parsed.checksum, 'engine': {'events_verified': True,
              'statistics_match': True, 'root_count': len(events), 'unmapped_events': 0,
              'reverted_events': False, 'events': events}}
    return sequence, result


def test_note_labels_use_actual_events_and_distinguish_low_grade_from_miss():
    sequence, result = labelled_chart()
    events = result['engine']['events']
    events[2].update(result='ok')
    events[3].update(result='miss', is_hit=False, combo_after=0)
    # A slider's final grade alone does not prove an inaccurate timed head.
    events[4].update(object_type='Slider', result='ok')
    events.append(dict(events[4], object_type='SliderTick', result='smalltickmiss', is_hit=False))
    y, mask = note_labels(result, sequence)
    assert np.array_equal(y[2], [0, 1])
    assert np.array_equal(y[3], [1, 1])
    assert np.array_equal(y[4], [1, 0]) and not mask[4, 1]
    assert y.sum() == 4 and mask[:, 0].all()
    for view in ('x', 'y', 'xy'):
        actual, observed = note_labels(result, mirror_sequence(sequence, view))
        assert np.array_equal(y, actual) and np.array_equal(mask, observed)


@pytest.mark.parametrize('change', ['incomplete', 'statistics', 'index', 'suffix', 'time', 'checksum'])
def test_incomplete_mismatched_or_misaligned_events_cannot_become_training_labels(change):
    sequence, result = labelled_chart()
    if change == 'incomplete': result['complete'] = False
    elif change == 'statistics': result['engine']['statistics_match'] = False
    elif change == 'index': result['engine']['events'][0]['root_index'] = 999
    elif change == 'suffix': result['engine']['events'].pop()
    elif change == 'time': result['engine']['events'][0]['object_time_ms'] += 200
    elif change == 'checksum': result['map_checksum'] = 'wrong'
    with pytest.raises(ValueError): note_labels(result, sequence)


def test_local_cnn_can_learn_note_labels_and_preserves_numeric_weights(tmp_path):
    torch = pytest.importorskip('torch')
    from app.sequence import FEATURES
    from app.sequence_data import CONTEXT_FEATURES
    from app.sequence_net import LocalPatternModel, save_weights, load_weights
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(6)
    model = LocalPatternModel(len(CONTEXT_FEATURES))
    x = torch.randn(4, len(FEATURES), 128)
    mask = torch.ones(4, 128, dtype=torch.bool)
    context = torch.zeros(4, len(CONTEXT_FEATURES))
    baseline = torch.zeros(4, len(TARGETS), 128)
    y = (x[:, :2] > 0).float()
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    criterion = torch.nn.BCEWithLogitsLoss()
    first = float(criterion(model(x, mask, context, baseline), y).detach())
    for _ in range(15):
        loss = criterion(model(x, mask, context, baseline), y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    model.eval()
    with torch.inference_mode():
        prediction = model(x, mask, context, baseline)
        assert float(criterion(prediction, y)) < first*.8
    save_weights(model, tmp_path/'local.npz')
    restored = load_weights(LocalPatternModel(len(CONTEXT_FEATURES)), tmp_path/'local.npz').eval()
    with torch.inference_mode():
        assert torch.equal(prediction, restored(x, mask, context, baseline))


def test_local_training_snapshot_and_audit_use_unique_root_predictions(tmp_path, monkeypatch):
    pytest.importorskip('torch')
    from app import sequence_local as local
    from app.ml import vector
    from app.sequence_data import Corpus, PROFILE_FEATURES, split_families
    from app.storage import Store
    store=Store(tmp_path/'test.db')
    records,maps,families,labels,masks,contexts=[],{},{},[],[],[]
    assignment=split_families([f'family-{i}' for i in range(30)])
    for i in range(30):
        mid=str(1000+i);parsed=parse_map(chart(int(mid),spacing=150+i*8))
        store.save_map(parsed)
        maps[mid]=parsed;families[mid]=f'family-{i}'
        analysis=analyze_map(parsed)
        records.append({'id':str(i),'map_id':mid,'mods':'NM','analysis':analysis,'accuracy':.95,
            'passed':True,'passed_known':True,'complete':True,'source':'fixture','played_at':'2026-10-24T00:00:00Z',
            'pass_mask':True,'accuracy_mask':True})
        contexts.append([*vector(analysis),*[0.]*len(PROFILE_FEATURES)])
        y=np.zeros((analysis['object_count'],len(TARGETS)),dtype=np.float32)
        y[::6,0]=1;y[::3,1]=1;labels.append(y);masks.append(np.ones_like(y,dtype=bool))
    manifest={'schema':'fixture','fingerprint':'frozen-test','profile_reference':sorted(p.checksum for p in maps.values()),'splits':{}}
    for split in ('train','validation','test'):
        ids=[i for i,p in enumerate(records) if assignment[families[p['map_id']]]==split]
        manifest['splits'][split]={'indices':ids,'attempts':len(ids),'families':[families[records[i]['map_id']] for i in ids]}
    corpus=Corpus(store,records,maps,families,assignment,np.asarray(contexts,dtype=np.float32),
                  np.ones(len(records),dtype=np.float32),'frozen-test',manifest)
    monkeypatch.setattr(local,'build_event_corpus',lambda *args:(corpus,labels,masks))
    run=tmp_path/'models/local-runs/run'
    report=local.train_local(store,'Fixture',run,steps=1,pretrain_steps=1,device='cpu')
    assert set(report['variants'])=={'original','four_views'}
    assert all(not r['accepted'] for r in report['release'].values())
    assert local.audit_local(run)['status']=='verified'
    frozen,ys,ms=local.load_snapshot(run)
    assert len(frozen.records)==30 and np.array_equal(ys[0],labels[0])
    # Counts refer to original roots, not overlapping or mirrored windows.
    expected=sum(len(labels[i]) for i in manifest['splits']['test']['indices'])
    assert report['variants']['original']['test'][TARGETS[0]]['notes']==expected
    from app import sequence_service
    monkeypatch.setattr(sequence_service,'model_folder',lambda *args:tmp_path/'models')
    target=store.analysis('1000')
    assert sequence_service.predict_local(store,'Fixture',target)['status']=='rejected'
    # Exercise the accepted runtime branch with a controlled fixture artifact.
    # This changes no release decision on actual player data.
    artifact=json.loads((run/'artifact.json').read_text(encoding='utf-8'))
    artifact['release'][TARGETS[0]]['accepted']=True
    artifact['support']['validated_mods'][TARGETS[0]]=['NM']
    (run/'artifact.json').write_text(json.dumps(artifact),encoding='utf-8')
    result=sequence_service.predict_local(store,'Fixture',target)
    assert result['status']=='trained' and result['windows']
    assert all(0<=w[TARGETS[0]]<=100 and TARGETS[1] not in w for w in result['windows'])
