"""Meaningful optional-CNN checks; fixtures never touch the player's database."""
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.analyzer import parse_map
from app.main import create_app
from app.storage import Store
from app import sequence_service
from app.predictor import predict
from test_app import chart


def test_cnn_private_endpoints_and_no_training_without_data(tmp_path, monkeypatch):
    public = TestClient(create_app(tmp_path / 'public.db', public_mode=True))
    for method, path in [('get', ''), ('get', '/report'), ('post', '/train')]:
        assert getattr(public, method)('/api/users/Player/cnn' + path).status_code == 403
        assert getattr(public, method)('/api/users/Player/cnn/local' + path).status_code == 403
    assert public.post('/api/users/Player/cnn/stop').status_code == 403
    assert public.post('/api/users/Player/cnn/forward').status_code == 403
    local = TestClient(create_app(tmp_path / 'local.db'))
    assert local.get('/api/users/Player/cnn').json()['status'] == 'not_trained'
    assert local.get('/api/users/Player/cnn/report').status_code == 404
    assert local.post('/api/users/Player/cnn/train', headers={'origin': 'https://evil.example'}).status_code == 403
    assert local.post('/api/users/Player/cnn/train').status_code == 400
    monkeypatch.setattr(sequence_service, 'start_training', lambda store, user: {'started': True, 'run_id': 'test'})
    assert local.post('/api/users/Player/cnn/train').json()['started']


def test_stop_worker_is_scoped_to_user_and_database(tmp_path, monkeypatch):
    store=Store(tmp_path/'db.sqlite')
    own=sequence_service.model_folder(store,'Player')/'local-runs'/'fixture'
    class Worker:
        stopped=False
        def cmdline(self): return ['python','-m','app.sequence_local','Player','--output',str(own)]
        def children(self,recursive=True): return []
        def terminate(self): self.stopped=True
    worker=Worker()
    monkeypatch.setattr(sequence_service,'_running_job',lambda:worker)
    monkeypatch.setattr(sequence_service.psutil,'wait_procs',lambda *args,**kwargs:([],[]))
    assert not sequence_service.stop_training(store,'Other')['stopped'] and not worker.stopped
    assert sequence_service.stop_training(store,'Player')['stopped'] and worker.stopped


def test_actual_training_worker_can_be_stopped_through_local_api(tmp_path, monkeypatch):
    pytest.importorskip('torch')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    app=create_app(tmp_path/'worker.db')
    store=app.state.store
    for i in range(30):
        mid=str(2000+i)
        raw=chart(int(mid),spacing=180+i*5).replace(b'Title: Test Song',f'Title: Worker {i}'.encode())
        store.save_map(parse_map(raw))
        for j in range(3):store.save_play('Fixture',mid,'NM',.9,j>0)
    client=TestClient(app)
    launched=client.post('/api/users/Fixture/cnn/train')
    assert launched.status_code==200 and launched.json()['started']
    try:
        assert client.get('/api/users/Fixture/cnn').json()['running']
        assert not client.post('/api/users/Other/cnn/stop').json()['stopped']
        assert client.post('/api/users/Fixture/cnn/stop').json()['stopped']
        status=client.get('/api/users/Fixture/cnn').json()
        assert not status['running'] and status['progress']['status']=='interrupted'
    finally:
        sequence_service.stop_training(store,'Fixture')


def test_accepted_cnn_accuracy_reaches_predictor_with_its_error_range(tmp_path, monkeypatch):
    store = Store(tmp_path / 'test.db')
    store.save_map(parse_map(chart(101)))
    for i in range(5):
        store.save_play('Player', '101', 'NM', .95, True)
    accepted = {'status': 'trained', 'accuracy': .876, 'pass_probability': None, 'accuracy_error_q90': .035}
    monkeypatch.setattr('app.predictor.predict_cnn', lambda *args: accepted)
    result = predict(store, 'Player', store.analysis('101'))
    assert result['accuracy_model'] == 'validated_pattern_cnn'
    assert result['expected_accuracy'] == 87.6
    assert result['accuracy_range'] == [84.1, 91.1]
    assert result['model'] == 'personal_history_posterior'


def test_broken_optional_artifact_falls_back_without_crashing(tmp_path, monkeypatch):
    store = Store(tmp_path / 'test.db')
    store.save_map(parse_map(chart(101)))
    def broken(*args):
        raise ValueError('Invalid numeric weights')
    monkeypatch.setattr(sequence_service, '_active_run', broken)
    assert sequence_service.predict_cnn(store, 'Player', store.analysis('101'), [])['status'] == 'invalid_artifact'


def test_runtime_profile_uses_the_training_reference_families(tmp_path):
    store=Store(tmp_path/'profile.db')
    first=parse_map(chart(101));second=parse_map(chart(102,spacing=110))
    store.save_map(first);store.save_map(second)
    store.save_play('Player','101','NM',.99,True)
    store.save_play('Player','102','NM',.5,False)
    profile=sequence_service._profile(str(store.path.resolve()),'Player',store.cache_token('Player'),0,(first.checksum,))
    assert profile[2]==pytest.approx(.99)
    assert profile[1] > .6


def test_numeric_weight_roundtrip_and_chunked_inference(tmp_path):
    torch = pytest.importorskip('torch')
    from app.sequence import FEATURES
    from app.sequence_data import CONTEXT_FEATURES
    from app.sequence_net import PatternModel, save_weights, load_weights
    torch.manual_seed(19)
    model = PatternModel(len(CONTEXT_FEATURES)).eval()
    # Nonzero heads ensure this checks the CNN, rather than its initial baseline.
    torch.nn.init.normal_(model.accuracy_delta.weight)
    torch.nn.init.normal_(model.pass_delta.weight)
    notes = torch.randn(6, len(FEATURES), 128)
    mask = torch.ones(6, 128, dtype=torch.bool)
    mask[-1, 30:] = False
    bags = torch.tensor([0, 0, 1, 1, 1, 1])
    context = torch.randn(2, len(CONTEXT_FEATURES))
    bp, ba = torch.tensor([1., -1.]), torch.tensor([.9, .8])
    with torch.inference_mode():
        full = model(notes, mask, bags, context, bp, ba)
        chunks = torch.cat([model.encode_chunks(notes[i:i+2], mask[i:i+2])[0] for i in range(0, 6, 2)])
        partial = model.from_chunks(chunks, bags, context, bp, ba)
        assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(full[:2], partial))
    path = tmp_path / 'weights.npz'
    save_weights(model, path)
    restored = load_weights(PatternModel(len(CONTEXT_FEATURES)), path).eval()
    with torch.inference_mode():
        actual = restored(notes, mask, bags, context, bp, ba)
        assert all(torch.equal(a, b) for a, b in zip(full, actual))
    np.savez(path, bad=np.ones(1))
    with pytest.raises(ValueError): load_weights(restored, path)


def test_small_training_freezes_inputs_and_audits_saved_models(tmp_path):
    pytest.importorskip('torch')
    from app.sequence_train import train_experiment, load_corpus_snapshot, audit_experiment
    from app.sequence_data import CONTEXT_FEATURES
    store = Store(tmp_path / 'training.db')
    for i in range(30):
        mid = str(1000 + i)
        raw = chart(int(mid), spacing=150+i*8).replace(b'Title: Test Song', f'Title: Song {i}'.encode())
        store.save_map(parse_map(raw))
        for j in range(3):
            store.save_play('Fixture', mid, 'NM', .88+j*.025, j > 0,
                            played_at=f'2026-10-{10+j:02d}T00:00:00Z', play_id=f'{mid}-{j}')
    output = tmp_path / 'run'
    report = train_experiment(store, 'Fixture', output, steps=1, pretrain_steps=1, batch_size=8, device='cpu')
    assert report['independent_attempts'] == 90
    frozen = load_corpus_snapshot(output)
    assert frozen.contexts.shape == (90, len(CONTEXT_FEATURES))
    assert set(report['variants']) == {'context_only', 'original', 'mirror_x', 'mirror_y', 'mirror_xy', 'four_views'}
    from app.sequence_forward import evaluate_forward
    assert evaluate_forward(store,'Fixture',output)['status']=='awaiting_new_plays'
    # Updating an account afterwards cannot change the frozen audit.
    store.save_play('Fixture', '1000', 'NM', .99, True)
    forward=evaluate_forward(store,'Fixture',output)
    assert forward['status']=='evaluated' and forward['attempts']==1
    assert not forward['pass_validation_sufficient'] and not forward['accuracy_validation_sufficient']
    assert len(frozen.records) == 90 and audit_experiment(output)['status'] == 'verified'
    for path in output.glob('*-predictions.npz'):
        with np.load(path, allow_pickle=False) as data:
            assert len(data['test_pass']) == report['dataset']['splits']['test']['attempts']
    samples = output / 'snapshot/samples.json'
    samples.write_text(samples.read_text(encoding='utf-8') + ' ', encoding='utf-8')
    with pytest.raises(ValueError, match='checksum'): load_corpus_snapshot(output)
