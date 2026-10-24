import hashlib
import html
import json
import struct
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.storage import Store
from app.local_client import read_scores
from app.web_session import parse_cookies, web_get
from app.predictor import predict
from app import osu_api
from test_app import chart


def osu_string(value):
    raw = value.encode()
    if not raw:
        return b'\0'
    length, chunks = len(raw), bytearray()
    while length >= 128:
        chunks.append((length & 127) | 128)
        length >>= 7
    chunks.append(length)
    return b'\x0b' + bytes(chunks) + raw


def score_database(raw, count=2):
    checksum = hashlib.md5(raw).hexdigest()
    rows = []
    for index in range(count):
        rows.append(
            struct.pack('<Bi', 0, 20260412) + osu_string(checksum) + osu_string('Player') + osu_string(f'replay-{index}') +
            struct.pack('<6H', 20, 3, 0, 0, 0, 1) + struct.pack('<iHBi', 100000 + index, 23, 0, 0) + osu_string('') +
            struct.pack('<qiq', 638000000000000000 + index * 10000000, -1, 900 + index)
        )
    return struct.pack('<ii', 20260412, 1) + osu_string(checksum) + struct.pack('<i', count) + b''.join(rows)


def test_local_client_reads_full_scores_incrementally_and_does_not_invent_pass(tmp_path):
    root = tmp_path / 'osu!'
    songs = root / 'Songs' / 'Test'
    songs.mkdir(parents=True)
    raw = chart(800)
    (songs / 'map.osu').write_bytes(raw)
    (root / 'osu!.test.cfg').write_text('Username = Player\nBeatmapDirectory = Songs\n', encoding='utf-8')
    (root / 'scores.db').write_bytes(score_database(raw))
    client = TestClient(create_app(tmp_path / 'store.db'))
    first = client.post('/api/local/import', json={'path': str(root)})
    assert first.status_code == 200, first.text
    assert first.json()['username'] == 'Player' and first.json()['imported'] == 2
    store = client.app.state.store
    rows = store.plays('Player')
    assert len(rows) == 2 and all(r['passed_known'] == 0 for r in rows)
    assert all(r['source'] == 'local_unknown' for r in rows)
    assert client.get('/api/users/Player/model').json()['plays'] == 0
    token = store.cache_token('Player')
    second = client.post('/api/local/import', json={'path': str(root)}).json()
    assert second['cached'] and second['imported'] == 0
    assert token == store.cache_token('Player')
    (root / 'scores.db').write_bytes(score_database(raw, 3))
    third = client.post('/api/local/import', json={'path': str(root)}).json()
    assert third['imported'] == 1 and store.play_count('Player') == 3
    assert token != store.cache_token('Player')
    assert (root / 'scores.db').read_bytes() == score_database(raw, 3)


def test_broken_scores_and_cross_origin_local_access_rejected(tmp_path):
    with pytest.raises(ValueError):
        read_scores(b'broken')
    root = tmp_path / 'osu!'
    root.mkdir()
    (root / 'scores.db').write_bytes(b'broken')
    client = TestClient(create_app(tmp_path / 'test.db'))
    assert client.post('/api/local/import', json={'path': str(root)}).status_code == 400
    assert client.post('/api/local/import', json={'path': str(root)}, headers={'Origin': 'https://foreign.example'}).status_code == 403
    assert client.post('/api/session/cookies', json={'content': 'osu_session=secret'}, headers={'Origin': 'https://foreign.example'}).status_code == 403


def test_lazer_relocated_store_imports_incrementally_without_touching_source(tmp_path, monkeypatch):
    from app import lazer_client
    root = tmp_path / 'moved-lazer'
    root.mkdir()
    pointer = tmp_path / 'default-lazer'
    pointer.mkdir()
    (pointer / 'storage.ini').write_text(f'FullPath = {root}\n', encoding='utf-8')
    (root / 'client.realm').write_bytes(b'realm-test-snapshot')
    raw = chart(901)
    digest = hashlib.sha256(raw).hexdigest()
    def save_file(content):
        checksum = hashlib.sha256(content).hexdigest()
        file = root / 'files' / checksum[0] / checksum[:2] / checksum
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(content)
        return checksum
    save_file(raw)
    replay = save_file(b'test replay reference')
    rows = [{'id': 'local-1', 'user': 'LazerPlayer', 'mode': 0, 'accuracy': .95,
             'map_file_hash': digest, 'mods': '[{"acronym":"HD"}]',
             'statistics': '{"great":20,"ok":3,"miss":1}',
             'replay_files': [{'hash': replay}], 'legacy_online_id': -1, 'online_id': -1}]
    monkeypatch.setattr(lazer_client, '_read_realm', lambda folder: rows)
    client = TestClient(create_app(tmp_path / 'lazer.db'))
    result = client.post('/api/local/import', json={'path': str(pointer)})
    assert result.status_code == 200, result.text
    data = result.json()
    assert data['provider'] == 'local_lazer' and data['imported'] == 1
    assert data['available_replays'] == 1
    store = client.app.state.store
    play = store.plays('LazerPlayer')[0]
    assert play['mods'] == 'HD' and play['passed_known'] == 0
    again = client.post('/api/local/import', json={'path': str(root)}).json()
    assert again['cached'] and again['imported'] == 0
    rows.append({**rows[0], 'id': 'local-2'})
    (root / 'client.realm').write_bytes(b'realm-test-snapshot-new')
    added = client.post('/api/local/import', json={'path': str(root)}).json()
    assert added['imported'] == 1 and store.play_count('LazerPlayer') == 2
    assert (root / 'client.realm').read_bytes() == b'realm-test-snapshot-new'


def test_lazer_rejects_missing_files_partial_scores_and_custom_mods(tmp_path, monkeypatch):
    from app import lazer_client
    root = tmp_path / 'lazer'
    root.mkdir()
    (root / 'client.realm').write_bytes(b'test')
    raw = chart(902)
    digest = hashlib.sha256(raw).hexdigest()
    path = root / 'files' / digest[0] / digest[:2] / digest
    path.parent.mkdir(parents=True)
    path.write_bytes(raw)
    base = {'id': '1', 'user': 'Player', 'mode': 0, 'accuracy': .9, 'map_file_hash': digest,
            'mods': '[]', 'statistics': '{"great":2}', 'replay_files': []}
    rows = [base, {**base, 'id': '2', 'mods': '[{"acronym":"DT","settings":{"speed_change":1.2}}]'},
            {**base, 'id': '3', 'map_file_hash': 'a' * 64}]
    monkeypatch.setattr(lazer_client, '_read_realm', lambda folder: rows)
    result = lazer_client.import_lazer_scores(Store(tmp_path / 'store.db'), root)
    assert result['imported'] == 0 and result['skipped'] == 2
    assert result['incomplete_scores'] == 1 and result['missing_maps'] == 1
    monkeypatch.setattr(lazer_client, '_read_realm', lambda folder: pytest.fail('Unchanged database must use cache'))
    assert lazer_client.import_lazer_scores(Store(tmp_path / 'store.db'), root)['cached']


def test_mod_features_do_not_apply_nm_model_to_hidden():
    from app.ml import vector, predict_accuracy, predict_pass
    analysis = {'stars': 4, 'aim': 2, 'speed': 2, 'ar': 8, 'od': 8, 'duration': 120, 'mods': 'HD'}
    values = vector(analysis)
    assert values[7] == 1 and values[14] == 2
    context = ({'mod_bounds': [[0, 0]] * 7}, {'status': 'trained'})
    for function in (predict_accuracy, predict_pass):
        value, status = function(None, 'Player', analysis, context=context)
        assert value is None and status['status'] == 'insufficient_mod_data'


@pytest.mark.parametrize('content', [
    '[{"name":"osu_session","value":"secret","domain":".osu.ppy.sh"}]',
    '{"cookies":[{"name":"osu_session","value":"secret","domain":"osu.ppy.sh"}]}',
    'Cookie: osu_session=secret',
    '# Netscape HTTP Cookie File\n#HttpOnly_.osu.ppy.sh\tTRUE\t/\tTRUE\t0\tosu_session\tsecret',
])
def test_cookie_formats_and_domain_filter(content):
    assert parse_cookies(content) == {'osu_session': 'secret'}
    with pytest.raises(ValueError):
        parse_cookies('[{"name":"bad","value":"secret","domain":"foreign.example"}]')
    with pytest.raises(ValueError):
        parse_cookies('osu_session=secret\r\nInjected: value')


def test_cookie_sessions_are_isolated_and_not_written_to_database(tmp_path):
    app = create_app(tmp_path / 'sessions.db')
    one, two = TestClient(app), TestClient(app)
    result = one.post('/api/session/cookies', json={'content': 'osu_session=highly-secret-example'})
    assert result.status_code == 200 and 'highly-secret-example' not in result.text
    assert one.get('/api/session/cookies').json()['active']
    assert not two.get('/api/session/cookies').json()['active']
    assert b'highly-secret-example' not in (tmp_path / 'sessions.db').read_bytes()
    assert 'HttpOnly' in result.headers['set-cookie'] and 'SameSite=strict' in result.headers['set-cookie']
    assert one.delete('/api/session/cookies').json()['active'] is False
    assert not one.get('/api/session/cookies').json()['active']


def test_web_cookie_import_paginates_and_reuses_cache_without_losing_attempts(tmp_path, monkeypatch):
    calls = []
    class Response:
        status_code = 200
        def __init__(self, data=None, text=''):
            self.data, self.text = data, text
        def raise_for_status(self): pass
        def json(self): return self.data
    class Client:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def get(self, url, headers=None, params=None):
            assert url.startswith('https://osu.ppy.sh/')
            assert headers['Cookie'] == 'osu_session=secret'
            calls.append((url, params))
            if '/scores/' not in url:
                payload = html.escape(json.dumps({'user': {'id': 123, 'username': 'Player'}}))
                return Response(text=f'<div data-react="profile-page" data-initial-data="{payload}"></div>')
            source = url.rsplit('/', 1)[-1]
            if params['offset']:
                return Response([])
            if source == 'recent':
                return Response([{'id': 1, 'beatmap_id': 1, 'accuracy': .9, 'passed': True},
                                 {'id': 2, 'beatmap_id': 1, 'accuracy': .7, 'passed': False}])
            return Response([{'id': 3, 'beatmap_id': 2, 'accuracy': .99, 'passed': True}])
    async def download(map_id): return chart(int(map_id))
    monkeypatch.setattr(osu_api.httpx, 'AsyncClient', Client)
    monkeypatch.setattr(osu_api, 'download_beatmap', download)
    monkeypatch.setattr(osu_api, 'SCORES_PER_SOURCE', 2)
    client = TestClient(create_app(tmp_path / 'web.db'))
    client.post('/api/session/cookies', json={'content': 'osu_session=secret'})
    body = {'username': 'Player', 'max_scores': 1000}
    first = client.post('/api/users/import', json=body)
    assert first.status_code == 200, first.text
    assert first.json()['provider'] == 'osu_web' and first.json()['imported'] == 3
    count = len(calls)
    token = client.app.state.store.cache_token('Player')
    second = client.post('/api/users/import', json=body).json()
    assert second['cached'] and second['imported'] == 0 and len(calls) == count
    fresh = client.post('/api/users/import', json={**body, 'force': True}).json()
    assert fresh['imported'] == 0 and len(calls) > count
    assert token == client.app.state.store.cache_token('Player')
    model = client.get('/api/users/Player/model').json()
    assert model['plays'] == 2 and model['fails'] == 1


def test_summary_and_model_inputs_cached_until_player_changes(tmp_path, monkeypatch):
    store = Store(tmp_path / 'cache.db')
    store.save_map(__import__('app.analyzer', fromlist=['parse_map']).parse_map(chart(999)))
    for i in range(3):store.save_play('Player', '999', 'NM', .95, True, play_id=str(i))
    original, calls = Store.plays_with_analysis, []
    def counted(self, user):
        calls.append(user)
        return original(self, user)
    monkeypatch.setattr(Store, 'plays_with_analysis', counted)
    target = store.analysis('999')
    first = predict(store, 'Player', target)
    calls.clear()
    second = predict(store, 'Player', target)
    assert not calls and first == second
    store.save_play('Player', '999', 'NM', .7, False, play_id='new')
    third = predict(store, 'Player', target)
    assert calls and third['observations'] == 4 and third['pass_probability'] < second['pass_probability']


def test_unchanged_map_keeps_analysis_and_online_score_confirms_local_outcome(tmp_path):
    from app.analyzer import parse_map
    store = Store(tmp_path / 'version.db')
    parsed = parse_map(chart(55))
    store.save_map(parsed)
    store.analysis('55')
    store.save_play('Player','55','NM',.9,False,source='local_unknown',play_id='osu-12',passed_known=False)
    token = store.cache_token('Player')
    store.save_map(parsed)
    assert store.cache_token('Player') == token
    store.save_play('Player','55','NM',.9,False,source='osu_web_recent',play_id='osu-12')
    row=store.plays('Player')[0]
    assert row['passed_known'] == 1 and row['source'] == 'osu_web_recent'
    assert store.cache_token('Player') != token
