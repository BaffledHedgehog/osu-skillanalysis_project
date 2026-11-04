import asyncio
import hashlib
import time

import httpx
from fastapi.testclient import TestClient
import pytest

from app import osu_api, player_dataset, map_discovery
from app.analyzer import parse_map
from app.main import create_app
from app.ml import _dataset, BEST_SOURCES
from app.predictor import recommendations, recommendation_bounds
from app.storage import Store
from test_app import chart


def test_all_lists_keep_full_payloads_and_never_train_pass_on_curated_or_nf(tmp_path, monkeypatch):
    calls = []
    payloads = {
        'recent': [{'id': 1, 'beatmap_id': 101, 'mods': [], 'passed': False, 'accuracy': .81,
                    'max_combo': 11, 'statistics': {'great': 10, 'miss': 4}},
                   {'id': 2, 'beatmap_id': 102, 'mods': ['NF'], 'passed': True, 'accuracy': .7}],
        'best': [{'id': 3, 'beatmap_id': 103, 'mods': [], 'passed': True, 'accuracy': .98}],
        'pinned': [{'id': 1, 'beatmap_id': 101, 'mods': [], 'passed': False, 'accuracy': .81,
                    'max_combo': 11, 'statistics': {'great': 10, 'miss': 4}},
                   {'id': 4, 'beatmap_id': 104, 'mods': [], 'passed': True, 'accuracy': .99}],
        'firsts': [{'id': 5, 'beatmap_id': 105, 'mods': [], 'passed': True, 'accuracy': 1.0}],
    }
    async def token(*args): return 'test-token'
    async def download(map_id): return chart(int(map_id))
    original_client = httpx.AsyncClient
    def handler(request):
        if request.url.path.endswith('/osu'):
            return httpx.Response(200, json={'id': 99, 'username': 'Player', 'statistics': {'pp': 4000}, 'private_field': 'not persisted'})
        source = request.url.path.rsplit('/', 1)[-1]
        calls.append(source)
        assert request.url.params.get('legacy_only') == '0'
        if source == 'recent': assert request.url.params['include_fails'] == '1'
        return httpx.Response(200, json=payloads[source])
    monkeypatch.setattr(osu_api, 'api_configured', lambda: True)
    monkeypatch.setattr(osu_api, '_token', token)
    monkeypatch.setattr(osu_api, 'download_beatmap', download)
    monkeypatch.setattr(osu_api.httpx, 'AsyncClient', lambda **kwargs: original_client(transport=httpx.MockTransport(handler), **kwargs))
    store = Store(tmp_path / 'test.db')
    result = asyncio.run(osu_api.import_user_scores(store, 'Player', expanded=True))
    assert calls == ['recent', 'best', 'pinned', 'firsts']
    assert result['received_by_source'] == {'recent': 2, 'best': 1, 'pinned': 2, 'firsts': 1}
    assert result['retained_scores'] == 5 and result['imported'] == 4
    rows = {r['score_key']: r for r in store.score_payloads('Player')}
    assert rows['1']['sources'] == ['pinned', 'recent']
    assert rows['1']['payload']['max_combo'] == 11 and rows['1']['payload']['statistics']['miss'] == 4
    assert rows['2']['reason'] == 'no_fail'
    assert '102' in store.excluded_map_ids('Player')
    assert _dataset(store, 'Player')[1].tolist() == [0]
    again = asyncio.run(osu_api.import_user_scores(store, 'Player', expanded=True))
    assert again['cached'] and again['imported'] == 0 and len(calls) == 4


def test_recent_promotes_pinned_and_firsts_without_duplicate_or_target_leak(tmp_path):
    store = Store(tmp_path / 'test.db')
    store.save_map(parse_map(chart(101)))
    store.save_play('Player', '101', 'NM', .99, True, 0, source='osu_api_firsts', play_id='osu-1')
    assert len(_dataset(store, 'Player')[1]) == 0
    store.save_play('Player', '101', 'NM', .99, True, 0, source='osu_api_recent', play_id='osu-1')
    assert store.play_count('Player') == 1 and len(_dataset(store, 'Player')[1]) == 1
    for provider in ('osu_api', 'osu_web', 'hinai'):
        assert all(f'{provider}_{s}' in BEST_SOURCES for s in ('firsts', 'pinned', 'best'))


@pytest.mark.parametrize('public', [True, False])
def test_unified_collection_uses_every_client_and_exact_requested_player(tmp_path, monkeypatch, public):
    store = Store(tmp_path / 'test.db')
    seen = []
    async def online(*args, **kwargs):
        assert kwargs['expanded'] and kwargs['max_scores'] == 1000
        return {'username': 'Canonical', 'imported': 2}
    def local(store, path, user, force):
        seen.append((path, user))
        if path.endswith('broken'): raise ValueError('No scores for this account')
        return {'username': user, 'provider': 'local_' + path.rsplit('/', 1)[-1], 'imported': 1}
    clients = [{'path': str(tmp_path / s), 'client': s, 'supported': True} for s in ('stable', 'broken', 'lazer')]
    monkeypatch.setattr(player_dataset, 'import_user_scores', online)
    monkeypatch.setattr(player_dataset, 'discover_clients', lambda: clients)
    monkeypatch.setattr(player_dataset, 'import_local_scores', local)
    monkeypatch.setattr(player_dataset, 'collect_owned_maps', lambda *args: 10)
    result = asyncio.run(player_dataset.collect_dataset(store, 'Canonical', allow_local=not public))
    assert result['username'] == 'Canonical'
    assert len(seen) == (0 if public else 3)
    assert all(user == 'Canonical' for _, user in seen)
    assert len(result['sources']) == (1 if public else 3)
    assert result['warnings']


def test_dataset_job_returns_progress_result_and_does_not_touch_local_files_in_public_mode(tmp_path, monkeypatch):
    async def collect(store, reference, **kwargs):
        assert kwargs['allow_local'] is False and kwargs['cookies'] is None
        kwargs['progress'](phase='online', message='progress')
        return {'username': 'Player', 'sources': [], 'warnings': [], 'imported': 0, 'total_plays': 0}
    monkeypatch.setattr('app.main.collect_dataset', collect)
    client = TestClient(create_app(tmp_path / 'test.db', public_mode=True))
    start = client.post('/api/dataset/jobs', json={'username': 'Player'})
    assert start.status_code == 200
    for _ in range(100):
        status = client.get('/api/dataset/jobs/' + start.json()['job_id']).json()
        if status['status'] != 'running': break
        time.sleep(.01)
    assert status['status'] == 'completed' and status['result']['username'] == 'Player'
    assert client.get('/api/dataset/jobs/nonexistent').status_code == 404
    assert client.post('/api/dataset/jobs', json={'username': 'https://foreign.example/user'}).status_code == 400


def test_bad_cookie_falls_back_to_public_source_without_forwarding_cookie(tmp_path, monkeypatch):
    calls = []
    async def online(*args, **kwargs):
        calls.append(kwargs)
        if not kwargs.get('use_public'):
            raise ValueError('Expired session')
        assert 'cookies' not in kwargs and 'oauth' not in kwargs
        return {'username': 'Player', 'imported': 0}
    monkeypatch.setattr(player_dataset, 'import_user_scores', online)
    result = asyncio.run(player_dataset.collect_dataset(Store(tmp_path / 'test.db'), 'Player', allow_local=False, cookies={'osu_session': 'private'}))
    assert len(calls) == 2 and result['sources'][0]['username'] == 'Player'


@pytest.mark.parametrize('provider', ['hinai', 'osu_api'])
def test_optional_lists_unavailable_do_not_discard_recent_and_best(tmp_path, monkeypatch, provider):
    original = httpx.AsyncClient
    async def token(*args): return 'test-token'
    async def download(map_id): return chart(int(map_id))
    def handler(request):
        if request.url.path.endswith('/osu') or request.url.path.endswith('/Player'):
            return httpx.Response(200, json={'id': 99, 'username': 'Player'})
        source = request.url.params.get('type') if provider == 'hinai' else request.url.path.rsplit('/', 1)[-1]
        if source in ('pinned', 'firsts'):
            return httpx.Response(404, json={})
        rows = [{'id': 1 if source == 'recent' else 2, 'beatmap_id': 101, 'mods': [], 'passed': True, 'accuracy': .95}]
        return httpx.Response(200, json={'scores': rows} if provider == 'hinai' else rows)
    monkeypatch.setattr(osu_api, 'api_configured', lambda: provider == 'osu_api')
    monkeypatch.setattr(osu_api, '_token', token)
    monkeypatch.setattr(osu_api, 'download_beatmap', download)
    monkeypatch.setattr(osu_api.httpx, 'AsyncClient', lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))
    store = Store(tmp_path / 'test.db')
    result = asyncio.run(osu_api.import_user_scores(store, 'Player', expanded=True))
    assert result['imported'] == 2 and result['source_status']['firsts'] == 'unavailable_http_404'
    assert result['source_status']['pinned'] == ('unsupported_by_provider' if provider == 'hinai' else 'unavailable_http_404')


def test_recommendations_exclude_nf_custom_raw_and_owned_and_do_not_pad_with_trivial_maps(tmp_path):
    store = Store(tmp_path / 'test.db')
    for map_id, spacing in [(101, 70), (102, 70), (103, 70), (104, 70), (105, 2000), (106, 70)]:
        store.save_map(parse_map(chart(map_id, spacing)))
    for i in range(5):
        store.save_play('Player', '101', 'NM', .96, True, 0, play_id=f'play-{i}')
    store.save_score_payload('Player', 'osu_api', 'recent', 'raw-nf', '102', {'mods': ['NF']}, 'no_fail')
    store.save_score_payload('Player', 'osu_api', 'recent', 'raw-custom', '103', {'mods': [{'acronym': 'DT', 'settings': {'speed_change': 1.1}}]})
    store.replace_owned_maps('Player', 'stable', ['104'])
    result = recommendations(store, 'Player')
    assert [r['id'] for r in result] == ['106']
    assert store.analysis('105', 'NM')['stars'] < recommendation_bounds(store, 'Player')[0]
    store.replace_owned_maps('Other', 'stable', ['106'])
    assert [r['id'] for r in recommendations(store, 'Player')] == ['106']
    store.save_play('Player', '106', 'HD', .96, True, 0)
    assert recommendations(store, 'Player') == []


def test_stable_inventory_excludes_unplayed_owned_map_and_checksum_only_history(tmp_path):
    root = tmp_path / 'osu'
    folder = root / 'Songs/example'
    folder.mkdir(parents=True)
    (folder / 'map.osu').write_bytes(chart(101))
    store = Store(tmp_path / 'test.db')
    player_dataset.collect_owned_maps(store, {'path': str(root), 'client': 'stable'}, 'Player')
    assert store.excluded_map_ids('Player') == {'101'}
    store.save_score_payload('Player', 'local_stable', 'local', 'checksum-only', '', {'checksum': hashlib.md5(chart(102)).hexdigest()})
    store.save_map(parse_map(chart(102)))
    assert store.excluded_map_ids('Player') == {'101', '102'}


def test_legacy_lazer_cache_does_not_skip_first_raw_score_export(tmp_path, monkeypatch):
    from app import lazer_client
    root = tmp_path / 'lazer'
    root.mkdir()
    (root / 'client.realm').write_bytes(b'snapshot')
    info = (root / 'client.realm').stat()
    store = Store(tmp_path / 'test.db')
    old_key = f'lazer-rank-v3:{root}:player:{info.st_size}:{info.st_mtime_ns}'
    store.cache_import(old_key, {'username': 'Player', 'imported': 3})
    rows = [{'id': 'offline-1', 'user': 'Player', 'mode': 0, 'map_online_id': 101,
             'map_file_hash': '0'*64, 'mods': '[]'},
            {'id': 'offline-2', 'user': 'Player', 'mode': 0, 'map_online_id': 102,
             'map_file_hash': '0'*64, 'mods': '[]', 'online_id': -1, 'legacy_online_id': -1}]
    monkeypatch.setattr(lazer_client, '_read_realm', lambda _: rows)
    result = lazer_client.import_lazer_scores(store, root, 'Player')
    assert not result['cached']
    assert len(store.score_payloads('Player')) == 2
    assert store.excluded_map_ids('Player') == {'101', '102'}


def test_discovery_filters_children_individually_and_reuses_cache(tmp_path, monkeypatch):
    store = Store(tmp_path / 'test.db')
    store.replace_owned_maps('Player', 'local', ['102'])
    downloaded = []
    async def download(map_id):
        downloaded.append(map_id)
        return chart(int(map_id))
    original_client = httpx.AsyncClient
    def handler(request):
        return httpx.Response(200, json=[{'ChildrenBeatmaps': [
            {'BeatmapID': 101, 'Mode': 0, 'DifficultyRating': 3},
            {'BeatmapID': 102, 'Mode': 0, 'DifficultyRating': 3},
            {'BeatmapID': 103, 'Mode': 0, 'DifficultyRating': 1},
            {'BeatmapID': 104, 'Mode': 3, 'DifficultyRating': 3},
        ]}])
    monkeypatch.setattr(map_discovery, 'recommendation_bounds', lambda *args: (2, 5))
    monkeypatch.setattr(map_discovery, 'download_beatmap', download)
    monkeypatch.setattr(map_discovery.httpx, 'AsyncClient', lambda **kwargs: original_client(transport=httpx.MockTransport(handler), **kwargs))
    first = asyncio.run(map_discovery.discover_maps(store, 'Player'))
    assert downloaded == ['101'] and first['candidates'] == 1
    assert asyncio.run(map_discovery.discover_maps(store, 'Player'))['cached']
