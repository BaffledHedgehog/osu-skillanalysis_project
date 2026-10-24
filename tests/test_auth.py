import json
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.web_session import parse_cookies, CookieVault, web_get
from app import oauth_session, osu_api


@pytest.mark.parametrize('expiry', ['2099-01-01T00:00:00.000Z', 'Thu, 01 Jan 2099 00:00:00 GMT', 4_070_908_800_000, 'Session', -1])
def test_fresh_parent_domain_exports_with_browser_expiry_formats(expiry):
    payload = [{'domain': '.ppy.sh', 'name': 'osu_session', 'value': 'fresh-session', 'expires': expiry}]
    assert parse_cookies(json.dumps(payload)) == {'osu_session': 'fresh-session'}


def test_cookie_export_bom_wrapped_json_and_session_flag():
    payload = [{'domain': '.osu.ppy.sh', 'name': 'osu_session', 'value': 'session', 'session': True, 'expires': 1}]
    for text in ['\ufeff' + json.dumps(payload), '```json\n' + json.dumps(payload) + '\n```', json.dumps(json.dumps(payload))]:
        assert parse_cookies(text) == {'osu_session': 'session'}
    with pytest.raises(ValueError, match='osu_session'):
        CookieVault().put('XSRF-TOKEN=not-a-login-session')
    for domain in ['evil.ppy.sh', 'ppy.sh.attacker.example', 'evil-osu.ppy.sh']:
        with pytest.raises(ValueError):
            parse_cookies(json.dumps([{'domain': domain, 'name': 'osu_session', 'value': 'secret'}]))


def test_expired_cookie_rejection_is_specific_and_never_echoes_value():
    with pytest.raises(ValueError) as error:
        parse_cookies(json.dumps([{'name': 'osu_session', 'value': 'do-not-echo-this', 'domain': '.ppy.sh', 'expires': time.time() - 10}]))
    assert '1' in str(error.value) and 'do-not-echo-this' not in str(error.value)


def mock_oauth(monkeypatch, *, status=200):
    calls = []
    class Client:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def post(self, url, **kwargs):
            calls.append((url, kwargs))
            return httpx.Response(status, json={'access_token': 'example-private-access-token', 'expires_in': 3600,
                                               'error_description': 'never forward this upstream text'},
                                  request=httpx.Request('POST', url))
    monkeypatch.setattr(oauth_session.httpx, 'AsyncClient', Client)
    return calls


def test_oauth_connect_validates_token_isolates_sessions_and_never_persists_secrets(tmp_path, monkeypatch):
    calls = mock_oauth(monkeypatch)
    app = create_app(tmp_path / 'oauth.db')
    one, two = TestClient(app), TestClient(app)
    before = one.get('/api/health').json()['api_configured']
    response = one.post('/api/session/oauth', json={'client_id': '123', 'client_secret': 'private-client-secret'})
    assert response.status_code == 200, response.text
    assert one.get('/api/session/oauth').json()['active']
    assert not two.get('/api/session/oauth').json()['active']
    assert one.get('/api/health').json()['api_configured']
    assert calls[0][0] == 'https://osu.ppy.sh/oauth/token'
    assert calls[0][1]['data']['grant_type'] == 'client_credentials'
    assert 'HttpOnly' in response.headers['set-cookie']
    assert 'private-client-secret' not in response.text and 'example-private-access-token' not in response.text
    assert b'private-client-secret' not in (tmp_path / 'oauth.db').read_bytes()
    captured = {}
    async def imported(store, username, **kwargs):
        captured.update(kwargs)
        return {'provider': 'osu_api'}
    monkeypatch.setattr('app.main.import_user_scores', imported)
    assert one.post('/api/users/import', json={'username': 'Player'}).json()['provider'] == 'osu_api'
    assert captured['oauth']['client_secret'] == 'private-client-secret'
    assert one.delete('/api/session/oauth').status_code == 200
    assert not one.get('/api/session/oauth').json()['active']
    assert one.get('/api/health').json()['api_configured'] == before


def test_oauth_rejection_preserves_old_session_and_hides_upstream_error(tmp_path, monkeypatch):
    mock_oauth(monkeypatch)
    client = TestClient(create_app(tmp_path / 'test.db'))
    assert client.post('/api/session/oauth', json={'client_id': '123', 'client_secret': 'old-secret'}).status_code == 200
    mock_oauth(monkeypatch, status=401)
    response = client.post('/api/session/oauth', json={'client_id': '123', 'client_secret': 'wrong-secret'})
    assert response.status_code == 400
    assert 'wrong-secret' not in response.text and 'upstream' not in response.text
    assert client.get('/api/session/oauth').json()['active']


def test_oauth_is_local_only_and_validation_does_not_echo_credentials(tmp_path, monkeypatch):
    client = TestClient(create_app(tmp_path / 'test.db', public_mode=True))
    response = client.post('/api/session/oauth', json={'client_id': '1', 'client_secret': 'private-secret'})
    assert response.status_code == 403
    client = TestClient(create_app(tmp_path / 'local.db'))
    assert client.post('/api/session/oauth', json={'client_id': '1', 'client_secret': 's'}, headers={'Origin': 'https://foreign.example'}).status_code == 403
    response = client.post('/api/session/oauth', json={'client_id': '1', 'client_secret': ['private-secret']})
    assert response.status_code == 422 and 'private-secret' not in response.text


def test_cookie_redirect_cannot_leak_to_external_domain():
    import asyncio
    calls = []
    class Client:
        async def get(self, url, **kwargs):
            calls.append(url)
            return httpx.Response(302, headers={'location': 'https://foreign.example/steal'}, request=httpx.Request('GET', url))
    with pytest.raises(ValueError):
        asyncio.run(web_get(Client(), 'https://osu.ppy.sh/home/account/edit', {'osu_session': 'secret'}))
    assert calls == ['https://osu.ppy.sh/home/account/edit']
