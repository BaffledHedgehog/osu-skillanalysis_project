"""Local, browser-specific OAuth app credentials, never persisted to disk."""
import time

import httpx

from .web_session import CookieVault


async def request_token(client, credentials):
    if credentials.get('access_token') and credentials.get('token_expires_at', 0) > time.time() + 30:
        return credentials['access_token']
    response = await client.post('https://osu.ppy.sh/oauth/token', data={
        'client_id': credentials['client_id'], 'client_secret': credentials['client_secret'],
        'grant_type': 'client_credentials', 'scope': 'public'}, headers={'Accept': 'application/json'})
    if response.status_code in (400, 401, 422):
        raise ValueError('osu! не принял Client ID или Client Secret. Скопируйте оба значения из одного OAuth-приложения. Нужен API v2, не старый API key.')
    if response.status_code == 403:
        raise ValueError('osu! заблокировал запрос токена (403). Это не ошибка формата ключей. Повторите позже или проверьте доступ к osu.ppy.sh с компьютера сервера.')
    if response.status_code == 429:
        raise ValueError('osu! ограничил запросы токена. Повторите подключение позже.')
    response.raise_for_status()
    try:
        payload = response.json()
        token = payload['access_token']
        ttl = int(payload['expires_in'])
        if not isinstance(token, str) or not token or not 0 < ttl <= 31_536_000:
            raise ValueError
    except (ValueError, TypeError, KeyError) as exc:
        raise ValueError('osu! вернул неожиданный ответ OAuth. Токен не сохранен.') from exc
    credentials.update(access_token=token, token_expires_at=time.time() + ttl)
    return token


class OAuthVault(CookieVault):
    async def connect(self, client_id, client_secret, previous=None):
        import secrets
        client_id, client_secret = client_id.strip(), client_secret.strip()
        if len(client_id) > 20 or not client_id.isascii() or not client_id.isdigit() or not 0 < int(client_id) < 2 ** 63:
            raise ValueError('Client ID должен быть положительным числом из настроек OAuth-приложения.')
        if not client_secret or len(client_secret) > 512 or any(c.isspace() for c in client_secret):
            raise ValueError('Вставьте Client Secret целиком, без пробелов и переносов строк.')
        credentials = {'client_id': client_id, 'client_secret': client_secret}
        async with httpx.AsyncClient(timeout=20) as client:
            await request_token(client, credentials)
        self.sessions = {key: entry for key, entry in self.sessions.items() if entry[0] >= time.time()}
        if len(self.sessions) >= 128 and previous not in self.sessions:
            raise ValueError('Слишком много сессий OAuth. Удалите старую или перезапустите сервер.')
        if previous:
            self.remove(previous)
        token = secrets.token_urlsafe(32)
        self.sessions[token] = (time.time() + 28800, credentials)
        return token
