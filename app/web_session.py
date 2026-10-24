"""Explicit cookie imports. Secrets live only in this process and only target osu.ppy.sh."""
from __future__ import annotations

import json
import re
import secrets
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse


def parse_cookies(text: str) -> dict[str, str]:
    if not text.strip() or len(text.encode()) > 256_000:
        raise ValueError('Вставьте cookie: JSON Cookie Editor, Netscape или строку Cookie (до 256 КБ).')
    result = {}
    rejected = {'domain': 0, 'expired': 0, 'invalid': 0}

    def add(name, value, domain=None, expires=None):
        if domain and str(domain).strip().lstrip('.').casefold() not in ('osu.ppy.sh', 'ppy.sh'):
            rejected['domain'] += 1
            return
        if expires is not None and str(expires).strip().lower() not in ('', 'session', 'null', 'none'):
            try:
                try:
                    expiry = float(expires)
                    if expiry > 1e12:
                        expiry /= 1000  # Some browser exports use milliseconds.
                except (ValueError, TypeError):
                    try:
                        date = datetime.fromisoformat(str(expires).replace('Z', '+00:00'))
                    except ValueError:
                        date = parsedate_to_datetime(str(expires))
                    expiry = date.replace(tzinfo=date.tzinfo or timezone.utc).timestamp()
                if expiry > 0 and expiry < time.time():
                    rejected['expired'] += 1
                    return
            except (ValueError, TypeError, OverflowError):
                rejected['invalid'] += 1
                return
        if not isinstance(name, str) or not re.fullmatch(r'[!#$%&\'*+.^_`|~0-9A-Za-z-]+', name):
            return
        if not isinstance(value, str) or not value or len(value) > 16_384 or any(ord(c) < 33 or ord(c) > 126 or c == ';' for c in value):
            return
        if name.lower() in {'path', 'domain', 'expires', 'max-age', 'secure', 'httponly', 'samesite'}:
            return
        result[name] = value

    stripped = text.lstrip('\ufeff').strip()
    if stripped.startswith('```'):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == '```':
            stripped = '\n'.join(lines[1:-1]).strip()
    # JSON encoded as a string can result from copying an export from another app.
    if stripped.startswith('"'):
        try:
            decoded = json.loads(stripped)
            if isinstance(decoded, str):
                stripped = decoded.strip()
        except ValueError:
            pass
    if stripped.startswith(('[', '{')):
        try:
            data = json.loads(stripped)
        except ValueError as exc:
            raise ValueError('Не удалось прочитать JSON cookie.') from exc
        if isinstance(data, dict) and isinstance(data.get('cookies'), list):
            data = data['cookies']
        elif isinstance(data, dict) and 'name' in data and 'value' in data:
            data = [data]
        if isinstance(data, list):
            for cookie in data:
                if isinstance(cookie, dict):
                    add(cookie.get('name'), cookie.get('value'), cookie.get('domain'),
                        None if cookie.get('session') is True else cookie.get('expirationDate', cookie.get('expires', cookie.get('expiry'))))
        elif isinstance(data, dict):
            for name, value in data.items():
                add(name, value)
    elif '\t' in stripped:
        for line in stripped.splitlines():
            line = line.removeprefix('#HttpOnly_')
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) == 7:
                add(parts[5], parts[6], parts[0], parts[4])
    else:
        for part in re.sub(r'^Cookie:\s*', '', stripped, flags=re.I).split(';'):
            name, sep, value = part.strip().partition('=')
            if sep:
                add(name, value)
    if not result or len(result) > 100:
        raise ValueError(f'Нет подходящих cookie для osu.ppy.sh. Другой домен: {rejected["domain"]}; истекшие: {rejected["expired"]}; неверная дата: {rejected["invalid"]}. Экспортируйте все cookie со страницы osu.ppy.sh, включая HttpOnly.')
    return result


class CookieVault:
    def __init__(self):
        self.sessions = {}

    def put(self, content: str, previous: str | None = None):
        cookies = parse_cookies(content)
        if not any(name.endswith('osu_session') for name in cookies):
            raise ValueError('В экспорте нет osu_session. Войдите на osu.ppy.sh и экспортируйте все cookie, включая HttpOnly. Одного XSRF-TOKEN недостаточно.')
        self.sessions = {key: entry for key, entry in self.sessions.items() if entry[0] >= time.time()}
        if len(self.sessions) >= 128 and previous not in self.sessions:
            raise ValueError('Слишком много локальных сессий. Удалите старую сессию или перезапустите сервер.')
        if previous:
            self.sessions.pop(previous, None)
        token = secrets.token_urlsafe(32)
        self.sessions[token] = (time.time() + 3600 * 8, cookies)
        return token, len(cookies)

    def get(self, token: str | None):
        entry = self.sessions.get(token)
        if not entry:
            return None
        expires, cookies = entry
        if expires < time.time():
            self.sessions.pop(token, None)
            return None
        return cookies.copy()

    def remove(self, token):
        self.sessions.pop(token, None)


class ProfileParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.profile = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get('data-react') == 'profile-page' and 'data-initial-data' in attrs:
            try:
                self.profile = json.loads(attrs['data-initial-data']).get('user')
            except (ValueError, AttributeError):
                pass


async def web_get(client, url: str, cookies: dict, params=None, accept_json=True):
    headers = {'Cookie': '; '.join(f'{name}={value}' for name, value in cookies.items()),
               'Accept': 'application/json' if accept_json else 'text/html',
               'Referer': 'https://osu.ppy.sh/'}
    for _ in range(4):
        parsed = urlparse(url)
        if parsed.scheme != 'https' or parsed.hostname != 'osu.ppy.sh' or parsed.port not in (None, 443):
            raise ValueError('Запрос cookie разрешен только к https://osu.ppy.sh.')
        response = await client.get(url, headers=headers, params=params)
        if response.status_code in (301, 302, 303, 307, 308):
            url, params = urljoin(url, response.headers.get('location', '')), None
            if '/login' in url or '/session' in url:
                raise ValueError('Сессия osu! истекла. Войдите на сайте и импортируйте cookie заново.')
            continue
        if response.status_code == 401:
            raise ValueError('osu! не принял сессию (401). Формат cookie прочитан, но вход не подтвержден. Войдите на сайте заново или подключите OAuth.')
        if response.status_code == 403:
            raise ValueError('osu! отклонил запрос (403). Даже свежие cookie могут не работать вне браузера из-за проверки сайта или привязки сессии. Подключите OAuth для официального API.')
        if response.status_code == 429:
            raise ValueError('osu! ограничил частоту запросов. Повторите загрузку позже.')
        response.raise_for_status()
        return response
    raise ValueError('osu! вернул слишком много перенаправлений.')
