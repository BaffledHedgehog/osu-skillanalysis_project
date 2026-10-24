"""FastAPI entry point for the local osu! skill analysis app."""

from __future__ import annotations

import io
import os
import re
import zipfile
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, Request, Response
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, SecretStr
from urllib.parse import urlparse

from .analyzer import normalize_mods, parse_map
from .ml import model_status, train_models
from .osu_api import api_configured, download_beatmap, import_user_scores
from .predictor import predict, recommendations, training_maps, user_summary
from .replays import import_replay
from .storage import Store
from .local_client import discover_clients, import_local_scores
from .web_session import CookieVault
from .oauth_session import OAuthVault
from .replay_health import calculate_health, register_replay
from . import sequence_service


STATIC = Path(__file__).resolve().parent.parent / "static"
STARTER_MAP_IDS = ("75", "4819932", "221777", "3601629")


class MapLink(BaseModel):
    value: str


class PlayInput(BaseModel):
    user: str = Field(min_length=1, max_length=50)
    map_id: str
    mods: str = "NM"
    accuracy: float = Field(ge=0, le=100)
    passed: bool = True
    misses: int = Field(default=0, ge=0)


class UserInput(BaseModel):
    username: str = Field(min_length=1, max_length=300)
    force: bool = False
    max_scores: int = Field(default=100, ge=1, le=1000)


class CookieInput(BaseModel):
    content: SecretStr


class OAuthInput(BaseModel):
    client_id: str
    client_secret: SecretStr


class LocalInput(BaseModel):
    path: str | None = Field(default=None, max_length=1000)
    user: str | None = Field(default=None, max_length=50)
    force: bool = False


class HealthInput(BaseModel):
    user: str = Field(min_length=1, max_length=50)
    play_id: str = Field(min_length=1, max_length=200)


def require_local(request: Request):
    if getattr(request.app.state, 'public_mode', False):
        raise HTTPException(403, 'Локальные файлы и cookie доступны только в локальном приложении.')
    if request.client and request.client.host not in ('127.0.0.1', '::1', 'testclient'):
        raise HTTPException(403, 'Локальный импорт доступен только на компьютере с сервером.')
    origin = request.headers.get('origin')
    if origin:
        parsed = urlparse(origin)
        if parsed.hostname not in ('127.0.0.1', 'localhost', '::1', 'testserver') or parsed.netloc != request.url.netloc:
            raise HTTPException(403, 'Запрос должен быть отправлен с локальной страницы приложения.')


def _map_id(value: str) -> str:
    value = value.strip()
    if value.isdigit() and len(value) <= 12:
        return value
    # Official beatmap links use a difficulty ID after #osu/ or /b/.
    match = re.search(r"(?:#osu/|/beatmaps/|/b/)(\d{1,12})(?:\D|$)", value)
    if match:
        return match.group(1)
    raise ValueError("Вставьте ID карты или ссылку вида osu.ppy.sh/beatmapsets/...#osu/ID.")


def create_app(db_path: str | Path | None = None, public_mode: bool | None = None) -> FastAPI:
    app = FastAPI(title="osu!skillanalysis", version="0.1.0")
    store = Store(db_path)
    vault = CookieVault()
    oauth_vault = OAuthVault()
    app.state.store = store
    app.state.public_mode = public_mode if public_mode is not None else os.getenv('PULSE_PUBLIC_MODE') == '1'
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc):
        # Never echo submitted cookie or credentials in validation responses.
        from fastapi.exception_handlers import request_validation_exception_handler
        if request.url.path.startswith('/api/session/'):
            return JSONResponse(status_code=422, content={'detail': 'Проверьте формат полей. Значения должны быть строками.'})
        return await request_validation_exception_handler(request, exc)

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/health")
    def health(request: Request):
        return {"ok": True, "api_configured": bool(oauth_vault.get(request.cookies.get('pulse-oauth'))) or api_configured(), "maps": len(store.list_maps()),
                "public_mode": app.state.public_mode}

    @app.get('/api/maps/{map_id}/training')
    def training(map_id: str, user: str = Query(min_length=1), mods: str = 'NM'):
        try:
            target = store.analysis(map_id, mods)
            if not target:
                raise HTTPException(404, 'Карта не найдена.')
            return training_maps(store, user, target)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get('/api/local/clients')
    def local_clients(request: Request):
        require_local(request)
        return {'clients': discover_clients()}

    @app.get('/api/local/nf-replays')
    def nf_replays(request: Request, user: str = Query(min_length=1, max_length=50)):
        require_local(request)
        return {'replays': store.no_fail_replays(user)}

    @app.post('/api/local/nf-health')
    def nf_health(body: HealthInput, request: Request):
        require_local(request)
        try:
            return calculate_health(store, body.user, body.play_id)
        except (ValueError, OSError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post('/api/local/nf-replay')
    async def upload_nf_replay(request: Request, user: str = Form(...), file: UploadFile = File(...)):
        require_local(request)
        try:
            data = await file.read(25_000_001)
            return {'play_id': register_replay(store, user, data)}
        except (ValueError, OSError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post('/api/local/import')
    def local_import(body: LocalInput, request: Request):
        require_local(request)
        try:
            return import_local_scores(store, body.path, body.user, body.force)
        except (ValueError, OSError, UnicodeError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post('/api/session/cookies')
    def cookie_import(body: CookieInput, request: Request, response: Response):
        require_local(request)
        try:
            token, count = vault.put(body.content.get_secret_value(), request.cookies.get('pulse-session'))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        response.set_cookie('pulse-session', token, httponly=True, samesite='strict', max_age=28800)
        return {'active': True, 'count': count, 'message': 'Формат cookie принят. Сессия проверяется сайтом при загрузке истории. Cookie хранятся в памяти 8 часов.'}

    @app.post('/api/session/oauth')
    async def oauth_connect(body: OAuthInput, request: Request, response: Response):
        require_local(request)
        try:
            token = await oauth_vault.connect(body.client_id, body.client_secret.get_secret_value(), request.cookies.get('pulse-oauth'))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except httpx.HTTPError as exc:
            raise HTTPException(502, 'Не удалось связаться с OAuth osu!. Ключи не сохранены. Повторите подключение позже.') from exc
        response.set_cookie('pulse-oauth', token, httponly=True, samesite='strict', max_age=28800,
                            secure=request.url.scheme == 'https')
        return {'active': True, 'message': 'OAuth проверен. Официальный API подключен на 8 часов; ключи только в памяти.'}

    @app.get('/api/session/oauth')
    def oauth_status(request: Request):
        require_local(request)
        return {'active': bool(oauth_vault.get(request.cookies.get('pulse-oauth'))), 'environment_configured': api_configured()}

    @app.delete('/api/session/oauth')
    def oauth_delete(request: Request, response: Response):
        require_local(request)
        oauth_vault.remove(request.cookies.get('pulse-oauth'))
        response.delete_cookie('pulse-oauth')
        return {'active': False, 'environment_configured': api_configured()}

    @app.get('/api/session/cookies')
    def cookie_status(request: Request):
        require_local(request)
        cookies = vault.get(request.cookies.get('pulse-session'))
        return {'active': bool(cookies), 'count': len(cookies or {})}

    @app.delete('/api/session/cookies')
    def cookie_delete(request: Request, response: Response):
        require_local(request)
        vault.remove(request.cookies.get('pulse-session'))
        response.delete_cookie('pulse-session')
        return {'active': False}

    @app.get("/api/maps")
    def maps():
        return store.list_maps()

    @app.post("/api/maps/link")
    async def add_map(body: MapLink):
        try:
            beatmap_id = _map_id(body.value)
            existing = store.get_map(beatmap_id)
            if existing:
                return {"id": existing.map_id, **existing.metadata}
            raw = await download_beatmap(beatmap_id)
            return store.save_map(parse_map(raw, beatmap_id))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except httpx.HTTPStatusError as exc:
            raise HTTPException(404, "Карта не найдена или недоступна на osu!." ) from exc
        except httpx.HTTPError as exc:
            raise HTTPException(502, "Не удалось загрузить карту с osu!.") from exc

    @app.post("/api/maps/starter")
    async def add_starter_maps():
        added = []
        for beatmap_id in STARTER_MAP_IDS:
            existing = store.get_map(beatmap_id)
            if existing:
                added.append({"id": beatmap_id, **existing.metadata})
                continue
            try:
                added.append(store.save_map(parse_map(await download_beatmap(beatmap_id), beatmap_id)))
            except (httpx.HTTPError, ValueError):
                continue
        if not added:
            raise HTTPException(502, "Не удалось загрузить примеры карт с osu!.")
        return {"maps": added}

    @app.post("/api/maps/upload")
    async def upload_map(file: UploadFile = File(...)):
        data = await file.read(25_000_001)
        if len(data) > 25_000_000:
            raise HTTPException(413, "Архив слишком большой (лимит 25 МБ).")
        try:
            if zipfile.is_zipfile(io.BytesIO(data)):
                saved = []
                extracted_size = 0
                with zipfile.ZipFile(io.BytesIO(data)) as archive:
                    for info in archive.infolist():
                        if len(saved) >= 50:
                            break
                        if not info.filename.lower().endswith(".osu") or info.file_size > 8_000_000:
                            continue
                        extracted_size += info.file_size
                        if extracted_size > 40_000_000:
                            raise ValueError("Слишком большой архив карт после распаковки (лимит 40 МБ).")
                        try:
                            saved.append(store.save_map(parse_map(archive.read(info))))
                        except ValueError:
                            continue
                if not saved:
                    raise ValueError("В архиве не найдено карт osu!standard.")
                return {"maps": saved}
            return {"maps": [store.save_map(parse_map(data))]}
        except (ValueError, zipfile.BadZipFile) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/maps/{map_id}/analysis")
    def map_analysis(map_id: str, user: str = "", mods: str = "NM"):
        try:
            analysis = store.analysis(map_id, mods)
            if not analysis:
                raise HTTPException(404, "Карта не найдена в каталоге.")
            return {**analysis, "prediction": predict(store, user, analysis) if user else None,
                    'local_prediction': sequence_service.predict_local(store, user, analysis) if user else None}
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/plays")
    def add_play(body: PlayInput):
        try:
            mods, _ = normalize_mods(body.mods)
            created = store.save_play(body.user, body.map_id, mods, body.accuracy / 100,
                                      body.passed, body.misses)
            return {"created": created, "summary": user_summary(store, body.user)}
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/plays/replay")
    async def add_replay(user: str = Form(...), passed: bool = Form(...), file: UploadFile = File(...)):
        if not user.strip() or len(user) > 50:
            raise HTTPException(400, "Укажите корректный ник игрока.")
        raw = await file.read(25_000_001)
        try:
            result = import_replay(store, raw, user, passed)
            return {**result, "summary": user_summary(store, user)}
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/users/{username}/summary")
    def summary(username: str):
        return user_summary(store, username)

    @app.get("/api/users/{username}/model")
    def user_model(username: str):
        return model_status(store, username)

    @app.post("/api/users/{username}/model/train")
    def train_user_models(username: str):
        if not username.strip() or len(username) > 50:
            raise HTTPException(400, "Укажите корректный ник игрока.")
        return train_models(store, username)

    @app.get('/api/users/{username}/cnn')
    def cnn_status(username: str, request: Request):
        require_local(request)
        return sequence_service.status(store, username)

    @app.post('/api/users/{username}/cnn/train')
    def cnn_train(username: str, request: Request):
        require_local(request)
        if not username.strip() or len(username) > 50:
            raise HTTPException(400, 'Укажите корректный ник игрока.')
        try:
            return sequence_service.start_training(store, username)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get('/api/users/{username}/cnn/report')
    def cnn_report(username: str, request: Request):
        require_local(request)
        state = sequence_service.status(store, username)
        report = state['report']
        if report is None:
            raise HTTPException(404, 'Завершенных экспериментов пока нет.')
        return JSONResponse({**report,'forward':state['forward']}, headers={'Content-Disposition': 'attachment; filename="cnn-report.json"'})

    @app.get('/api/users/{username}/cnn/local')
    def local_cnn_status(username: str, request: Request):
        require_local(request)
        return sequence_service.status(store, username, local=True)

    @app.post('/api/users/{username}/cnn/local/train')
    def local_cnn_train(username: str, request: Request):
        require_local(request)
        if not username.strip() or len(username) > 50:
            raise HTTPException(400, 'Укажите корректный ник игрока.')
        try:
            return sequence_service.start_training(store, username, local=True)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get('/api/users/{username}/cnn/local/report')
    def local_cnn_report(username: str, request: Request):
        require_local(request)
        report = sequence_service.status(store, username, local=True)['report']
        if report is None:
            raise HTTPException(404, 'Завершенных экспериментов пока нет.')
        return JSONResponse(report, headers={'Content-Disposition': 'attachment; filename="local-cnn-report.json"'})

    @app.post('/api/users/{username}/cnn/stop')
    def cnn_stop(username: str, request: Request):
        require_local(request)
        return sequence_service.stop_training(store, username)

    @app.post('/api/users/{username}/cnn/forward')
    def cnn_forward(username: str, request: Request):
        require_local(request)
        try:
            return sequence_service.check_forward(store,username)
        except (ValueError,OSError) as exc:
            raise HTTPException(400,str(exc)) from exc

    @app.post("/api/users/import")
    async def import_user(body: UserInput, request: Request):
        try:
            cookies = vault.get(request.cookies.get('pulse-session'))
            oauth = oauth_vault.get(request.cookies.get('pulse-oauth'))
            if cookies or oauth:
                require_local(request)
            return await import_user_scores(store, body.username, force=body.force,
                                            max_scores=body.max_scores, cookies=cookies, oauth=oauth)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise HTTPException(404, "Игрок не найден на osu!.") from exc
            raise HTTPException(502, "Источник результатов недоступен. Попробуйте позже.") from exc
        except httpx.HTTPError as exc:
            raise HTTPException(502, "Не удалось получить результаты игрока.") from exc

    @app.get("/api/recommendations")
    def recommend(user: str, goal: str = Query("comfort", pattern="^(warmup|comfort|challenge)$"),
                  mods: str = "NM"):
        try:
            normalize_mods(mods)
            return {"goal": goal, "maps": recommendations(store, user, goal, mods),
                    "catalog_size": len(store.list_maps())}
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    return app


app = create_app()
