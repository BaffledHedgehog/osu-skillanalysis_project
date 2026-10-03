"""FastAPI entry point for the local osu! skill analysis app."""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .analyzer import normalize_mods, parse_map
from .ml import model_status
from .osu_api import api_configured, download_beatmap, import_user_scores
from .predictor import predict, recommendations, user_summary
from .replays import import_replay
from .storage import Store


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
    username: str = Field(min_length=1, max_length=50)


def _map_id(value: str) -> str:
    value = value.strip()
    if value.isdigit() and len(value) <= 12:
        return value
    # Official beatmap links use a difficulty ID after #osu/ or /b/.
    match = re.search(r"(?:#osu/|/beatmaps/|/b/)(\d{1,12})(?:\D|$)", value)
    if match:
        return match.group(1)
    raise ValueError("Вставьте ID карты или ссылку вида osu.ppy.sh/beatmapsets/...#osu/ID.")


def create_app(db_path: str | Path | None = None) -> FastAPI:
    app = FastAPI(title="osu!skillanalysis", version="0.1.0")
    store = Store(db_path)
    app.state.store = store
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/health")
    def health():
        return {"ok": True, "api_configured": api_configured(), "maps": len(store.list_maps())}

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
            return {**analysis, "prediction": predict(store, user, analysis) if user else None}
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

    @app.post("/api/users/import")
    async def import_user(body: UserInput):
        try:
            return await import_user_scores(store, body.username)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise HTTPException(404, "Игрок не найден на osu!.") from exc
            raise HTTPException(502, "osu! API вернул ошибку. Попробуйте позже.") from exc
        except httpx.HTTPError as exc:
            raise HTTPException(502, "Не удалось связаться с osu! API.") from exc

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
