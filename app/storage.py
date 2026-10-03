"""Small local SQLite store. Raw map charts remain on the user's machine."""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .analyzer import ParsedMap, analyze_map


ANALYZER_VERSION = 1


class Store:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or os.getenv("DATABASE_PATH", ".data/osu_skill.db"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS maps (
                    id TEXT PRIMARY KEY, checksum TEXT NOT NULL, content TEXT NOT NULL,
                    metadata TEXT NOT NULL, added_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS analyses (
                    map_id TEXT NOT NULL, mods TEXT NOT NULL, version INTEGER NOT NULL,
                    result TEXT NOT NULL, PRIMARY KEY (map_id, mods, version)
                );
                CREATE TABLE IF NOT EXISTS plays (
                    id TEXT PRIMARY KEY, user_key TEXT NOT NULL, map_id TEXT NOT NULL,
                    mods TEXT NOT NULL, accuracy REAL NOT NULL, passed INTEGER NOT NULL,
                    misses INTEGER NOT NULL, played_at TEXT NOT NULL, source TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS plays_user_date ON plays(user_key, played_at DESC);
                """
            )

    @contextmanager
    def _connection(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA journal_mode=WAL")
            with db:
                yield db
        finally:
            db.close()

    def save_map(self, parsed: ParsedMap) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        with self._connection() as db:
            db.execute(
                "INSERT INTO maps(id,checksum,content,metadata,added_at) VALUES(?,?,?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET checksum=excluded.checksum, content=excluded.content, "
                "metadata=excluded.metadata, added_at=excluded.added_at",
                (parsed.map_id, parsed.checksum, parsed.content, json.dumps(parsed.metadata), now),
            )
            db.execute("DELETE FROM analyses WHERE map_id=?", (parsed.map_id,))
        return {"id": parsed.map_id, **parsed.metadata}

    def get_map(self, map_id: str) -> ParsedMap | None:
        with self._connection() as db:
            row = db.execute("SELECT * FROM maps WHERE id=?", (map_id,)).fetchone()
        if not row:
            return None
        return ParsedMap(row["id"], row["checksum"], row["content"], json.loads(row["metadata"]))

    def get_map_by_checksum(self, checksum: str) -> ParsedMap | None:
        with self._connection() as db:
            row = db.execute("SELECT * FROM maps WHERE checksum=? LIMIT 1", (checksum,)).fetchone()
        if not row:
            return None
        return ParsedMap(row["id"], row["checksum"], row["content"], json.loads(row["metadata"]))

    def list_maps(self) -> list[dict[str, Any]]:
        with self._connection() as db:
            rows = db.execute("SELECT id,metadata FROM maps ORDER BY added_at DESC").fetchall()
        return [{"id": row["id"], **json.loads(row["metadata"])} for row in rows]

    def analysis(self, map_id: str, mods: str = "NM") -> dict[str, Any] | None:
        from .analyzer import normalize_mods

        mods, _ = normalize_mods(mods)
        with self._connection() as db:
            row = db.execute(
                "SELECT result FROM analyses WHERE map_id=? AND mods=? AND version=?",
                (map_id, mods, ANALYZER_VERSION),
            ).fetchone()
        if row:
            return json.loads(row["result"])
        parsed = self.get_map(map_id)
        if not parsed:
            return None
        result = analyze_map(parsed, mods)
        with self._connection() as db:
            db.execute(
                "INSERT OR REPLACE INTO analyses(map_id,mods,version,result) VALUES(?,?,?,?)",
                (map_id, mods, ANALYZER_VERSION, json.dumps(result)),
            )
        return result

    def save_play(
        self, user: str, map_id: str, mods: str, accuracy: float, passed: bool,
        misses: int = 0, played_at: str | None = None, source: str = "manual", play_id: str | None = None,
    ) -> bool:
        if not self.get_map(map_id):
            raise ValueError("Сначала добавьте эту карту в каталог.")
        played_at = played_at or datetime.now(timezone.utc).isoformat()
        play_id = play_id or str(uuid.uuid4())
        with self._connection() as db:
            cursor = db.execute(
                "INSERT OR IGNORE INTO plays(id,user_key,map_id,mods,accuracy,passed,misses,played_at,source) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (play_id, user.strip().casefold(), map_id, mods, accuracy, int(passed), misses, played_at, source),
            )
            return cursor.rowcount > 0

    def plays(self, user: str, limit: int = 300) -> list[dict[str, Any]]:
        with self._connection() as db:
            rows = db.execute(
                "SELECT * FROM plays WHERE user_key=? ORDER BY played_at DESC LIMIT ?",
                (user.strip().casefold(), limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def play_count(self, user: str) -> int:
        with self._connection() as db:
            row = db.execute("SELECT count(*) n FROM plays WHERE user_key=?", (user.strip().casefold(),)).fetchone()
        return row["n"]
