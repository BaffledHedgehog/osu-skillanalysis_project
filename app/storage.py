"""Small local SQLite store. Raw map charts remain on the user's machine."""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
import time
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .analyzer import ParsedMap, analyze_map


ANALYZER_VERSION = 6


class Store:
    def __init__(self, path: str | Path | None = None):
        self._batch = threading.local()
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
                    misses INTEGER NOT NULL, played_at TEXT NOT NULL, source TEXT NOT NULL,
                    pp REAL
                );
                CREATE INDEX IF NOT EXISTS plays_user_date ON plays(user_key, played_at DESC);
                CREATE INDEX IF NOT EXISTS maps_checksum ON maps(checksum);
                CREATE TABLE IF NOT EXISTS user_state (user_key TEXT PRIMARY KEY, revision INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS player_limits (user_key TEXT PRIMARY KEY, settings TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS local_replays (play_id TEXT PRIMARY KEY, path TEXT NOT NULL, sha256 TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS replay_health (cache_key TEXT PRIMARY KEY, result TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS cache_state (id INTEGER PRIMARY KEY, revision INTEGER NOT NULL DEFAULT 0);
                INSERT OR IGNORE INTO cache_state(id,revision) VALUES(1,0);
                CREATE TABLE IF NOT EXISTS import_cache (
                    cache_key TEXT PRIMARY KEY, result TEXT NOT NULL, saved_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS score_payloads (
                    user_key TEXT NOT NULL, provider TEXT NOT NULL, score_key TEXT NOT NULL,
                    map_id TEXT NOT NULL, payload TEXT NOT NULL, sources TEXT NOT NULL,
                    reason TEXT NOT NULL, saved_at REAL NOT NULL,
                    PRIMARY KEY(user_key,provider,score_key)
                );
                CREATE TABLE IF NOT EXISTS owned_maps (
                    user_key TEXT NOT NULL, source TEXT NOT NULL, map_id TEXT NOT NULL,
                    PRIMARY KEY(user_key,source,map_id)
                );
                CREATE TABLE IF NOT EXISTS public_profiles (
                    user_key TEXT PRIMARY KEY, payload TEXT NOT NULL, saved_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS local_files (
                    path TEXT PRIMARY KEY, mtime_ns INTEGER NOT NULL, size INTEGER NOT NULL, checksum TEXT NOT NULL
                );
                CREATE TRIGGER IF NOT EXISTS plays_revision_insert AFTER INSERT ON plays BEGIN
                    INSERT INTO user_state(user_key,revision) VALUES(NEW.user_key,1)
                    ON CONFLICT(user_key) DO UPDATE SET revision=revision+1;
                END;
                CREATE TRIGGER IF NOT EXISTS maps_revision_insert AFTER INSERT ON maps BEGIN
                    UPDATE cache_state SET revision=revision+1 WHERE id=1;
                END;
                CREATE TRIGGER IF NOT EXISTS maps_revision_update AFTER UPDATE ON maps BEGIN
                    UPDATE cache_state SET revision=revision+1 WHERE id=1;
                END;
                """
            )
            if "pp" not in {row[1] for row in db.execute("PRAGMA table_info(plays)")}:
                db.execute("ALTER TABLE plays ADD COLUMN pp REAL")
            if "passed_known" not in {row[1] for row in db.execute("PRAGMA table_info(plays)")}:
                db.execute("ALTER TABLE plays ADD COLUMN passed_known INTEGER NOT NULL DEFAULT 1")
            columns = {row[1] for row in db.execute('PRAGMA table_info(plays)')}
            if 'complete' not in columns:
                db.execute('ALTER TABLE plays ADD COLUMN complete INTEGER NOT NULL DEFAULT 1')
            if 'grade' not in columns:
                db.execute('ALTER TABLE plays ADD COLUMN grade TEXT')
            if db.execute("SELECT 1 FROM sqlite_master WHERE type='trigger' AND name='plays_revision_update'").fetchone():
                db.execute('DROP TRIGGER plays_revision_update')
            db.execute('''CREATE TRIGGER IF NOT EXISTS plays_revision_update_v2 AFTER UPDATE ON plays
                WHEN OLD.source != NEW.source OR OLD.accuracy != NEW.accuracy OR OLD.passed != NEW.passed
                  OR OLD.passed_known != NEW.passed_known OR OLD.mods != NEW.mods OR OLD.map_id != NEW.map_id
                  OR OLD.pp IS NOT NEW.pp OR OLD.complete != NEW.complete OR OLD.grade IS NOT NEW.grade
                BEGIN
                  INSERT INTO user_state(user_key,revision) VALUES(NEW.user_key,1)
                  ON CONFLICT(user_key) DO UPDATE SET revision=revision+1;
                END''')

    @contextmanager
    def _connection(self):
        active = getattr(self._batch, 'connection', None)
        if active is not None:
            yield active
            return
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA journal_mode=WAL")
            with db:
                yield db
        finally:
            db.close()

    @contextmanager
    def batch(self):
        if getattr(self._batch, 'connection', None) is not None:
            yield
            return
        with self._connection() as db:
            self._batch.connection = db
            try:
                yield
            finally:
                del self._batch.connection

    def save_map(self, parsed: ParsedMap) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        with self._connection() as db:
            existing = db.execute("SELECT checksum FROM maps WHERE id=?", (parsed.map_id,)).fetchone()
            if existing and existing["checksum"] == parsed.checksum:
                return {"id": parsed.map_id, **parsed.metadata}
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
        pp: float | None = None, passed_known: bool = True, complete: bool = True, grade: str | None = None,
    ) -> bool:
        if not self.get_map(map_id):
            raise ValueError("Сначала добавьте эту карту в каталог.")
        played_at = played_at or datetime.now(timezone.utc).isoformat()
        play_id = play_id or str(uuid.uuid4())
        with self._connection() as db:
            cursor = db.execute(
                "INSERT OR IGNORE INTO plays(id,user_key,map_id,mods,accuracy,passed,misses,played_at,source,pp,passed_known,complete,grade) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (play_id, user.strip().casefold(), map_id, mods, accuracy, int(passed), misses, played_at, source, pp, int(passed_known), int(complete), grade),
            )
            if cursor.rowcount == 0:
                if source == 'local_lazer':
                    # A mirror may expose legacy/head-only accuracy for this
                    # same score ID. Realm holds the accuracy shown by lazer,
                    # including its slider judgements. Keep one attempt while
                    # preserving that local scoring value and the raw payloads.
                    db.execute('''UPDATE plays SET passed=?,passed_known=?,complete=?,grade=?,mods=?,
                                  accuracy=?,misses=?,played_at=?,source='local_lazer'
                                  WHERE id=? AND user_key=? AND map_id=?
                                  AND (passed_known=0 OR ?=1)''',
                               (int(passed),int(passed_known),int(complete),grade,mods,accuracy,misses,played_at,
                                play_id,user.strip().casefold(),map_id,int(passed_known)))
                if source.startswith('local_'):
                    db.execute('''UPDATE plays SET passed=?,passed_known=?,complete=?,grade=?,mods=?,accuracy=?,misses=?
                                  WHERE id=? AND user_key=? AND map_id=? AND source LIKE 'local_%'
                                  AND (passed_known=0 OR ?=1)''',
                               (int(passed), int(passed_known), int(complete), grade, mods, accuracy, misses,
                                play_id, user.strip().casefold(), map_id, int(passed_known)))
                if source.startswith(('osu_api_', 'hinai_', 'osu_web_')):
                    db.execute("UPDATE plays SET source=?,passed=?,passed_known=1 WHERE id=? "
                               "AND (source='osu_api' OR passed_known=0 OR "
                               "((source LIKE '%_best' OR source LIKE '%_firsts' OR source LIKE '%_pinned') AND ? LIKE '%_recent'))",
                               (source, int(passed), play_id, source))
                if pp is not None:
                    db.execute("UPDATE plays SET pp=COALESCE(pp, ?) WHERE id=?", (pp, play_id))
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

    def save_score_payload(self, user, provider, source, score_key, map_id, payload, reason='pending'):
        """Retain score fields and list membership, including unusable training rows."""
        key = user.strip().casefold()
        with self._connection() as db:
            old = db.execute('SELECT sources FROM score_payloads WHERE user_key=? AND provider=? AND score_key=?',
                             (key, provider, str(score_key))).fetchone()
            sources = set(json.loads(old['sources'])) if old else set()
            sources.add(source)
            db.execute('''INSERT INTO score_payloads VALUES(?,?,?,?,?,?,?,?)
                          ON CONFLICT(user_key,provider,score_key) DO UPDATE SET
                          map_id=excluded.map_id,payload=excluded.payload,sources=excluded.sources,
                          reason=excluded.reason,saved_at=excluded.saved_at''',
                       (key, provider, str(score_key), str(map_id), json.dumps(payload, ensure_ascii=False, allow_nan=False),
                        json.dumps(sorted(sources)), reason, time.time()))

    def save_public_profile(self, user, profile):
        allowed = ('id', 'username', 'previous_usernames', 'country_code', 'statistics',
                   'rank_history', 'monthly_playcounts', 'scores_first_count', 'scores_best_count',
                   'scores_recent_count', 'playmode', 'join_date')
        payload = {k: profile[k] for k in allowed if k in profile}
        with self._connection() as db:
            db.execute('INSERT OR REPLACE INTO public_profiles VALUES(?,?,?)',
                       (user.strip().casefold(), json.dumps(payload, ensure_ascii=False), time.time()))

    def score_payloads(self, user):
        with self._connection() as db:
            rows = db.execute('SELECT * FROM score_payloads WHERE user_key=?', (user.strip().casefold(),)).fetchall()
        return [{**dict(row), 'payload': json.loads(row['payload']), 'sources': json.loads(row['sources'])} for row in rows]

    def replace_owned_maps(self, user, source, map_ids):
        with self._connection() as db:
            db.execute('DELETE FROM owned_maps WHERE user_key=? AND source=?', (user.strip().casefold(), source))
            db.executemany('INSERT OR IGNORE INTO owned_maps VALUES(?,?,?)',
                           ((user.strip().casefold(), source, str(m)) for m in set(map_ids)))

    def excluded_map_ids(self, user):
        key = user.strip().casefold()
        with self._connection() as db:
            rows = db.execute('''SELECT map_id FROM plays WHERE user_key=?
                                 UNION SELECT map_id FROM score_payloads WHERE user_key=?
                                 UNION SELECT map_id FROM owned_maps WHERE user_key=?
                                 UNION SELECT m.id FROM maps m JOIN score_payloads s
                                 ON m.checksum=json_extract(s.payload,'$.checksum')
                                 WHERE s.user_key=?''', (key, key, key, key)).fetchall()
        return {row['map_id'] for row in rows if row['map_id']}

    def save_local_replay(self, play_id: str, path: Path, digest: str):
        with self._connection() as db:
            db.execute('INSERT INTO local_replays VALUES(?,?,?) ON CONFLICT(play_id) DO UPDATE SET path=excluded.path,sha256=excluded.sha256',
                       (play_id, str(path.resolve()), digest))

    def no_fail_replays(self, user: str):
        with self._connection() as db:
            rows = db.execute('''SELECT p.id,p.map_id,p.mods,p.accuracy,p.played_at,m.metadata
                                 FROM plays p JOIN local_replays r ON p.id=r.play_id JOIN maps m ON p.map_id=m.id
                                 WHERE p.user_key=? AND p.mods LIKE '%NF%' ORDER BY p.played_at DESC''',
                              (user.strip().casefold(),)).fetchall()
        return [{**dict(r), 'metadata': json.loads(r['metadata'])} for r in rows]

    def replay_input(self, user: str, play_id: str):
        with self._connection() as db:
            row = db.execute('''SELECT p.*,r.path,r.sha256 FROM plays p JOIN local_replays r ON p.id=r.play_id
                                WHERE p.id=? AND p.user_key=?''', (play_id, user.strip().casefold())).fetchone()
        return dict(row) if row else None

    def health_cache(self, key: str):
        with self._connection() as db:
            row = db.execute('SELECT result FROM replay_health WHERE cache_key=?', (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def save_health_cache(self, key: str, result: dict):
        with self._connection() as db:
            db.execute('INSERT OR REPLACE INTO replay_health VALUES(?,?)', (key, json.dumps(result)))

    def player_limits(self, user: str) -> dict:
        with self._connection() as db:
            row = db.execute('SELECT settings FROM player_limits WHERE user_key=?', (user.strip().casefold(),)).fetchone()
        return json.loads(row[0]) if row else {}

    def save_player_limits(self, user: str, limits: dict) -> dict:
        with self._connection() as db:
            db.execute('INSERT INTO player_limits VALUES(?,?) ON CONFLICT(user_key) DO UPDATE SET settings=excluded.settings',
                       (user.strip().casefold(), json.dumps(limits)))
        return limits

    def cache_token(self, user: str) -> tuple[int, int]:
        with self._connection() as db:
            row = db.execute(
                "SELECT COALESCE((SELECT revision FROM user_state WHERE user_key=?),0), "
                "(SELECT revision FROM cache_state WHERE id=1)", (user.strip().casefold(),)
            ).fetchone()
        return tuple(row)

    def plays_with_analysis(self, user: str) -> list[dict[str, Any]]:
        with self._connection() as db:
            rows = db.execute(
                "SELECT p.*, a.result AS cached_analysis FROM plays p LEFT JOIN analyses a "
                "ON a.map_id=p.map_id AND a.mods=p.mods AND a.version=? "
                "WHERE p.user_key=? ORDER BY p.played_at DESC,p.id",
                (ANALYZER_VERSION, user.strip().casefold()),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            cached = item.pop("cached_analysis")
            item["analysis"] = json.loads(cached) if cached else self.analysis(item["map_id"], item["mods"])
            result.append(item)
        return result

    def cached_import(self, key: str, ttl: float | None = 300) -> dict[str, Any] | None:
        with self._connection() as db:
            row = db.execute("SELECT result,saved_at FROM import_cache WHERE cache_key=?", (key,)).fetchone()
        if not row or (ttl is not None and time.time() - row["saved_at"] > ttl):
            return None
        return {**json.loads(row["result"]), "cached_at": row["saved_at"]}

    def cache_import(self, key: str, result: dict[str, Any]) -> None:
        with self._connection() as db:
            db.execute("INSERT INTO import_cache VALUES(?,?,?) ON CONFLICT(cache_key) DO UPDATE "
                       "SET result=excluded.result,saved_at=excluded.saved_at",
                       (key, json.dumps(result, ensure_ascii=False), time.time()))

    def cached_file_hash(self, path: Path, mtime_ns: int, size: int) -> str | None:
        with self._connection() as db:
            row = db.execute("SELECT checksum FROM local_files WHERE path=? AND mtime_ns=? AND size=?",
                             (str(path), mtime_ns, size)).fetchone()
        return row[0] if row else None

    def cache_file_hash(self, path: Path, mtime_ns: int, size: int, checksum: str) -> None:
        with self._connection() as db:
            db.execute("INSERT INTO local_files VALUES(?,?,?,?) ON CONFLICT(path) DO UPDATE SET "
                       "mtime_ns=excluded.mtime_ns,size=excluded.size,checksum=excluded.checksum",
                       (str(path), mtime_ns, size, checksum))
