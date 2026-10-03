from pathlib import Path
from datetime import datetime, timezone
import sqlite3

from fastapi.testclient import TestClient
from osrparse import GameMode, Key, Mod, Replay, ReplayEventOsu

from app.analyzer import normalize_mods, parse_map
from app.main import create_app
from app.ml import predict_pass, train
from app.storage import Store
from app import osu_api


def chart(map_id: int, spacing: int = 190) -> bytes:
    objects = "\n".join(
        f"{64 + (index % 4) * 120},{64 + (index % 3) * 100},{1000 + index * spacing},1,0,0:0:0:0:"
        for index in range(24)
    )
    return f"""osu file format v14

[General]
AudioFilename: track.mp3
Mode: 0

[Metadata]
Title: Test Song
Artist: Test Artist
Creator: Tester
Version: Test {map_id}
BeatmapID: {map_id}

[Difficulty]
HPDrainRate: 5
CircleSize: 4
OverallDifficulty: 7
ApproachRate: 9
SliderMultiplier: 1.4
SliderTickRate: 1

[TimingPoints]
0,500,4,1,0,100,1,0

[HitObjects]
{objects}
""".encode()


def test_chart_analysis_and_mods(tmp_path: Path):
    client = TestClient(create_app(tmp_path / "test.db"))
    uploaded = client.post("/api/maps/upload", files={"file": ("test.osu", chart(101), "text/plain")})
    assert uploaded.status_code == 200
    assert uploaded.json()["maps"][0]["id"] == "101"
    normal = client.get("/api/maps/101/analysis").json()
    faster = client.get("/api/maps/101/analysis", params={"mods": "DT"}).json()
    assert normal["object_count"] == 24
    assert normal["stars"] > 0
    assert len(normal["windows"]) > 0
    assert faster["bpm"] > normal["bpm"]
    assert faster["duration"] < normal["duration"]
    assert normalize_mods("HDDT")[0] == "HDDT"
    assert client.get("/api/maps/101/analysis", params={"mods": "HRHTEZ"}).status_code == 400


def test_personal_result_feedback_and_recommendation(tmp_path: Path):
    client = TestClient(create_app(tmp_path / "test.db"))
    for map_id, spacing in [(101, 190), (102, 130)]:
        response = client.post("/api/maps/upload", files={"file": ("test.osu", chart(map_id, spacing), "text/plain")})
        assert response.status_code == 200
    no_history = client.get("/api/maps/102/analysis", params={"user": "Player"}).json()["prediction"]
    assert no_history["pass_probability"] is None
    for _ in range(3):
        response = client.post("/api/plays", json={"user": "Player", "map_id": "101", "accuracy": 96, "passed": True})
        assert response.status_code == 200
    before = client.get("/api/maps/102/analysis", params={"user": "Player"}).json()["prediction"]
    for _ in range(4):
        client.post("/api/plays", json={"user": "Player", "map_id": "101", "accuracy": 73, "passed": False, "misses": 12})
    after = client.get("/api/maps/102/analysis", params={"user": "Player"}).json()["prediction"]
    assert before["pass_probability"] is not None
    assert after["pass_probability"] < before["pass_probability"]
    suggestions = client.get("/api/recommendations", params={"user": "Player"}).json()["maps"]
    assert [item["id"] for item in suggestions] == ["102"]
    assert client.get("/api/users/Player/summary").json()["plays"] == 7


def test_other_mode_rejected():
    raw = chart(103).replace(b"Mode: 0", b"Mode: 3")
    try:
        parse_map(raw)
    except ValueError as exc:
        assert "osu!standard" in str(exc)
    else:
        raise AssertionError("Mania chart should be rejected")


def test_replay_import_matches_exact_chart_and_deduplicates(tmp_path: Path):
    client = TestClient(create_app(tmp_path / "test.db"))
    raw = chart(201)
    client.post("/api/maps/upload", files={"file": ("map.osu", raw, "text/plain")})
    checksum = parse_map(raw).checksum
    replay = Replay(
        GameMode.STD, 20210101, checksum, "Player", "unique-replay-hash",
        90, 7, 1, 0, 0, 2, 100000, 95, False, Mod.Hidden, None,
        datetime.now(timezone.utc), [ReplayEventOsu(16, 256, 192, Key(0))], 123, None,
    )
    files = {"file": ("play.osr", replay.pack(), "application/octet-stream")}
    response = client.post("/api/plays/replay", data={"user": "Player", "passed": "false"}, files=files)
    assert response.status_code == 200
    assert response.json()["created"] is True
    assert response.json()["mods"] == "HD"
    assert response.json()["passed"] is False
    assert response.json()["accuracy"] < 100
    again = client.post("/api/plays/replay", data={"user": "Player", "passed": "false"}, files=files)
    assert again.status_code == 200
    assert again.json()["created"] is False
    assert client.get("/api/users/Player/summary").json()["plays"] == 1


def test_personal_ml_trains_on_real_stored_outcomes_and_unseen_maps(tmp_path: Path):
    store = Store(tmp_path / "ml.db")
    # Distinct charts let the validation split hold out complete maps.
    for index in range(12):
        map_id = str(1000 + index)
        store.save_map(parse_map(chart(1000 + index, spacing=275 - index * 17)))
        for attempt in range(5):
            passed = index < 6
            store.save_play("Player", map_id, "NM", 0.95 if passed else 0.72,
                            passed, play_id=f"{map_id}-{attempt}")
    result = train(store, "Player")
    assert result["status"] == "trained"
    assert result["metrics"]["brier"] < result["metrics"]["baseline_brier"]
    easy = store.analysis("1000")
    hard = store.analysis("1011")
    assert easy is not None and hard is not None
    easy_p, _ = predict_pass(store, "Player", easy)
    hard_p, _ = predict_pass(store, "Player", hard)
    assert easy_p is not None and hard_p is not None and easy_p > hard_p


def test_profile_link_imports_100_score_windows_without_40_chart_cap(tmp_path: Path, monkeypatch):
    class Response:
        def __init__(self, data):
            self.data = data
            self.status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return self.data

    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get(self, url, headers=None, params=None):
            if "/api/v2/users/12345/osu" in url:
                return Response({"id": 12345, "username": "ActualPlayer"})
            assert params["limit"] == 100
            if url.endswith("/recent"):
                assert params["include_fails"] == 1
                return Response([score(i, i == 1) for i in range(1, 46)])
            if url.endswith("/best"):
                return Response([score(i + 1000, True, map_id=i) for i in range(25, 56)])
            raise AssertionError(url)

    def score(score_id, passed, map_id=None):
        return {"id": score_id, "beatmap": {"id": map_id or score_id}, "mods": [],
                "accuracy": 0.91, "passed": passed, "ended_at": "2026-10-01T12:00:00Z",
                "statistics": {"miss": 1}}

    async def token(client):
        return "test-token"

    async def beatmap(map_id):
        return chart(int(map_id))

    monkeypatch.setattr(osu_api, "api_configured", lambda: True)
    monkeypatch.setattr(osu_api, "_token", token)
    monkeypatch.setattr(osu_api, "download_beatmap", beatmap)
    monkeypatch.setattr(osu_api.httpx, "AsyncClient", Client)
    client = TestClient(create_app(tmp_path / "profile.db"))
    body = {"username": "https://osu.ppy.sh/users/12345/osu?from=profile"}
    first = client.post("/api/users/import", json=body)
    assert first.status_code == 200, first.text
    data = first.json()
    assert data["username"] == "ActualPlayer"
    assert data["highlight_map_id"] == "1"
    assert data["unique_maps"] == 55
    assert data["recent_unique"] == 45 and data["best_unique"] == 10
    assert data["duplicates"] == 21
    assert data["imported"] == 55 and data["maps_fetched"] == 55
    plays = client.app.state.store.plays("ActualPlayer")
    assert len(plays) == 55
    repeated_map = next(play for play in plays if play["map_id"] == "25")
    assert repeated_map["passed"] == 0 and repeated_map["source"] == "osu_api_recent"
    assert repeated_map["played_at"] == "2026-10-01T12:00:00Z"
    assert next(play for play in plays if play["map_id"] == "55")["source"] == "osu_api_best"
    model = client.get("/api/users/ActualPlayer/model").json()
    assert model["status"] == "insufficient_data" and model["plays"] == 45
    again = client.post("/api/users/import", json=body).json()
    assert again["imported"] == 0 and again["maps_fetched"] == 0


def test_profile_import_pages_until_100_unique_maps_per_list(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(osu_api, "SCORES_PER_SOURCE", 4)
    requests = []

    def score(score_id, map_id):
        return {"id": score_id, "beatmap_id": map_id, "mods": [],
                "accuracy": 0.9, "passed": True}

    pages = {
        ("recent", 0): [score(i, 1) for i in range(1, 5)],
        ("recent", 4): [score(i, i - 3) for i in range(5, 9)],
        ("best", 0): [score(10, 1), score(11, 2), score(12, 6), score(13, 7)],
        ("best", 4): [score(14, 8), score(15, 9), score(16, 10), score(17, 11)],
    }

    class Response:
        status_code = 200

        def __init__(self, value):
            self.value = value

        def raise_for_status(self):
            pass

        def json(self):
            return self.value

    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get(self, url, headers=None, params=None):
            if url.endswith("/osu"):
                return Response({"id": 123, "username": "Pager"})
            source = url.rsplit("/", 1)[-1]
            requests.append((source, params["offset"]))
            return Response(pages.get((source, params["offset"]), []))

    async def token(client):
        return "test-token"

    async def beatmap(map_id):
        return chart(int(map_id))

    monkeypatch.setattr(osu_api, "api_configured", lambda: True)
    monkeypatch.setattr(osu_api, "_token", token)
    monkeypatch.setattr(osu_api, "download_beatmap", beatmap)
    monkeypatch.setattr(osu_api.httpx, "AsyncClient", Client)
    client = TestClient(create_app(tmp_path / "pages.db"))
    result = client.post("/api/users/import", json={"username": "https://osu.ppy.sh/users/123"})
    assert result.status_code == 200, result.text
    body = result.json()
    assert body["recent_unique"] == 4 and body["best_unique"] == 4
    assert body["unique_maps"] == 8 and body["imported"] == 8
    assert requests == [("recent", 0), ("recent", 4), ("best", 0), ("best", 4)]


def test_profile_reference_rejects_other_hosts():
    assert osu_api.parse_player_reference("osu.ppy.sh/users/12345") == "12345"
    assert osu_api.parse_player_reference("Player_Name") == "Player_Name"
    try:
        osu_api.parse_player_reference("https://evil.example/users/12345")
    except ValueError:
        pass
    else:
        raise AssertionError("A non-osu! URL must not be accepted")


def test_public_profile_import_works_without_oauth(tmp_path: Path, monkeypatch):
    class Response:
        status_code = 200

        def __init__(self, value):
            self.value = value

        def raise_for_status(self):
            pass

        def json(self):
            return self.value

    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get(self, url, headers=None, params=None):
            assert "Authorization" not in headers
            if url.endswith("/player/123"):
                return Response({"id": 123, "username": "PublicPlayer"})
            if params["type"] == "recent":
                return Response({"scores": [{"id": 10, "beatmap_id": 101,
                    "mods": [], "accuracy": 93.5, "passed": False,
                    "statistics": {"count_miss": 4}}]})
            return Response({"scores": [{"id": 11, "beatmap_id": 102,
                "mods": ["HD"], "accuracy": 0.98, "passed": True, "pp": 225.3}]})

    async def beatmap(map_id):
        return chart(int(map_id))

    monkeypatch.setattr(osu_api, "api_configured", lambda: False)
    monkeypatch.setattr(osu_api, "download_beatmap", beatmap)
    monkeypatch.setattr(osu_api.httpx, "AsyncClient", Client)
    client = TestClient(create_app(tmp_path / "public.db"))
    response = client.post("/api/users/import", json={"username": "https://osu.ppy.sh/users/123"})
    assert response.status_code == 200, response.text
    assert response.json()["provider"] == "hinai"
    assert response.json()["imported"] == 2
    plays = client.app.state.store.plays("PublicPlayer")
    assert {play["source"] for play in plays} == {"hinai_recent", "hinai_best"}
    recent = next(play for play in plays if play["source"] == "hinai_recent")
    assert recent["accuracy"] == 0.935 and recent["misses"] == 4 and recent["passed"] == 0
    assert client.get("/api/users/PublicPlayer/summary").json()["best_pp"] == 225.3


def test_existing_database_adds_pp_column(tmp_path: Path):
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as db:
        db.execute("""CREATE TABLE plays (
            id TEXT PRIMARY KEY, user_key TEXT NOT NULL, map_id TEXT NOT NULL,
            mods TEXT NOT NULL, accuracy REAL NOT NULL, passed INTEGER NOT NULL,
            misses INTEGER NOT NULL, played_at TEXT NOT NULL, source TEXT NOT NULL
        )""")
    Store(path)
    with sqlite3.connect(path) as db:
        columns = {row[1] for row in db.execute("PRAGMA table_info(plays)")}
    assert "pp" in columns
