from pathlib import Path
from datetime import datetime, timezone

from fastapi.testclient import TestClient
from osrparse import GameMode, Key, Mod, Replay, ReplayEventOsu

from app.analyzer import normalize_mods, parse_map
from app.main import create_app
from app.ml import predict_pass, train
from app.storage import Store


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
