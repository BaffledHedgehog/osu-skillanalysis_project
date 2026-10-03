"""Train a local pass model from a player's observed attempts.

The model is deliberately serialized as plain JSON, never as an executable pickle.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from .storage import Store


FEATURES = ("stars", "aim", "speed", "ar", "od", "duration_log_minutes")
MIN_PLAYS = 40
MIN_CLASS = 8
MIN_MAPS = 10


def vector(analysis: dict[str, Any]) -> list[float]:
    return [
        float(analysis["stars"]), float(analysis["aim"]), float(analysis["speed"]),
        float(analysis["ar"]), float(analysis["od"]),
        math.log1p(float(analysis["duration"]) / 60),
    ]


def _dataset(store: Store, user: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    rows = []
    identity = []
    for play in store.plays(user, limit=100_000):
        analysis = store.analysis(play["map_id"], play["mods"])
        if analysis is None:
            continue
        features = vector(analysis)
        rows.append((features, int(play["passed"]), play["map_id"]))
        identity.append((play["id"], play["map_id"], play["mods"], int(play["passed"]), features))
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if not rows:
        return np.empty((0, len(FEATURES))), np.array([], dtype=int), np.array([], dtype=str), fingerprint
    return (np.asarray([row[0] for row in rows], dtype=float),
            np.asarray([row[1] for row in rows], dtype=int),
            np.asarray([row[2] for row in rows], dtype=str), fingerprint)


def _artifact_path(store: Store, user: str) -> Path:
    key = hashlib.sha256(f"{store.path.resolve()}:{user.strip().casefold()}".encode()).hexdigest()[:24]
    return store.path.parent / "models" / f"{key}.json"


def _read_model(path: Path, fingerprint: str) -> dict[str, Any] | None:
    try:
        model = json.loads(path.read_text(encoding="utf-8"))
        if model.get("schema") == 1 and model.get("fingerprint") == fingerprint and model.get("features") == list(FEATURES):
            return model
    except (OSError, ValueError, TypeError):
        pass
    return None


def _fit(x: np.ndarray, y: np.ndarray) -> tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler().fit(x)
    classifier = LogisticRegression(C=0.35, max_iter=1000).fit(scaler.transform(x), y)
    return scaler, classifier


def _probabilities(model: tuple[StandardScaler, LogisticRegression], x: np.ndarray) -> np.ndarray:
    scaler, classifier = model
    return classifier.predict_proba(scaler.transform(x))[:, 1]


def train(store: Store, user: str) -> dict[str, Any]:
    x, y, groups, fingerprint = _dataset(store, user)
    counts = np.bincount(y, minlength=2)
    unique_maps = len(set(groups))
    status = {"status": "insufficient_data", "plays": len(y), "passes": int(counts[1]),
              "fails": int(counts[0]), "maps": unique_maps, "required_plays": MIN_PLAYS,
              "required_per_class": MIN_CLASS, "required_maps": MIN_MAPS}
    if len(y) < MIN_PLAYS or min(counts) < MIN_CLASS or unique_maps < MIN_MAPS:
        return status

    # No chart appears in both fitting and validation. Try deterministic splits until
    # both outcomes appear on each side; otherwise the dataset cannot be evaluated.
    split = None
    for seed in range(50):
        candidate = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed).split(x, y, groups))
        train_ids, test_ids = candidate
        if len(set(y[train_ids])) == 2 and len(set(y[test_ids])) == 2 and len(test_ids) >= 8:
            split = candidate
            break
    if split is None:
        return {**status, "status": "insufficient_validation", "message": "Нужны успехи и неудачи на разных картах."}

    train_ids, test_ids = split
    model = _fit(x[train_ids], y[train_ids])
    predicted = _probabilities(model, x[test_ids])
    prevalence = float(y[train_ids].mean())
    baseline = np.full(len(test_ids), prevalence)
    metrics = {
        "holdout_plays": len(test_ids), "holdout_maps": len(set(groups[test_ids])),
        "brier": round(float(brier_score_loss(y[test_ids], predicted)), 4),
        "baseline_brier": round(float(brier_score_loss(y[test_ids], baseline)), 4),
        "log_loss": round(float(log_loss(y[test_ids], predicted, labels=[0, 1])), 4),
        "baseline_log_loss": round(float(log_loss(y[test_ids], baseline, labels=[0, 1])), 4),
    }
    # A prevalence-only estimate is more honest when the trained model fails on
    # unseen maps. Retain the metrics so the reason is inspectable.
    if metrics["brier"] >= metrics["baseline_brier"]:
        return {**status, "status": "rejected", "metrics": metrics,
                "message": "Модель не улучшила простой базовый прогноз на отложенных картах."}

    scaler, classifier = _fit(x, y)
    artifact = {
        "schema": 1, "features": list(FEATURES), "fingerprint": fingerprint,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "plays": len(y), "maps": unique_maps, "metrics": metrics,
        "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
        "coef": classifier.coef_[0].tolist(), "intercept": float(classifier.intercept_[0]),
    }
    path = _artifact_path(store, user)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": "trained", "plays": len(y), "passes": int(counts[1]),
            "fails": int(counts[0]), "maps": unique_maps, "metrics": metrics,
            "trained_at": artifact["trained_at"]}


def _score(model: dict[str, Any], analysis: dict[str, Any]) -> float:
    values = vector(analysis)
    linear = model["intercept"] + sum(
        coefficient * (value - mean) / scale
        for value, mean, scale, coefficient in zip(values, model["mean"], model["scale"], model["coef"])
    )
    return 1 / (1 + math.exp(-max(-25, min(25, linear))))


@lru_cache(maxsize=64)
def _ensure_cached(database: str, user: str, fingerprint: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    store = Store(database)
    path = _artifact_path(store, user)
    model = _read_model(path, fingerprint)
    if model:
        return model, {"status": "trained", "plays": model["plays"], "maps": model["maps"],
                       "metrics": model["metrics"], "trained_at": model["trained_at"]}
    result = train(store, user)
    return (_read_model(path, fingerprint) if result["status"] == "trained" else None), result


def model_status(store: Store, user: str) -> dict[str, Any]:
    _, _, _, fingerprint = _dataset(store, user)
    return _ensure_cached(str(store.path.resolve()), user.strip().casefold(), fingerprint)[1]


def predict_pass(store: Store, user: str, analysis: dict[str, Any]) -> tuple[float | None, dict[str, Any]]:
    _, _, _, fingerprint = _dataset(store, user)
    model, status = _ensure_cached(str(store.path.resolve()), user.strip().casefold(), fingerprint)
    return (_score(model, analysis) if model else None), status


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a local osu!standard pass model from stored attempts")
    parser.add_argument("user", help="Player name used when importing attempts")
    parser.add_argument("--database", default=None, help="SQLite path (defaults to DATABASE_PATH)")
    args = parser.parse_args()
    print(json.dumps(train(Store(args.database), args.user), ensure_ascii=False, indent=2))
