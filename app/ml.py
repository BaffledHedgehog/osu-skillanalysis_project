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
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from .storage import Store
from .recency import POLICY_VERSION, HALF_LIFE_DAYS, as_of_day, sample_weights


FEATURES = ("stars", "aim", "speed", "ar", "od", "duration_log_minutes", "cs",
            "hidden", "hard_rock", "double_time", "half_time", "easy", "flashlight", "classic", "hidden_low_ar",
            "stream_bpm", "burst_bpm", "slider_fraction", "rhythm_variation",
            "ar_squared", "hidden_ar", "hidden_ar_squared", "stream_bpm_squared", "hidden_rhythm",
            "peak_aim", "peak_speed", "peak_load", "peak_to_mean", "ending_peak_load", "ending_peak_fraction", "no_fail")
SIMILARITY_SCALES = (1.3, 1.2, 1.2, 1.8, 2.0, 0.9, 1.5, 1, 1, 1, 1, 1, 1, 1, 3,
                     60, 75, .3, 1, 35, 10, 100, 30000, 1, 100, 100, 200, 3, 200, 1, 1)
MOD_FEATURES = ('hidden', 'hard_rock', 'double_time', 'half_time', 'easy', 'flashlight', 'classic', 'no_fail')
MIN_PLAYS = 40
MIN_CLASS = 8
MIN_MAPS = 10
BEST_SOURCES = {"osu_api_best", "hinai_best", "osu_web_best", "osu_api"}


def vector(analysis: dict[str, Any]) -> list[float]:
    mods = analysis.get('mods', 'NM')
    hd = float('HD' in mods)
    return [
        float(analysis["stars"]), float(analysis["aim"]), float(analysis["speed"]),
        float(analysis["ar"]), float(analysis["od"]),
        math.log1p(float(analysis["duration"]) / 60),
        float(analysis.get('cs', 4)), hd, float('HR' in mods), float('DT' in mods or 'NC' in mods),
        float('HT' in mods), float('EZ' in mods), float('FL' in mods), float('CL' in mods),
        hd * max(0, 10 - float(analysis['ar'])),
        float(analysis.get('stream_bpm', 0)), float(analysis.get('burst_bpm', 0)),
        float(analysis.get('slider_fraction', 0)), float(analysis.get('rhythm_variation', 0)),
        float(analysis['ar']) ** 2, hd * float(analysis['ar']), hd * float(analysis['ar']) ** 2,
        float(analysis.get('stream_bpm', 0)) ** 2, hd * float(analysis.get('rhythm_variation', 0)),
        *(float(analysis.get(key, 0)) for key in ('peak_aim', 'peak_speed', 'peak_load', 'peak_to_mean', 'ending_peak_load', 'ending_peak_fraction')),
        float('NF' in mods),
    ]


def _dataset(store: Store, user: str, return_weights=False):
    rows = []
    identity = []
    included = []
    for play in store.plays_with_analysis(user):
        if play["source"] in BEST_SOURCES or not play.get('passed_known', 1) or 'NF' in play['mods']:
            # Best scores were selected for success and cannot estimate pass rate.
            # Older mixed API rows have no recoverable source distinction.
            continue
        analysis = play["analysis"]
        if analysis is None:
            continue
        features = vector(analysis)
        rows.append((features, int(play["passed"]), play["map_id"]))
        included.append(play)
        identity.append((play["id"], play["map_id"], play["mods"], int(play["passed"]), features))
    weights = np.asarray(sample_weights(included), dtype=float)
    fingerprint = hashlib.sha256(json.dumps([POLICY_VERSION, as_of_day().isoformat(), identity, weights.tolist()], sort_keys=True).encode()).hexdigest()
    if not rows:
        result = np.empty((0, len(FEATURES))), np.array([], dtype=int), np.array([], dtype=str), fingerprint
        return (*result, weights) if return_weights else result
    result = (np.asarray([row[0] for row in rows], dtype=float),
            np.asarray([row[1] for row in rows], dtype=int),
            np.asarray([row[2] for row in rows], dtype=str), fingerprint)
    return (*result, weights) if return_weights else result


def _artifact_path(store: Store, user: str, task: str = "pass") -> Path:
    key = hashlib.sha256(f"{store.path.resolve()}:{user.strip().casefold()}".encode()).hexdigest()[:24]
    suffix = "-accuracy" if task == "accuracy" else ""
    return store.path.parent / "models" / f"{key}{suffix}.json"


def _read_model(path: Path, fingerprint: str) -> dict[str, Any] | None:
    try:
        model = json.loads(path.read_text(encoding="utf-8"))
        if model.get("schema") == 1 and model.get("fingerprint") == fingerprint and model.get("features") == list(FEATURES):
            return model
    except (OSError, ValueError, TypeError):
        pass
    return None


def _fit(x: np.ndarray, y: np.ndarray, weights=None) -> tuple[StandardScaler, LogisticRegression]:
    fit_weights = _relative_fit_weights(weights)
    scaler = StandardScaler().fit(x, sample_weight=fit_weights)
    classifier = LogisticRegression(C=0.35, max_iter=1000).fit(scaler.transform(x), y, sample_weight=fit_weights)
    return scaler, classifier


def _relative_fit_weights(weights):
    if weights is None or np.ptp(weights) < 1e-12:
        return None
    effective_n = weights.sum() ** 2 / np.square(weights).sum()
    return weights * effective_n / weights.sum()


def _probabilities(model: tuple[StandardScaler, LogisticRegression], x: np.ndarray) -> np.ndarray:
    scaler, classifier = model
    return classifier.predict_proba(scaler.transform(x))[:, 1]


def train(store: Store, user: str) -> dict[str, Any]:
    x, y, groups, fingerprint, weights = _dataset(store, user, return_weights=True)
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
    model = _fit(x[train_ids], y[train_ids], weights[train_ids])
    predicted = _probabilities(model, x[test_ids])
    prevalence = float(np.average(y[train_ids], weights=weights[train_ids]))
    baseline = np.full(len(test_ids), prevalence)
    metrics = {
        "holdout_plays": len(test_ids), "holdout_maps": len(set(groups[test_ids])),
        "brier": round(float(brier_score_loss(y[test_ids], predicted, sample_weight=weights[test_ids])), 4),
        "baseline_brier": round(float(brier_score_loss(y[test_ids], baseline, sample_weight=weights[test_ids])), 4),
        "log_loss": round(float(log_loss(y[test_ids], predicted, labels=[0, 1], sample_weight=weights[test_ids])), 4),
        "baseline_log_loss": round(float(log_loss(y[test_ids], baseline, labels=[0, 1], sample_weight=weights[test_ids])), 4),
        "recency_half_life_days": HALF_LIFE_DAYS, "weighted_plays": round(float(weights.sum()), 2),
    }
    # A prevalence-only estimate is more honest when the trained model fails on
    # unseen maps. Retain the metrics so the reason is inspectable.
    if metrics["brier"] >= metrics["baseline_brier"]:
        return {**status, "status": "rejected", "metrics": metrics,
                "message": "Модель не улучшила простой базовый прогноз на отложенных картах."}

    scaler, classifier = _fit(x, y, weights)
    artifact = {
        "schema": 1, "features": list(FEATURES), "fingerprint": fingerprint,
        "mod_bounds": [[float(x[:, FEATURES.index(key)].min()), float(x[:, FEATURES.index(key)].max())] for key in MOD_FEATURES],
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
    return {**model_context(store, user)[1], "accuracy": accuracy_model_context(store, user)[1]}


def model_context(store: Store, user: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    return _pass_context_by_revision(str(store.path.resolve()), user.strip().casefold(), store.cache_token(user), as_of_day().isoformat())


@lru_cache(maxsize=64)
def _pass_context_by_revision(database: str, user: str, revision, day):
    store = Store(database)
    _, _, _, fingerprint = _dataset(store, user)
    return _ensure_cached(database, user, fingerprint)


def predict_pass(store: Store, user: str, analysis: dict[str, Any],
                 context: tuple[dict[str, Any] | None, dict[str, Any]] | None = None
                 ) -> tuple[float | None, dict[str, Any]]:
    model, status = context if context is not None else model_context(store, user)
    if model and not _mods_supported(model, analysis):
        return None, {**status, "status": "insufficient_mod_data"}
    return (_score(model, analysis) if model else None), status


def _mods_supported(model, analysis):
    features = vector(analysis)
    values = [features[FEATURES.index(key)] for key in MOD_FEATURES]
    return all(lower <= value <= upper for value, (lower, upper) in zip(values, model.get('mod_bounds', [])))


def _accuracy_dataset(store: Store, user: str, return_weights=False):
    attempts, best, local = [], [], []
    for play in store.plays_with_analysis(user):
        if play['source'] == 'replay_nf_analysis' or not play.get('complete', 1) or (not play["passed"] and play.get('passed_known', 1)) or not 0 <= play["accuracy"] <= 1:
            continue
        analysis = play["analysis"]
        if analysis is None:
            continue
        row = (play, vector(analysis))
        (local if not play.get('passed_known', 1) else best if play["source"] in BEST_SOURCES else attempts).append(row)
    enough_attempts = len(attempts) >= MIN_PLAYS and len({p[0]["map_id"] for p in attempts}) >= MIN_MAPS
    enough_best = len(best) >= MIN_PLAYS and len({p[0]["map_id"] for p in best}) >= MIN_MAPS
    # Prefer ordinary passed attempts; best-only accuracy is a separate, biased task.
    enough_local = len(local) >= MIN_PLAYS and len({p[0]["map_id"] for p in local}) >= MIN_MAPS
    source_mode = "attempts" if enough_attempts else "local_scores" if enough_local else "best_scores" if enough_best else "attempts"
    rows = {"attempts": attempts, "best_scores": best, "local_scores": local}[source_mode]
    weights = np.asarray(sample_weights([p for p, _ in rows]), dtype=float)
    identity = [(p["id"], p["map_id"], p["mods"], p["accuracy"], features) for p, features in rows]
    fingerprint = hashlib.sha256(json.dumps(
        ["accuracy-ridge-v2", POLICY_VERSION, as_of_day().isoformat(), source_mode, identity, weights.tolist()], sort_keys=True).encode()).hexdigest()
    x = np.asarray([features for _, features in rows], dtype=float).reshape(-1, len(FEATURES))
    y = np.asarray([play["accuracy"] for play, _ in rows], dtype=float)
    groups = np.asarray([play["map_id"] for play, _ in rows], dtype=str)
    result = x, y, groups, fingerprint, source_mode
    return (*result, weights) if return_weights else result


def _accuracy_status(model: dict[str, Any]) -> dict[str, Any]:
    return {key: model[key] for key in (
        "plays", "maps", "source_mode", "metrics", "trained_at", "stars_range"
    )} | {"status": "trained", "algorithm": "ridge_regression"}


def train_accuracy(store: Store, user: str) -> dict[str, Any]:
    x, y, groups, fingerprint, source_mode, weights = _accuracy_dataset(store, user, return_weights=True)
    status = {"status": "insufficient_data", "plays": len(y), "maps": len(set(groups)),
              "source_mode": source_mode, "required_plays": MIN_PLAYS, "required_maps": MIN_MAPS}
    if len(y) < MIN_PLAYS or status["maps"] < MIN_MAPS:
        return status
    # Fixed split and fixed alpha: holdout outcomes never choose parameters or seed.
    train_ids, test_ids = next(GroupShuffleSplit(
        n_splits=1, test_size=0.25, random_state=0).split(x, y, groups))
    if len(test_ids) < 8:
        return {**status, "status": "insufficient_validation"}
    fit_weights = _relative_fit_weights(weights[train_ids])
    scaler = StandardScaler().fit(x[train_ids], sample_weight=fit_weights)
    regressor = Ridge(alpha=10).fit(scaler.transform(x[train_ids]), y[train_ids], sample_weight=fit_weights)
    predicted = np.clip(regressor.predict(scaler.transform(x[test_ids])), 0, 1)
    model_error = float(mean_absolute_error(y[test_ids], predicted, sample_weight=weights[test_ids]))
    baseline_error = float(mean_absolute_error(y[test_ids], np.full(len(test_ids), np.average(y[train_ids], weights=weights[train_ids])), sample_weight=weights[test_ids]))
    ordered = np.argsort(np.abs(y[test_ids] - predicted))
    q90_index = ordered[min(len(ordered) - 1, np.searchsorted(np.cumsum(weights[test_ids][ordered]), .9 * weights[test_ids].sum()))]
    metrics = {
        "holdout_plays": len(test_ids), "holdout_maps": len(set(groups[test_ids])),
        "mae_pp": round(model_error * 100, 4), "baseline_mae_pp": round(baseline_error * 100, 4),
        "validation": "unseen_maps", "split_seed": 0,
        "holdout_abs_error_q90": float(np.abs(y[test_ids] - predicted)[q90_index]),
        "recency_half_life_days": HALF_LIFE_DAYS, "weighted_plays": round(float(weights.sum()), 2),
    }
    if model_error >= baseline_error - 1e-8:
        return {**status, "status": "rejected", "metrics": metrics,
                "message": "Прогноз точности не улучшил среднее значение на отложенных картах."}
    fit_weights = _relative_fit_weights(weights)
    scaler = StandardScaler().fit(x, sample_weight=fit_weights)
    regressor = Ridge(alpha=10).fit(scaler.transform(x), y, sample_weight=fit_weights)
    artifact = {
        "schema": 1, "task": "accuracy", "features": list(FEATURES), "fingerprint": fingerprint,
        "mod_bounds": [[float(x[:, FEATURES.index(key)].min()), float(x[:, FEATURES.index(key)].max())] for key in MOD_FEATURES],
        "trained_at": datetime.now(timezone.utc).isoformat(), "plays": len(y), "maps": len(set(groups)),
        "source_mode": source_mode, "metrics": metrics,
        "stars_range": [float(x[:, 0].min()), float(x[:, 0].max())],
        "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
        "coef": regressor.coef_.tolist(), "intercept": float(regressor.intercept_),
    }
    path = _artifact_path(store, user, "accuracy")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    return _accuracy_status(artifact)


@lru_cache(maxsize=64)
def _ensure_accuracy_cached(database: str, user: str, fingerprint: str):
    store = Store(database)
    path = _artifact_path(store, user, "accuracy")
    model = _read_model(path, fingerprint)
    if model:
        return model, _accuracy_status(model)
    status = train_accuracy(store, user)
    return (_read_model(path, fingerprint) if status["status"] == "trained" else None), status


def accuracy_model_context(store: Store, user: str):
    return _accuracy_context_by_revision(str(store.path.resolve()), user.strip().casefold(), store.cache_token(user), as_of_day().isoformat())


@lru_cache(maxsize=64)
def _accuracy_context_by_revision(database: str, user: str, revision, day):
    store = Store(database)
    _, _, _, fingerprint, _ = _accuracy_dataset(store, user)
    return _ensure_accuracy_cached(database, user, fingerprint)


def predict_accuracy(store: Store, user: str, analysis: dict[str, Any], context=None):
    model, status = context if context is not None else accuracy_model_context(store, user)
    if model and not _mods_supported(model, analysis):
        return None, {**status, "status": "insufficient_mod_data"}
    if model is None:
        return None, status
    lower, upper = model["stars_range"]
    if not lower <= analysis["stars"] <= upper:
        return None, {**status, "status": "out_of_range",
                      "message": "Сложность карты вне диапазона обучения; используется предварительная оценка."}
    values = vector(analysis)
    predicted = model["intercept"] + sum(
        coefficient * (value - mean) / scale
        for value, mean, scale, coefficient in zip(values, model["mean"], model["scale"], model["coef"])
    )
    return float(np.clip(predicted, 0, 1)), status


def train_models(store: Store, user: str):
    result = {"pass": train(store, user), "accuracy": train_accuracy(store, user)}
    _ensure_cached.cache_clear()
    _ensure_accuracy_cached.cache_clear()
    _pass_context_by_revision.cache_clear()
    _accuracy_context_by_revision.cache_clear()
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train local osu!standard models from stored results")
    parser.add_argument("user", help="Player name used when importing attempts")
    parser.add_argument("--database", default=None, help="SQLite path (defaults to DATABASE_PATH)")
    args = parser.parse_args()
    print(json.dumps(train_models(Store(args.database), args.user), ensure_ascii=False, indent=2))
