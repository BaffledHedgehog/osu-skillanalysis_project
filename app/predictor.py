"""Experimental personalized difficulty model, trained only on actual player results."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from functools import lru_cache
import time
from typing import Any

import numpy as np

from .ml import BEST_SOURCES, SIMILARITY_SCALES, accuracy_model_context, model_context, predict_accuracy, predict_pass, vector
from .storage import Store
from .capabilities import coverage_warnings, practice_distance
from .recency import sample_weights, progress_summary
from .sequence_service import predict_cnn


def _clamp(value: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, value))


def _sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-_clamp(value, -25, 25)))


def _weighted_quantile(values: list[float], weights: list[float], q: float) -> float:
    ranked = sorted(zip(values, weights), key=lambda item: item[0])
    goal = q * sum(weights)
    total = 0.0
    for value, weight in ranked:
        total += weight
        if total >= goal:
            return value
    return ranked[-1][0]


def _records(store: Store, user: str) -> list[dict[str, Any]]:
    return _records_cached(str(store.path.resolve()), user.strip().casefold(), store.cache_token(user), int(time.time() // 600))


@lru_cache(maxsize=64)
def _records_cached(database: str, user: str, revision, time_bucket: int) -> list[dict[str, Any]]:
    store = Store(database)
    records = []
    plays = store.plays_with_analysis(user)
    for play, recency_weight in zip(plays, sample_weights(plays)):
        analysis = play["analysis"]
        if not analysis:
            continue
        play["analysis"] = analysis
        source_weight = 0.35 if play["source"] in ("osu_api_best", "hinai_best", "osu_web_best", "osu_api") else 1.0
        play["weight"] = source_weight * recency_weight
        records.append(play)
    return records


def _summary_from_records(user: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    passed = [r for r in records if r["passed"] and r.get("passed_known", 1) and 'NF' not in r['mods']]
    if not records:
        return {"user": user, "plays": 0, "passed": 0, "accuracy": None,
                "comfort_stars": None, "aim_level": None, "speed_level": None,
                "best_pp": None, "sources": {}, "recent": []}
    weights = [r["weight"] for r in records]
    source_counts: dict[str, int] = {}
    for r in records:
        source_counts[r["source"]] = source_counts.get(r["source"], 0) + 1
    return {
        "user": user,
        "plays": len(records),
        "passed": len(passed),
        "unknown_outcomes": sum(not r.get("passed_known", 1) for r in records),
        "failed": sum(r.get('passed_known', 1) and not r['passed'] and 'NF' not in r['mods'] for r in records),
        "no_fail_plays": sum('NF' in r['mods'] for r in records),
        "accuracy": round(sum(r["accuracy"] * r["weight"] for r in records) / sum(weights) * 100, 1),
        "best_pp": round(max((r["pp"] for r in records if r["pp"] is not None), default=0), 1)
                   if any(r["pp"] is not None for r in records) else None,
        "comfort_stars": round(_weighted_quantile(
            [r["analysis"]["stars"] for r in passed], [r["weight"] for r in passed], 0.7), 2) if passed else None,
        "aim_level": round(_weighted_quantile(
            [r["analysis"]["aim"] for r in passed], [r["weight"] for r in passed], 0.7), 2) if passed else None,
        "speed_level": round(_weighted_quantile(
            [r["analysis"]["speed"] for r in passed], [r["weight"] for r in passed], 0.7), 2) if passed else None,
        "sources": source_counts,
        "progress": progress_summary(records),
        "recent": [{"map_id": r["map_id"], "title": r["analysis"]["title"], "mods": r["mods"],
                    "accuracy": round(r["accuracy"] * 100, 1), "passed": bool(r["passed"]) if r.get('passed_known', 1) else None,
                    "grade": r.get('grade'), "complete": bool(r.get('complete', 1))} for r in records[:5]],
    }


def user_summary(store: Store, user: str) -> dict[str, Any]:
    return _summary_from_records(user, _records(store, user))


def predict(store: Store, user: str, target: dict[str, Any],
            records: list[dict[str, Any]] | None = None,
            summary: dict[str, Any] | None = None,
            ml_context: tuple[dict[str, Any] | None, dict[str, Any]] | None = None,
            accuracy_context=None) -> dict[str, Any]:
    records = records if records is not None else _records(store, user)
    records = [r for r in records if 'NF' not in r['mods']]
    if len(records) < 3:
        return {
            "pass_probability": None, "expected_accuracy": None, "accuracy_range": None,
            "confidence": "none", "model": "insufficient_data", "observations": len(records),
            "message": "Добавьте хотя бы три результата. До этого персональный прогноз не рассчитывается.",
            "factors": [],
        }
    stars = [r["analysis"]["stars"] for r in records]
    weights = [r["weight"] for r in records]
    # A score's accuracy adjusts the player's demonstrated comfortable difficulty.
    abilities = [s + (r["accuracy"] - 0.9) * 3.5 - (0.65 if r.get("passed_known", 1) and not r["passed"] else 0)
                 for s, r in zip(stars, records)]
    ability = _weighted_quantile(abilities, weights, 0.7)
    prior = _sigmoid((ability - target["stars"]) * 1.65)

    # Similar played maps are the local, data-driven part of the estimate.
    target_vector = np.asarray(vector(target))
    candidates = []
    for r in records:
        v = np.asarray(vector(r["analysis"]))
        scaled = (v - target_vector) / np.asarray(SIMILARITY_SCALES)
        distance = float(np.sqrt(np.sum(scaled ** 2)))
        w = r["weight"] * math.exp(-0.5 * distance * distance)
        candidates.append((r, w))
    pass_candidates = [(r, w) for r, w in candidates if r.get("passed_known", 1)]
    nearby_weight = sum(w for _, w in candidates)
    p = (3 * prior + sum(w * r["passed"] for r, w in pass_candidates)) / (3 + sum(w for _, w in pass_candidates))
    mean_acc = sum(r["accuracy"] * r["weight"] for r in records) / sum(weights)
    baseline_acc = _clamp(mean_acc - (target["stars"] - ability) * 0.042, 0.5, 0.995)
    acc = (2 * baseline_acc + sum(w * _clamp(
        r["accuracy"] - (target["stars"] - r["analysis"]["stars"]) * 0.025, 0.3, 1)
        for r, w in candidates)) / (2 + nearby_weight)

    known_outcomes = [r for r in records if r.get("passed_known", 1)]
    failed = sum(not r["passed"] for r in known_outcomes)
    model = "similarity_and_prior"
    ml_p, ml_status = predict_pass(store, user, target, ml_context)
    if ml_p is not None:
        p = ml_p
        model = "validated_personal_logistic"
    cnn = predict_cnn(store, user, target, records)
    if cnn.get('pass_probability') is not None:
        p = cnn['pass_probability']
        model = 'validated_pattern_cnn'
    # Beta posterior: observed attempts of this exact chart update the model's
    # prior. This is evidence-weighting, not a hard skill/probability threshold.
    same_chart = [r for r in records if r.get('passed_known', 1) and r['source'] not in BEST_SOURCES
                  and r['analysis'].get('checksum') == target.get('checksum') and r['mods'] == target['mods']]
    same_chart_passes = sum(r['passed'] for r in same_chart)
    if same_chart:
        history_weight = sum(r['weight'] for r in same_chart)
        p = (3 * p + sum(r['weight'] * r['passed'] for r in same_chart)) / (3 + history_weight)
        model = 'personal_history_posterior'
    ml_acc, accuracy_status = predict_accuracy(store, user, target, accuracy_context)
    accuracy_model = "similarity_and_prior"
    if ml_acc is not None:
        acc = ml_acc
        accuracy_model = "validated_personal_ridge"
    if cnn.get('accuracy') is not None:
        acc = cnn['accuracy']
        accuracy_model = 'validated_pattern_cnn'
    has_both = failed > 0 and failed < len(known_outcomes)
    confidence = "medium" if len(records) >= 25 and has_both and nearby_weight >= 2 else "low"
    half_width = 0.065 if confidence == "medium" else 0.105
    if ml_acc is not None:
        half_width = max(0.005, accuracy_status["metrics"]["holdout_abs_error_q90"])
    if cnn.get('accuracy') is not None:
        half_width = max(.005, cnn['accuracy_error_q90'])
    factors = []
    if same_chart:
        factors.append(f'В сохраненной истории этой версии карты с теми же модами: {len(same_chart)} попыток, проходов {same_chart_passes}, провалов {len(same_chart) - same_chart_passes}. Это не полный счетчик всех запусков карты.')
    if target.get('peak_to_mean', 0) > 1:
        factors.append(f'Пик нагрузки в {target["peak_to_mean"]:.1f} раза выше средней нагрузки. Короткий сложный участок учитывается отдельными признаками модели.')
    delta = target["stars"] - ability
    if delta > 0.35:
        factors.append(f"Сложность карты выше твоего текущего ориентира примерно на {delta:.1f} звезды.")
    elif delta < -0.35:
        factors.append(f"Сложность карты ниже твоего текущего ориентира примерно на {-delta:.1f} звезды.")
    summary = summary or _summary_from_records(user, records)
    if summary["aim_level"] is not None and target["aim"] > summary["aim_level"] + 0.45:
        factors.append("Aim нагрузка выше, чем на большинстве успешно пройденных тобой карт.")
    if summary["speed_level"] is not None and target["speed"] > summary["speed_level"] + 0.45:
        factors.append("Скоростная нагрузка выше, чем на большинстве успешно пройденных тобой карт.")
    if not has_both:
        factors.append("В истории мало неудачных попыток, поэтому вероятность pass особенно ненадёжна.")
    if not factors:
        factors.append("Общая нагрузка близка к твоим недавним результатам.")
    warnings = coverage_warnings(target, records)
    unsupported = ml_status.get('status') == 'insufficient_mod_data'
    unknown = bool(warnings) or unsupported or 'NF' in target['mods']
    if 'NF' in target['mods']:
        factors.insert(0, 'No Fail отключает обычный провал. Вероятность pass без NF по такому результату не определяется.')
    if unknown:
        confidence = 'low'
        factors = [w['reason'] for w in warnings] + factors
        if unsupported:
            factors.insert(0, 'Для выбранных модов нет примеров в обучении модели прохождения.')
    return {
        "pass_probability": None if unknown else round(_clamp(p, 0, 1) * 100, 1),
        "coverage_warnings": warnings,
        "same_chart_history": {'attempts': len(same_chart), 'passes': same_chart_passes, 'fails': len(same_chart) - same_chart_passes},
        "prediction_status": 'unknown_skill_range' if unknown else 'estimated',
        "too_hard": False if unknown else p <= .1,
        "suggest_training": unknown or p <= .1,
        "expected_accuracy": round(acc * 100, 1),
        "accuracy_range": [round(_clamp(acc - half_width, 0, 1) * 100, 1),
                           round(_clamp(acc + half_width, 0, 1) * 100, 1)],
        "confidence": confidence, "model": model, "observations": len(records),
        "ml_status": ml_status,
        "accuracy_ml_status": accuracy_status, "accuracy_model": accuracy_model,
        "cnn_status": cnn,
        "message": "Экспериментальная оценка по вашим результатам; вероятность пока не откалибрована на независимой выборке.",
        "factors": factors,
    }


def recommendations(store: Store, user: str, goal: str = "comfort", mods: str = "NM", limit: int = 12) -> list[dict[str, Any]]:
    ranges = {"warmup": (80, 100), "comfort": (55, 85), "challenge": (25, 60)}
    lo, hi = ranges.get(goal, ranges["comfort"])
    records = _records(store, user)
    played_ids = {r['map_id'] for r in records}
    summary = _summary_from_records(user, records)
    ml_context = model_context(store, user) if len(records) >= 3 else None
    accuracy_context = accuracy_model_context(store, user) if len(records) >= 3 else None
    ranked = []
    for item in store.list_maps():
        if item["id"] in played_ids:
            continue
        try:
            analysis = store.analysis(item["id"], mods)
        except (ValueError, RuntimeError):
            continue
        if not analysis:
            continue
        prediction = predict(store, user, analysis, records, summary, ml_context, accuracy_context)
        p = prediction["pass_probability"]
        if p is None:
            # With no history, simply sort by chart stars and avoid invented personalization.
            distance = 1000 + analysis["stars"]
        else:
            distance = abs(p - (lo + hi) / 2) + (25 if p < lo or p > hi else 0)
        ranked.append({**{key: value for key, value in analysis.items() if key not in ("windows", "peaks")},
                       "prediction": prediction, "_distance": distance})
    ranked.sort(key=lambda row: row["_distance"])
    for item in ranked:
        item.pop("_distance")
    return ranked[:limit]


def training_maps(store: Store, user: str, target, limit=6):
    records = _records(store, user)
    if len(records) < 3:
        return {'maps': [], 'message': 'Для персонального подбора нужны хотя бы три результата.'}
    summary = _summary_from_records(user, records)
    contexts = model_context(store, user), accuracy_model_context(store, user)
    ranked = []
    for item in store.list_maps():
        if item['id'] == target['id']:
            continue
        analysis = store.analysis(item['id'], target['mods'])
        if not analysis or analysis['stars'] >= target['stars']:
            continue
        result = predict(store, user, analysis, records, summary, *contexts)
        if result['pass_probability'] is None or result['expected_accuracy'] < 95 or result['pass_probability'] < 60:
            continue
        distance = practice_distance(target, analysis)
        if distance > 1.5:
            continue
        ranked.append({'id': analysis['id'], 'artist': analysis['artist'], 'title': analysis['title'],
                       'version': analysis['version'], 'stars': analysis['stars'], 'mods': analysis['mods'],
                       'prediction': result, 'similarity_distance': round(distance, 3)})
    ranked.sort(key=lambda item: (item['similarity_distance'], -item['stars']))
    return {'maps': ranked[:limit], 'message': 'Похожие карты из локального каталога с прогнозом от 95% accuracy. Это оценка, не гарантия.' if ranked else
            'В каталоге пока нет похожих карт с прогнозом от 95%. Добавь более простые карты или импортируй локальную историю.'}
