"""Fit and evaluate intent, admission, and time-aware user interests."""

import math
import re
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

from backend.app.dataset_format import FEATURES, INTENTS, summary

TOKEN = re.compile(r"[a-z][a-z0-9]{2,}")
STOP = set("the and for from into with your about how this that of use guide best page compare review".split())
VARIANTS = ("behavior", "behavior_text", "multimodal")


def words(text):
    return [word for word in TOKEN.findall(text.lower()) if word not in STOP]


def feature_rows(rows, variant, context=None):
    facts = []
    for index, row in enumerate(rows):
        item = {key: math.log1p(row[key]) for key in FEATURES if key != "screenshot_count"}
        if variant != "behavior":
            tokens = set(words(" ".join(row["page_titles"])))
            item.update({f"text:{token}": 1.0 for token in tokens})
            if context is not None:
                item.update(context[index])
        if variant == "multimodal":
            image = row["image_embedding_mean"]
            item["visual:screenshot_count"] = math.log1p(row["screenshot_count"])
            item["visual:available"] = float(bool(image))
            item.update({f"visual:{i}": value for i, value in enumerate(image) if value})
        facts.append(item)
    return facts


def classifier(name, seed):
    estimator = {
        "logistic": LogisticRegression(max_iter=500),
        "random_forest": RandomForestClassifier(n_estimators=70, min_samples_leaf=3,
                                               random_state=seed, n_jobs=2),
        "xgboost": XGBClassifier(n_estimators=55, max_depth=3, learning_rate=0.12,
                                 random_state=seed, n_jobs=2, tree_method="hist"),
    }[name]
    return make_pipeline(DictVectorizer(), estimator)


def _score(actual, predicted, *, labels=None, probability=None):
    binary = labels is None
    labels = [0, 1] if binary else labels
    options = {"average": "binary" if binary else "macro", "labels": labels, "zero_division": 0}
    metrics = {
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision": float(precision_score(actual, predicted, **options)),
        "recall": float(recall_score(actual, predicted, **options)),
        "f1": float(f1_score(actual, predicted, **options)),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=labels).tolist(),
    }
    if binary:
        metrics["positive_prediction_rate"] = float(np.mean(predicted))
    if probability is not None:
        metrics["roc_auc"] = float(roc_auc_score(actual, probability)) if len(set(actual)) == 2 else None
    return metrics


def _select(train_x, train_y, validation_x, validation_y, seed, labels=None):
    candidates, scores = {}, {}
    for name in ("logistic", "random_forest", "xgboost"):
        model = classifier(name, seed).fit(train_x, train_y)
        candidates[name] = model
        probability = model.predict_proba(validation_x)[:, 1] if labels is None else None
        scores[name] = _score(validation_y, model.predict(validation_x), labels=labels,
                              probability=probability)
    selected = max(scores, key=lambda name: scores[name]["f1"])
    return selected, candidates[selected], scores


def _intent_context(model, rows, variant="multimodal"):
    predictions = model.predict_proba(feature_rows(rows, variant))
    return [{f"intent:{INTENTS[int(label)]}": float(value)
             for label, value in zip(model.classes_, probabilities, strict=True)}
            for probabilities in predictions]


def _training_context(model, train, labels, grouped):
    if not grouped:
        # Skip stacking on a single participant rather than predicting training rows in-sample.
        return [{} for _ in train]
    people = [row["participant_id"] for row in train]
    context = [{} for _ in train]
    for fit, holdout in GroupKFold(min(5, len(set(people)))).split(train, labels, groups=people):
        if len(set(labels[fit])) != len(INTENTS):
            return [{} for _ in train]
        fold = clone(model).fit(feature_rows([train[i] for i in fit], "behavior_text"), labels[fit])
        predictions = _intent_context(fold, [train[i] for i in holdout], "behavior_text")
        for i, prediction in zip(holdout, predictions, strict=True):
            context[i] = prediction
    return context


def _rule(row):
    return int(row["dwell_sec"] >= 120 and (row["visit_count"] >= 3 or row["repeat_visit_count"] > 0))


def _topics(row, allowed=None):
    terms = set(words(" ".join(row["page_titles"])))
    return terms if allowed is None else terms & allowed


def _history(rows, vocabulary, *, only_seen=False, at_time=None, targets=None):
    """Predict the next session's topics using only preceding sessions."""
    history, previous, samples = {}, None, []
    for row in sorted(rows, key=lambda entry: (entry["started_at"], entry["session_id"])):
        at = datetime.fromisoformat(row["started_at"])
        current = _topics(row, vocabulary)
        for token, (strength, last, count) in list(history.items()):
            elapsed = max(0.0, (at - previous).total_seconds() / 86400)
            history[token] = (strength * 0.5 ** (elapsed / 7), last, count)
        if previous is not None and not only_seen and current and (targets is None or row["session_id"] in targets):
            for topic in sorted(history):
                strength, last, count = history[topic]
                days = max(0, (at - last).total_seconds() / 86400)
                samples.append(([strength, math.log1p(days), math.log1p(count)], int(topic in current)))
        engagement = min(1.0, 0.25 + row["dwell_sec"] / 600)
        for topic in current:
            old, _, count = history.get(topic, (0.0, at, 0))
            history[topic] = (old + engagement, at, count + 1)
        previous = at
    if only_seen:
        if previous is None:
            return {}
        at = at_time or previous
        gap = max(0, (at - previous).total_seconds() / 86400)
        return {topic: [strength * 0.5 ** (gap / 7),
                        math.log1p(max(0, (at - last).total_seconds() / 86400)), math.log1p(count)]
                for topic, (strength, last, count) in history.items()}
    return samples


def _interest_examples(rows, vocabulary, prior=()):
    participants = defaultdict(list)
    targets = {row["session_id"] for row in rows}
    for row in (*prior, *rows):
        participants[row["participant_id"]].append(row)
    return [sample for person in participants.values()
            for sample in _history(person, vocabulary, targets=targets)]


def _fit_interests(train, validation, test):
    vocabulary = set().union(*(_topics(row) for row in train))
    training = _interest_examples(train, vocabulary)
    validating = _interest_examples(validation, vocabulary, prior=train)
    testing = _interest_examples(test, vocabulary, prior=[*train, *validation])
    if len({label for _, label in training}) < 2:
        return None, vocabulary, {"available": False}
    model = LogisticRegression(max_iter=500).fit(*zip(*training))
    metrics = {"available": True, "test_pairs": len(testing)}
    if testing:
        def probabilities(examples):
            matrix, labels = zip(*examples)
            return labels, {
                "exponential": np.asarray([1 - math.exp(-row[0]) for row in matrix]),
                "learned": model.predict_proba(matrix)[:, 1],
            }

        test_labels, test_scores = probabilities(testing)
        validation_labels, validation_scores = probabilities(validating) if validating else ([], {})
        for name in ("exponential", "learned"):
            threshold = 0.5
            if len(set(validation_labels)) == 2:
                threshold = max((0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
                                key=lambda cut: f1_score(validation_labels, validation_scores[name] >= cut))
            metrics[name] = {"threshold": threshold, **_score(
                test_labels, test_scores[name] >= threshold, probability=test_scores[name])}
    return model, vocabulary, metrics


def _admission_context(rows, intent=None, prior=()):
    """Use topic engagement from completed sessions preceding each example."""
    ordered = defaultdict(list)
    for row in (*prior, *rows):
        ordered[row.get("participant_id")].append(row)
    context = []
    for index, row in enumerate(rows):
        start = datetime.fromisoformat(row["started_at"])
        previous = [candidate for candidate in ordered[row.get("participant_id")]
                    if candidate["session_id"] != row["session_id"]
                    and datetime.fromisoformat(candidate["ended_at"]) <= start]
        topics = _topics(row)
        history = _history(previous, topics, only_seen=True, at_time=start) if topics else {}
        strengths = [1 - math.exp(-history[topic][0]) if topic in history else 0 for topic in topics]
        context.append({**(intent[index] if intent is not None else {}),
                        "interest:max": max(strengths, default=0.0),
                        "interest:mean": sum(strengths) / max(1, len(strengths))})
    return context


def train_sessions(partitions, method, seed):
    train, validation, test = (partitions[name] for name in ("train", "validation", "test"))
    intent_labels = list(range(len(INTENTS)))
    encode = lambda rows: np.array([INTENTS.index(row["intent"]) for row in rows])
    amount = lambda rows: np.array([row["memory_worthy"] for row in rows])
    ti, vi, ei = map(encode, (train, validation, test))
    ta, va, ea = map(amount, (train, validation, test))
    if set(ti) != set(intent_labels) or any(len(set(values)) < 2 for values in (ta, va)):
        raise ValueError("Training needs all seven intent classes and both admission labels. Validation needs both admission labels.")

    name, intent, scores = _select(feature_rows(train, "multimodal"), ti,
                                  feature_rows(validation, "multimodal"), vi, seed, intent_labels)
    report = {
        "partitions": {key: summary(rows) for key, rows in partitions.items()},
        "intent": {"selected": name, "labels": list(INTENTS), "validation": scores,
                   "test": _score(ei, intent.predict(feature_rows(test, "multimodal")), labels=intent_labels)},
    }

    grouped = method == "participant_holdout" and len({row["participant_id"] for row in train}) >= 2
    tc = _training_context(intent, train, ti, grouped)
    stacked = bool(tc[0])
    text_intent = clone(intent).fit(feature_rows(train, "behavior_text"), ti) if stacked else None
    vc = _intent_context(text_intent, validation, "behavior_text") if stacked else None
    ec = _intent_context(text_intent, test, "behavior_text") if stacked else None
    rule = [_rule(row) for row in test]
    train_context = _admission_context(train, tc)
    validation_context = _admission_context(validation, vc, prior=train)
    test_context = _admission_context(test, ec, prior=[*train, *validation])
    always = np.ones_like(ea)
    admission_report = {"rule": {"test": _score(ea, rule, probability=rule)},
                        "always_yes": {"test": _score(ea, always, probability=always)}}
    finalists = {}
    for variant in VARIANTS:
        name, model, scores = _select(feature_rows(train, variant, train_context), ta,
                                      feature_rows(validation, variant, validation_context), va, seed)
        validation_probability = model.predict_proba(feature_rows(validation, variant, validation_context))[:, 1]
        # Choose the operating point on validation only. Prefer the midpoint for ties.
        threshold = max((0.25, 0.35, 0.5, 0.65, 0.75),
                        key=lambda cutoff: (f1_score(va, validation_probability >= cutoff), -abs(cutoff - 0.5)))
        probability = model.predict_proba(feature_rows(test, variant, test_context))[:, 1]
        admission_report[variant] = {"selected": name, "threshold": threshold, "validation": scores,
                                     "validation_f1": float(f1_score(va, validation_probability >= threshold)),
                                     "test": _score(ea, probability >= threshold, probability=probability)}
        finalists[variant] = (model, threshold)
    winner = max(VARIANTS, key=lambda variant: admission_report[variant]["validation_f1"])
    report["admission"] = {"selected": winner, "uses_predicted_intent": stacked, "variants": admission_report}

    interest, vocabulary, report["interests"] = _fit_interests(train, validation, test)
    return {"intent": intent, "admissions": finalists, "selected": winner, "stacked": stacked,
            "text_intent": text_intent, "interest": interest, "vocabulary": vocabulary}, report


def predict_session(bundle, row, history=()):
    probabilities = _intent_context(bundle["intent"], [row])[0]
    confidence = {label: probabilities[f"intent:{label}"] for label in INTENTS}
    best = max(confidence, key=confidence.get)
    predicted_intent = _intent_context(bundle["text_intent"], [row], "behavior_text") if bundle["stacked"] else None
    context = _admission_context([row], predicted_intent, history)
    decisions = {}
    for variant, (model, cutoff) in bundle["admissions"].items():
        features = feature_rows([row], variant, context)
        score = float(model.predict_proba(features)[0, 1])
        decisions[variant] = {"probability": score, "threshold": cutoff, "remember": score >= cutoff}
    return {"intent": best, "intent_probabilities": confidence, "rule": bool(_rule(row)),
            "selected": bundle["selected"], "admission": decisions}


def interest_scores(bundle, rows, *, at_time=None):
    features = _history(rows, bundle["vocabulary"], only_seen=True, at_time=at_time)
    if not features:
        return []
    topics, values = zip(*features.items())
    learned = bundle["interest"].predict_proba(values)[:, 1] if bundle["interest"] is not None else [None] * len(values)
    rows = [{"topic": topic, "baseline": float(1 - math.exp(-features[topic][0])),
             "learned": float(score) if score is not None else None}
            for topic, score in zip(topics, learned, strict=True)]
    return sorted(rows, key=lambda item: -(item["learned"] if item["learned"] is not None else item["baseline"]))[:10]


def interest_trends(bundle, rows):
    history = sorted(rows, key=lambda row: row["started_at"])
    current = interest_scores(bundle, history)
    topics = [item["topic"] for item in current[:3]]
    if not topics:
        return []
    last = datetime.fromisoformat(history[-1]["ended_at"])
    samples = []
    for days in range(7, -1, -1):
        at = last - timedelta(days=days)
        seen = [row for row in history if datetime.fromisoformat(row["ended_at"]) <= at]
        features = _history(seen, bundle["vocabulary"], only_seen=True, at_time=at)
        samples.append({"date": at.date().isoformat(),
                        "scores": {topic: float(1 - math.exp(-features[topic][0])) if topic in features else 0
                                   for topic in topics}})
    return samples
