"""Grouped pairwise ranking, retrieval baselines, and feedback evaluation."""

import math

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRanker

from backend.app.experiments.sessions import words

CANDIDATES = 20
TOP_K = 5


def features(query, candidates):
    terms = set(words(query))
    return np.asarray([
        [row["sim"], row["importance"], row["recency"],
         len(terms.intersection(words(row["content"]))) / max(1, len(terms))]
        for row in candidates
    ], dtype=float)


def vector_candidates(encoder, query, memories):
    vectors = encoder.transform([query, *(row["content"] for row in memories)])
    similarity = (vectors[1:] @ vectors[0].T).toarray().ravel()
    ranked = [{**row, "sim": float(sim)} for row, sim in zip(memories, similarity, strict=True)]
    return sorted(ranked, key=lambda row: (-row["sim"], row["id"]))[:CANDIDATES]


def _groups(encoder, prompts):
    return [(prompt, vector_candidates(encoder, prompt["query"], prompt["memories"])) for prompt in prompts]


def training_groups(encoder, prompts):
    groups = []
    for prompt, candidates in _groups(encoder, prompts):
        labels = [row["relevant"] for row in candidates]
        if len(set(labels)) == 2:
            groups.append((features(prompt["query"], candidates), labels))
    return groups


def fit_ranker(base, feedback=(), seed=42):
    groups = list(base) + list(feedback)
    if not groups:
        raise ValueError("At least one query needs both relevant and irrelevant judgments.")
    x, labels = zip(*groups)
    model = XGBRanker(objective="rank:pairwise", n_estimators=65, learning_rate=0.16, max_depth=3,
                      tree_method="hist", n_jobs=2, random_state=seed)
    weights = np.asarray([1.0] * len(base) + [5.0] * len(feedback))
    return model.fit(np.vstack(x), np.concatenate(labels), group=[len(y) for y in labels],
                     sample_weight=weights)


def order(query, candidates, method, model=None):
    if not candidates:
        return []
    if method == "vector":
        scores = [row["sim"] for row in candidates]
    elif method == "weighted":
        scores = [0.6 * row["sim"] + 0.25 * row["importance"] + 0.15 * row["recency"] for row in candidates]
    else:
        matrix = features(query, candidates)
        scores = model.predict_proba(matrix)[:, 1] if method == "logistic" else model.predict(matrix)
    ranked = [{**row, "ranking_score": float(score)} for row, score in zip(candidates, scores, strict=True)]
    return sorted(ranked, key=lambda row: -row["ranking_score"])


def _metrics(groups, method, model=None):
    precision, recall, mrr, ndcg, candidate_recall = [], [], [], [], []
    for prompt, candidates in groups:
        relevant = sum(row["relevant"] for row in prompt["memories"])
        if not relevant:
            continue
        ranked = order(prompt["query"], candidates, method, model)
        hits = [index for index, row in enumerate(ranked[:TOP_K], 1) if row["relevant"]]
        precision.append(len(hits) / TOP_K)
        recall.append(len(hits) / relevant)
        mrr.append(1 / hits[0] if hits else 0)
        ideal = sum(1 / math.log2(index + 1) for index in range(1, min(relevant, TOP_K) + 1))
        ndcg.append(sum(1 / math.log2(index + 1) for index in hits) / ideal)
        candidate_recall.append(sum(row["relevant"] for row in candidates) / relevant)
    return {key: float(np.mean(series)) if series else None for key, series in
            {"precision_at_5": precision, "recall_at_5": recall, "mrr_at_5": mrr,
             "ndcg_at_5": ndcg, "candidate_recall_at_20": candidate_recall}.items()}


def train_retrieval(train, test, seed):
    encoder = TfidfVectorizer(ngram_range=(1, 2)).fit(
        [prompt["query"] for prompt in train] + [row["content"] for p in train for row in p["memories"]]
    )
    base = training_groups(encoder, train)
    x, y = zip(*base)
    logistic = LogisticRegression(class_weight="balanced", max_iter=500).fit(np.vstack(x), np.concatenate(y))
    model = fit_ranker(base, seed=seed)
    groups = _groups(encoder, test)
    report = {
        "queries": len(test), "vectorizer": "tfidf", "candidate_count": CANDIDATES, "top_k": TOP_K,
        "methods": {method: _metrics(groups, method, estimator) for method, estimator in
                    (("vector", None), ("weighted", None), ("logistic", logistic), ("xgboost", model))},
    }
    return {"encoder": encoder, "base": base, "model": model, "logistic": logistic}, report


def feedback_comparison(retrieval, feedback, test, seed):
    if not feedback:
        return retrieval["model"], None
    model = fit_ranker(retrieval["base"], feedback, seed)
    groups = _groups(retrieval["encoder"], test)
    return model, {"before": _metrics(groups, "xgboost", retrieval["model"]),
                   "after": _metrics(groups, "xgboost", model), "training_query_groups": len(feedback)}
