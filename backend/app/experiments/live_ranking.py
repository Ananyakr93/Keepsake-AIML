"""Fit the local pairwise reranker with a fixed held-out prompt group."""

import hashlib
from itertools import groupby
from threading import Lock

from backend.app import config, db
from backend.app.experiments import ranking
from backend.app.experiments.workspace import _read_artifact, save_artifact

_lock = Lock()


class Ranker:
    def __init__(self, method, model=None):
        self.method = method
        self.model = model

    def rank(self, query, candidates):
        return ranking.order(query, candidates, self.method, self.model)


def _artifact(user_id):
    return config.MODEL_DIR / f"ranker_xgb_{user_id}.joblib"


def build_live_ranker():
    user = db.query_one("SELECT id FROM users WHERE username=%s", [config.DEFAULT_USERNAME])
    artifact = _artifact(user["id"])
    if not artifact.is_file():
        raise FileNotFoundError("The live pairwise ranker has no validated update.")
    return Ranker("xgboost", _read_artifact(artifact, artifact.stat().st_mtime_ns)["model"])


def clear_trained_model(user_id):
    from backend.app import ml

    with _lock:
        _artifact(user_id).unlink(missing_ok=True)
        ml.get_ranker.cache_clear()


def split_judgments(groups):
    training, validation = [], []
    for prompt, candidates in groups:
        digest = hashlib.sha256(str(prompt["id"]).encode()).digest()
        split = validation if digest[0] % 4 == 0 else training
        split.append((prompt, candidates))
    return training, validation


def refit(user_id):
    from backend.app import ml

    with _lock:
        rows = db.query("""
            SELECT p.id AS prompt_id,p.text AS query,m.id,m.content,m.importance,f.vote,
                   1 - (p.embedding <=> m.embedding) AS sim,
                   exp(-extract(epoch from (now() - m.updated_at)) / 86400.0) AS recency
            FROM memory_feedback f JOIN training_prompts p ON p.id=f.prompt_id
            JOIN memories m ON m.id=f.memory_id AND m.user_id=p.user_id
            WHERE p.user_id=%s AND m.archived_at IS NULL
            ORDER BY p.id,m.id
        """, [user_id])
        groups = []
        for prompt_id, group in groupby(rows, key=lambda row: row["prompt_id"]):
            candidates = [{**entry, "relevant": int(entry["vote"] > 0)} for entry in group]
            if len({entry["relevant"] for entry in candidates}) != 2:
                continue
            candidates.sort(key=lambda entry: (-entry["sim"], str(entry["id"])))
            groups.append(({"id": prompt_id, "query": candidates[0]["query"], "memories": candidates}, candidates))
        training, validation = split_judgments(groups)
        if len(training) < 4 or len(validation) < 2:
            return {"retrained": False, "votes": len(rows),
                    "reason": f"Need four training prompts and two held-out prompts with mixed votes. Have {len(training)} and {len(validation)}."}
        pairs = [(ranking.features(prompt["query"], candidates), [row["relevant"] for row in candidates])
                 for prompt, candidates in training]
        model = ranking.fit_ranker(pairs)
        path = _artifact(user_id)
        try:
            current = _read_artifact(path, path.stat().st_mtime_ns)["model"]
        except FileNotFoundError:
            current = None
        baseline = ranking._metrics(validation, "xgboost" if current is not None else "vector", current)["ndcg_at_5"]
        candidate = ranking._metrics(validation, "xgboost", model)["ndcg_at_5"]
        quality = {"baseline": baseline, "candidate": candidate}
        if candidate <= baseline:
            return {"retrained": False, "votes": len(rows), "validation": quality,
                    "reason": "The candidate did not improve held-out ranking."}
        save_artifact({"model": model}, path)
        ml.get_ranker.cache_clear()
        return {"retrained": True, "votes": sum(len(labels) for _, labels in pairs),
                "query_groups": len(pairs), "validation": quality}
