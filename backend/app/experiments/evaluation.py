"""Participant-separated evaluation for the synthetic phase 6 experiments."""

import hashlib
import platform
import time

import numpy as np
from sklearn.model_selection import GroupKFold

from backend.app.dataset_format import INTENTS, dataset_csv, read_datasets, summary
from backend.app.experiments import fixtures, ranking, sessions

RANK_METRICS = (
    "precision_at_5", "recall_at_5", "mrr_at_5", "ndcg_at_5", "candidate_recall_at_20",
)
RANK_FEATURES = ("cosine", "importance", "recency", "term_coverage")
ABLATIONS = {"xgboost_cosine": (0,), "xgboost_without_term_coverage": (0, 1, 2)}


def participant_folds(rows, count, seed):
    participants = sorted({row["participant_id"] for row in rows})
    if not 3 <= count <= len(participants):
        raise ValueError("Choose between three folds and the number of participants.")
    rng = np.random.default_rng(seed)
    groups = [row["participant_id"] for row in rows]
    folds = GroupKFold(n_splits=count, shuffle=True, random_state=seed)
    for fit, holdout in folds.split(rows, groups=groups):
        fit_people = sorted({groups[i] for i in fit})
        validation_count = max(1, round(len(fit_people) * 0.2))
        validation_people = set(rng.choice(fit_people, validation_count, replace=False))
        yield {
            "train": [rows[i] for i in fit if groups[i] not in validation_people],
            "validation": [rows[i] for i in fit if groups[i] in validation_people],
            "test": [rows[i] for i in holdout],
        }


class _SelectedFeatures:
    def __init__(self, model, columns):
        self.model = model
        self.columns = columns

    def predict(self, values):
        return self.model.predict(values[:, self.columns])


def _latency(encoder, prompts, ranker):
    for prompt in prompts[:2]:
        candidates = ranking.vector_candidates(encoder, prompt["query"], prompt["memories"])
        ranking.order(prompt["query"], candidates, "xgboost", ranker)
    timings = []
    for prompt in prompts:
        started = time.perf_counter()
        candidates = ranking.vector_candidates(encoder, prompt["query"], prompt["memories"])
        ranking.order(prompt["query"], candidates, "xgboost", ranker)
        timings.append((time.perf_counter() - started) * 1000)
    return timings


def _evaluate_ranking(partition, seed):
    examples = {name: fixtures.synthetic_relevance(rows, seed + i, name)
                for i, (name, rows) in enumerate(partition.items())}
    retrieval, result = ranking.train_retrieval(examples["train"], examples["test"], seed)
    encoder = retrieval["encoder"]
    groups = ranking._groups(encoder, examples["test"])
    for name, columns in ABLATIONS.items():
        base = [(features[:, columns], labels) for features, labels in retrieval["base"]]
        model = _SelectedFeatures(ranking.fit_ranker(base, seed=seed), columns)
        result["methods"][name] = ranking._metrics(groups, "xgboost", model)
    feedback = ranking.training_groups(encoder, examples["validation"])[:3]
    _, comparison = ranking.feedback_comparison(retrieval, feedback, examples["test"], seed)
    judgments = [label for _, labels in feedback for label in labels]
    comparison.update(judgments=len(judgments), useful_vote_rate=sum(judgments) / len(judgments))
    result["feedback"] = comparison
    return result, _latency(encoder, examples["test"], retrieval["model"])


def _aggregate(records, metrics):
    aggregate = {}
    for name in metrics:
        values = [entry[name] for entry in records if entry.get(name) is not None]
        if values:
            aggregate[name] = {"mean": float(np.mean(values)), "fold_sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                               "folds": len(values)}
    return aggregate


UNSEEN_TITLES = {
    "research": "What are the other explanations",
    "learning": "Trying this with a practice example",
    "purchase": "Would this be a sensible choice",
    "planning": "Options for getting it done later",
    "entertainment": "Something interesting to look at",
    "casual": "A quick look before moving on",
    "task": "One item to get through today",
}


def _stress_test(model, rows):
    altered = []
    for row in rows:
        title = row["page_titles"][0].split(" | ", 1)[0]
        altered.append({**row, "page_titles": [title, UNSEEN_TITLES[row["intent"]]]})
    labels = [INTENTS.index(row["intent"]) for row in rows]
    output = model.predict(sessions.feature_rows(altered, "multimodal"))
    return sessions._score(labels, output, labels=list(range(len(INTENTS))))


def run_evaluation(seed, fold_count, status=None):
    import sklearn
    import xgboost

    contents = dataset_csv(fixtures.synthetic_sessions(seed))
    rows = read_datasets([("synthetic_sessions.csv", contents)])
    results, times = [], []
    for index, partition in enumerate(participant_folds(rows, fold_count, seed), start=1):
        if status:
            status(index, fold_count)
        fold_seed = seed + 10 * index
        models, session_report = sessions.train_sessions(partition, "participant_holdout", fold_seed)
        session_report["intent_stress"] = _stress_test(models["intent"], partition["test"])
        retrieval_report, latency = _evaluate_ranking(partition, fold_seed)
        times.extend(latency)
        results.append({
            "fold": index, "fit_seed": fold_seed,
            "participants": {name: sorted({row["participant_id"] for row in subset})
                             for name, subset in partition.items()},
            **session_report, "retrieval": retrieval_report,
        })
    classification = ("accuracy", "precision", "recall", "f1", "roc_auc", "positive_prediction_rate")
    admission_variants = results[0]["admission"]["variants"]
    ranking_methods = results[0]["retrieval"]["methods"]
    feedback = [fold["retrieval"]["feedback"] for fold in results]
    combined_confusion = np.sum([fold["intent"]["test"]["confusion_matrix"] for fold in results], axis=0)
    return {
        "source": "synthetic",
        "seed": seed,
        "fold_count": fold_count,
        "dataset": summary(rows),
        "dataset_sha256": hashlib.sha256(contents.encode()).hexdigest(),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scikit_learn": sklearn.__version__, "xgboost": xgboost.__version__},
        "intent": _aggregate([fold["intent"]["test"] for fold in results], classification),
        "intent_stress": _aggregate([fold["intent_stress"] for fold in results], classification),
        "intent_validation": {name: _aggregate([fold["intent"]["validation"][name] for fold in results], classification)
                              for name in results[0]["intent"]["validation"]},
        "intent_labels": list(INTENTS),
        "intent_confusion": combined_confusion.tolist(),
        "admission": {name: _aggregate([fold["admission"]["variants"][name]["test"] for fold in results], classification)
                      for name in admission_variants},
        "interests": {name: _aggregate([fold["interests"][name] for fold in results], classification)
                      for name in ("exponential", "learned")},
        "ranking": {name: _aggregate([fold["retrieval"]["methods"][name] for fold in results], RANK_METRICS)
                    for name in ranking_methods},
        "ranking_ablations": {name: [RANK_FEATURES[i] for i in columns] for name, columns in ABLATIONS.items()},
        "ranking_queries": sum(fold["retrieval"]["queries"] for fold in results),
        "feedback": {when: _aggregate([row[when] for row in feedback], RANK_METRICS)
                     for when in ("before", "after")},
        "feedback_counts": {"training_query_groups": sum(item["training_query_groups"] for item in feedback),
                            "judgments": sum(item["judgments"] for item in feedback),
                            "useful": round(sum(item["judgments"] * item["useful_vote_rate"] for item in feedback))},
        "latency_local_cpu_ms": {"median": float(np.median(times)), "p95": float(np.percentile(times, 95)),
                                 "queries": len(times)},
        "folds": results,
    }
