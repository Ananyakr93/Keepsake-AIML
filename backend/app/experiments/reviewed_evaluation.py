"""Evaluate reviewed consented sessions and independent relevance judgments."""

import hashlib
import json
import math
from pathlib import Path

from backend.app.dataset_format import read_datasets, split_dataset
from backend.app.experiments import ranking, sessions


def _optional_rate(items, key):
    values = [item[key] for item in items if item.get(key) is not None]
    return {"rate": sum(values) / len(values) if values else None, "observed": len(values)}


def _reviewed_groups(path):
    prompts = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("Relevance judgments must be a nonempty JSON array.")
    result, ids = [], set()
    for prompt in prompts:
        if not isinstance(prompt, dict) or any(not isinstance(prompt.get(key), str) or not prompt[key]
                                               for key in ("id", "participant_id", "query")):
            raise ValueError("Each prompt requires an id, participant_id and query.")
        if prompt["id"] in ids:
            raise ValueError("Prompt ids must be unique.")
        ids.add(prompt["id"])
        if type(prompt.get("gate_open")) is not bool:
            raise ValueError("Record the live gate_open decision on every prompt.")
        for key in ("accepted", "corrected"):
            if prompt.get(key) is not None and type(prompt[key]) is not bool:
                raise ValueError(f"{key} must be true, false or null.")
        candidates = prompt.get("memories")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("Each prompt requires a judged memory pool.")
        memory_ids = set()
        for row in candidates:
            if (not isinstance(row, dict) or not isinstance(row.get("id"), str)
                    or not isinstance(row.get("content"), str) or type(row.get("relevant")) is not int
                    or row["relevant"] not in {0, 1} or row["id"] in memory_ids):
                raise ValueError("Each judged memory requires a unique id within the prompt, content, and relevant=0 or 1.")
            memory_ids.add(row["id"])
            for key in ("sim", "importance", "recency"):
                value = row.get(key)
                if type(value) not in {float, int} or not math.isfinite(value) or not -1 <= value <= 1:
                    raise ValueError(f"Invalid live {key} feature.")
        candidates = sorted(candidates, key=lambda row: (-row["sim"], row["id"]))[:ranking.CANDIDATES]
        result.append((prompt, candidates if prompt["gate_open"] else []))
    return result


def _retrieval(partitions, judgments, seed):
    people = {row["participant_id"]: key for key, rows in partitions.items() for row in rows}
    if sum(len({row["participant_id"] for row in rows}) for rows in partitions.values()) != len(people):
        raise ValueError("Reviewed relevance evaluation requires separate participant splits.")
    groups = {"train": [], "validation": [], "test": []}
    for prompt, candidates in _reviewed_groups(judgments):
        partition = people.get(prompt["participant_id"])
        if partition is None:
            raise ValueError("A relevance participant is missing from the reviewed session input.")
        groups[partition].append((prompt, candidates))
    if any(not group for group in groups.values()):
        raise ValueError("Provide independent relevance judgments for each participant partition.")

    def train_groups(group):
        output = []
        for prompt, candidates in group:
            labels = [row["relevant"] for row in candidates]
            if len(set(labels)) == 2:
                output.append((ranking.features(prompt["query"], candidates), labels))
        return output

    base = train_groups(groups["train"])
    model = ranking.fit_ranker(base, seed=seed)
    feedback = train_groups(groups["validation"])
    test = groups["test"]
    report = {
        "queries": len(test), "source": "live cosine features with recorded gate decisions",
        "judgments_sha256": hashlib.sha256(judgments.read_bytes()).hexdigest(),
        "methods": {
            "vector": ranking._metrics(test, "vector"),
            "weighted": ranking._metrics(test, "weighted"),
            "xgboost": ranking._metrics(test, "xgboost", model),
        },
        "feedback_acceptance": _optional_rate([prompt for prompt, _ in test], "accepted"),
        "memory_correction": _optional_rate([prompt for prompt, _ in test], "corrected"),
    }
    if feedback:
        adjusted = ranking.fit_ranker(base, feedback, seed)
        report["after_validation_feedback"] = ranking._metrics(test, "xgboost", adjusted)
    return report


def run_reviewed_evaluation(inputs: list[Path], judgments: Path | None, seed: int):
    sources = [(str(path), path.read_text(encoding="utf-8-sig")) for path in inputs]
    rows = read_datasets(sources)
    partitions, method = split_dataset(rows, seed)
    _, metrics = sessions.train_sessions(partitions, method, seed)
    metrics.update(source="reviewed", seed=seed, split=method,
                   inputs_sha256=[hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs])
    if judgments:
        metrics["retrieval"] = _retrieval(partitions, judgments, seed)
    return metrics


def render_summary(report):
    lines = ["# AIML-EL reviewed evaluation", "",
             f"Split **{report['split']}** · Dataset provenance declared reviewed", "",
             f"Selected intent classifier **{report['intent']['selected']}** achieved "
             f"held-out macro F1 **{report['intent']['test']['f1']:.3f}**.", "",
             "| Admission variant | Held-out F1 | ROC-AUC | Share admitted |",
             "|---|---:|---:|---:|"]
    for name, variant in report["admission"]["variants"].items():
        score = variant["test"]
        auc = f"{score['roc_auc']:.3f}" if score["roc_auc"] is not None else "—"
        lines.append(f"| {name} | {score['f1']:.3f} | {auc} | {score['positive_prediction_rate']:.1%} |")
    retrieval = report.get("retrieval")
    if retrieval:
        lines += ["", "## Retrieval and user observations", "",
                  "| Method | NDCG@5 | Recall@5 |", "|---|---:|---:|"]
        for name, values in retrieval["methods"].items():
            ndcg, recall = values["ndcg_at_5"], values["recall_at_5"]
            lines.append(f"| {name} | {ndcg if ndcg is not None else '—'} | {recall if recall is not None else '—'} |")
        for key in ("feedback_acceptance", "memory_correction"):
            score = retrieval[key]
            label = f"{score['rate']:.1%}" if score["rate"] is not None else "unmeasured"
            lines.append(f"- {key.replace('_', ' ').capitalize()} · {label} across {score['observed']} recorded observations.")
    else:
        lines += ["", "Retrieval and user feedback require an independent relevance judgment file."]
    lines += ["", "See [metrics.json](metrics.json) for split composition and detailed scores.", ""]
    return "\n".join(lines)
