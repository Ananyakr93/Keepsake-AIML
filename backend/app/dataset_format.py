"""Versioned CSV schema and deterministic dataset partitions."""

import csv
import io
import json
import math
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

SCHEMA_VERSION = "1"
IMAGE_EMBED_DIM = 1024
INTENTS = ("research", "learning", "purchase", "planning", "entertainment", "casual", "task")
FEATURES = (
    "duration_sec", "dwell_sec", "visit_count", "page_count", "domain_count",
    "repeat_visit_count", "screenshot_count",
)
COLUMNS = (
    "schema_version", "participant_id", "session_id", "started_at", "ended_at",
    "intent", "memory_worthy", *FEATURES, "page_titles", "image_embedding_mean",
)
PARTITIONS = ("train", "validation", "test")
PSEUDONYM = re.compile(r"[a-f0-9]{64}")


def dataset_csv(rows: list[dict]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=COLUMNS)
    writer.writeheader()
    for row in rows:
        writer.writerow({**row, "page_titles": json.dumps(row["page_titles"], ensure_ascii=False),
                         "image_embedding_mean": json.dumps(row["image_embedding_mean"])})
    return buffer.getvalue()


def _timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.utcoffset() is None:
        raise ValueError("timestamps must specify a time zone")
    return parsed.astimezone(timezone.utc)


def _parse_row(row: dict) -> dict:
    if None in row or any(value is None for value in row.values()):
        raise ValueError("row has the wrong number of fields")
    if row["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported schema version")
    if not all(PSEUDONYM.fullmatch(row[key]) for key in ("participant_id", "session_id")):
        raise ValueError("invalid participant or session identifier")
    if row["intent"] not in INTENTS or row["memory_worthy"] not in {"0", "1"}:
        raise ValueError("invalid intent or memory-worthiness label")
    start, end = _timestamp(row["started_at"]), _timestamp(row["ended_at"])
    if end < start:
        raise ValueError("end precedes start")
    features = {name: int(row[name]) for name in FEATURES}
    if any(value < 0 for value in features.values()):
        raise ValueError("features must be non-negative integers")
    if features["duration_sec"] != int((end - start).total_seconds()):
        raise ValueError("duration does not match the session interval")
    if not 1 <= features["page_count"] <= features["visit_count"]:
        raise ValueError("invalid visit or page count")
    if features["repeat_visit_count"] != features["visit_count"] - features["page_count"]:
        raise ValueError("invalid repeat visit count")
    titles = json.loads(row["page_titles"])
    if not isinstance(titles, list) or not all(isinstance(title, str) for title in titles):
        raise ValueError("page_titles must be a JSON list of strings")
    image = json.loads(row["image_embedding_mean"])
    if not isinstance(image, list) or len(image) not in {0, IMAGE_EMBED_DIM}:
        raise ValueError("image_embedding_mean must be empty or contain 1024 numbers")
    if not all(type(value) in {float, int} and math.isfinite(value) for value in image):
        raise ValueError("image embedding must contain finite numbers")
    if image and not features["screenshot_count"]:
        raise ValueError("a visual embedding requires a matched screenshot")
    return {**row, **features, "started_at": start.isoformat(), "ended_at": end.isoformat(),
            "memory_worthy": int(row["memory_worthy"]), "page_titles": titles,
            "image_embedding_mean": image}


def read_datasets(sources: list[tuple[str, str]]) -> list[dict]:
    rows = []
    seen = set()
    csv.field_size_limit(2 * 1024 * 1024)
    for name, contents in sources:
        reader = csv.DictReader(io.StringIO(contents, newline=""), strict=True)
        try:
            if tuple(reader.fieldnames or ()) != COLUMNS:
                raise ValueError("CSV header does not match the export schema")
            for raw in reader:
                row = _parse_row(raw)
                if row["session_id"] in seen:
                    raise ValueError("duplicate session identifier")
                seen.add(row["session_id"])
                rows.append(row)
        except (ValueError, csv.Error, OverflowError) as exc:
            raise ValueError(f"{name}, line {reader.line_num}: {exc}") from exc
    if not rows:
        raise ValueError("No labeled sessions to split")
    return rows


def _chronological_groups(rows: list[dict]) -> list[list[dict]]:
    by_day = defaultdict(list)
    for row in rows:
        by_day[_timestamp(row["started_at"]).date()].append(row)
    groups = []
    previous_end = None
    for _, day_rows in sorted(by_day.items()):
        start = min(_timestamp(row["started_at"]) for row in day_rows)
        end = max(_timestamp(row["ended_at"]) for row in day_rows)
        if previous_end is not None and start - previous_end < timedelta(minutes=30):
            groups[-1].extend(day_rows)
            previous_end = max(previous_end, end)
        else:
            groups.append(day_rows)
            previous_end = end
    return groups


def split_dataset(rows: list[dict], seed: int) -> tuple[dict[str, list[dict]], str]:
    by_participant = defaultdict(list)
    for row in sorted(rows, key=lambda entry: (entry["started_at"], entry["session_id"])):
        by_participant[row["participant_id"]].append(row)

    if len(by_participant) >= 3:
        participants = sorted(by_participant)
        random.Random(seed).shuffle(participants)
        groups = [by_participant[key] for key in participants]
        method = "participant_holdout"
    elif len(by_participant) == 1:
        groups = _chronological_groups(next(iter(by_participant.values())))
        method = "single_participant_chronological"
    else:
        raise ValueError("Participant holdout needs three participants; add another consented participant")

    if len(groups) < 3:
        raise ValueError("Chronological splitting needs at least three separate UTC day groups")
    validation = max(1, round(len(groups) * 0.15))
    test = max(1, round(len(groups) * 0.15))
    train = len(groups) - validation - test
    subsets = (groups[:train], groups[train:train + validation], groups[train + validation:])
    return {name: [row for group in subset for row in group]
            for name, subset in zip(PARTITIONS, subsets, strict=True)}, method


def summary(rows: list[dict]) -> dict:
    intents = Counter(row["intent"] for row in rows)
    labels = Counter(row["memory_worthy"] for row in rows)
    return {
        "sessions": len(rows),
        "participants": len({row["participant_id"] for row in rows}),
        "intent_counts": {intent: intents[intent] for intent in INTENTS},
        "memory_worthy_counts": {str(label): labels[label] for label in (0, 1)},
    }
