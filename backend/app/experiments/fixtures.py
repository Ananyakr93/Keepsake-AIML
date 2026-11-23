"""Deterministic fictional browsing activity and query relevance judgments."""

import hashlib
from datetime import datetime, timedelta, timezone

import numpy as np

from backend.app.dataset_format import IMAGE_EMBED_DIM, INTENTS, SCHEMA_VERSION

TOPICS = {
    "laptop": ("battery life and weight", "budget and shortlisted models", "keyboard and screen"),
    "python": ("testing and debugging", "dataframe joins and missing values", "async tasks and timeout"),
    "cycling": ("commuter routes and road safety", "helmet and repair kit", "training distance and hills"),
    "travel": ("train tickets and dates", "packing and luggage limits", "hotel and neighborhood"),
    "cooking": ("ingredient substitutions", "oven temperature and timing", "menu for a gathering"),
    "photography": ("low light exposure", "lens choice and focal length", "portrait editing"),
    "gardening": ("watering schedule", "soil and containers", "pest prevention"),
    "music": ("live venue and dates", "practice playlist", "recording microphone"),
}
INTENT_WORDS = {
    "research": ("independent evidence and surveys", "published findings and alternatives"),
    "learning": ("hands-on tutorial and exercises", "worked examples and first lesson"),
    "purchase": ("prices and seller comparison", "stock and shipping policy"),
    "planning": ("schedule and checklist", "calendar and itinerary"),
    "entertainment": ("highlights and funny clips", "interview and favorite moments"),
    "casual": ("quick answer and top results", "latest updates and short posts"),
    "task": ("work queue and instructions", "check status and submit changes"),
}
SIGNALS = (
    ("saved shortlist with notes", "decision log and preferences"),
    ("related reference", "overview and examples"),
    ("quick glance", "general results"),
)


def _id(text):
    return hashlib.sha256(text.encode()).hexdigest()


def synthetic_sessions(seed=42):
    rng = np.random.default_rng(seed)
    topic_names = list(TOPICS)
    visual_signals = rng.normal(0, 0.13, (len(TOPICS) + len(SIGNALS), IMAGE_EMBED_DIM))
    origin = datetime(2026, 7, 1, 9, tzinfo=timezone.utc)
    rows = []
    for person in range(28):
        preferences = rng.choice(len(TOPICS), 3, replace=False)
        ordering = np.concatenate([rng.permutation(len(INTENTS)) for _ in range(4)])
        for index, intent_index in enumerate(ordering):
            intent = INTENTS[int(intent_index)]
            topic_index = int(rng.choice(preferences) if rng.random() < 0.72 else rng.integers(len(TOPICS)))
            topic = topic_names[topic_index]
            detail = str(rng.choice(TOPICS[topic]))
            signal = int(rng.choice(len(SIGNALS), p=[0.34, 0.37, 0.29]))
            visits = max(1, int(rng.poisson(2 + 0.35 * (2 - signal))))
            pages = max(1, visits - int(rng.integers(0, 3)))
            dwell = int(np.clip(rng.lognormal(4.75 + (2 - signal) * 0.24, 0.8), 10, 1400))
            duration = dwell + int(rng.integers(0, 500))
            probability = [0.8, 0.51, 0.21][signal] + (0.1 if intent in {"learning", "planning"} else 0)
            screenshots = int(rng.integers(1, 4)) if rng.random() < 0.70 else 0
            vector = visual_signals[topic_index] + visual_signals[len(TOPICS) + signal]
            if screenshots:
                vector = vector + rng.normal(0, 0.17, IMAGE_EMBED_DIM)
                vector = (vector / np.linalg.norm(vector)).round(5).tolist()
            start = origin + timedelta(days=index // 2, hours=7 * (index % 2), minutes=person)
            rows.append({
                "schema_version": SCHEMA_VERSION, "participant_id": _id(f"synthetic-participant-{person}"),
                "session_id": _id(f"synthetic-session-{person}-{index}"),
                "started_at": start.isoformat(), "ended_at": (start + timedelta(seconds=duration)).isoformat(),
                "intent": intent, "memory_worthy": int(rng.random() < probability),
                "duration_sec": duration, "dwell_sec": dwell, "visit_count": visits,
                "page_count": pages, "domain_count": min(pages, int(rng.integers(1, 4))),
                "repeat_visit_count": visits - pages, "screenshot_count": screenshots,
                "page_titles": [f"{topic.title()} {detail} | {rng.choice(INTENT_WORDS[intent])}",
                                str(rng.choice(SIGNALS[signal])) if rng.random() < 0.75 else "notes and resources"],
                "image_embedding_mean": vector if screenshots else [],
            })
    return rows


def synthetic_relevance(rows, seed, split):
    """Each fictional prompt has its own candidate pool and explicit oracle."""
    rng = np.random.default_rng(seed)
    prompts = []
    people = sorted({item["participant_id"] for item in rows})
    for person in people:
        for index in range(4):
            topic = list(TOPICS)[int(rng.integers(len(TOPICS)))]
            facet = TOPICS[topic][index % 3]
            query = f"What did I keep about {topic} {facet}?"
            memories = []
            for item, facets in TOPICS.items():
                for detail in facets:
                    for scope in ("personal", "general"):
                        condition = detail == facet and item == topic
                        content = (f"My {item} notes cover {detail}. {rng.choice(SIGNALS[0])}." if scope == "personal"
                                   else f"General information on {item} {detail}. {rng.choice(SIGNALS[1])}.")
                        # Broad pages can contain query terms without answering the personal question.
                        if scope == "general" and rng.random() < 0.35:
                            content += f" Related search: {topic} {facet}."
                        memories.append({
                            "id": f"{item}-{facets.index(detail)}-{scope}", "content": content,
                            "importance": round(float(rng.uniform(0.12, 0.95)), 4),
                            "recency": round(float(rng.uniform(0.05, 1)), 4),
                            "relevant": int(condition and scope == "personal"),
                        })
            prompts.append({"id": _id(f"{person}:{split}:{index}"), "query": query, "memories": memories})
    return prompts
