"""Read session evidence and produce one versioned example per reviewed session."""

import hashlib
import json
from datetime import datetime, timezone

from backend.app import config, db
from backend.app.dataset_format import SCHEMA_VERSION
from backend.app.ml.session_builder import BrowsingSession, build_sessions

def sessions_for(user_id, *, include_visual: bool = False) -> list[BrowsingSession]:
    with db.repeatable_read() as conn:
        events = conn.execute(
            """
            SELECT b.id, b.url, url_domain(b.url) AS domain, b.title, b.visited_at,
                   b.time_spent_sec, b.session_id, b.client_visit_id, b.dataset_consent_since,
                   a.evidence_id AS reviewed_evidence, a.intent, a.memory_worthy, a.memory_note
            FROM browsing_events b
            LEFT JOIN dataset_annotations a ON a.anchor_event_id = b.id
            WHERE b.user_id = %s AND b.dataset_consent_since IS NOT NULL
            """, [user_id],
        ).fetchall()
        shots = conn.execute(
            """
            SELECT s.id, s.captured_at, s.source_app, s.source_url, s.dataset_consent_since,
                   CASE WHEN %s THEN o.image_embedding::text END AS image_embedding,
                   o.image_embedding IS NOT NULL AS embedded
            FROM screenshots s
            LEFT JOIN observations o ON o.screenshot_id = s.id AND o.user_id = s.user_id
            WHERE s.user_id = %s AND s.dataset_consent_since IS NOT NULL
            ORDER BY s.captured_at, s.id
            """, [include_visual, user_id],
        ).fetchall()
    return build_sessions(events, shots, now=datetime.now(timezone.utc),
                          browser_names=config.BROWSER_TITLE_TAILS)


def review_row(session: BrowsingSession) -> dict:
    pages = {}
    for event in session.events:
        key = event["client_visit_id"]
        if key not in pages:
            pages[key] = {"domain": event["domain"], "title": event["title"], "dwell_sec": 0}
        pages[key]["dwell_sec"] += event["time_spent_sec"]
    return {
        "id": session.anchor["id"], "evidence_id": session.evidence_id,
        "started_at": session.start, "ended_at": session.end,
        "features": session.features(), "pages": list(pages.values()),
        "screenshots": [{k: shot[k] for k in ("id", "captured_at", "embedded")}
                        for shot in session.screenshots],
        "label": {"intent": session.anchor["intent"], "memory_worthy": session.anchor["memory_worthy"],
                  "memory_note": session.anchor["memory_note"]}
                 if session.reviewed else None,
        "needs_review": session.anchor["reviewed_evidence"] is not None and not session.reviewed,
    }


def _mean_embedding(session: BrowsingSession) -> list[float]:
    vectors = [json.loads(shot["image_embedding"]) for shot in session.screenshots
               if shot["image_embedding"]]
    return [sum(values) / len(vectors) for values in zip(*vectors, strict=True)] if vectors else []


def session_features(session, *, include_text=False, include_visual=False):
    return {
        "session_id": str(session.anchor["id"]),
        "started_at": session.start.isoformat(),
        "ended_at": session.end.isoformat(),
        **session.features(),
        "page_titles": list(dict.fromkeys(event["title"] for event in session.events if event["title"]))
                       if include_text else [],
        "image_embedding_mean": _mean_embedding(session) if include_visual else [],
    }


def participant_id(user_id):
    return hashlib.sha256(f"aiml-el:{user_id}".encode()).hexdigest()


def export_rows(user_id, *, include_text: bool = False, include_visual: bool = False) -> list[dict]:
    participant = participant_id(user_id)
    return [
        {
            **session_features(session, include_text=include_text, include_visual=include_visual),
            "schema_version": SCHEMA_VERSION,
            "participant_id": participant,
            "session_id": hashlib.sha256(f"{participant}:{session.anchor['id']}".encode()).hexdigest(),
            "intent": session.anchor["intent"],
            "memory_worthy": int(session.anchor["memory_worthy"]),
        }
        for session in sessions_for(user_id, include_visual=include_visual) if session.reviewed
    ]

