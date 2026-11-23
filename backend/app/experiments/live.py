"""Apply reviewed session models to local browsing before creating memories."""

import asyncio

from backend.app import config, db
from backend.app.experiments import sessions
from backend.app.experiments.workspace import _read_artifact
from backend.app.ml.dataset_builder import session_features, sessions_for


def reviewed_models():
    path = config.SESSION_MODELS_PATH
    if not path.is_file():
        return None
    bundle = _read_artifact(path, path.stat().st_mtime_ns)
    if bundle["source"] != "reviewed" or bundle["format"] != 1:
        return None
    return bundle["models"]


def analyze(user_id, anchor_id):
    model = reviewed_models()
    if model is None:
        raise ValueError("Train an artifact from reviewed sessions to enable live analysis.")
    history = sessions_for(user_id, include_visual=True)
    session = next((item for item in history if item.anchor["id"] == anchor_id), None)
    if session is None:
        raise KeyError("Session not found or still open.")
    current = session_features(session, include_text=True, include_visual=True)
    previous = [session_features(item, include_text=True) for item in history if item.start <= session.start]
    return {"prediction": sessions.predict_session(model, current, previous),
            "interests": sessions.interest_scores(model, previous, at_time=session.end),
            "trends": sessions.interest_trends(model, previous)}


def _closed_sessions(user_id):
    row = db.query_one("SELECT dataset_consent_since FROM users WHERE id=%s", [user_id])
    if not row or row["dataset_consent_since"] is None:
        return []
    consent = row["dataset_consent_since"]
    return [item for item in sessions_for(user_id, include_visual=True)
            if item.anchor["dataset_consent_since"] == consent]


def _content(result, session):
    if session.reviewed and session.anchor["memory_note"]:
        return session.anchor["memory_note"]
    titles = list(dict.fromkeys(event["title"] for event in session.events if event["title"]))
    evidence = " | ".join(titles[:5]) or " | ".join(
        dict.fromkeys(event["domain"] for event in session.events if event["domain"]))
    return f"{result['intent'].capitalize()} session. {evidence}"[:1000]


def revoke_memory(conn, user_id, anchor):
    record = conn.execute(
        "UPDATE session_memory_links SET revoked_at=COALESCE(revoked_at,now()) "
        "WHERE anchor_event_id=%s RETURNING memory_id", [anchor],
    ).fetchone()
    if record and record["memory_id"]:
        conn.execute("UPDATE memories SET archived_at=COALESCE(archived_at,now()) WHERE id=%s AND user_id=%s",
                     [record["memory_id"], user_id])


def reconcile_memory(user_id, session):
    anchor = session.anchor["id"]
    with db.transaction() as conn:
        conn.execute("SELECT id FROM users WHERE id=%s FOR UPDATE", [user_id])
        link = conn.execute("SELECT memory_id,evidence_id,revoked_at FROM session_memory_links WHERE anchor_event_id=%s",
                            [anchor]).fetchone()
        if link is None:
            return False
        if link["memory_id"] is None:  # honor a user's manual deletion
            return True
        if link["evidence_id"] is None:
            conn.execute("UPDATE session_memory_links SET evidence_id=%s WHERE anchor_event_id=%s",
                         [session.evidence_id, anchor])
        changed = link["evidence_id"] not in {None, session.evidence_id}
        if changed or (session.reviewed and not session.anchor["memory_worthy"]):
            revoke_memory(conn, user_id, anchor)
        if (changed or link["revoked_at"] is not None) and session.reviewed and session.anchor["memory_worthy"]:
            conn.execute("DELETE FROM session_memory_links WHERE anchor_event_id=%s", [anchor])
            return False
        return True


def _persist(user_id, session, result, content, vector, probability):
    ids = [event["id"] for event in session.events]
    anchor = session.anchor["id"]
    with db.transaction() as conn:
        row = conn.execute("SELECT dataset_consent_since FROM users WHERE id=%s FOR UPDATE", [user_id]).fetchone()
        if row is None or row["dataset_consent_since"] != session.anchor["dataset_consent_since"]:
            return False
        source = conn.execute(
            "SELECT id FROM browsing_events WHERE user_id=%s AND id=ANY(%s) FOR KEY SHARE", [user_id, ids],
        ).fetchall()
        if len(source) != len(ids):
            return False
        processed = conn.execute(
            "SELECT 1 FROM session_memory_links WHERE anchor_event_id=%s "
            "UNION ALL SELECT 1 FROM memory_observations mo JOIN observations o ON o.id=mo.observation_id "
            "JOIN memories m ON m.id=mo.memory_id "
            "WHERE m.archived_at IS NULL AND o.user_id=%s AND o.browsing_event_id=ANY(%s) LIMIT 1", [anchor, user_id, ids],
        ).fetchone()
        if processed:
            return False
        label = conn.execute("SELECT intent,memory_worthy,evidence_id,memory_note FROM dataset_annotations WHERE anchor_event_id=%s",
                             [anchor]).fetchone()
        if label and label["evidence_id"] == session.evidence_id:
            if not label["memory_worthy"] or label["intent"] != result["intent"]:
                return False
            if label["memory_note"] != session.anchor["memory_note"]:
                return False
        record = conn.execute(
            "INSERT INTO memories (user_id,content,embedding,importance) VALUES (%s,%s,%s::vector,%s) RETURNING id",
            [user_id, content, db.vec_lit(vector), probability],
        ).fetchone()
        for event in session.events:
            conn.execute(
                "INSERT INTO observations (user_id,browsing_event_id,caption,observed_at) VALUES (%s,%s,%s,%s) "
                "ON CONFLICT (browsing_event_id) WHERE browsing_event_id IS NOT NULL DO NOTHING",
                [user_id, event["id"], event["title"], event["visited_at"]],
            )
        conn.execute(
            "INSERT INTO memory_observations (memory_id,observation_id) "
            "SELECT %s,id FROM observations WHERE user_id=%s "
            "AND (browsing_event_id=ANY(%s) OR screenshot_id=ANY(%s::uuid[]))",
            [record["id"], user_id, ids, [shot["id"] for shot in session.screenshots]],
        )
        conn.execute(
            "INSERT INTO session_memory_links (anchor_event_id,memory_id,admission_score,intent,evidence_id) VALUES (%s,%s,%s,%s,%s)",
            [anchor, record["id"], probability, result["intent"], session.evidence_id],
        )
    return True


async def consolidate_sessions(user_id):
    from backend.app import lifecycle
    from backend.app.embed import service

    model = await asyncio.to_thread(reviewed_models)
    if model is None:
        raise ValueError("Session admission requires a trained artifact from reviewed sessions.")
    stats = {"memories_created": 0, "sessions_skipped": 0, "already_processed": 0, "archived": 0}
    if not service.ready:
        return {**stats, "reason": "Embedding model is still unavailable."}
    history = await asyncio.to_thread(_closed_sessions, user_id)
    processed = await asyncio.to_thread(db.query,
        "SELECT l.anchor_event_id FROM session_memory_links l JOIN browsing_events e ON e.id=l.anchor_event_id "
        "WHERE e.user_id=%s", [user_id])
    anchors = {row["anchor_event_id"] for row in processed}
    feature_history = [session_features(item, include_text=True) for item in history]
    for item in history:
        if item.anchor["id"] in anchors and await asyncio.to_thread(reconcile_memory, user_id, item):
            stats["already_processed"] += 1
            continue
        row = session_features(item, include_text=True, include_visual=True)
        result = await asyncio.to_thread(sessions.predict_session, model, row, feature_history)
        admission = result["admission"][result["selected"]]
        remember = item.anchor["memory_worthy"] if item.reviewed else admission["remember"]
        if not remember:
            stats["sessions_skipped"] += 1
            continue
        if item.reviewed:
            result["intent"] = item.anchor["intent"]
        content = _content(result, item)
        vector = await service.embed_document(content)
        if await asyncio.to_thread(_persist, user_id, item, result, content, vector, admission["probability"]):
            stats["memories_created"] += 1
    if config.DECAY != "off":
        stats["archived"] = await asyncio.to_thread(lifecycle.archive_stale_memories, user_id)
    return stats
