"""Local dataset review API, called by the Electron main process."""

from ipaddress import ip_address
from typing import Literal
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field, StrictBool

from backend.app import dataset_format, db
from backend.app.ml import dataset_builder

Intent = Literal["research", "learning", "purchase", "planning", "entertainment", "casual", "task"]


def require_local_client(request: Request):
    try:
        hostname = urlsplit(f"//{request.headers.get('host', '')}").hostname
        loopback = bool(request.client and ip_address(request.client.host).is_loopback)
    except ValueError:
        loopback = False
        hostname = None
    if (not loopback or hostname not in {"localhost", "127.0.0.1", "::1"}
            or "origin" in request.headers or "sec-fetch-site" in request.headers):
        raise HTTPException(403, "Dataset operations require a local desktop client")


router = APIRouter(prefix="/api/dataset", tags=["dataset"], dependencies=[Depends(require_local_client)])


class CollectionPatch(BaseModel):
    enabled: StrictBool


class Annotation(BaseModel):
    evidence_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    intent: Intent
    memory_worthy: StrictBool
    memory_note: str | None = Field(default=None, max_length=1000)


@router.get("/sessions")
def sessions(request: Request, offset: int = Query(0, ge=0), limit: int = Query(20, ge=1, le=100)):
    user_id = request.app.state.user["id"]
    user = db.query_one("SELECT dataset_consent_since FROM users WHERE id = %s", [user_id])
    available = dataset_builder.sessions_for(user_id)
    return {
        "collection_enabled": user["dataset_consent_since"] is not None,
        "total": len(available),
        "reviewed": sum(session.reviewed for session in available),
        "sessions": [dataset_builder.review_row(session) for session in available[offset:offset + limit]],
    }


@router.post("/collection")
def set_collection(payload: CollectionPatch, request: Request):
    user_id = request.app.state.user["id"]
    db.execute(
        """
        UPDATE users
        SET dataset_consent_since = CASE WHEN %s THEN coalesce(dataset_consent_since, now()) END
        WHERE id = %s
        """, [payload.enabled, user_id],
    )
    return {"collection_enabled": payload.enabled}


@router.post("/sessions/{anchor_id}")
def label_session(anchor_id: int, payload: Annotation, request: Request):
    user_id = request.app.state.user["id"]
    note = payload.memory_note.strip() or None if payload.memory_note else None
    if not payload.memory_worthy:
        note = None
    with db.transaction() as conn:
        conn.execute("SELECT id FROM users WHERE id=%s FOR UPDATE", [user_id])
        session = next((item for item in dataset_builder.sessions_for(user_id)
                        if item.anchor["id"] == anchor_id), None)
        if session is None or session.evidence_id != payload.evidence_id:
            raise HTTPException(409, "Session evidence changed. Refresh and review it again.")
        row = conn.execute(
            """
            INSERT INTO dataset_annotations (anchor_event_id, evidence_id, intent, memory_worthy,memory_note)
            SELECT id, %s, %s, %s,%s FROM browsing_events WHERE id = %s AND user_id = %s
            ON CONFLICT (anchor_event_id) DO UPDATE SET evidence_id=excluded.evidence_id,
                intent=excluded.intent, memory_worthy=excluded.memory_worthy,
                memory_note=excluded.memory_note, labeled_at=now()
            RETURNING anchor_event_id
            """, [payload.evidence_id, payload.intent, payload.memory_worthy, note, anchor_id, user_id],
        ).fetchone()
        if row is None:
            raise HTTPException(409, "Session evidence was deleted. Refresh the queue.")
        if not payload.memory_worthy or (session.reviewed and (session.anchor["memory_note"] != note
                                                               or session.anchor["intent"] != payload.intent)):
            from backend.app.experiments.live import revoke_memory
            revoke_memory(conn, user_id, anchor_id)
    return {"saved": True}


@router.delete("/sessions/{anchor_id}")
def remove_label(anchor_id: int, request: Request):
    db.execute(
        """
        DELETE FROM dataset_annotations a USING browsing_events e
        WHERE a.anchor_event_id = e.id AND e.id = %s AND e.user_id = %s
        """, [anchor_id, request.app.state.user["id"]],
    )
    return {"removed": True}


class ExportOptions(BaseModel):
    include_text: StrictBool = False
    include_visual: StrictBool = False


@router.post("/export")
def export(request: Request, payload: ExportOptions):
    rows = dataset_builder.export_rows(request.app.state.user["id"], **payload.model_dump())
    if not rows:
        raise HTTPException(409, "Review and label at least one session before exporting.")
    return {"rows": len(rows), "csv": dataset_format.dataset_csv(rows)}


@router.get("/analysis/{anchor_id}")
def analyze_session(anchor_id: int, request: Request):
    from backend.app.experiments.live import analyze

    try:
        return analyze(request.app.state.user["id"], anchor_id)
    except KeyError as exc:
        raise HTTPException(404, exc.args[0]) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
