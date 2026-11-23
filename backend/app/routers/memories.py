"""Memory management (view / edit / delete + evidence tracing)."""

import asyncio
import uuid

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from backend.app import db
from backend.app.embed import service as embed_service

router = APIRouter(prefix="/api", tags=["memories"])


@router.get("/memories")
async def list_memories(request: Request, limit: int = 200):
    user = request.app.state.user
    rows = db.query(
        """
        SELECT m.id, m.content, m.importance, u.hit_count, u.last_retrieved_at,
               m.created_at, m.updated_at, m.source, m.archived_at,
               (SELECT count(*) FROM memory_observations mo
                 WHERE mo.memory_id = m.id) AS evidence_count,
               (SELECT coalesce(array_agg(t.topic), '{}') FROM
                 (SELECT topic FROM v_memory_topics vt WHERE vt.memory_id = m.id
                  ORDER BY uses DESC, topic LIMIT 5) t) AS topics
        FROM memories m
        JOIN v_memory_usage u ON u.memory_id = m.id
        WHERE m.user_id = %s
        ORDER BY m.updated_at DESC
        LIMIT %s
        """,
        [user["id"], min(limit, 500)],
    )
    return {"memories": rows}


@router.get("/memories/{mid}/evidence")
async def memory_evidence(mid: uuid.UUID, request: Request):
    user = request.app.state.user
    rows = db.query(
        """
        SELECT observation_id, observation_kind, observation_caption, observed_at,
               contribution, screenshot_id, source_app, domain, url, page_title,
               time_spent_sec
        FROM v_memory_evidence
        WHERE memory_id = %s AND memory_id IN
              (SELECT id FROM memories WHERE user_id = %s)
        ORDER BY observed_at DESC
        """,
        [mid, user["id"]],
    )
    return {"evidence": rows}


class MemoryPatch(BaseModel):
    content: str | None = Field(default=None, min_length=1, max_length=5000)
    importance: float | None = Field(default=None, ge=0, le=1)


@router.patch("/memories/{mid}")
async def patch_memory(mid: uuid.UUID, payload: MemoryPatch, request: Request):
    user = request.app.state.user
    existing = db.query_one(
        "SELECT id, content, importance FROM memories WHERE id = %s AND user_id = %s",
        [mid, user["id"]],
    )
    if existing is None:
        raise HTTPException(404, "Memory not found")

    if payload.content is not None and payload.content.strip() != existing["content"]:
        content = payload.content.strip()
        if not content:
            raise HTTPException(422, "Enter a nonempty memory.")
        if not embed_service.ready:
            raise HTTPException(503, "Start the local embedding model before editing a memory.")
        vector = await embed_service.embed_document(content)
        with db.transaction() as conn:
            updated = conn.execute(
                """UPDATE memories SET content=%s,embedding=%s::vector,importance=coalesce(%s,importance)
                   WHERE id=%s AND user_id=%s AND content=%s RETURNING id""",
                [content, db.vec_lit(vector), payload.importance, mid, user["id"], existing["content"]],
            ).fetchone()
            if not updated:
                raise HTTPException(409, "The memory changed. Refresh and edit it again.")
            conn.execute("DELETE FROM memory_feedback WHERE memory_id=%s", [mid])
        from backend.app.experiments.live_ranking import clear_trained_model
        await asyncio.to_thread(clear_trained_model, user["id"])
    elif payload.importance is not None:
        db.execute(
            "UPDATE memories SET importance = %s WHERE id = %s",
            [payload.importance, mid],
        )
    return {"updated": str(mid)}


@router.delete("/memories/{mid}")
async def delete_memory(mid: uuid.UUID, request: Request):
    user = request.app.state.user
    row = db.query_one(
        "DELETE FROM memories WHERE id = %s AND user_id = %s RETURNING id",
        [mid, user["id"]],
    )
    if row is None:
        raise HTTPException(404, "Memory not found")
    from backend.app.experiments.live_ranking import clear_trained_model
    await asyncio.to_thread(clear_trained_model, user["id"])
    return {"deleted": str(mid)}
