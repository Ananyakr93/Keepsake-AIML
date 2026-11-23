"""Chat query retrieval — called by the Chrome extension right before a
message is sent on a chatbot site."""

import uuid

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from backend.app import db, retrieval
from backend.app.embed import service as embed_service

router = APIRouter(prefix="/api", tags=["queries"])


class ContextIn(BaseModel):
    query: str = Field(min_length=1, max_length=10000)
    site: str | None = None


@router.post("/context")
async def get_context(payload: ContextIn, request: Request):
    user = request.app.state.user
    extension_preview = request.state.client_role == "extension"
    epoch = None
    if extension_preview:
        row = db.query_one("SELECT dataset_consent_since FROM users WHERE id=%s", [user["id"]])
        epoch = row["dataset_consent_since"]
        if not epoch:
            return {"context": "", "count": 0}
    if not user["settings"].get("injection_enabled", True):
        return {"context": "", "count": 0, "injection_enabled": False, "memories": []}
    if not embed_service.ready:
        if embed_service.error is not None or not embed_service.wait_ready(timeout=30):
            raise HTTPException(503, "Embedding model is not available")
    result = await retrieval.retrieve_context(user["id"], payload.query, payload.site, record=not extension_preview)
    if extension_preview:
        row = db.query_one("SELECT dataset_consent_since,injection_enabled FROM users WHERE id=%s", [user["id"]])
        if row["dataset_consent_since"] != epoch or not row["injection_enabled"]:
            return {"context": "", "count": 0}
    return result


@router.get("/queries")
async def list_queries(request: Request, limit: int = 50):
    user = request.app.state.user
    rows = db.query(
        """
        SELECT q.id, q.query_text, q.site, q.context_text, q.created_at,
               coalesce((SELECT array_agg(qm.memory_id ORDER BY qm.rank)
                         FROM query_memories qm WHERE qm.query_id = q.id), '{}')
                   AS retrieved_memory_ids
        FROM chat_queries q WHERE q.user_id = %s
        ORDER BY q.created_at DESC LIMIT %s
        """,
        [user["id"], min(limit, 200)],
    )
    return {"queries": rows}
