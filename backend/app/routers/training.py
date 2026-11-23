"""Train memories tab: the user types example chat prompts, sees which memories the app
would retrieve, and votes thumbs up/down. The votes become training labels for the
learned ranker (scripts/train_retrieval.py ranker)."""

import asyncio
import uuid

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from backend.app import config, db, retrieval
from backend.app.embed import service as embed_service

router = APIRouter(prefix="/api/training", tags=["training"])

SHOWN_MEMORIES = 20


class PromptIn(BaseModel):
    text: str


class VoteIn(BaseModel):
    memory_id: uuid.UUID
    vote: int  # 1, -1, or 0 to clear


async def _show(request: Request, prompt_id: uuid.UUID, text: str, embedding: str) -> dict:
    user = request.app.state.user
    qvec = [float(x) for x in embedding.strip("[]").split(",")]
    ranked, gate_open = await retrieval.rank_memories(
        user["id"], text, qvec, apply_gate=False, top_k=SHOWN_MEMORIES)
    db.execute("UPDATE training_prompts SET gate_open=%s WHERE id=%s AND gate_open IS NULL", [gate_open, prompt_id])
    votes = {r["memory_id"]: r["vote"] for r in db.query(
        "SELECT memory_id, vote FROM memory_feedback WHERE prompt_id = %s", [prompt_id])}
    return {
        "id": str(prompt_id), "text": text, "gate_open": gate_open,
        "memories": [{"id": str(r["id"]), "content": r["content"],
                      "similarity": round(float(r["sim"]), 3), "importance": r["importance"],
                      "vote": votes.get(r["id"], 0)} for r in ranked],
    }


@router.post("/prompts")
async def add_prompt(payload: PromptIn, request: Request):
    text = payload.text.strip()
    if not text or len(text) > 500:
        raise HTTPException(400, "Prompt must be 1-500 characters")
    qvec = await embed_service.embed_query(text)
    row = db.query_one(
        "INSERT INTO training_prompts (user_id, text, embedding) VALUES (%s, %s, %s::vector) "
        "RETURNING id, embedding::text AS embedding",
        [request.app.state.user["id"], text, db.vec_lit(qvec)])
    return await _show(request, row["id"], text, row["embedding"])


@router.get("/prompts")
async def list_prompts(request: Request):
    rows = db.query(
        "SELECT p.id, p.text, p.created_at, "
        "count(f.*) FILTER (WHERE f.vote = 1) AS up, count(f.*) FILTER (WHERE f.vote = -1) AS down "
        "FROM training_prompts p LEFT JOIN memory_feedback f ON f.prompt_id = p.id "
        "WHERE p.user_id = %s GROUP BY p.id ORDER BY p.created_at DESC LIMIT 100",
        [request.app.state.user["id"]])
    return {"prompts": rows}


@router.get("/judgments")
def export_judgments(request: Request):
    import json
    from backend.app.ml.dataset_builder import participant_id

    user_id = request.app.state.user["id"]
    rows = db.query(
        """SELECT p.id AS prompt_id,p.text,p.gate_open,m.id AS memory_id,m.content,m.importance,
                  1-(p.embedding <=> m.embedding) AS sim,
                  exp(-extract(epoch from (now()-m.updated_at))/86400.0) AS recency,f.vote
           FROM training_prompts p JOIN memory_feedback f ON f.prompt_id=p.id
           JOIN memories m ON m.id=f.memory_id AND m.user_id=p.user_id
           WHERE p.user_id=%s AND p.gate_open IS NOT NULL ORDER BY p.id,m.id""", [user_id],
    )
    if not rows:
        raise HTTPException(409, "Save relevance judgments on a training prompt first.")
    grouped = {}
    for row in rows:
        prompt = grouped.setdefault(row["prompt_id"], {
            "id": str(row["prompt_id"]), "participant_id": participant_id(user_id),
            "query": row["text"], "gate_open": row["gate_open"], "accepted": None, "corrected": None,
            "memories": [],
        })
        prompt["memories"].append({
            "id": str(row["memory_id"]), "content": row["content"], "importance": row["importance"],
            "sim": float(row["sim"]), "recency": float(row["recency"]), "relevant": int(row["vote"] > 0),
        })
    return {"json": json.dumps(list(grouped.values()), ensure_ascii=False, indent=2)}


@router.get("/prompts/{prompt_id}")
async def open_prompt(prompt_id: uuid.UUID, request: Request):
    row = db.query_one(
        "SELECT id, text, embedding::text AS embedding FROM training_prompts "
        "WHERE id = %s AND user_id = %s", [prompt_id, request.app.state.user["id"]])
    if row is None:
        raise HTTPException(404, "Prompt not found")
    return await _show(request, row["id"], row["text"], row["embedding"])


@router.put("/prompts/{prompt_id}/votes")
async def vote(prompt_id: uuid.UUID, payload: VoteIn, request: Request):
    if payload.vote not in (-1, 0, 1):
        raise HTTPException(400, "vote must be -1, 0 or 1")
    owned = db.query_one("SELECT 1 FROM training_prompts WHERE id = %s AND user_id = %s",
                         [prompt_id, request.app.state.user["id"]])
    if owned is None:
        raise HTTPException(404, "Prompt not found")
    previous = db.query_one("SELECT vote FROM memory_feedback WHERE prompt_id=%s AND memory_id=%s",
                            [prompt_id, payload.memory_id])
    if payload.vote == 0:
        db.execute("DELETE FROM memory_feedback WHERE prompt_id = %s AND memory_id = %s",
                   [prompt_id, payload.memory_id])
    else:
        row = db.query_one(
            "INSERT INTO memory_feedback (prompt_id, memory_id, vote) "
            "SELECT p.id, m.id, %s FROM training_prompts p JOIN memories m ON m.user_id = p.user_id "
            "WHERE p.id = %s AND m.id = %s AND m.user_id = %s "
            "ON CONFLICT (prompt_id, memory_id) DO UPDATE SET vote = EXCLUDED.vote, voted_at = now() "
            "RETURNING memory_id",
            [payload.vote, prompt_id, payload.memory_id, request.app.state.user["id"]])
        if row is None:
            raise HTTPException(404, "Memory or prompt not found")
    if config.RANKER == "xgboost":
        from backend.app.experiments.live_ranking import clear_trained_model
        if previous is not None and previous["vote"] != payload.vote:
            await asyncio.to_thread(clear_trained_model, request.app.state.user["id"])
    return {"ok": True}


@router.delete("/prompts/{prompt_id}")
async def delete_prompt(prompt_id: uuid.UUID, request: Request):
    row = db.query_one("DELETE FROM training_prompts WHERE id = %s AND user_id = %s RETURNING id",
                       [prompt_id, request.app.state.user["id"]])
    if row is None:
        raise HTTPException(404, "Prompt not found")
    from backend.app.experiments.live_ranking import clear_trained_model
    await asyncio.to_thread(clear_trained_model, request.app.state.user["id"])
    return {"deleted": str(prompt_id)}


@router.post("/retrain")
async def retrain(request: Request):
    """Finish button / leaving the tab: refit the learned ranker from the votes, no restart needed."""
    if config.RANKER == "xgboost":
        from backend.app.experiments.live_ranking import refit
        return await asyncio.to_thread(refit, request.app.state.user["id"])
    return {"retrained": False, "reason": "Select AIML_RANKER=xgboost to validate feedback updates."}
