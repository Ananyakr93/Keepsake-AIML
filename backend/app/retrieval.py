"""Retrieve vector candidates, rerank them, and build the context block the
Chrome extension appends after the user's message."""

import asyncio
import logging
import uuid

from backend.app import config, db, ml
from backend.app.embed import service as embed_service

log = logging.getLogger("aiml-el.retrieval")

CONTEXT_HEADER = "[Personal memory context — retrieved automatically by AIML-EL]"
CONTEXT_FOOTER = "[End of memory context]"


async def rank_memories(user_id: uuid.UUID, query: str, qvec: list[float],
                        *, apply_gate: bool = True, top_k: int | None = None) -> tuple[list[dict], bool]:
    """Candidates by cosine distance, then the configured ranker. -> (ranked top-K, gate_open).
    With apply_gate=False the gate is skipped (the Train memories tab shows memories regardless)."""
    qlit = db.vec_lit(qvec)
    candidates = await asyncio.to_thread(db.query,
        """
        SELECT id, content, importance,
               1 - (embedding <=> %s::vector) AS sim,
               exp(-extract(epoch from (now() - updated_at)) / 86400.0) AS recency
        FROM memories
        WHERE user_id = %s AND archived_at IS NULL
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        [qlit, user_id, qlit, config.RETRIEVAL_CANDIDATES],
    )
    gate_open = ml.get_gate().should_inject(query, qvec, candidates)
    if not gate_open and apply_gate:
        return [], False
    ranked = await asyncio.to_thread(ml.get_ranker().rank, query, candidates)
    return ranked[:top_k or config.RETRIEVAL_TOP_K], gate_open


async def retrieve_context(user_id: uuid.UUID, query: str, site: str | None, *, record: bool = True) -> dict:
    qvec = await embed_service.embed_query(query)
    qlit = db.vec_lit(qvec)
    rows, _ = await rank_memories(user_id, query, qvec)
    used = rows
    confident = bool(used)

    if used:
        body = "\n".join(f"- {r['content']}" for r in used)
        context_text = f"{CONTEXT_HEADER}\n{body}\n{CONTEXT_FOOTER}"
    else:
        context_text = ""

    if record:
        with db.transaction() as conn:
            qrow = conn.execute(
                """INSERT INTO chat_queries (user_id,query_text,query_embedding,site,context_text)
                   VALUES (%s,%s,%s::vector,%s,%s) RETURNING id""",
                [user_id, query, qlit, site, context_text],
            ).fetchone()
            for rank, row in enumerate(used, start=1):
                conn.execute(
                    """INSERT INTO query_memories (query_id,memory_id,rank,similarity)
                       SELECT %s,id,%s,%s FROM memories WHERE id=%s""",
                    [qrow["id"], rank, float(row["sim"]), row["id"]],
                )
    return {
        "context": context_text,
        "count": len(used),
        "confident": confident,
        "memories": [
            {"id": str(r["id"]), "content": r["content"],
             "similarity": round(float(r["sim"]), 4), "importance": r["importance"]}
            for r in used
        ],
    }
