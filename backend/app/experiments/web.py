"""Loopback-only dashboard for the trained synthetic workspace."""

from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field, field_validator

from backend.app.experiments.workspace import DemoWorkspace


class Vote(BaseModel):
    prompt: str
    memory: str
    vote: Literal[0, 1] | None


class Retrain(BaseModel):
    scripted: bool = False


class Label(BaseModel):
    intent: Literal["research", "learning", "purchase", "planning", "entertainment", "casual", "task"]
    memory_worthy: bool


class MemoryUpdate(BaseModel):
    content: str | None = Field(default=None, min_length=1, max_length=2000)
    archived: bool | None = None

    @field_validator("content")
    @classmethod
    def nonblank(cls, value):
        if value is not None and not value.strip():
            raise ValueError("Memory text is required.")
        return value.strip() if value is not None else None


class Search(BaseModel):
    query: str = Field(min_length=1, max_length=500)


class Preference(BaseModel):
    ranking: Literal["vector", "weighted", "xgboost"]


def local(request: Request):
    url = request.url
    origin = request.headers.get("origin")
    if url.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise HTTPException(403, "Open the demo on localhost.")
    if origin and urlsplit(origin).netloc != url.netloc:
        raise HTTPException(403, "Open the demo directly to change feedback.")


def create_demo(directory: Path):
    workspace = DemoWorkspace(directory)
    app = FastAPI(docs_url=None, redoc_url=None, dependencies=[Depends(local)])

    @app.get("/")
    def index():
        return FileResponse(Path(__file__).with_name("demo.html"))

    @app.get("/state")
    def state():
        try:
            return workspace.summary()
        except FileNotFoundError as exc:
            raise HTTPException(503, "Train the synthetic experiment bundle first.") from exc

    @app.get("/sessions/{session_id}")
    def session(session_id: str):
        try:
            return workspace.analyze(session_id)
        except KeyError as exc:
            raise HTTPException(404, "Session not found.") from exc

    @app.get("/prompts/{prompt_id}")
    def prompt(prompt_id: str):
        try:
            return workspace.retrieve(prompt_id)
        except KeyError as exc:
            raise HTTPException(404, "Prompt not found.") from exc

    @app.post("/votes")
    def vote(body: Vote):
        try:
            workspace.vote(body.prompt, body.memory, body.vote)
        except KeyError as exc:
            raise HTTPException(404, "Prompt or candidate not found.") from exc
        return {"saved": True}

    @app.post("/retrain")
    def retrain(body: Retrain):
        try:
            return workspace.retrain(body.scripted)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/activity")
    def activity():
        return workspace.activity()

    @app.post("/sessions/{session_id}/label")
    def label(session_id: str, body: Label):
        try:
            workspace.label(session_id, body.intent, body.memory_worthy)
        except KeyError as exc:
            raise HTTPException(404, "Session not found.") from exc
        return {"saved": True}

    @app.get("/labels.csv")
    def export_labels():
        return Response(workspace.label_csv(), media_type="text/csv",
                        headers={"Content-Disposition": 'attachment; filename="synthetic-practice-labels.csv"'})

    @app.get("/memories")
    def memories():
        return workspace.memories()

    @app.post("/sessions/{session_id}/admit")
    def admit(session_id: str):
        try:
            return {"id": workspace.admit(session_id)}
        except KeyError as exc:
            raise HTTPException(404, "Session not found.") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/memories/{memory_id}")
    def update_memory(memory_id: str, body: MemoryUpdate):
        try:
            workspace.update_memory(memory_id, body.content, body.archived)
        except KeyError as exc:
            raise HTTPException(404, "Saved memory not found.") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"saved": True}

    @app.post("/recall")
    def recall(body: Search):
        query = body.query.strip()
        if not query:
            raise HTTPException(422, "Enter a search query.")
        return workspace.search(query)

    @app.get("/preferences")
    def preferences():
        return workspace.preferences()

    @app.post("/preferences")
    def set_preferences(body: Preference):
        return workspace.preferences(body.ranking)

    return app
