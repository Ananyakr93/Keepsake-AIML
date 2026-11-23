"""Access, research evaluation, and optional disposable PostgreSQL integration."""

import base64
import io
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from PIL import Image

from backend.app import access, config
from backend.app.experiments import evaluation
from backend.app.experiments.reviewed_evaluation import _reviewed_groups


class LocalAccessTests(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.middleware("http")(access.protect)
        app.state.setup_key = "a" * 64
        app.state.desktop_sessions = {}
        self.token = access.issue_session(app)
        self.pairing = access.extension_key(app)

        @app.get("/api/memories")
        def desktop():
            return {"ok": True}

        @app.get("/api/capture-policy")
        def capture_policy(request: Request):
            return {"role": request.state.client_role}

        self.app = app
        self.client = TestClient(app, base_url="http://localhost", client=("127.0.0.1", 1234))

    def test_session_and_extension_have_separate_permissions(self):
        self.assertEqual(self.client.get("/api/memories").status_code, 401)
        self.assertEqual(self.client.get("/api/memories", headers={"x-aiml-token": self.pairing}).status_code, 403)
        self.assertEqual(self.client.get("/api/capture-policy", headers={"x-aiml-token": self.pairing}).json(),
                         {"role": "extension"})
        self.assertEqual(self.client.get("/api/memories", headers={"x-aiml-token": self.token}).status_code, 200)
        self.app.state.desktop_sessions.clear()
        self.assertEqual(self.client.get("/api/memories", headers={"x-aiml-token": self.token}).status_code, 401)

    def test_nonlocal_client_is_rejected(self):
        client = TestClient(self.app, base_url="http://localhost", client=("192.0.2.5", 1234))
        self.assertEqual(client.get("/api/memories", headers={"x-aiml-token": self.token}).status_code, 403)


class EvaluationTests(unittest.TestCase):
    def test_participant_folds_are_disjoint(self):
        rows = [{"participant_id": f"participant-{i}"} for i in range(20) for _ in range(3)]
        seen = set()
        for fold in evaluation.participant_folds(rows, 5, 42):
            train, val, test = [{row["participant_id"] for row in fold[name]}
                                for name in ("train", "validation", "test")]
            self.assertFalse(train & val or train & test or val & test or seen & test)
            seen |= test
        self.assertEqual(len(seen), 20)

    def test_reviewed_input_uses_gate_and_keeps_relevant_pool(self):
        with tempfile.TemporaryDirectory() as path:
            sample = Path(path) / "judgments.json"
            prompt = {"id": "q", "participant_id": "p", "query": "query", "gate_open": False,
                      "accepted": None, "corrected": False,
                      "memories": [{"id": "m", "content": "text", "sim": 0.2,
                                    "importance": 0.4, "recency": 0.5, "relevant": 1}]}
            sample.write_text(json.dumps([prompt]), encoding="utf-8")
            full, selected = _reviewed_groups(sample)[0]
            self.assertEqual(full["memories"][0]["relevant"], 1)
            self.assertEqual(selected, [])


@unittest.skipUnless(os.getenv("AIML_TEST_DATABASE_URL"), "Requires a disposable pgvector database.")
class StorageIntegrationTests(unittest.TestCase):
    def test_consent_private_capture_correction_and_purge(self):
        # The fixture uses a deterministic embedder to avoid downloading a model or using personal data.
        shim = ModuleType("backend.app.embed")
        shim.service = SimpleNamespace(ready=False)
        with patch.dict(sys.modules, {"backend.app.embed": shim}):
            from backend.app import db
            from backend.app.experiments.live import _persist, reconcile_memory
            from backend.app.main import app
            from backend.app.ml.dataset_builder import sessions_for
            from backend.app.routers import auth, events, screenshots

            with tempfile.TemporaryDirectory() as directory:
                data = Path(directory)
                shots = data / "screenshots"
                shots.mkdir()
                with (patch.object(config, "DATABASE_URL", os.environ["AIML_TEST_DATABASE_URL"]),
                      patch.object(config, "DATA_DIR", data),
                      patch.object(config, "SCREENSHOT_DIR", shots),
                      patch.object(config, "MODEL_DIR", data / "models"),
                      patch.object(config, "EXPERIMENT_DIR", data / "experiments"),
                      patch.object(config, "SESSION_MODELS_PATH", data / "experiments/reviewed/session_models.joblib"),
                      patch.object(config, "DEFAULT_USERNAME", "test-" + uuid4().hex),
                      patch.object(events.blacklist, "phrases", return_value=["excluded"]),
                      patch.object(screenshots.crypto, "encrypt", side_effect=lambda data: data)):
                    db.init_pool()
                    self.addCleanup(db.close_pool)
                    db.apply_schema(config.REPO_ROOT / "sql/schema.sql")
                    app.state.user = db.get_default_user()
                    uid = app.state.user["id"]
                    self.addCleanup(db.execute, "DELETE FROM users WHERE id=%s", [uid])
                    app.state.setup_key = "b" * 64
                    app.state.desktop_sessions = {}
                    app.state.login_failures = []
                    desktop = {"x-aiml-token": access.issue_session(app)}
                    extension = {"x-aiml-token": access.extension_key(app)}
                    client = TestClient(app, base_url="http://localhost", client=("127.0.0.1", 1234))

                    self.assertEqual(client.get("/api/memories").status_code, 401)
                    class Accounts:
                        entry = None
                        def insert_one(self, value):
                            self.entry = value
                        def find_one(self, query):
                            return self.entry if self.entry and query["username"] == self.entry["username"] else None

                    setup = {"x-aiml-setup": app.state.setup_key}
                    credentials = {"username": app.state.user["username"], "pin": "123456"}
                    with patch.object(auth.mongo, "db", return_value=SimpleNamespace(accounts=Accounts())):
                        self.assertEqual(client.post("/api/auth/signup", json=credentials).status_code, 403)
                        self.assertEqual(client.post("/api/auth/signup", headers=setup,
                                         json=dict(credentials, username="another-person")).status_code, 403)
                        self.assertEqual(client.post("/api/auth/signup", headers=setup,
                                         json=credentials).status_code, 200)
                        logged_in = client.post("/api/auth/login", headers=setup, json=credentials)
                        self.assertEqual(logged_in.status_code, 200)
                        desktop = {"x-aiml-token": logged_in.json()["token"]}
                        self.assertEqual(client.post("/api/auth/forgot", headers=desktop).status_code, 404)
                    self.assertEqual(client.delete("/api/data/all", headers=extension).status_code, 403)
                    client.patch("/api/settings", headers=desktop,
                                 json={"tracking_enabled": True, "screenshots_enabled": True})
                    now = datetime.now(timezone.utc)
                    consent = now - timedelta(hours=2)
                    db.execute("UPDATE users SET dataset_consent_since=%s WHERE id=%s", [consent, uid])
                    body = {"url": "https://example.test/review?token=hidden", "title": "Notes",
                            "visited_at": (consent + timedelta(seconds=1)).isoformat(), "time_spent_sec": 125,
                            "client_visit_id": str(uuid4()), "client_event_id": str(uuid4()),
                            "session_id": "local-fixture", "consent_id": consent.isoformat()}
                    result = client.post("/api/events", headers=extension, json={"events": [body]})
                    self.assertEqual(result.json(), {"inserted": 1, "tracking": True}, result.text)
                    excluded = dict(body, title="Excluded site", client_event_id=str(uuid4()))
                    result = client.post("/api/events", headers=extension, json={"events": [excluded]})
                    self.assertEqual(result.json()["inserted"], 0)

                    png = io.BytesIO()
                    Image.new("RGB", (20, 20)).save(png, "PNG")
                    capture = {"image_b64": base64.b64encode(png.getvalue()).decode(),
                               "source_url": body["url"], "title": "Notes", "captured_at": now.isoformat(),
                               "consent_id": consent.isoformat()}
                    invalid = client.post("/api/screenshots", headers=extension, json=dict(capture, image_b64="%%%"))
                    self.assertEqual(invalid.status_code, 422, invalid.text)
                    response = client.post("/api/screenshots", headers=extension, json=capture)
                    self.assertEqual(response.status_code, 200, response.text)
                    sid = response.json()["id"]
                    self.assertEqual(client.get(f"/api/screenshots/{sid}/image", headers=extension).status_code, 403)
                    with patch.object(screenshots.crypto, "decrypt", side_effect=lambda data: data):
                        self.assertEqual(client.get(f"/api/screenshots/{sid}/image", headers=desktop).content,
                                         png.getvalue())
                    self.assertEqual(db.query_one("SELECT source_url FROM screenshots WHERE id=%s", [sid])["source_url"],
                                     "https://example.test/review")

                    session = sessions_for(uid)[0]
                    label = {"intent": "research", "memory_worthy": True, "evidence_id": session.evidence_id,
                             "memory_note": "Use the reviewed note."}
                    url = f"/api/dataset/sessions/{session.anchor['id']}"
                    self.assertEqual(client.post(url, headers=desktop, json=label).status_code, 200)
                    session = sessions_for(uid)[0]
                    vector = [1, 0] + [0] * 1022
                    self.assertTrue(_persist(uid, session, {"intent": "research"}, "Use the reviewed note.", vector, 0.9))
                    self.assertEqual(client.post(url, headers=desktop, json=dict(label, memory_worthy=False)).status_code, 200)
                    self.assertIsNotNone(db.query_one("SELECT archived_at FROM memories WHERE user_id=%s", [uid])["archived_at"])
                    self.assertEqual(client.post(url, headers=desktop, json=label).status_code, 200)
                    session = sessions_for(uid)[0]
                    self.assertFalse(reconcile_memory(uid, session))
                    self.assertTrue(_persist(uid, session, {"intent": "research"}, "Use the reviewed note.", vector, 0.9))
                    self.assertEqual(db.query_one("SELECT count(*) AS n FROM memories WHERE user_id=%s AND archived_at IS NULL", [uid])["n"], 1)
                    import asyncio
                    from backend.app import consolidation

                    observation = db.query_one(
                        "INSERT INTO observations (user_id,screenshot_id,caption) "
                        "VALUES (%s,%s,'image note') RETURNING id", [uid, sid],
                    )
                    memory = db.query_one("SELECT * FROM memories WHERE user_id=%s AND archived_at IS NULL", [uid])
                    baseline = db.query_one(
                        "SELECT count(*) AS n FROM memory_observations WHERE memory_id=%s", [memory["id"]],
                    )["n"]
                    stats = {"evidence_links": 0, "memories_merged": 0, "clusters_unchanged": 0}
                    recent = [{"id": observation["id"], "caption": "image note"}]
                    with patch.object(config, "ALLOW_CLOUD_PROCESSING", True):
                        with patch.object(consolidation.caption, "merge_memory", side_effect=TimeoutError("fixture")):
                            asyncio.run(consolidation._merge_into_existing(memory, recent, None, True, stats))
                        self.assertEqual(db.query_one(
                            "SELECT count(*) AS n FROM memory_observations WHERE memory_id=%s",
                            [memory["id"]])["n"], baseline)
                        with (patch.object(consolidation.caption, "merge_memory", return_value="Updated note"),
                              patch.object(shim.service, "embed_document", AsyncMock(return_value=vector), create=True)):
                            scorer = SimpleNamespace(after_merge=lambda *_: 0.8)
                            asyncio.run(consolidation._merge_into_existing(memory, recent, scorer, True, stats))
                    self.assertEqual(db.query_one("SELECT content FROM memories WHERE id=%s", [memory["id"]])["content"],
                                     "Updated note")
                    self.assertEqual(stats["evidence_links"], 1)
                    db.execute("INSERT INTO training_prompts (user_id,text,embedding) VALUES (%s,'private',%s::vector)",
                               [uid, db.vec_lit(vector)])
                    exports = data / "datasets"
                    exports.mkdir()
                    (exports / "participant.csv").write_text("fixture", encoding="utf-8")
                    response = client.delete("/api/data/all", headers=desktop)
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertEqual(response.json()["training_prompts"], 1)
                    self.assertFalse(exports.exists())
                    self.assertEqual(list(shots.iterdir()), [])
                    policy = client.get("/api/capture-policy", headers=extension).json()
                    self.assertFalse(policy["browsing"] or policy["screenshots"])
