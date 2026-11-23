"""Train model artifacts and run an isolated demo on the same inference and ranking code."""

import hashlib
import json
import sqlite3
from contextlib import closing
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import RLock

import joblib
import sklearn
import xgboost

from backend.app import dataset_format
from backend.app.experiments import fixtures, ranking, sessions

_CACHED_LOCK = RLock()


def save_artifact(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as f:
        staged = Path(f.name)
    try:
        joblib.dump(payload, staged, compress=3)
        staged.replace(path)
    finally:
        staged.unlink(missing_ok=True)


def run_training(directory, inputs=(), seed=42):
    directory.mkdir(parents=True, exist_ok=True)
    synthetic = not inputs
    source = "synthetic" if synthetic else "reviewed"
    report_path = directory / "report.json"
    if report_path.is_file() and json.loads(report_path.read_text(encoding="utf-8"))["source"] != source:
        raise ValueError("Choose separate output directories for synthetic and reviewed experiments.")
    if synthetic:
        contents = dataset_format.dataset_csv(fixtures.synthetic_sessions(seed))
        (directory / "synthetic_sessions.csv").write_text(contents, encoding="utf-8", newline="")
        sources = [("synthetic_sessions.csv", contents)]
    else:
        sources = [(str(path), path.read_text(encoding="utf-8-sig")) for path in inputs]
    rows = dataset_format.read_datasets(sources)
    partitions, method = dataset_format.split_dataset(rows, seed)
    model, report = sessions.train_sessions(partitions, method, seed)
    report.update(source=source, split=method, seed=seed,
                  input_sha256=[hashlib.sha256(content.encode()).hexdigest() for _, content in sources],
                  versions={"sklearn": sklearn.__version__, "xgboost": xgboost.__version__})
    save_artifact({"format": 1, "source": report["source"], "models": model}, directory / "session_models.joblib")
    if synthetic:
        retrieval_sets = {part: fixtures.synthetic_relevance(items, seed + offset, part)
                          for offset, (part, items) in enumerate(partitions.items())}
        retrieval, report["retrieval"] = ranking.train_retrieval(retrieval_sets["train"], retrieval_sets["test"], seed)
        validation = partitions["validation"]
        person = validation[0]["participant_id"]
        sample = [row for row in validation if row["participant_id"] == person]
        save_artifact({"retrieval": retrieval, "prompts": retrieval_sets["validation"],
                       "test": retrieval_sets["test"], "sessions": sample, "seed": seed,
                       "revision": report["input_sha256"][0], "session_models": model, "report": report},
                      directory / "demo.joblib")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


class DemoWorkspace:
    def __init__(self, directory):
        self.directory = directory

    def _load(self, name):
        path = self.directory / name
        return _read_artifact(path, path.stat().st_mtime_ns)

    def summary(self):
        fixture = self._load("demo.joblib")
        return {"report": fixture["report"], "sessions": [{"id": row["session_id"],
                 "title": row["page_titles"][0]} for row in fixture["sessions"]],
                "prompts": [{"id": row["id"], "query": row["query"]} for row in fixture["prompts"]]}

    def activity(self):
        fixture = self._load("demo.joblib")
        with closing(self._db(fixture)) as conn:
            labels = {key: {"intent": intent, "memory_worthy": bool(worthy)}
                      for key, intent, worthy in conn.execute("SELECT session,intent,worthy FROM labels")}
            memories = conn.execute("SELECT count(*) FROM memories WHERE archived=0").fetchone()[0]
            votes = conn.execute("SELECT count(*) FROM votes").fetchone()[0]
        return {"sessions": [
            {"id": row["session_id"], "title": row["page_titles"][0], "started_at": row["started_at"],
             "dwell_sec": row["dwell_sec"], "visit_count": row["visit_count"],
             "screenshot_count": row["screenshot_count"], "page_titles": row["page_titles"],
             "label": labels.get(row["session_id"])}
            for row in fixture["sessions"]
        ], "reviewed": len(labels), "memories": memories, "votes": votes}

    def _session_row(self, fixture, session_id):
        row = next((item for item in fixture["sessions"] if item["session_id"] == session_id), None)
        if row is None:
            raise KeyError("Session not found")
        return row

    def label(self, session_id, intent, worthy):
        fixture = self._load("demo.joblib")
        self._session_row(fixture, session_id)
        with closing(self._db(fixture)) as conn, conn:
            conn.execute(
                "INSERT INTO labels VALUES (?,?,?) ON CONFLICT(session) DO UPDATE "
                "SET intent=excluded.intent,worthy=excluded.worthy", (session_id, intent, int(worthy)),
            )
            if not worthy:
                conn.execute("UPDATE memories SET archived=1 WHERE session=?", (session_id,))

    def label_csv(self):
        fixture = self._load("demo.joblib")
        with closing(self._db(fixture)) as conn:
            labels = {key: (intent, worthy) for key, intent, worthy in conn.execute("SELECT * FROM labels")}
        rows = [{**row, "intent": labels[row["session_id"]][0],
                 "memory_worthy": labels[row["session_id"]][1]}
                for row in fixture["sessions"] if row["session_id"] in labels]
        return dataset_format.dataset_csv(rows)

    def admit(self, session_id):
        fixture = self._load("demo.joblib")
        row = self._session_row(fixture, session_id)
        analysis = self.analyze(session_id)
        prediction = analysis["prediction"]
        decision = prediction["admission"][prediction["selected"]]
        memory_id = "session-" + session_id
        with closing(self._db(fixture)) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            label = conn.execute("SELECT intent,worthy FROM labels WHERE session=?", (session_id,)).fetchone()
            remember = bool(label[1]) if label else decision["remember"]
            if not remember:
                raise ValueError("This session is marked to skip. Review its evidence in Label dataset.")
            intent = label[0] if label else prediction["intent"]
            content = f"{intent.capitalize()} session: {' | '.join(row['page_titles'])}"
            conn.execute(
                "INSERT INTO memories VALUES (?,?,?,?,0) ON CONFLICT(id) DO UPDATE SET archived=0",
                (memory_id, session_id, content, decision["probability"]),
            )
        return memory_id

    def memories(self):
        fixture = self._load("demo.joblib")
        with closing(self._db(fixture)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM memories ORDER BY archived,rowid DESC").fetchall()
        return [dict(row) for row in rows]

    def update_memory(self, memory_id, content, archived):
        fixture = self._load("demo.joblib")
        with closing(self._db(fixture)) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            if archived is False and conn.execute(
                "SELECT 1 FROM memories JOIN labels USING (session) WHERE memories.id=? AND labels.worthy=0",
                (memory_id,),
            ).fetchone():
                raise ValueError("This session is marked to skip. Update its label before restoring the memory.")
            changed = conn.execute(
                "UPDATE memories SET content=COALESCE(?,content), archived=COALESCE(?,archived) WHERE id=?",
                (content, archived, memory_id),
            )
            if not changed.rowcount:
                raise KeyError("Saved memory not found")

    def preferences(self, method=None):
        fixture = self._load("demo.joblib")
        with closing(self._db(fixture)) as conn, conn:
            if method is not None:
                conn.execute("INSERT OR REPLACE INTO preferences VALUES ('recall',?)", (method,))
            existing = conn.execute("SELECT value FROM preferences WHERE key='recall'").fetchone()
        return {"ranking": existing[0] if existing else "xgboost"}

    def search(self, query):
        fixture = self._load("demo.joblib")
        retrieval = fixture["retrieval"]
        saved = [{**row, "recency": 1.0} for row in self.memories() if not row["archived"]]
        pool = saved + fixture["prompts"][0]["memories"]
        candidates = ranking.vector_candidates(retrieval["encoder"], query, pool)
        candidates = [row for row in candidates if row["sim"] > 0]
        matched_ids = {row["id"] for row in candidates}
        query_terms = set(sessions.words(query))
        candidates.extend(
            {**row, "sim": 0.0} for row in saved
            if row["id"] not in matched_ids and query_terms.intersection(sessions.words(row["content"]))
        )
        method = self.preferences()["ranking"]
        updated = self.directory / f"demo_ranker-{fixture['revision'][:16]}.joblib"
        current = self._load(updated.name) if updated.is_file() else None
        model = current["model"] if current else retrieval["model"]
        ordered = ranking.order(query, candidates, method, model)
        return {"ranking": method, "candidate_count": len(candidates),
                "results": [{"id": row["id"], "content": row["content"], "similarity": row["sim"],
                             "session": row.get("session")} for row in ordered[:ranking.TOP_K]]}

    def analyze(self, session_id):
        fixture = self._load("demo.joblib")
        model = fixture["session_models"]
        row = self._session_row(fixture, session_id)
        history = [entry for entry in fixture["sessions"] if entry["started_at"] <= row["started_at"]]
        prediction = sessions.predict_session(model, row, history)
        memory = None
        if prediction["admission"][prediction["selected"]]["remember"]:
            memory = f"{prediction['intent'].capitalize()} session: {' | '.join(row['page_titles'])}"
        recall = None
        if memory:
            retrieval = fixture["retrieval"]
            pool = [{"id": "admitted-session", "content": memory, "importance": prediction["admission"][prediction["selected"]]["probability"],
                     "recency": 1.0}]
            pool.extend(fixture["prompts"][0]["memories"])
            query = row["page_titles"][0]
            candidates = ranking.vector_candidates(retrieval["encoder"], query, pool)
            ranked = ranking.order(query, candidates, "xgboost", retrieval["model"])[:ranking.TOP_K]
            recall = {"query": query, "admitted_rank": next((i for i, item in enumerate(ranked, 1)
                       if item["id"] == "admitted-session"), None),
                      "context": [item["content"] for item in ranked]}
        return {"page_titles": row["page_titles"], "prediction": prediction, "fixture_labels": {"intent": row["intent"], "memory_worthy": row["memory_worthy"]},
                "features": {name: row[name] for name in dataset_format.FEATURES}, "memory": memory, "recall": recall,
                "trends": sessions.interest_trends(model, history),
                "interests": sessions.interest_scores(model, history,
                                                      at_time=datetime.fromisoformat(row["ended_at"]))}

    def _db(self, fixture):
        connection = sqlite3.connect(self.directory / f"feedback-{fixture['revision'][:16]}.sqlite")
        connection.execute("""
            CREATE TABLE IF NOT EXISTS votes (
                prompt TEXT NOT NULL, memory TEXT NOT NULL, vote INTEGER NOT NULL CHECK (vote IN (0,1)),
                PRIMARY KEY (prompt, memory)
            )
        """)
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS labels (
                session TEXT PRIMARY KEY, intent TEXT NOT NULL, worthy INTEGER NOT NULL CHECK(worthy IN (0,1))
            );
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY, session TEXT NOT NULL, content TEXT NOT NULL,
                importance REAL NOT NULL, archived INTEGER NOT NULL CHECK(archived IN (0,1))
            );
            CREATE TABLE IF NOT EXISTS preferences (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """)
        return connection

    def _prompt(self, prompt_id):
        fixture = self._load("demo.joblib")
        prompt = next((prompt for prompt in fixture["prompts"] if prompt["id"] == prompt_id), None)
        if prompt is None:
            raise KeyError("Prompt not found")
        return fixture, prompt

    def retrieve(self, prompt_id):
        fixture, prompt = self._prompt(prompt_id)
        retrieval = fixture["retrieval"]
        updated = self.directory / f"demo_ranker-{fixture['revision'][:16]}.joblib"
        current = self._load(updated.name) if updated.is_file() else None
        candidates = ranking.vector_candidates(retrieval["encoder"], prompt["query"], prompt["memories"])
        ordered = ranking.order(prompt["query"], candidates, "xgboost", current["model"] if current else retrieval["model"])
        with closing(self._db(fixture)) as conn:
            votes = dict(conn.execute("SELECT memory, vote FROM votes WHERE prompt = ?", (prompt_id,)))
        return {"query": prompt["query"], "comparison": current["comparison"] if current else None,
                "candidates": [{"id": row["id"], "content": row["content"], "similarity": row["sim"],
                                "top_five": index < ranking.TOP_K, "vote": votes.get(row["id"])}
                               for index, row in enumerate(ordered)]}

    def vote(self, prompt_id, memory_id, vote):
        fixture, _ = self._prompt(prompt_id)
        current = self.retrieve(prompt_id)
        if memory_id not in {row["id"] for row in current["candidates"]}:
            raise KeyError("Candidate not found")
        with closing(self._db(fixture)) as conn, conn:
            if vote is None:
                conn.execute("DELETE FROM votes WHERE prompt=? AND memory=?", (prompt_id, memory_id))
            else:
                conn.execute("INSERT OR REPLACE INTO votes VALUES (?,?,?)", (prompt_id, memory_id, vote))

    def retrain(self, scripted=False):
        fixture = self._load("demo.joblib")
        with _CACHED_LOCK, closing(self._db(fixture)) as conn, conn:
            retrieval = fixture["retrieval"]
            prompts = fixture["prompts"]
            if scripted:
                for prompt in prompts[:3]:
                    candidates = ranking.vector_candidates(retrieval["encoder"], prompt["query"], prompt["memories"])
                    conn.executemany("INSERT OR IGNORE INTO votes VALUES (?,?,?)",
                                     [(prompt["id"], row["id"], row["relevant"]) for row in candidates])
                conn.commit()
            votes = {}
            for prompt_id, memory_id, vote in conn.execute("SELECT prompt, memory, vote FROM votes"):
                votes.setdefault(prompt_id, {})[memory_id] = vote
            feedback = []
            for prompt in prompts:
                labels = votes.get(prompt["id"], {})
                if len(set(labels.values())) < 2:
                    continue
                candidates = ranking.vector_candidates(retrieval["encoder"], prompt["query"], prompt["memories"])
                rows = [row for row in candidates if row["id"] in labels]
                feedback.append((ranking.features(prompt["query"], rows), [labels[row["id"]] for row in rows]))
            if not feedback:
                raise ValueError("Mark at least one useful and one irrelevant memory for the same query.")
            model, comparison = ranking.feedback_comparison(retrieval, feedback, fixture["test"], fixture["seed"])
            judged = [label for _, labels in feedback for label in labels]
            comparison.update(judgments=len(judged), useful_vote_rate=sum(judged) / len(judged))
            save_artifact({"model": model, "comparison": comparison}, self.directory / f"demo_ranker-{fixture['revision'][:16]}.joblib")
            return comparison


@lru_cache(maxsize=8)
def _read_artifact(path, modified):
    return joblib.load(path)
