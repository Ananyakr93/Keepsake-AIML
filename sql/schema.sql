-- AIML-EL storage for activity, annotations, memory evidence, and retrieval feedback.
CREATE EXTENSION IF NOT EXISTS vector;

-- Recreate dependent views in order on startup.
DROP VIEW IF EXISTS v_memory_evidence;
DROP VIEW IF EXISTS v_memory_usage;
DROP VIEW IF EXISTS v_memory_topics;
DROP VIEW IF EXISTS v_observations;
DROP VIEW IF EXISTS v_browsing_events;

-- Derive the site hostname from the URL.
CREATE OR REPLACE FUNCTION url_domain(u TEXT) RETURNS TEXT AS $$
    SELECT lower(substring(u from '^[A-Za-z][A-Za-z0-9+.-]*://(?:[^/?#@]*@)?([^/:?#]+)'))
$$ LANGUAGE sql IMMUTABLE;

-- ----------------------------------------------------------------------------
-- Local participant and collection settings.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username                TEXT NOT NULL UNIQUE,
    tracking_enabled        BOOLEAN NOT NULL DEFAULT FALSE,
    screenshots_enabled     BOOLEAN NOT NULL DEFAULT FALSE,
    screenshot_interval_sec INTEGER NOT NULL DEFAULT 30 CHECK (screenshot_interval_sec BETWEEN 10 AND 600),
    injection_enabled       BOOLEAN NOT NULL DEFAULT FALSE,
    retention_days          INTEGER NOT NULL DEFAULT 30 CHECK (retention_days BETWEEN 1 AND 365),
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ----------------------------------------------------------------------------
-- Browsing intervals sent in batches by the Chrome extension.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS browsing_events (
    id             BIGSERIAL PRIMARY KEY,
    user_id        UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    url            TEXT NOT NULL,
    title          TEXT,
    visited_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    time_spent_sec INTEGER NOT NULL DEFAULT 0 CHECK (time_spent_sec >= 0),
    session_id     TEXT
);

-- ----------------------------------------------------------------------------
-- screenshots — captured by the Electron app. Pixels live in an AES-256-GCM
-- encrypted file on disk; this table holds metadata + the reference.
-- browsing_event_id = the page that was on screen (browser in foreground).
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS screenshots (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id           UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    browsing_event_id BIGINT REFERENCES browsing_events(id) ON DELETE SET NULL,
    captured_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    file_path         TEXT NOT NULL,
    sha256            CHAR(64),
    phash             TEXT,
    size_bytes        INTEGER,
    width             INTEGER,
    height            INTEGER,
    source_app        TEXT,
    encrypted         BOOLEAN NOT NULL DEFAULT TRUE
);

-- ----------------------------------------------------------------------------
-- observations — what was understood about ONE screenshot or ONE browsing
-- event: caption, OCR text (encrypted bytea) and its embeddings in the shared
-- 1024-d Jina v5 text/image space. Exactly one source FK is set; each source
-- has at most one observation (UNIQUE), so each source FK is a candidate key.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS observations (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id           UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    screenshot_id     UUID REFERENCES screenshots(id) ON DELETE CASCADE,
    browsing_event_id BIGINT REFERENCES browsing_events(id) ON DELETE CASCADE,
    caption           TEXT,
    ocr_encrypted     BYTEA,
    importance_hint   REAL CHECK (importance_hint >= 0 AND importance_hint <= 1),
    observed_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    embedding         vector(1024),
    image_embedding   vector(1024),
    CONSTRAINT obs_exactly_one_source
        CHECK ((screenshot_id IS NOT NULL) <> (browsing_event_id IS NOT NULL))
);

-- Topic tags linked to each observation.
CREATE TABLE IF NOT EXISTS observation_topics (
    observation_id UUID NOT NULL REFERENCES observations(id) ON DELETE CASCADE,
    topic          TEXT NOT NULL CHECK (length(btrim(topic)) > 0),
    PRIMARY KEY (observation_id, topic)
);

-- ----------------------------------------------------------------------------
-- memories — consolidated, higher-level facts with importance. Each memory
-- links back to its evidence (observations) via memory_observations.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS memories (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    content     TEXT NOT NULL CONSTRAINT memories_content_nonempty CHECK (length(btrim(content)) > 0),
    embedding   vector(1024) NOT NULL,
    importance  REAL NOT NULL DEFAULT 0.5 CHECK (importance >= 0 AND importance <= 1),
    source      TEXT NOT NULL DEFAULT 'auto' CHECK (source IN ('auto', 'manual')),
    archived_at TIMESTAMPTZ,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- M:N evidence link: a memory traces back to the observations that produced it.
CREATE TABLE IF NOT EXISTS memory_observations (
    memory_id      UUID NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    observation_id UUID NOT NULL REFERENCES observations(id) ON DELETE CASCADE,
    contribution   REAL NOT NULL DEFAULT 1.0,
    linked_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (memory_id, observation_id)
);

-- ----------------------------------------------------------------------------
-- chat_queries — every chatbot query that asked for memory context, with its
-- embedding. Retrieved memories are rows of query_memories (M:N).
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chat_queries (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    query_text      TEXT NOT NULL,
    query_embedding vector(1024) NOT NULL,
    site            TEXT,
    context_text    TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS query_memories (
    query_id   UUID NOT NULL REFERENCES chat_queries(id) ON DELETE CASCADE,
    memory_id  UUID NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    rank       SMALLINT NOT NULL,
    similarity REAL,
    PRIMARY KEY (query_id, memory_id)
);

-- "Train memories" tab: example prompts and the user's thumbs up/down per retrieved memory
CREATE TABLE IF NOT EXISTS training_prompts (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id    UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    text       TEXT NOT NULL,
    embedding  vector(1024) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS memory_feedback (
    prompt_id UUID NOT NULL REFERENCES training_prompts(id) ON DELETE CASCADE,
    memory_id UUID NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    vote      SMALLINT NOT NULL CHECK (vote IN (-1, 1)),
    voted_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (prompt_id, memory_id)
);

-- ----------------------------------------------------------------------------
-- Indexes
-- ----------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_browsing_events_user_visited
    ON browsing_events (user_id, visited_at DESC);
CREATE INDEX IF NOT EXISTS idx_browsing_events_domain
    ON browsing_events (url_domain(url));

CREATE INDEX IF NOT EXISTS idx_screenshots_user_captured
    ON screenshots (user_id, captured_at DESC);
CREATE INDEX IF NOT EXISTS idx_screenshots_user_sha
    ON screenshots (user_id, sha256);
CREATE INDEX IF NOT EXISTS idx_screenshots_browsing_event
    ON screenshots (browsing_event_id);

CREATE INDEX IF NOT EXISTS idx_observations_user_observed
    ON observations (user_id, observed_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS uq_observations_screenshot
    ON observations (screenshot_id) WHERE screenshot_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS uq_observations_browsing_event
    ON observations (browsing_event_id) WHERE browsing_event_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_observations_embedding
    ON observations USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_observations_image_embedding
    ON observations USING hnsw (image_embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_observation_topics_topic
    ON observation_topics (topic);

CREATE INDEX IF NOT EXISTS idx_memories_user_updated
    ON memories (user_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_memories_embedding
    ON memories USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_memory_observations_obs
    ON memory_observations (observation_id);

CREATE INDEX IF NOT EXISTS idx_chat_queries_user_created
    ON chat_queries (user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_query_memories_memory
    ON query_memories (memory_id);

-- ----------------------------------------------------------------------------
-- updated_at trigger
-- ----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION touch_updated_at() RETURNS trigger AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_users_touch_updated_at ON users;
CREATE TRIGGER trg_users_touch_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION touch_updated_at();

DROP TRIGGER IF EXISTS trg_memories_touch_updated_at ON memories;
CREATE TRIGGER trg_memories_touch_updated_at BEFORE UPDATE ON memories
    FOR EACH ROW
    WHEN (OLD.content IS DISTINCT FROM NEW.content OR OLD.importance IS DISTINCT FROM NEW.importance)
    EXECUTE FUNCTION touch_updated_at();

-- ----------------------------------------------------------------------------
-- Views used by activity history, retrieval, and evidence inspection
-- ----------------------------------------------------------------------------
CREATE VIEW v_browsing_events AS
SELECT b.id, b.user_id, b.url, b.title, b.visited_at, b.time_spent_sec, b.session_id,
       url_domain(b.url) AS domain
FROM browsing_events b;

-- observation + derived kind, page url, and topic array
CREATE VIEW v_observations AS
SELECT o.id, o.user_id, o.screenshot_id, o.browsing_event_id,
       CASE WHEN o.screenshot_id IS NOT NULL THEN 'screenshot' ELSE 'browsing' END AS kind,
       o.caption, o.ocr_encrypted, o.importance_hint, o.observed_at,
       o.embedding, o.image_embedding,
       COALESCE(be.url, sb.url) AS url,
       COALESCE((SELECT array_agg(t.topic ORDER BY t.topic)
                 FROM observation_topics t WHERE t.observation_id = o.id), '{}') AS topics
FROM observations o
LEFT JOIN screenshots s       ON s.id  = o.screenshot_id
LEFT JOIN browsing_events sb  ON sb.id = s.browsing_event_id
LEFT JOIN browsing_events be  ON be.id = o.browsing_event_id;

-- per-memory retrieval usage, derived from the query log
CREATE VIEW v_memory_usage AS
SELECT m.id AS memory_id,
       count(qm.query_id)  AS hit_count,
       max(q.created_at)   AS last_retrieved_at
FROM memories m
LEFT JOIN query_memories qm ON qm.memory_id = m.id
LEFT JOIN chat_queries q    ON q.id = qm.query_id
GROUP BY m.id;

-- per-memory topics, derived from its evidence (uses = evidence count)
CREATE VIEW v_memory_topics AS
SELECT mo.memory_id, ot.topic, count(*) AS uses
FROM memory_observations mo
JOIN observation_topics ot ON ot.observation_id = mo.observation_id
GROUP BY mo.memory_id, ot.topic;

CREATE VIEW v_memory_evidence AS
SELECT
    m.id                AS memory_id,
    m.content           AS memory_content,
    m.importance        AS importance,
    mo.contribution     AS contribution,
    o.id                AS observation_id,
    o.kind              AS observation_kind,
    o.caption           AS observation_caption,
    o.url               AS observation_url,
    o.observed_at       AS observed_at,
    s.id                AS screenshot_id,
    s.captured_at       AS screenshot_captured_at,
    s.source_app        AS source_app,
    b.id                AS browsing_event_id,
    b.domain            AS domain,
    b.url               AS url,
    b.title             AS page_title,
    b.time_spent_sec    AS time_spent_sec
FROM memories m
JOIN memory_observations mo ON mo.memory_id = m.id
JOIN v_observations o       ON o.id = mo.observation_id
LEFT JOIN screenshots s     ON s.id = o.screenshot_id
LEFT JOIN v_browsing_events b ON b.id = COALESCE(o.browsing_event_id, s.browsing_event_id);

-- AIML dataset collection is opt-in and applies to newly captured evidence.
ALTER TABLE users ADD COLUMN IF NOT EXISTS dataset_consent_since TIMESTAMPTZ;
ALTER TABLE users ALTER COLUMN retention_days SET DEFAULT 30;
ALTER TABLE users ALTER COLUMN tracking_enabled SET DEFAULT FALSE;
ALTER TABLE users ALTER COLUMN screenshots_enabled SET DEFAULT FALSE;
ALTER TABLE users ALTER COLUMN injection_enabled SET DEFAULT FALSE;

ALTER TABLE browsing_events ADD COLUMN IF NOT EXISTS client_event_id UUID;
ALTER TABLE browsing_events ADD COLUMN IF NOT EXISTS client_visit_id UUID;
ALTER TABLE browsing_events ADD COLUMN IF NOT EXISTS dataset_consent_since TIMESTAMPTZ;
ALTER TABLE screenshots ADD COLUMN IF NOT EXISTS dataset_consent_since TIMESTAMPTZ;

CREATE UNIQUE INDEX IF NOT EXISTS uq_browsing_client_event
    ON browsing_events (user_id, client_event_id);
CREATE INDEX IF NOT EXISTS idx_browsing_dataset
    ON browsing_events (user_id, visited_at, id) WHERE dataset_consent_since IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_screenshots_dataset
    ON screenshots (user_id, captured_at) WHERE dataset_consent_since IS NOT NULL;

CREATE TABLE IF NOT EXISTS dataset_annotations (
    anchor_event_id BIGINT PRIMARY KEY REFERENCES browsing_events(id) ON DELETE CASCADE,
    evidence_id TEXT NOT NULL,
    intent TEXT NOT NULL CHECK (intent IN
        ('research', 'learning', 'purchase', 'planning', 'entertainment', 'casual', 'task')),
    memory_worthy BOOLEAN NOT NULL,
    labeled_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- A session stays processed when its memory is manually removed.
CREATE TABLE IF NOT EXISTS session_memory_links (
    anchor_event_id BIGINT PRIMARY KEY REFERENCES browsing_events(id) ON DELETE CASCADE,
    memory_id UUID REFERENCES memories(id) ON DELETE SET NULL,
    admission_score REAL NOT NULL CHECK (admission_score BETWEEN 0 AND 1),
    intent TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE screenshots ADD COLUMN IF NOT EXISTS source_url TEXT;
ALTER TABLE screenshots ADD COLUMN IF NOT EXISTS processing_attempts INT NOT NULL DEFAULT 0;
ALTER TABLE screenshots ADD COLUMN IF NOT EXISTS processing_after TIMESTAMPTZ NOT NULL DEFAULT now();
ALTER TABLE session_memory_links ADD COLUMN IF NOT EXISTS evidence_id TEXT;
ALTER TABLE session_memory_links ADD COLUMN IF NOT EXISTS revoked_at TIMESTAMPTZ;
ALTER TABLE dataset_annotations ADD COLUMN IF NOT EXISTS memory_note TEXT;
ALTER TABLE training_prompts ADD COLUMN IF NOT EXISTS gate_open BOOLEAN;
