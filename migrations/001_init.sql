-- Postgres owns operational state only: what we know exists, what we have
-- fetched, where discovery got to, and which blobs are already stored.
--
-- It deliberately does NOT hold capture records. The bucket is the record and
-- ClickHouse is the query layer; duplicating them here would create a third
-- copy to keep consistent for no gain.

CREATE TABLE IF NOT EXISTS crawl_queue (
    source        text        NOT NULL,
    artifact_id   text        NOT NULL,
    post_id       text,
    author_id     text,
    discovered_at timestamptz NOT NULL DEFAULT now(),
    published_at  timestamptz,
    -- Higher runs first. The tip queues at 100, backlog at 0, so a backlog
    -- sweep never starves the tip even though they share one worker.
    priority      int         NOT NULL DEFAULT 0,
    -- Why we queued it: hasMeta / toolIds / container hints from the listing.
    -- Kept so a prefilter change can be evaluated against real decisions.
    prefilter     jsonb,
    state         text        NOT NULL DEFAULT 'pending',
    attempts      int         NOT NULL DEFAULT 0,
    lease_until   timestamptz,
    last_error    text,
    updated_at    timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (source, artifact_id),
    CONSTRAINT crawl_queue_state_check
        CHECK (state IN ('pending', 'leased', 'done', 'failed', 'skipped'))
);

-- The claim query: pending work, best first. Partial so the index stays small
-- once most rows are done.
CREATE INDEX IF NOT EXISTS crawl_queue_pending
    ON crawl_queue (priority DESC, published_at DESC NULLS LAST)
    WHERE state = 'pending';

-- Expired leases get swept back to pending by the same claim query.
CREATE INDEX IF NOT EXISTS crawl_queue_leased
    ON crawl_queue (lease_until)
    WHERE state = 'leased';

-- "Have we already captured this artifact's post?" — one-artifact-per-post is a
-- policy the discoverer applies, and it needs this lookup to be cheap.
CREATE INDEX IF NOT EXISTS crawl_queue_post
    ON crawl_queue (source, post_id)
    WHERE post_id IS NOT NULL;


-- Discovery position. One row per (source, stream).
CREATE TABLE IF NOT EXISTS crawl_cursor (
    source      text        NOT NULL,
    stream      text        NOT NULL,
    cursor      text,
    -- Newest artifact id seen. The tip job pages until it meets this, then stops.
    frontier_id text,
    updated_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (source, stream)
);


-- Backlog partitions.
--
-- This table exists because paging `image.getInfinite` backwards from the tip
-- caps at ~4,400 items: after 44 pages the cursor stops advancing and the
-- endpoint returns a fixed jumbled fallback set. History is therefore only
-- reachable one partition at a time, and paging *within* a partition works.
--
-- The tip crawl discovers partitions (every artifact carries a userId), the
-- backlog crawl drains them. The backlog is a breadth-first expansion out of
-- the tip rather than a walk backwards through it.
CREATE TABLE IF NOT EXISTS crawl_partition (
    source        text        NOT NULL,
    kind          text        NOT NULL,
    key           text        NOT NULL,
    discovered_at timestamptz NOT NULL DEFAULT now(),
    swept_at      timestamptz,
    cursor        text,
    exhausted     boolean     NOT NULL DEFAULT false,
    pages_read    int         NOT NULL DEFAULT 0,
    images_seen   int         NOT NULL DEFAULT 0,
    lease_until   timestamptz,
    last_error    text,
    PRIMARY KEY (source, kind, key),
    CONSTRAINT crawl_partition_kind_check
        CHECK (kind IN ('user', 'post', 'model_version'))
);

CREATE INDEX IF NOT EXISTS crawl_partition_undrained
    ON crawl_partition (source, swept_at NULLS FIRST, discovered_at)
    WHERE NOT exhausted;


-- Content-addressed blob index, so an unchanged workflow is stored once no
-- matter how many artifacts carry it. Measured: 60% of Civitai workflows are
-- structural duplicates and 6% are byte-identical.
CREATE TABLE IF NOT EXISTS blob (
    sha256     bytea       PRIMARY KEY,
    bytes      int         NOT NULL,
    kind       text        NOT NULL,
    bucket_key text        NOT NULL,
    stored_at  timestamptz NOT NULL DEFAULT now(),
    -- Byte prefixes are kept only long enough to make a parser fix a re-parse
    -- instead of a re-crawl; workflow blobs are kept forever.
    expires_at timestamptz,
    CONSTRAINT blob_kind_check CHECK (kind IN ('workflow', 'prompt', 'prefix'))
);

CREATE INDEX IF NOT EXISTS blob_expiring ON blob (expires_at)
    WHERE expires_at IS NOT NULL;


-- One row per completed capture batch, for operational visibility. Not the
-- record of what was captured — that is the manifest in the bucket.
CREATE TABLE IF NOT EXISTS crawl_run (
    run_id      uuid        PRIMARY KEY,
    job         text        NOT NULL,
    source      text        NOT NULL,
    started_at  timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz,
    items       int         NOT NULL DEFAULT 0,
    workflows   int         NOT NULL DEFAULT 0,
    requests    int         NOT NULL DEFAULT 0,
    bytes_down  bigint      NOT NULL DEFAULT 0,
    manifest_key text,
    error       text
);

CREATE INDEX IF NOT EXISTS crawl_run_recent ON crawl_run (job, started_at DESC);
