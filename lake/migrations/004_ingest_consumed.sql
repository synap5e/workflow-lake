-- Ingest tracked "what have I loaded" as a single watermark key, compared
-- lexicographically against manifest keys of the form
-- raw/<source>/<YYYY-MM>/<uuid4>-<part>.jsonl.gz. Random UUIDs do not sort in
-- arrival order, so the watermark ratcheted toward the lexicographic max of
-- every UUID seen and silently dropped whatever sorted below it — 87% of
-- everything fetched over 31 days (393k records written, 50k rows landed).
-- On 2026-09-03 it landed on ffde02ea..., the top 0.05% of hex space, and
-- consumption stopped entirely while the CronJob stayed green. At the October
-- partition rollover it would have "recovered" on its own, permanently
-- skipping the rest of September and hiding that anything happened.
--
-- A set has no order to get wrong: one row per consumed manifest, and
-- membership — not comparison — decides what is new. The table starts empty
-- on purpose. Engines are idempotent and the bucket is the system of record,
-- so the first ticks after this migration re-read everything and backfill
-- every manifest the watermark ever dropped.
CREATE TABLE IF NOT EXISTS ingest_manifest (
    manifest_key text        PRIMARY KEY,
    ingested_at  timestamptz NOT NULL DEFAULT now()
);

-- The stale watermark row would otherwise sit in crawl_cursor looking
-- authoritative. Nothing reads it after this migration.
DELETE FROM crawl_cursor WHERE source = '_ingest' AND stream = 'ingest';
