-- ComfyUI workflow lake — ClickHouse schema.
--
-- Two layers, on purpose:
--
--   raw_*      a faithful mirror of what the crawler put in the bucket. Never
--              edited, never back-filled, never "cleaned". One row per capture
--              event, carrying the untouched source JSON alongside the parsed
--              columns. This is the layer that makes a schema mistake cheap.
--   derived_*  everything computed from the raw layer by the ingest job. Any of
--              it can be dropped and rebuilt from the bucket in minutes. A new
--              column is a rebuild, not a recrawl.
--
-- The dimension tables (packs, pack_versions, pack_nodes) are reference data
-- fetched on their own schedule, not per-crawl.
--
-- Verified against clickhouse-local 26.6 by analysis/e6_clickhouse.sh.

-- ---------------------------------------------------------------------------
-- RAW LAYER
-- ---------------------------------------------------------------------------

-- One row per (artifact, channel) fetch. An artifact commonly produces several:
-- a listing row, an API row, a bytes-head row, and — for video — a bytes-tail
-- row. Keeping them separate is what let the E1 table compare channels at all;
-- collapsing them at capture time would have hidden that neither dominates.
CREATE TABLE IF NOT EXISTS raw_artifacts
(
    capture_id          String,                     -- uuid of this fetch
    crawled_at          DateTime,
    crawler_version     LowCardinality(String),

    source              LowCardinality(String),     -- civitai | civitai_models | github | comfyworkflows.wayback | ...
    channel             LowCardinality(String),     -- listing | api | bytes_head | bytes_tail | asset | archive
    source_artifact_id  String,                     -- the id in the source's own namespace
    parent_url          String,                     -- the page a human would open
    fetch_url           String,                     -- what we actually requested

    -- what the source says about it (the sidecar the lake is built to keep)
    author              String,
    author_id           String,
    title               String,
    description         String,                     -- free text as published
    tags                Array(String),
    published_at        Nullable(DateTime),
    stats               Map(String, Int64),         -- likes/downloads/views/...
    source_meta         String,                     -- untouched source JSON

    -- what the fetch cost and found
    http_status         UInt16,
    content_type        LowCardinality(String),
    container           LowCardinality(String),     -- png | webp | jpeg | isobmff | json | none
    total_size          Nullable(UInt64),           -- Content-Range total
    bytes_fetched       UInt64,
    needed_bytes        Nullable(UInt64),           -- offset at which metadata ended
    windows             Array(UInt32),              -- range windows actually issued
    from_tail           Bool,

    -- the payload, by reference: blobs live in the bucket, not in ClickHouse
    payload_kind        LowCardinality(String),     -- workflow | prompt | none
    payload_ref         String,                     -- bucket key
    payload_bytes       UInt32,
    workflow_id         String                      -- FK to derived_workflows, '' if none
)
ENGINE = MergeTree
PARTITION BY (source, toYYYYMM(crawled_at))
ORDER BY (source, source_artifact_id, channel, crawled_at);

-- In production this is the same data read straight off the bucket, so a
-- rebuild never depends on ClickHouse having kept anything:
--
--   CREATE TABLE raw_artifacts_s3 AS raw_artifacts
--   ENGINE = S3('s3://workflow-lake/raw/{source}/{yyyymm}/*.jsonl.zst', JSONEachRow, 'zstd');

-- ---------------------------------------------------------------------------
-- DERIVED LAYER
-- ---------------------------------------------------------------------------

-- One row per distinct workflow, not per artifact. Three hashes because three
-- different questions are asked of them:
--   exact_hash       byte-identical JSON — provenance, "is this the same file"
--   canonical_hash   ids/layout stripped — "the same workflow, re-saved"
--   structural_hash  parameters stripped too — "the same graph, another seed",
--                    which is the level the Civitai corpus actually collapses at
--                    (63% of sampled workflows) and the level a pack-set or
--                    snapshot cache key belongs at.
CREATE TABLE IF NOT EXISTS derived_workflows
(
    workflow_id         String,                     -- = exact_hash
    canonical_hash      String,
    structural_hash     String,

    format              LowCardinality(String),     -- save | api
    node_count          UInt32,
    link_count          UInt32,
    group_count         UInt32,
    muted_or_bypassed   UInt32,
    model_edges         UInt32,
    hidden_prompt_nodes UInt32,

    frontend_version    LowCardinality(String),
    graph_version       String,

    stamped_nodes       UInt32,                     -- nodes carrying cnr_id or aux_id
    cnr_nodes           UInt32,
    aux_nodes           UInt32,
    pack_set            Array(String),
    pack_set_size       UInt16,

    first_source        LowCardinality(String),
    first_seen          DateTime,
    ingest_version      UInt32                      -- bump to rebuild in place
)
ENGINE = ReplacingMergeTree(ingest_version)
ORDER BY workflow_id;

-- Which artifacts carried which workflow. Many-to-one: a popular workflow shows
-- up under hundreds of images, and the same image can carry both formats.
CREATE TABLE IF NOT EXISTS derived_artifact_workflows
(
    workflow_id         String,
    source              LowCardinality(String),
    source_artifact_id  String,
    channel             LowCardinality(String),
    payload_kind        LowCardinality(String),
    author              String,
    published_at        Nullable(DateTime)
)
ENGINE = ReplacingMergeTree
ORDER BY (workflow_id, source, source_artifact_id, channel, payload_kind);

-- One row per node instance. The projection is what makes "every workflow using
-- node Y" a point lookup instead of a scan of the whole table.
CREATE TABLE IF NOT EXISTS derived_workflow_nodes
(
    workflow_id     String,
    node_id         String,
    class_type      String,
    is_core         Bool,
    cnr_id          String,                         -- '' when unstamped
    ver             String,
    aux_id          String,
    mode            Int8,                           -- 0 normal, 2 muted, 4 bypassed
    packs           Array(String),
    pack_ambiguous  Bool,                           -- name maps to >1 pack, unstamped
    pack_unknown    Bool,                           -- name maps to no known pack
    PROJECTION by_class_type
    (
        SELECT class_type, workflow_id, cnr_id, packs, pack_ambiguous, pack_unknown
        ORDER BY class_type
    )
)
ENGINE = MergeTree
ORDER BY (workflow_id, node_id);

-- Loader inputs, one row per (node, input). `binding` is the flagship column:
-- 'literal' means the workflow names the file, 'link' means an upstream node
-- picks it at run time and the name is not knowable statically.
CREATE TABLE IF NOT EXISTS derived_workflow_bindings
(
    workflow_id     String,
    class_type      String,
    input           LowCardinality(String),         -- ckpt_name, unet_name, lora_name, ...
    binding         LowCardinality(String),         -- literal | link
    evidence        LowCardinality(String)          -- api.inputs | save.inputs | save.widget
)
ENGINE = MergeTree
ORDER BY (input, binding, class_type, workflow_id);

-- ---------------------------------------------------------------------------
-- PACK DIMENSION
-- ---------------------------------------------------------------------------

-- The Registry and ComfyUI-Manager indexes are not nested: the Registry knows
-- versions and `cnr_id` (which is what the frontend stamps), Manager knows
-- class names and covers packs that never registered. `repo` is the join key,
-- and `in_registry`/`in_manager` say which index a row came from.
CREATE TABLE IF NOT EXISTS packs
(
    pack_id         String,                         -- cnr_id when registered, else normalised repo
    cnr_id          String,
    repo            String,                         -- host/owner/name, normalised
    repo_url        String,
    title           String,
    author          String,
    publisher_id    String,
    description     String,
    downloads       Nullable(UInt64),
    stars           Nullable(UInt32),
    last_update     Nullable(DateTime),
    created_at      Nullable(DateTime),
    latest_version  String,
    in_registry     Bool,
    in_manager      Bool
)
ENGINE = ReplacingMergeTree
ORDER BY pack_id;

CREATE TABLE IF NOT EXISTS pack_versions
(
    pack_id                     String,
    version                     String,
    published_at                Nullable(DateTime),
    deprecated                  Bool,
    status                      LowCardinality(String),
    download_url                String,
    dependencies                Array(String),
    supported_comfyui_version   String
)
ENGINE = ReplacingMergeTree
ORDER BY (pack_id, version);

-- Which class_types a pack ships. `version` is '' for rows seeded from
-- ComfyUI-Manager's node map, which is a latest-only snapshot with no version
-- axis (empty rather than NULL because sorting keys may not be nullable, and
-- this column has to be in the key); a later job that unpacks each registry
-- version's node.zip fills the versioned rows, and `schema_hash` lets a node's
-- signature change be detected across versions. The ORDER BY leads with
-- class_type because the hot query is attribution: given a name, which pack.
CREATE TABLE IF NOT EXISTS pack_nodes
(
    class_type      String,
    pack_id         String,
    repo            String,
    version         String,                         -- '' = not version-resolved
    schema_hash     String,
    source          LowCardinality(String)          -- manager-node-map | registry-unpack
)
ENGINE = ReplacingMergeTree
ORDER BY (class_type, pack_id, version);
