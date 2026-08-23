-- The politeness latch.
--
-- `lake/polite.py` counts 401/403 per process and gives up. Run by hand that is
-- enough; run as a CronJob it is not, because the counter dies with the pod —
-- a source that starts refusing us would be hit again at the next tick, forever,
-- which is precisely the behaviour the politeness design exists to prevent.
--
-- So the refusal has to outlive the pod. Imported from k8s-homelab's
-- flight-prices scraper, which latches on `api_change` / `persistent_captcha`
-- and stays latched until a human clears it. Sticky by design: auto-clearing
-- would just rediscover the block on a slower loop.
CREATE TABLE IF NOT EXISTS crawl_latch (
    host        text        PRIMARY KEY,
    latched_at  timestamptz NOT NULL DEFAULT now(),
    reason      text        NOT NULL,
    -- What tripped it, kept so the operator can tell a hard block from a blip.
    status_code int,
    consecutive int         NOT NULL DEFAULT 0,
    detail      text,
    -- Set by hand to release. Never set by the crawler.
    cleared_at  timestamptz,
    cleared_by  text
);

-- Only un-cleared rows gate a run.
CREATE INDEX IF NOT EXISTS crawl_latch_active ON crawl_latch (host)
    WHERE cleared_at IS NULL;

-- Consecutive-refusal tally, kept separately so a single 403 between successes
-- does not accumulate toward a latch across days.
CREATE TABLE IF NOT EXISTS crawl_host_health (
    host              text        PRIMARY KEY,
    consecutive_refusals int      NOT NULL DEFAULT 0,
    last_refusal_at   timestamptz,
    last_success_at   timestamptz,
    last_status       int
);
