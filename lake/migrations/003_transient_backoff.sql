-- Graduated backoff for TRANSIENT source failures, distinct from the latch.
--
-- The latch means "a source told us to stop" and only a human clears it. That
-- is the right cost for 401/403/429 and the wrong one for 503: five discover
-- jobs failed over four hours against Civitai 503s that resolved on their own,
-- and a latch would have stopped the crawler until someone noticed a problem
-- that had already fixed itself.
--
-- Knocking every 15 minutes for four hours was impolite; stopping permanently
-- would have been impolite in the other direction. What a 503 asks for is to be
-- left alone for a while and then tried again, so the backoff doubles and
-- clears itself on the first success.
CREATE TABLE IF NOT EXISTS crawl_host_backoff (
    host                  text        PRIMARY KEY,
    consecutive_transient int         NOT NULL DEFAULT 0,
    backoff_until         timestamptz,
    last_status           int,
    updated_at            timestamptz NOT NULL DEFAULT now()
);
