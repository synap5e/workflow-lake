"""Discovery: deciding what to fetch, without fetching it.

Two streams, because Civitai's cursor caps at ~4,400 items and they therefore
need completely different shapes:

* **tip** — page `sort=Newest` until it meets the newest id we already know,
  then stop. Bounded by how much was published since the last run, so a 15-minute
  cadence never comes near the cursor cap.
* **backlog** — drain one partition at a time. Partitions come from the tip:
  every artifact carries a `userId`, so the tip crawl accumulates the partition
  list and the backlog crawl works through it. History is a breadth-first
  expansion out of the tip rather than a walk backwards through it.

Both write into `crawl_queue` and nothing else. Fetching is a separate job so
that a discovery bug never costs bandwidth.
"""

from __future__ import annotations

import datetime as dt
import sys

from lake import civitai
from lake.config import Config
from lake.db import Frontier, QueueItem
from lake.latch import PostgresLatch, gate
from lake.polite import PoliteClient

SOURCE = "civitai"
TIP_PRIORITY = 100
BACKLOG_PRIORITY = 0


def _to_items(rows: list[dict], frontier: Frontier, one_per_post: bool) -> list[QueueItem]:
    """Turn listing items into queue rows, applying the one-per-post policy.

    91% of multi-image posts share a single workflow topology, so the second
    artifact of a post is nearly always a duplicate graph. Dropping it costs 8%
    of distinct topologies and 2.5x fewer fetches.
    """
    seen_posts: set[str] = set()
    if one_per_post:
        post_ids = [str(r["postId"]) for r in rows if r.get("postId")]
        seen_posts = frontier.posts_already_captured(SOURCE, post_ids)

    out: list[QueueItem] = []
    for row in rows:
        post_id = str(row["postId"]) if row.get("postId") else None
        if one_per_post and post_id:
            if post_id in seen_posts:
                continue
            seen_posts.add(post_id)
        side = civitai.sidecar(row)
        published = side.get("published_at")
        out.append(
            QueueItem(
                source=SOURCE,
                artifact_id=str(row["id"]),
                post_id=post_id,
                author_id=str(row.get("userId")) if row.get("userId") else None,
                published_at=dt.datetime.fromisoformat(published.replace("Z", "+00:00"))
                if published
                else None,
                prefilter=civitai.prefilter(row),
            )
        )
    return out


def discover_tip(
    cfg: Config,
    frontier: Frontier,
    *,
    mode: str = "complete",
    max_pages: int = civitai.TIP_PAGE_CAP,
    one_per_post: bool = True,
) -> dict:
    """Page Newest until we reach known ground."""
    gate(frontier, civitai.EGRESS_HOSTS)
    # Recorded like fetch: `crawl_run` is what a "no successful run in N hours"
    # alert reads, and discovery silently absent from it would make a stalled
    # crawler look healthy.
    run_id = frontier.start_run("discover-tip", SOURCE)
    client = PoliteClient(rps=cfg.api_rps, user_agent=cfg.user_agent, latch=PostgresLatch(frontier))
    state = frontier.get_cursor(SOURCE, "tip") or {}
    frontier_id = int(state.get("frontier_id") or 0)

    cursor = None
    queued = partitions = pages = 0
    newest_seen = frontier_id
    caught_up = False

    try:
        for _ in range(max_pages):
            items, cursor = civitai.page_images(client, cursor=cursor)
            if not items:
                break
            pages += 1
            newest_seen = max(newest_seen, max(int(i["id"]) for i in items))

            fresh = [i for i in items if int(i["id"]) > frontier_id]
            if len(fresh) < len(items):
                caught_up = True

            # Partitions come from every artifact seen, wanted or not: a user who
            # posts anything is a user whose whole history is reachable.
            partitions += frontier.add_partitions(
                SOURCE,
                "user",
                sorted({str(i["userId"]) for i in fresh if i.get("userId")}),
            )
            keep = [i for i in fresh if civitai.wanted(i, mode=mode)]
            queued += frontier.enqueue(
                _to_items(keep, frontier, one_per_post), priority=TIP_PRIORITY
            )
            if caught_up or not cursor:
                break
    finally:
        client.close()

    frontier.set_cursor(SOURCE, "tip", None, str(newest_seen) if newest_seen else None)
    frontier.finish_run(
        run_id, items=queued, requests=client.stats.requests, bytes_down=client.stats.bytes_down
    )
    return {
        "pages": pages,
        "queued": queued,
        "new_partitions": partitions,
        "frontier_id": newest_seen,
        "caught_up": caught_up,
        "requests": client.stats.requests,
    }


def discover_backlog(
    cfg: Config,
    frontier: Frontier,
    *,
    mode: str = "comfy",
    partitions: int = 5,
    max_pages_per_partition: int = 40,
    one_per_post: bool = True,
) -> dict:
    """Drain un-swept partitions, one at a time, resuming where each left off."""
    gate(frontier, civitai.EGRESS_HOSTS)
    run_id = frontier.start_run("discover-backlog", SOURCE)
    client = PoliteClient(rps=cfg.api_rps, user_agent=cfg.user_agent, latch=PostgresLatch(frontier))
    queued = drained = exhausted = 0
    try:
        for _ in range(partitions):
            part = frontier.claim_partition(SOURCE, "user")
            if part is None:
                break
            drained += 1
            cursor = part["cursor"]
            pages = part["pages_read"]
            images = part["images_seen"]
            done = False
            error = None
            try:
                for _page in range(max_pages_per_partition):
                    items, cursor = civitai.page_images(
                        client, cursor=cursor, userId=int(part["key"])
                    )
                    if not items:
                        done = True
                        break
                    pages += 1
                    images += len(items)
                    keep = [i for i in items if civitai.wanted(i, mode=mode)]
                    queued += frontier.enqueue(
                        _to_items(keep, frontier, one_per_post), priority=BACKLOG_PRIORITY
                    )
                    if not cursor:
                        done = True
                        break
            except Exception as exc:  # one bad partition must not kill the run
                error = f"{type(exc).__name__}: {exc}"
                print(f"  partition {part['key']}: {error}", file=sys.stderr)
            exhausted += 1 if done else 0
            frontier.update_partition(
                SOURCE,
                "user",
                part["key"],
                cursor=cursor,
                exhausted=done,
                pages_read=pages,
                images_seen=images,
                error=error,
            )
    finally:
        client.close()

    frontier.finish_run(
        run_id, items=queued, requests=client.stats.requests, bytes_down=client.stats.bytes_down
    )
    return {
        "partitions_drained": drained,
        "partitions_exhausted": exhausted,
        "queued": queued,
        "requests": client.stats.requests,
    }
