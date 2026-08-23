"""One entrypoint, one subcommand per CronJob.

    lake migrate           [--check]
    lake discover-tip      [--mode complete|comfy|all] [--max-pages N]
    lake discover-backlog  [--partitions N]
    lake fetch             [--batch N] [--no-prefix]
    lake ingest            [--since KEY]
    lake gc
    lake unlatch <host>
    lake selftest          # packaged data present? no DB, no network
    lake status

Every subcommand is a bounded, idempotent batch that exits. That is what makes a
CronJob the right container for it: no long-running consumer to supervise, and a
pod that dies mid-batch just lets its Postgres lease expire.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from lake import migrate
from lake.config import Config
from lake.db import Frontier
from lake.latch import Latched

# Resolved from the package, not from this file's parent: an installed console
# script puts its bin directory on sys.path, so `parent.parent` lands in
# site-packages and a repo-root `migrations/` is invisible.
MIGRATIONS = None  # `lake.migrate` resolves its own packaged default


def _frontier(cfg: Config) -> Frontier:
    return Frontier(cfg.database_url)


def cmd_migrate(cfg: Config, args: argparse.Namespace) -> dict:
    """Apply pending migrations. Safe to run concurrently and on every pod start.

    This runs as an initContainer on every job rather than as a Job someone has
    to remember, because "the manifests assume a schema nothing creates" is a
    failure mode that only shows up in production.
    """
    frontier = _frontier(cfg)
    try:
        return migrate.run(frontier, dry_run=args.check)
    finally:
        frontier.close()


def cmd_discover_tip(cfg: Config, args: argparse.Namespace) -> dict:
    from pipeline.discover import discover_tip

    frontier = _frontier(cfg)
    try:
        return discover_tip(
            cfg,
            frontier,
            mode=args.mode,
            max_pages=args.max_pages,
            one_per_post=not args.every_asset,
        )
    finally:
        frontier.close()


def cmd_discover_backlog(cfg: Config, args: argparse.Namespace) -> dict:
    from pipeline.discover import discover_backlog

    frontier = _frontier(cfg)
    try:
        return discover_backlog(
            cfg,
            frontier,
            mode=args.mode,
            partitions=args.partitions,
            max_pages_per_partition=args.max_pages,
            one_per_post=not args.every_asset,
        )
    finally:
        frontier.close()


def cmd_fetch(cfg: Config, args: argparse.Namespace) -> dict:
    from pipeline import fetch

    frontier = _frontier(cfg)
    try:
        return fetch.run(
            cfg,
            frontier,
            batch=args.batch,
            keep_prefix=not args.no_prefix,
            use_api=not args.no_api,
        )
    finally:
        frontier.close()


def cmd_ingest(cfg: Config, args: argparse.Namespace) -> dict:
    from pipeline import ingest

    return ingest.run(cfg, since=args.since, limit_manifests=args.limit)


def cmd_gc(cfg: Config, args: argparse.Namespace) -> dict:
    """Drop byte prefixes past their retention window.

    Workflow blobs are kept forever; prefixes exist only to make a parser fix a
    re-parse instead of a re-crawl, so they age out.
    """
    from lake.storage import open_store

    frontier = _frontier(cfg)
    store = open_store(cfg.blob_root)
    removed = 0
    try:
        for row in frontier.expired_blobs(limit=args.limit):
            store.delete(row["bucket_key"])
            frontier.forget_blob(row["sha256"])
            removed += 1
    finally:
        frontier.close()
    return {"prefixes_removed": removed}


def cmd_unlatch(cfg: Config, args: argparse.Namespace) -> dict:
    """Release a politeness latch. Deliberately the only way one clears.

    Nothing in the crawler ever calls this. If a host is latched, someone has to
    look at why before it starts again.
    """
    frontier = _frontier(cfg)
    try:
        cleared = frontier.clear_latch(args.host, args.by)
        return {"host": args.host, "cleared": cleared, "by": args.by}
    finally:
        frontier.close()


def cmd_selftest(cfg: Config, args: argparse.Namespace) -> dict:
    """Verify the package carries the data it needs. No database, no network.

    Exists because `migrate --help` proved the *flag* existed while the
    migrations themselves were absent from the image. A build can be perfectly
    importable and still be missing everything that matters, so CI runs this
    against the built image.
    """
    from lake.derive import Reference
    from lake.migrate import load

    migrations = [m.filename for m in load()]
    ref = Reference.load()
    for name, value in (
        ("core node list", ref.core),
        ("hidden-PROMPT list", ref.hidden_prompt),
        ("core input table", ref.core_inputs),
        ("class_type -> pack index", ref.node_to_packs),
    ):
        if not value:
            raise RuntimeError(f"{name} is empty — the package is missing reference data")
    return {
        "migrations": migrations,
        "core_class_types": len(ref.core),
        "hidden_prompt_class_types": len(ref.hidden_prompt),
        "pack_index_class_types": len(ref.node_to_packs),
        "ok": True,
    }


def cmd_status(cfg: Config, args: argparse.Namespace) -> dict:
    frontier = _frontier(cfg)
    try:
        latches = frontier.active_latches()
        return {
            # First, because a latched crawler looks healthy by every other measure.
            "latched": latches or None,
            "queue": frontier.queue_stats("civitai"),
            "partitions": frontier.partition_stats("civitai"),
            "tip_cursor": frontier.get_cursor("civitai", "tip"),
            "blob_root": cfg.blob_root,
        }
    finally:
        frontier.close()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="lake")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("migrate", help="apply pending schema migrations (idempotent)")
    p.add_argument("--check", action="store_true", help="report pending, apply nothing")
    p.set_defaults(fn=cmd_migrate)
    # Kept so anything that documented the old name keeps working.
    p = sub.add_parser("init-db", help="alias for `migrate`")
    p.add_argument("--check", action="store_true")
    p.set_defaults(fn=cmd_migrate)

    for name, fn in (
        ("discover-tip", cmd_discover_tip),
        ("discover-backlog", cmd_discover_backlog),
    ):
        p = sub.add_parser(name)
        p.add_argument(
            "--mode",
            default="complete" if name == "discover-tip" else "comfy",
            choices=["complete", "comfy", "all"],
            help="complete: hasMeta OR ComfyUI-tagged (99.2%% recall, 76.8%% fetched); "
            "comfy: ComfyUI-tagged only (87.4%% recall, 31.8%% fetched); all: no prefilter",
        )
        p.add_argument("--max-pages", type=int, default=40)
        p.add_argument(
            "--every-asset",
            action="store_true",
            help="queue every artifact rather than one per post (costs 2.5x fetches "
            "for 8%% more distinct topologies)",
        )
        if name == "discover-backlog":
            p.add_argument("--partitions", type=int, default=5)
        p.set_defaults(fn=fn)

    p = sub.add_parser("fetch")
    p.add_argument("--batch", type=int, default=200)
    p.add_argument("--no-prefix", action="store_true", help="do not retain byte prefixes")
    p.add_argument("--no-api", action="store_true", help="bytes channel only")
    p.set_defaults(fn=cmd_fetch)

    p = sub.add_parser("ingest")
    p.add_argument("--since", default=None, help="only manifests with a key after this")
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(fn=cmd_ingest)

    p = sub.add_parser("gc")
    p.add_argument("--limit", type=int, default=1000)
    p.set_defaults(fn=cmd_gc)

    p = sub.add_parser("unlatch", help="release a politeness latch (humans only)")
    p.add_argument("host")
    p.add_argument("--by", default=os.environ.get("USER", "unknown"))
    p.set_defaults(fn=cmd_unlatch)

    sub.add_parser(
        "selftest", help="verify packaged migrations and reference data are present"
    ).set_defaults(fn=cmd_selftest)

    sub.add_parser("status").set_defaults(fn=cmd_status)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = Config.from_env()
    try:
        result = args.fn(cfg, args)
    except Latched as exc:
        # Exit 2, not 1: "a source told us to stop" is a different operational
        # state from "the job broke", and the alert rules distinguish them.
        print(json.dumps({"latched": str(exc)}), file=sys.stderr)
        return 2
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        return 1
    print(json.dumps(result, default=str, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
