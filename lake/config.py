"""Runtime configuration, all from the environment.

Kept deliberately small. Anything a k8s CronJob needs to vary between the tip
job and the backlog job is a flag on the command, not a setting here; this holds
only the things that are the same for every job in a deployment.
"""

from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass

CRAWLER_VERSION = "lake-crawler/0.1"
# Bumped whenever lake/wfmeta.py changes in a way that could alter extraction.
# Stamped into every capture record so a targeted re-parse is possible instead
# of a full recrawl — this project lost a day to an mp4 bug for want of it.
PARSER_VERSION = "wfmeta/2"


def _secret(name: str) -> str | None:
    """Read a secret from $NAME, or from the file named by $NAME_FILE.

    The file form is what k8s secret mounts give you, and it keeps the value off
    the process command line and out of `env` dumps in logs.
    """
    value = os.environ.get(name)
    if value:
        return value.strip()
    path = os.environ.get(f"{name}_FILE")
    if path and pathlib.Path(path).exists():
        return pathlib.Path(path).read_text().strip()
    return None


@dataclass(frozen=True)
class Config:
    database_url: str
    blob_root: str
    user_agent: str
    civitai_api_key: str | None
    api_rps: float
    cdn_rps: float
    keep_prefix_days: int
    fetch_deadline: float

    @classmethod
    def from_env(cls) -> Config:
        return cls(
            database_url=os.environ.get("LAKE_DATABASE_URL", "postgresql:///workflow_lake"),
            # s3://bucket/prefix or a local path. Local is what the tests use.
            blob_root=os.environ.get("LAKE_BLOB_ROOT", "./data/lake"),
            user_agent=os.environ.get(
                "LAKE_USER_AGENT",
                "comfy-workflow-lake/0.1",
            ),
            civitai_api_key=_secret("CIVITAI_API_KEY"),
            # One pod per host, so these are the real per-host rates. See BUILD.md
            # §4 before raising them or running more than one fetcher.
            api_rps=float(os.environ.get("LAKE_API_RPS", "1.5")),
            cdn_rps=float(os.environ.get("LAKE_CDN_RPS", "5.0")),
            keep_prefix_days=int(os.environ.get("LAKE_KEEP_PREFIX_DAYS", "90")),
            # Below the job's activeDeadlineSeconds, so the batch stops itself
            # and returns its leases rather than being SIGKILLed holding them.
            fetch_deadline=float(os.environ.get("LAKE_FETCH_DEADLINE", "600")),
        )
