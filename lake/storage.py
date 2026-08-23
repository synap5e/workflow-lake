"""The raw capture store: content-addressed blobs plus per-run manifests.

The bucket is the system of record. ClickHouse is rebuilt from it, Postgres only
tracks work state, and both can be dropped without losing anything. That is the
whole point of the layout, so the two rules here are worth stating:

* **Blobs are content-addressed.** The same workflow captured under a thousand
  artifacts is stored once. Measured on the sample: 6% of Civitai workflow blobs
  are byte-identical to another.
* **Manifests are append-only JSONL.** One line per (artifact, channel) fetch,
  carrying the untouched `source_meta`. Nothing rewrites a manifest.

Works against a local directory or S3. Local is not a toy path — it is what the
tests use, and a small deployment can run on a PVC before anyone provisions a
bucket.
"""

from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import io
import json
import os
import pathlib
from typing import Any, Protocol
from urllib.parse import urlparse


def sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def blob_key(sha: bytes, kind: str) -> str:
    hexed = sha.hex()
    suffix = "bin" if kind == "prefix" else "json"
    return f"blob/{hexed[:2]}/{hexed}.{suffix}.gz"


def manifest_key(source: str, run_id: str, when: dt.datetime, part: int = 0) -> str:
    return f"raw/{source}/{when:%Y-%m}/{run_id}-{part:04d}.jsonl.gz"


class Store(Protocol):
    def put(self, key: str, data: bytes) -> None: ...
    def get(self, key: str) -> bytes: ...
    def exists(self, key: str) -> bool: ...
    def delete(self, key: str) -> None: ...
    def list(self, prefix: str) -> list[str]: ...


class LocalStore:
    def __init__(self, root: str | pathlib.Path) -> None:
        self.root = pathlib.Path(root)

    def _path(self, key: str) -> pathlib.Path:
        return self.root / key

    def put(self, key: str, data: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename so a killed pod never leaves a half blob behind.
        tmp = path.with_suffix(path.suffix + ".part")
        tmp.write_bytes(data)
        tmp.replace(path)

    def get(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)

    def list(self, prefix: str) -> list[str]:
        base = self._path(prefix)
        if not base.exists():
            return []
        return sorted(str(p.relative_to(self.root)) for p in base.rglob("*") if p.is_file())


class GCSStore:
    """Native GCS, authenticating with a dedicated service-account key.

    medina grants this workload its own SA with objectAdmin on one bucket, key
    in a SOPS secret, mounted and pointed at by GOOGLE_APPLICATION_CREDENTIALS.
    Deliberately NOT the node SA via the metadata server: the per-app egress
    policy blocks 169.254.0.0/16, and the node SA's grants are effectively
    cluster-wide anyway.
    """

    def __init__(self, url: str) -> None:
        from google.cloud import storage  # lazy: local runs need no GCP deps

        parsed = urlparse(url)
        self.prefix = parsed.path.strip("/")

        # Build the client explicitly from the key file rather than letting
        # Application Default Credentials discover things. ADC probes the GCE
        # metadata server at 169.254.169.254 for identity and project — and this
        # workload's egress policy blackholes link-local by design, so every
        # probe waits for a timeout instead of failing fast. Explicit
        # credentials and an explicit project skip that path entirely.
        key_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if key_file and pathlib.Path(key_file).exists():
            client = storage.Client.from_service_account_json(key_file)
        else:
            client = storage.Client()
        self.bucket = client.bucket(parsed.netloc)

    def _key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def put(self, key: str, data: bytes) -> None:
        self.bucket.blob(self._key(key)).upload_from_string(data)

    def get(self, key: str) -> bytes:
        return self.bucket.blob(self._key(key)).download_as_bytes()

    def exists(self, key: str) -> bool:
        return self.bucket.blob(self._key(key)).exists()

    def delete(self, key: str) -> None:
        self.bucket.blob(self._key(key)).delete()

    def list(self, prefix: str) -> list[str]:
        base = self._key(prefix)
        names = (b.name for b in self.bucket.list_blobs(prefix=base))
        cut = len(self.prefix) + 1 if self.prefix else 0
        return sorted(n[cut:] for n in names)


class S3Store:
    """S3, and GCS's S3-compatible XML API.

    The second is the fallback medina flagged: if an org policy forbids SA key
    creation, the bucket is reached with a GCS HMAC key instead, which is the
    same containment with a different credential shape. Set
    `LAKE_S3_ENDPOINT=https://storage.googleapis.com` and the usual
    AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY to the HMAC pair.
    """

    def __init__(self, url: str) -> None:
        import boto3  # imported lazily so local runs need no AWS deps

        parsed = urlparse(url)
        self.bucket = parsed.netloc
        self.prefix = parsed.path.strip("/")
        self.client = boto3.client("s3", endpoint_url=os.environ.get("LAKE_S3_ENDPOINT"))

    def _key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def put(self, key: str, data: bytes) -> None:
        self.client.put_object(Bucket=self.bucket, Key=self._key(key), Body=data)

    def get(self, key: str) -> bytes:
        return self.client.get_object(Bucket=self.bucket, Key=self._key(key))["Body"].read()

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self.client.head_object(Bucket=self.bucket, Key=self._key(key))
            return True
        except ClientError:
            return False

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=self._key(key))

    def list(self, prefix: str) -> list[str]:
        paginator = self.client.get_paginator("list_objects_v2")
        out: list[str] = []
        base = self._key(prefix)
        for page in paginator.paginate(Bucket=self.bucket, Prefix=base):
            for obj in page.get("Contents") or []:
                key = obj["Key"]
                out.append(key[len(self.prefix) + 1 :] if self.prefix else key)
        return sorted(out)


def open_store(root: str) -> Store:
    """`gs://` native GCS, `s3://` S3 or the GCS S3-compat fallback, else local."""
    if root.startswith("gs://"):
        return GCSStore(root)
    if root.startswith("s3://"):
        return S3Store(root)
    return LocalStore(root)


class ManifestWriter:
    """A run's capture records, written as one or more gzipped parts.

    Parts rather than a single object at the end, because a single object is
    only written if the process survives to write it. A fetch job killed
    mid-batch had already streamed its blobs to the bucket but lost every record
    describing them — the blobs were orphaned and unreferenceable, and the
    artifacts were marked done, so nothing would ever fetch them again.

    Ingest globs `raw/**.jsonl.gz`, so multiple parts per run need no special
    handling downstream.
    """

    def __init__(self, store: Store, source: str, run_id: str) -> None:
        self.store = store
        self.source = source
        self.run_id = run_id
        self.when = dt.datetime.now(dt.UTC)
        self.buffer = io.StringIO()
        self.pending = 0
        self.count = 0
        self.part = 0
        self.keys: list[str] = []

    def write(self, record: dict[str, Any]) -> None:
        self.buffer.write(json.dumps(record, default=str) + "\n")
        self.pending += 1
        self.count += 1

    def flush(self) -> str | None:
        """Persist what is buffered. Callers must not mark work done until this
        has returned for the records covering it."""
        if not self.pending:
            return None
        key = manifest_key(self.source, self.run_id, self.when, self.part)
        self.store.put(key, gzip.compress(self.buffer.getvalue().encode()))
        self.keys.append(key)
        self.buffer = io.StringIO()
        self.pending = 0
        self.part += 1
        return key

    def commit(self) -> str | None:
        self.flush()
        return self.keys[-1] if self.keys else None


def read_manifest(store: Store, key: str) -> list[dict]:
    raw = gzip.decompress(store.get(key))
    return [json.loads(line) for line in raw.decode().splitlines() if line.strip()]


class BlobWriter:
    """Content-addressed writes, skipping anything already stored."""

    def __init__(self, store: Store, frontier: Any, keep_prefix_days: int) -> None:
        self.store = store
        self.frontier = frontier
        self.keep_prefix_days = keep_prefix_days
        self.written = 0
        self.deduped = 0

    def put(self, data: bytes, kind: str) -> tuple[str, str, int]:
        """Store `data`, returning (hex digest, bucket key, size)."""
        digest = sha256(data)
        key = blob_key(digest, kind)
        if self.frontier is not None and self.frontier.blob_exists(digest):
            self.deduped += 1
            return digest.hex(), key, len(data)
        self.store.put(key, gzip.compress(data))
        self.written += 1
        if self.frontier is not None:
            expires = (
                dt.datetime.now(dt.UTC) + dt.timedelta(days=self.keep_prefix_days)
                if kind == "prefix"
                else None
            )
            self.frontier.record_blob(
                digest, size=len(data), kind=kind, key=key, expires_at=expires
            )
        return digest.hex(), key, len(data)
