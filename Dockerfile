# Single image, one entrypoint, a subcommand per CronJob.
#
# No clickhouse-client: ingest speaks ClickHouse's HTTP interface with the httpx
# this project already depends on, which removes a whole apt source and GPG key
# dance from the build and a ~200MB binary from the image.
FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY lake ./lake
COPY pipeline ./pipeline
COPY migrations ./migrations
COPY schema.sql ./schema.sql
# gcs: the dedicated-SA path medina grants us. s3: the HMAC fallback if an org
# policy forbids SA key creation. Both installed so the credential decision is
# a config change rather than a rebuild.
RUN pip install --no-cache-dir '.[gcs,s3]'

# Reference tables the derivation needs, baked at build time. Kept in
# reference/ rather than data/ precisely because data/ is gitignored — a CI
# checkout has no data/, so a COPY from there builds locally and fails in CI.
COPY reference ./reference

RUN useradd --uid 10001 --create-home lake
USER 10001
ENTRYPOINT ["lake"]
CMD ["status"]
