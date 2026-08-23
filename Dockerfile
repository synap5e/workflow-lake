# Single image, one entrypoint, a subcommand per CronJob.
#
# No clickhouse-client: ingest speaks ClickHouse's HTTP interface with the httpx
# this project already depends on, which removes a whole apt source and GPG key
# dance from the build and a ~200MB binary from the image.
FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
# migrations/ and reference/ live INSIDE lake/ so they install with the package.
# Copying them to /app would put them where an installed console script cannot
# see them, which is precisely the defect this layout fixes.
COPY lake ./lake
COPY pipeline ./pipeline
# gcs: the dedicated-SA path medina grants us. s3: the HMAC fallback if an org
# policy forbids SA key creation. Both installed so the credential decision is
# a config change rather than a rebuild.
RUN pip install --no-cache-dir '.[gcs,s3]'

RUN useradd --uid 10001 --create-home lake
USER 10001
# Fails the build if the wheel did not carry its migrations or reference data.
RUN lake selftest
ENTRYPOINT ["lake"]
CMD ["status"]
