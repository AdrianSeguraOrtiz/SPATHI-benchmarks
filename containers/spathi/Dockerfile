# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE=python:3.13-slim@sha256:7ce4b6dfe35e55397b7cda544f8a13f191b7ae28dc5aad71fe664dbc9bc2623f
FROM ${BASE_IMAGE}

ARG SPATHI_COMMIT=8bb14c82ac5cbb0eb45c82408e40acae9b0fbcaa
ARG BENCH_BUILD_FINGERPRINT

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/spathi/src

COPY containers/spathi/constraints.txt /tmp/constraints.txt
RUN python -m pip install --no-cache-dir -r /tmp/constraints.txt \
    && rm /tmp/constraints.txt

COPY --from=spathi src /opt/spathi/src
COPY containers/spathi/run_tool.py /usr/local/bin/run_tool.py
COPY containers/spathi/run_convergence.py /usr/local/bin/run_spathi_convergence.py

LABEL org.opencontainers.image.title="SPATHI ANDREA adapter" \
      org.opencontainers.image.source="https://github.com/AdrianSeguraOrtiz/SPATHI" \
      org.opencontainers.image.revision="${SPATHI_COMMIT}" \
      dev.spathi.adapter.source="https://github.com/AdrianSeguraOrtiz/SPATHI-benchmarks" \
      dev.spathi.benchmark.build-fingerprint="${BENCH_BUILD_FINGERPRINT}"

ENTRYPOINT ["python", "/usr/local/bin/run_tool.py"]
