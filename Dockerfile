FROM python:3.11-slim

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        ca-certificates \
        clang \
        cmake \
        git \
        golang-go \
        lld \
        ninja-build && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/orbit

COPY requirements.txt requirements-dev.txt ./
RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements-dev.txt

ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd --gid "${GROUP_ID}" orbit && \
    useradd --uid "${USER_ID}" --gid "${GROUP_ID}" --create-home --shell /bin/bash orbit

COPY . .
RUN chmod +x scripts/reproduce.sh scripts/setup_dependencies.sh && \
    chown -R orbit:orbit /opt/orbit

USER orbit

CMD ["./scripts/reproduce.sh", "quick"]
