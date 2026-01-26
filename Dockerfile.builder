# Database builder container for Stash Face Recognition
# Runs periodic builds and publishes to GitHub Releases

FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

WORKDIR /app

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create venv and install dependencies
RUN python3.11 -m venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy requirements first for layer caching
COPY api/requirements.txt ./api/
RUN pip install --no-cache-dir -r api/requirements.txt

# Install GitHub CLI for publishing releases
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && apt-get update \
    && apt-get install gh -y \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY api/ ./api/

# Data directory for database output and caches
VOLUME /app/data
VOLUME /app/cache

# Environment variables
ENV STASHDB_URL=https://stashdb.org/graphql
ENV DATA_DIR=/app/data
ENV CACHE_DIR=/app/cache
ENV PYTHONUNBUFFERED=1

# Default: run build script
COPY scripts/build-database.sh /app/
RUN chmod +x /app/build-database.sh

ENTRYPOINT ["/app/build-database.sh"]
