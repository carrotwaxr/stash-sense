#!/bin/bash
# Unraid scheduled build script
# Add this to User Scripts plugin to run weekly/monthly
#
# Required: Set STASHDB_API_KEY in the script or as Unraid variable

STASHDB_API_KEY="${STASHDB_API_KEY:-your-api-key-here}"
PROJECT_DIR="/mnt/user/appdata/stash-face-recognition"

cd "$PROJECT_DIR"

docker-compose -f docker-compose.builder.yml up --build

echo "Build complete. Database updated at: $PROJECT_DIR/data/"
echo "Restart the face recognition API to load the new database."
