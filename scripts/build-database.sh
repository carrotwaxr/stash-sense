#!/bin/bash
set -e

# Database build script for Stash Face Recognition
# Can be run manually or via cron/scheduled task
#
# Environment variables:
#   STASHDB_API_KEY     - Required: API key for StashDB
#   STASHDB_URL         - StashDB GraphQL endpoint (default: https://stashdb.org/graphql)
#   MAX_PERFORMERS      - Max performers to index (default: 200000 = all)
#   MAX_IMAGES          - Max images per performer (default: 5)
#   RATE_LIMIT          - Seconds between requests (default: 0.25)
#   DATA_DIR            - Output directory (default: /app/data)
#   PUBLISH             - Set to "true" to publish release after build
#   GH_TOKEN            - GitHub token for publishing releases
#   GH_REPO             - GitHub repo (e.g., "username/stash-face-recognition")

# Defaults
STASHDB_URL="${STASHDB_URL:-https://stashdb.org/graphql}"
MAX_PERFORMERS="${MAX_PERFORMERS:-200000}"
MAX_IMAGES="${MAX_IMAGES:-5}"
RATE_LIMIT="${RATE_LIMIT:-0.25}"
DATA_DIR="${DATA_DIR:-/app/data}"
PUBLISH="${PUBLISH:-false}"

echo "=========================================="
echo "Stash Face Recognition Database Builder"
echo "=========================================="
echo "StashDB URL: $STASHDB_URL"
echo "Max performers: $MAX_PERFORMERS"
echo "Max images/performer: $MAX_IMAGES"
echo "Rate limit: ${RATE_LIMIT}s"
echo "Output: $DATA_DIR"
echo "Publish: $PUBLISH"
echo "=========================================="

# Check required env vars
if [ -z "$STASHDB_API_KEY" ]; then
    echo "ERROR: STASHDB_API_KEY is required"
    exit 1
fi

# Create output directory
mkdir -p "$DATA_DIR"

# Run the database builder
cd /app/api
python database_builder.py \
    --max-performers "$MAX_PERFORMERS" \
    --max-images "$MAX_IMAGES" \
    --rate-limit "$RATE_LIMIT" \
    --output "$DATA_DIR" \
    --resume

BUILD_EXIT=$?

if [ $BUILD_EXIT -ne 0 ]; then
    echo "ERROR: Database build failed with exit code $BUILD_EXIT"
    exit $BUILD_EXIT
fi

echo ""
echo "Build complete!"

# Read manifest for version info
if [ -f "$DATA_DIR/manifest.json" ]; then
    VERSION=$(python3 -c "import json; print(json.load(open('$DATA_DIR/manifest.json'))['version'])")
    PERFORMER_COUNT=$(python3 -c "import json; print(json.load(open('$DATA_DIR/manifest.json'))['performer_count'])")
    FACE_COUNT=$(python3 -c "import json; print(json.load(open('$DATA_DIR/manifest.json'))['face_count'])")

    echo "Version: $VERSION"
    echo "Performers: $PERFORMER_COUNT"
    echo "Faces: $FACE_COUNT"
fi

# Publish release if requested
if [ "$PUBLISH" = "true" ]; then
    echo ""
    echo "Publishing release..."

    if [ -z "$GH_TOKEN" ]; then
        echo "ERROR: GH_TOKEN required for publishing"
        exit 1
    fi

    if [ -z "$GH_REPO" ]; then
        echo "ERROR: GH_REPO required for publishing (e.g., username/repo)"
        exit 1
    fi

    # Create tarball
    TARBALL="stash-face-db-${VERSION}.tar.gz"
    echo "Creating $TARBALL..."
    tar -czf "/tmp/$TARBALL" -C "$DATA_DIR" \
        face_facenet.voy \
        face_arcface.voy \
        performers.json \
        faces.json \
        manifest.json

    TARBALL_SIZE=$(du -h "/tmp/$TARBALL" | cut -f1)
    echo "Tarball size: $TARBALL_SIZE"

    # Create GitHub release
    echo "Creating GitHub release v$VERSION..."
    export GH_TOKEN

    RELEASE_NOTES="## Stash Face Recognition Database v$VERSION

- **Performers:** $PERFORMER_COUNT
- **Faces:** $FACE_COUNT
- **Source:** StashDB

### Installation

1. Download \`$TARBALL\`
2. Extract to your data directory:
   \`\`\`bash
   tar -xzf $TARBALL -C /path/to/data
   \`\`\`
3. Restart the face recognition API

### Checksums
\`\`\`
$(sha256sum /tmp/$TARBALL)
\`\`\`
"

    gh release create "v$VERSION" \
        --repo "$GH_REPO" \
        --title "Database v$VERSION" \
        --notes "$RELEASE_NOTES" \
        "/tmp/$TARBALL"

    echo "Release published: https://github.com/$GH_REPO/releases/tag/v$VERSION"

    # Cleanup
    rm -f "/tmp/$TARBALL"
fi

echo ""
echo "Done!"
