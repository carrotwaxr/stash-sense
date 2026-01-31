# Stash-Box Endpoints

> **Note:** Database building has moved to the **stash-sense-trainer** repository.
> This document is preserved for reference but the build commands below apply to the trainer repo.

This document describes the different stash-box endpoints supported for building face recognition databases, their characteristics, and considerations for each.

## Supported Endpoints

| Endpoint | API Type | Performers | Images/Performer | Rate Limit |
|----------|----------|------------|------------------|------------|
| [StashDB](https://stashdb.org) | GraphQL | ~100,000+ | Multiple | 240/min safe |
| [ThePornDB](https://theporndb.net) | **REST** | ~10,000 | Multiple | 240/min safe |
| [PMVStash](https://pmvstash.org) | GraphQL | ~6,500 | 3.7 avg | 300/min+ |
| [JAVStash](https://javstash.org) | GraphQL | ~21,700 | 1.0 avg | 300/min+ |
| [FansDB](https://fansdb.cc) | GraphQL | Unknown | Unknown | Untested |

## Endpoint Details

### StashDB (Primary)

- **URL**: `https://stashdb.org/graphql`
- **API**: Standard stash-box GraphQL
- **Use**: `database_builder.py` directly
- **Notes**: Largest database, highest quality metadata

### ThePornDB

- **URL**: `https://api.theporndb.net` (REST API)
- **API**: REST, NOT GraphQL (despite Stash config showing `/graphql`)
- **Use**: `build_theporndb.py` (custom script)
- **Notes**:
  - Has pre-cropped face thumbnails (`face_url` field)
  - Many performers cross-reference StashDB IDs
  - Rate limit ~275/min, safe at 240/min

### PMVStash

- **URL**: `https://pmvstash.org/graphql`
- **API**: Standard stash-box GraphQL
- **Use**: `database_builder.py` with env override
- **Notes**:
  - PMV-focused content
  - Good image coverage (100% have images)
  - 34% have multiple images (avg 3.7)
  - Portrait images (1280x1920)

### JAVStash

- **URL**: `https://javstash.org/graphql`
- **API**: Standard stash-box GraphQL
- **Use**: `database_builder.py` with env override
- **Notes**:
  - JAV-focused content with Japanese names
  - **Limited to ~1 image per performer** (see below)
  - Square images (705x705)

## Running Builds

```bash
# These commands apply to the stash-sense-trainer repo
cd /home/carrot/code/stash-sense-trainer/api
source ../.venv/bin/activate

# StashDB (default)
python database_builder.py --output ./data --resume

# PMVStash
STASHDB_URL=$PMVSTASH_URL STASHDB_API_KEY=$PMVSTASH_API_KEY \
  python database_builder.py --output ./data-pmvstash --resume

# JAVStash
STASHDB_URL=$JAVSTASH_URL STASHDB_API_KEY=$JAVSTASH_API_KEY \
  python database_builder.py --output ./data-javstash --resume

# ThePornDB (different script)
python build_theporndb.py --output ./data-theporndb
```

---

## JAVStash: Single Image Limitation

**Problem**: JAVStash performers typically have only 1 image, which limits face recognition accuracy. Our standard approach of building multiple embeddings per performer (target: 5) doesn't work here.

### Why This Matters

- Single embedding = higher false positive/negative rates
- No redundancy for poor-quality source images
- Can't leverage voting across multiple embeddings

### Potential Solutions (TODO: Discuss)

1. **Scene Frame Extraction**
   - JAVStash has scene data with fingerprints
   - Could extract frames from matched scenes in user's library
   - Build embeddings from actual video content
   - Requires: scene matching pipeline, user's local content

2. **External Image Sources**
   - Cross-reference with other databases (e.g., JAV actress databases)
   - Scrape additional images from linked URLs in performer metadata
   - Ethical/legal considerations for scraping

3. **Data Augmentation**
   - Generate synthetic variations of single image
   - Horizontal flip, slight rotations, color adjustments
   - Risk: may not improve real-world accuracy

4. **Lower Confidence Threshold**
   - Accept that JAV matching will be lower confidence
   - Use separate threshold for JAV vs other sources
   - Let users decide on match acceptance

5. **Ensemble with ThePornDB**
   - Some JAV content exists on ThePornDB
   - Cross-reference and merge embeddings where possible
   - ThePornDB has StashDB links that might map to JAVStash

6. **User-Contributed Images**
   - Allow users to contribute additional performer images
   - Build community-sourced image database
   - Privacy/consent considerations

### Recommended Approach

**Short-term:** Options 1 (scene frames) and 4 (adjusted thresholds):
- Don't require external data sources
- Leverage data users already have
- Can be implemented incrementally

**Long-term:** Option 7 - **Crowd-Sourced Cloud Service**

The ultimate solution is a public cloud service (similar to stash-box) where users can submit face embeddings extracted from their identified scenes. This would:
- Bypass the single-image limitation entirely
- Build models from real scene content, not just promo images
- Scale with the community (more users = better models)
- Preserve privacy (only embeddings, never raw images)

See [project-status-and-vision.md](2026-01-26-project-status-and-vision.md#crowd-sourced-face-database-cloud-service) for full details on the cloud service vision.

---

## Rate Limit Reference

Tested January 2026:

| Endpoint | Tested Rate | Result |
|----------|-------------|--------|
| StashDB | 240/min | ✅ No issues (running 24h+) |
| ThePornDB | 240/min | ✅ Safe (limit ~275/min) |
| ThePornDB | 360/min | ❌ 429s at ~275 requests |
| PMVStash | 300/min | ✅ No issues |
| JAVStash | 300/min | ✅ No issues (slower response) |

**Recommendation**: Use 240/min (0.25s delay) for all endpoints as a safe default.
