# Reference Site Enrichment Strategy

**Date:** 2026-01-26
**Status:** Designed

---

## Overview

Reference sites (Babepedia, IAFD, Boobpedia, FreeOnes, etc.) serve as **embedding enrichment sources** to improve face recognition accuracy. Unlike stash-boxes, these sites:

- Are not authoritative for metadata
- Do not provide stash_ids for cross-linking
- Often have more/better images per performer (professional headshots, multiple angles, different eras)
- Are less curated and may have errors

**Primary Purpose:** Add more face embeddings to existing performers, not create new performers.

**Related:** See [Performer Identity Graph](2026-01-27-performer-identity-graph.md) for the overall cross-source linking strategy. Reference sites contribute to the identity graph by:
1. Adding face embeddings (improving match reliability)
2. Providing external URLs (enabling cross-referencing)

---

## Why More Embeddings = Better Accuracy

A performer with diverse embeddings matches more reliably because the model has seen:

- Different angles (profile vs front-facing)
- Different lighting conditions
- Different ages/eras of their career
- Different makeup/styling
- Professional photos vs scene screenshots

A performer with 10 diverse embeddings will match more reliably than one with 2 similar ones.

---

## Trust Hierarchy

```
1. Face embedding match (primary gate - required)
2. Fuzzy name validation (confidence boost - not required but increases trust)
3. Stash-box data always authoritative for metadata
```

Reference sites are **never** trusted for:
- Creating new performers
- Overriding metadata from stash-boxes
- Linking performers across databases (that's what face embeddings do)

---

## Source Categories

### Tier 1: Stash-Boxes (Authoritative)
- StashDB, ThePornDB, PMVStash, FansDB, JAVStash
- Provide stash_ids for cross-linking
- Can create new performers
- Metadata is trusted

### Tier 2: Reference Sites (Enrichment Only)
- Babepedia, IAFD, Boobpedia, FreeOnes, etc.
- No stash_ids
- Only add embeddings to existing performers
- Metadata used only for fuzzy name validation

---

## Scraping Engine Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Scraping Engine                          │
├─────────────────────────────────────────────────────────────┤
│  ScraperRegistry                                            │
│  ├─ BabepediaScraper                                        │
│  ├─ IAFDScraper                                             │
│  ├─ BoobpediaScraper                                        │
│  └─ FreeOnesScraper                                         │
├─────────────────────────────────────────────────────────────┤
│  Common Interface (per scraper):                            │
│  ├─ list_performers() → Iterator[PerformerStub]            │
│  ├─ get_performer(id) → PerformerDetail                    │
│  └─ get_images(id) → Iterator[ImageURL]                    │
├─────────────────────────────────────────────────────────────┤
│  Per-Scraper Configuration:                                 │
│  ├─ rate_limit: float (requests per second)                │
│  ├─ needs_flaresolverr: bool                               │
│  ├─ base_url: str                                          │
│  ├─ retry_config: RetryConfig                              │
│  └─ enabled: bool                                          │
├─────────────────────────────────────────────────────────────┤
│  Request Layer:                                             │
│  ├─ RateLimiter (per-scraper, configurable)                │
│  ├─ FlareSolverr proxy (optional, for Cloudflare sites)    │
│  ├─ Retry logic (exponential backoff on 429/503)           │
│  └─ Session management (cookies, headers)                  │
└─────────────────────────────────────────────────────────────┘
```

### Scraper Interface

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator

@dataclass
class ScraperConfig:
    rate_limit: float  # requests per second
    needs_flaresolverr: bool
    base_url: str
    enabled: bool = True

@dataclass
class PerformerStub:
    id: str
    name: str
    url: str

@dataclass
class PerformerDetail:
    id: str
    name: str
    aliases: list[str]
    image_urls: list[str]
    external_urls: dict[str, str]  # source -> url (twitter, IAFD, instagram, etc.)

class BaseScraper(ABC):
    config: ScraperConfig

    @abstractmethod
    def list_performers(self) -> Iterator[PerformerStub]:
        """Iterate through all performers on the site."""
        pass

    @abstractmethod
    def get_performer(self, performer_id: str) -> PerformerDetail:
        """Get detailed info for a specific performer."""
        pass

    @abstractmethod
    def get_images(self, performer_id: str) -> Iterator[str]:
        """Get all image URLs for a performer."""
        pass
```

### Rate Limiting

Each scraper has its own rate limit configuration:

| Site | Estimated Rate Limit | FlareSolverr |
|------|---------------------|--------------|
| Babepedia | 1 req/sec | No |
| IAFD | 2 req/sec | No |
| Boobpedia | 1 req/sec | Maybe |
| FreeOnes | 0.5 req/sec | Yes |

Rate limits should be:
- Configurable per scraper
- Adaptive based on response codes (slow down on 429)
- Respectful of site resources

### FlareSolverr Integration

Some sites use Cloudflare protection. FlareSolverr handles these challenges:

```
┌──────────┐     ┌──────────────┐     ┌─────────────┐
│ Scraper  │────►│ FlareSolverr │────►│ Target Site │
└──────────┘     │ (headless    │     └─────────────┘
                 │  browser)    │
                 └──────────────┘
```

- FlareSolverr runs as optional Docker container
- Scrapers that need it route requests through it
- Adds latency but bypasses Cloudflare challenges

---

## Enrichment Flow

```python
def enrich_from_reference_site(scraper: BaseScraper):
    """Add embeddings and external URLs from a reference site to existing performers.

    Reference sites contribute to the identity graph (see performer-identity-graph.md)
    by providing additional face embeddings and cross-reference URLs.
    """

    for performer_stub in scraper.list_performers():
        performer_detail = scraper.get_performer(performer_stub.id)
        images = performer_detail.image_urls

        for image_url in images:
            # Download and process image
            image = download_image(image_url)
            faces = detect_faces(image)

            for face in faces:
                embedding = generate_embedding(face)

                # Primary gate: face match against database
                matches = search_database(
                    embedding,
                    threshold=ENRICHMENT_THRESHOLD  # stricter than normal
                )

                if not matches:
                    # No match - skip (don't create new performers)
                    continue

                best_match = matches[0]

                # Confidence validation: fuzzy name check
                name_similarity = fuzzy_name_match(
                    query_name=performer_stub.name,
                    candidate_names=[best_match.name] + best_match.aliases
                )

                # Decision logic
                if embedding_distance < VERY_CONFIDENT_THRESHOLD:
                    # Very confident face match - add regardless of name
                    add_embedding(best_match, embedding, source=scraper.name)

                elif name_similarity > NAME_SIMILARITY_THRESHOLD:
                    # Good face match + name matches - add
                    add_embedding(best_match, embedding, source=scraper.name)

                else:
                    # Face matches but name doesn't - log for review
                    log_suspicious_match(
                        source_performer=performer_stub,
                        matched_performer=best_match,
                        embedding_distance=embedding_distance,
                        name_similarity=name_similarity
                    )

        # Also extract external URLs to add to identity graph
        for source, url in performer_detail.external_urls.items():
            if best_match and source not in best_match.external_urls:
                add_external_url(best_match, source, url)
```

---

## External URL Extraction

Reference sites often link to other profiles, contributing to the identity graph:

| Site | External Links Available |
|------|-------------------------|
| Babepedia | IAFD, Twitter, Instagram, OnlyFans |
| IAFD | Some social links |
| FreeOnes | Twitter, Instagram, Snapchat, OnlyFans |
| Indexxx | Various |

```python
def extract_external_urls(html: str, base_url: str) -> dict[str, str]:
    """Parse reference site page for external profile links."""
    urls = {}

    # Common patterns to look for
    patterns = {
        "twitter": r"(?:twitter\.com|x\.com)/(\w+)",
        "instagram": r"instagram\.com/(\w+)",
        "onlyfans": r"onlyfans\.com/(\w+)",
        "iafd": r"iafd\.com/person\.rme/perfid=(\w+)",
        "freeones": r"freeones\.com/([^/]+)",
    }

    for source, pattern in patterns.items():
        match = re.search(pattern, html)
        if match:
            urls[source] = match.group(0)

    return urls
```

These URLs are added to the performer's `external_urls` field, enabling cross-referencing across the identity graph.

---

## Fuzzy Name Matching

Names can differ in many ways:
- "Jane Doe" vs "Doe, Jane" (ordering)
- "Jane Doe" vs "Jane Marie Doe" (middle names)
- "Jane Doe" vs "Janie Doe" (nicknames)
- "Müller" vs "Mueller" (transliteration)

### Algorithm Options

| Algorithm | Pros | Cons |
|-----------|------|------|
| Levenshtein | Simple, well-understood | Sensitive to length differences |
| Jaro-Winkler | Good for names, weights prefix | May miss reordering |
| Token-based (Jaccard) | Handles word reordering | Ignores character similarity |

**Recommendation:** Hybrid approach
1. Normalize names (lowercase, remove punctuation)
2. Split into tokens
3. Try both orderings for two-word names
4. Use Jaro-Winkler on best token alignment

```python
def fuzzy_name_match(query_name: str, candidate_names: list[str]) -> float:
    """Return similarity score 0-1 for best matching name."""
    query_normalized = normalize_name(query_name)

    best_score = 0.0
    for candidate in candidate_names:
        candidate_normalized = normalize_name(candidate)

        # Try direct comparison
        score = jaro_winkler(query_normalized, candidate_normalized)

        # Try reversed token order for "First Last" vs "Last, First"
        query_tokens = query_normalized.split()
        if len(query_tokens) == 2:
            reversed_query = f"{query_tokens[1]} {query_tokens[0]}"
            score = max(score, jaro_winkler(reversed_query, candidate_normalized))

        best_score = max(best_score, score)

    return best_score
```

---

## Thresholds

| Threshold | Value | Purpose |
|-----------|-------|---------|
| ENRICHMENT_THRESHOLD | TBD (stricter than normal) | Face match required to consider adding |
| VERY_CONFIDENT_THRESHOLD | TBD | Face match so good we skip name check |
| NAME_SIMILARITY_THRESHOLD | 0.7 (tentative) | Fuzzy name match required if not very confident |

These need tuning based on:
- False positive rate we're willing to accept
- Quality of reference site data
- Embedding model performance

---

## Suspicious Match Handling

When face matches but name doesn't:

```python
@dataclass
class SuspiciousMatch:
    source_site: str
    source_performer_id: str
    source_performer_name: str
    source_image_url: str
    matched_performer_id: str
    matched_performer_name: str
    embedding_distance: float
    name_similarity: float
    timestamp: datetime
```

Options:
1. **Log only** - Write to file for manual review later
2. **Skip entirely** - Conservative, may miss valid matches
3. **Add with flag** - Add embedding but mark as "unverified"

**Recommendation:** Log only (option 1) initially. Review logs to tune thresholds, then potentially move to option 3.

---

## FlareSolverr Deployment

FlareSolverr is an optional dependency for sites with Cloudflare protection.

```yaml
# docker-compose.yml addition
services:
  flaresolverr:
    image: ghcr.io/flaresolverr/flaresolverr:latest
    container_name: flaresolverr
    environment:
      - LOG_LEVEL=info
    ports:
      - "8191:8191"
    restart: unless-stopped
```

Configuration:
- `FLARESOLVERR_URL`: URL of FlareSolverr instance (default: `http://flaresolverr:8191/v1`)
- Per-scraper `needs_flaresolverr` flag

---

## Smart Enrichment Strategies

### Prioritized Enrichment Queue

Not all performers benefit equally from enrichment. Process in priority order:

```python
# Priority scoring for enrichment queue
def enrichment_priority(performer) -> float:
    score = 0.0

    # High-activity performers first (more likely to appear in user libraries)
    score += min(performer.scene_count / 100, 1.0) * 40

    # Low embedding count = high need
    if performer.face_count == 1:
        score += 30
    elif performer.face_count < 5:
        score += 20

    # Multi-stash-box presence = higher value
    score += len(performer.stash_ids) * 10

    # Has reference site URLs we can use
    score += len(performer.enrichable_urls) * 5

    return score
```

| Priority | Criteria | Rationale |
|----------|----------|-----------|
| **P0** | scene_count > 50, face_count = 1 | High activity but unreliable matching |
| **P1** | face_count < 3 | Need diversity for reliable matching |
| **P2** | In multiple stash-boxes | Higher value, more cross-linking potential |
| **P3** | Has unused reference URLs | Easy wins, URLs already known |
| **P4** | Everyone else | Completeness |

### Alias-Aware Scraping

Performers have multiple names. Search reference sites with ALL known aliases:

```python
def search_reference_sites(performer):
    # Collect all name variations
    search_names = [performer.name] + performer.aliases

    # JAV performers: add romanization variations
    if is_japanese_name(performer.name):
        search_names.extend([
            romanize(performer.name),                    # 波多野結衣 → Hatano Yui
            western_order(romanize(performer.name)),     # Hatano Yui → Yui Hatano
        ])

    # Search each reference site with each name
    for site in reference_sites:
        for name in search_names:
            results = site.search(name)
            if results:
                return results  # Found match

    return None
```

**Special Cases:**
- **JAV:** Japanese (波多野結衣) vs Romanized (Hatano Yui) vs Western order (Yui Hatano)
- **Stage names:** Legal name vs performer name
- **Spelling variations:** Kenzie vs Kenzi vs Kinzie

### Temporal Diversity

Performers change over their careers. Seek embeddings from different eras:

```python
def select_diverse_images(images, performer):
    # If we have career dates, bucket images by era
    if performer.career_start_year and performer.career_end_year:
        career_length = performer.career_end_year - performer.career_start_year
        if career_length > 5:
            # Try to get images from early, mid, and late career
            # (Requires image dates or inference from context)
            pass

    # At minimum, ensure visual diversity
    # Different: angles, lighting, hair color/style, makeup
    return select_visually_diverse(images, target_count=10)
```

**Goal:** An embedding set that covers how a performer looks across their career, not just one photoshoot.

### Image Quality Scoring

Not all images produce equally reliable embeddings:

```python
def image_quality_score(image) -> float:
    score = 1.0

    # Resolution (higher = better embeddings)
    if image.width < 256 or image.height < 256:
        score *= 0.5  # Low-res images are noisy
    elif image.width >= 512 and image.height >= 512:
        score *= 1.2  # High-res bonus

    # Face size relative to image
    face_ratio = face_bbox_area / image_area
    if face_ratio < 0.05:
        score *= 0.7  # Face too small
    elif face_ratio > 0.3:
        score *= 1.1  # Good face coverage

    # Detection confidence
    score *= face_detection_confidence

    return score

# Use quality scores when matching
def weighted_match(query_embedding, performer_embeddings):
    distances = []
    for emb, quality in performer_embeddings:
        dist = cosine_distance(query_embedding, emb)
        # Weight by quality: high-quality embeddings count more
        distances.append((dist, quality))

    return weighted_average(distances)
```

### Embedding Diversity via Clustering

Avoid redundant embeddings from the same photoshoot:

```python
def dedupe_embeddings(embeddings, target_count=10):
    """Keep only diverse, representative embeddings."""

    # Cluster similar embeddings
    clusters = cluster_by_similarity(embeddings, threshold=0.3)

    # Keep one representative from each cluster
    # Prefer highest quality from each cluster
    representatives = []
    for cluster in clusters:
        best = max(cluster, key=lambda e: e.quality_score)
        representatives.append(best)

    # If too many, keep most diverse
    if len(representatives) > target_count:
        representatives = select_most_diverse(representatives, target_count)

    return representatives
```

**Benefit:** 10 diverse embeddings > 50 near-duplicate embeddings from same photoshoot.

---

## Open Questions

1. **Threshold tuning:** Need empirical data to set ENRICHMENT_THRESHOLD and VERY_CONFIDENT_THRESHOLD
2. **Suspicious match review UI:** How to present suspicious matches for manual review?
3. **Embedding deduplication:** If reference site has same image as stash-box, do we detect and skip?
4. **Incremental updates:** How to track which performers/images we've already processed?
5. **Site-specific challenges:** Each site may have unique scraping challenges (pagination, lazy loading, etc.)

---

## Implementation Priority

1. **Scraping engine framework** - Base classes, rate limiting, retry logic
2. **FlareSolverr integration** - Optional proxy support
3. **First scraper (Babepedia)** - Most straightforward, good test case
4. **Enrichment pipeline** - Face matching + name validation logic
5. **Additional scrapers** - IAFD, Boobpedia, FreeOnes as needed

---

*This document will be updated as implementation progresses.*
