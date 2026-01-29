# Performer Identity Graph

**Date:** 2026-01-27
**Status:** Design

---

## Overview

Build a unified identity graph that links performers across all stash-box endpoints and reference sites. The goal is: given any performer ID from any source, return all known IDs for that person.

## Schema Consistency

All stash-box endpoints (StashDB, PMVStash, JAVStash, FansDB) use the **identical GraphQL schema**:

```graphql
Performer {
  id, name, aliases, gender, birth_date, country,
  career_start_year, career_end_year,
  urls { url, site { name } },
  images { url },
  merged_ids,
  # ... and more
}
```

This means we can use the same client code for all endpoints.

---

## Cross-Linking Between Stash-Boxes

Stash-box endpoints already link to each other via the `urls` field:

| Source | Links To | Site Name in URLs |
|--------|----------|-------------------|
| **PMVStash** | StashDB | "StashDB performer" |
| **JAVStash** | StashDB | "StashDB" |
| **JAVStash** | ThePornDB | "ThePornDB" |
| **StashDB** | ThePornDB | "ThePornDB" |
| **ThePornDB** | StashDB | `extras.links.StashDB` (~3.3%) |

**Example JAVStash performer URLs (波多野結衣):**
- StashDB: `https://stashdb.org/performers/{uuid}`
- ThePornDB: `https://theporndb.net/performers/{slug}`
- IAFD, IMDb, Twitter, Instagram, Wikipedia, Wikidata
- JAV-specific: DMM/FANZA, R18.dev, Minnano-AV, XCity, XsList, WAPdB

---

## URL Sites by Endpoint

### StashDB (~103k performers)
High-frequency sites from top performers:
- IAFD, AFDB, DATA18, dbNaked, Pornhub, IMDb
- Twitter, Instagram, OnlyFans, Fansly
- Wikidata, Wikipedia
- Studio Profile (various)

### PMVStash (~6k performers)
- StashDB performer ← **direct link to StashDB!**
- Babepedia, IAFD profile, Pornhub profile
- Instagram, TikTok, YouTube
- x.com (twitter), OnlyFans
- "link" (generic external links)

### JAVStash (~18k performers)
- StashDB, ThePornDB ← **direct links to other stash-boxes!**
- IAFD, IMDb, FreeOnes, Babepedia, Boobpedia
- Twitter, Instagram, YouTube
- Wikidata, Wikipedia (often Japanese)
- **JAV-specific:**
  - DMM/FANZA, R18.dev (performer pages)
  - Minnano-AV, XCity, XsList
  - WAPdB, SougouWiki, av-wiki
  - 1pondo.tv, Caribbeancom, HEYZO, Tokyo-Hot
  - AV LEAGUE, Modeling Agency

### FansDB (needs API key)
- Likely: OnlyFans, Fansly, Instagram, Twitter, TikTok
- May link to StashDB

---

## URL-First Resolution Strategy

**Key Insight:** Face matching is slow and error-prone. URL matching is fast and reliable. We should exhaust URL-based linking before falling back to face recognition.

### Phase 1: Build Global URL Index

Before processing any performer, build a complete URL index across ALL sources:

```python
# Pseudocode for URL-first approach
url_index = {}  # normalized_url -> identity_id

# Pass 1: Index all URLs from all stash-boxes
for endpoint in ['stashdb', 'pmvstash', 'javstash', 'fansdb', 'theporndb']:
    for performer in endpoint.all_performers():
        for url in performer.urls:
            normalized = normalize_url(url)
            url_index[normalized] = performer.id

# Pass 2: Cluster by shared URLs
# Performers sharing ANY URL are the same person
clusters = union_find_by_shared_urls(url_index)

# Pass 3: Face matching ONLY for performers with no URL overlap
unlinked = performers_not_in_any_cluster(clusters)
for performer in unlinked:
    # Now face matching is a fallback, not primary strategy
    face_match(performer)
```

**Why This Is Better:**
- URL matching is O(1) lookup vs O(n) face comparison
- URLs are deterministic - no threshold tuning needed
- StashDB already has rich URL data (IAFD, Twitter, IMDb, etc.)
- Face matching becomes fallback for edge cases only

### Phase 2: Incremental URL Discovery

As we process reference sites, new URLs get added:

```python
# When enriching from Babepedia, we might discover:
# - IAFD link we didn't have
# - Twitter handle that matches another performer
# Each new URL potentially links previously-unlinked performers
```

---

## Identity Resolution Priority

### Priority 1: Explicit Cross-Links
When a stash-box performer has a URL pointing to another stash-box:

```python
# PMVStash performer has "StashDB performer" URL
stashdb_url = "https://stashdb.org/performers/abc-123"
stashdb_id = stashdb_url.split("/performers/")[-1]
# → Direct identity link established
```

### Priority 2: Shared External URLs
When performers share the same external URL:

```python
# Both StashDB and JAVStash link to same IAFD page
iafd_url = "https://www.iafd.com/person.rme/perfid=miaalkhalifa/Mia-Khalifa.htm"
# → High confidence same person
```

**High-confidence shared URL types:**
- IAFD (unique performer ID)
- IMDb (unique ID)
- Wikidata (unique Q-number)
- Twitter/X (unique username)
- Instagram (unique username)
- OnlyFans/Fansly (unique username)

**Lower-confidence (may have duplicates):**
- Studio Profile URLs
- Generic "link" entries

### Priority 3: Name + Metadata Matching
When no URL overlap exists:

```python
# Match by: exact name + same country + similar birth_date
if (name_match > 0.95 and
    country_match and
    birth_date_within_1_year):
    # → Medium confidence, may need face verification
```

### Priority 4: Face Recognition
When metadata doesn't match or is missing:

```python
# Compare face embeddings
if face_distance < 0.45 and name_similarity > 0.7:
    # → Medium confidence, add to review queue
```

---

## Handling New URLs from Other Endpoints

When processing a performer from a secondary endpoint (PMVStash, JAVStash, FansDB):

### Case 1: Has explicit stash-box link
```python
if has_stashdb_url(performer):
    stashdb_id = extract_stashdb_id(performer)
    existing = lookup_by_stashdb_id(stashdb_id)
    # Merge all new URLs into existing identity
    for url in performer.urls:
        if url not in existing.urls:
            existing.add_url(url)
            # New URL can help match OTHER performers later
            add_to_url_index(url, existing)
```

### Case 2: Has shared external URL (IAFD, Twitter, etc.)
```python
for url in performer.urls:
    normalized = normalize_url(url)
    existing = lookup_by_url(normalized)
    if existing:
        # Same person, different stash-box entry
        existing.add_stashbox_id(endpoint, performer.id)
        merge_new_urls(existing, performer.urls)
        break
```

### Case 3: No URL overlap - try face matching
```python
face_embeddings = extract_faces(performer.images)
matches = query_face_index(face_embeddings, threshold=0.45)
if matches:
    # Review queue - need human confirmation
    add_to_review_queue(performer, matches)
else:
    # New performer identity
    create_new_identity(performer)
```

### Case 4: New performer (not in any other stash-box)
```python
identity = PerformerIdentity(
    name=performer.name,
    aliases=performer.aliases,
    # Primary ID is this endpoint
    **{f"{endpoint}_id": performer.id}
)
# Index all their URLs for future matching
for url in performer.urls:
    add_to_url_index(url, identity)
```

---

## URL Normalization Rules

| Site | URL Pattern | Normalized ID |
|------|-------------|---------------|
| StashDB | `/performers/{uuid}` | UUID |
| ThePornDB | `/performers/{slug}` | slug |
| PMVStash | `/performers/{uuid}` | UUID |
| JAVStash | `/performers/{uuid}` | UUID |
| FansDB | `/performers/{uuid}` | UUID |
| IAFD | `perfid={slug}` | slug |
| IMDb | `/name/nm{id}` | nm{id} |
| Wikidata | `/wiki/Q{id}` | Q{id} |
| Twitter | `twitter.com/{handle}` or `x.com/{handle}` | lowercase handle |
| Instagram | `instagram.com/{handle}` | lowercase handle |
| OnlyFans | `onlyfans.com/{handle}` | lowercase handle |
| Fansly | `fansly.com/{handle}` | lowercase handle |
| Pornhub | `/pornstar/{slug}` | slug |
| DMM/FANZA | `/digital/videoa/-/list/=/article=actress/id={id}` | id |
| R18.dev | `/idols/{slug}` | slug |

---

## Database Schema

```sql
-- Identity graph stored in SQLite for efficient querying

CREATE TABLE performer_identities (
    id INTEGER PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    gender TEXT,
    country TEXT,
    birth_date TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE stashbox_ids (
    identity_id INTEGER REFERENCES performer_identities(id),
    endpoint TEXT NOT NULL,  -- 'stashdb', 'pmvstash', 'javstash', 'fansdb', 'theporndb'
    performer_id TEXT NOT NULL,
    PRIMARY KEY (endpoint, performer_id)
);

CREATE TABLE external_urls (
    identity_id INTEGER REFERENCES performer_identities(id),
    site TEXT NOT NULL,  -- 'iafd', 'imdb', 'twitter', etc.
    url TEXT NOT NULL,
    normalized_id TEXT,  -- Extracted ID where applicable
    source_endpoint TEXT,  -- Which stash-box provided this URL
    PRIMARY KEY (site, normalized_id)
);

CREATE TABLE aliases (
    identity_id INTEGER REFERENCES performer_identities(id),
    alias TEXT NOT NULL,
    source_endpoint TEXT  -- Where we learned this alias
);

-- Index for fast lookup by any identifier
CREATE INDEX idx_stashbox_performer ON stashbox_ids(performer_id);
CREATE INDEX idx_external_urls_normalized ON external_urls(site, normalized_id);
CREATE INDEX idx_aliases_alias ON aliases(alias COLLATE NOCASE);
```

---

## Statistics (Current)

| Endpoint | Performers | Avg URLs | Has StashDB Link |
|----------|------------|----------|------------------|
| StashDB | 103,580 | ~10 | - |
| PMVStash | 5,812 | ~20 | Yes ("StashDB performer") |
| JAVStash | 18,528 | ~30 | Yes ("StashDB") |
| FansDB | Unknown | Unknown | Unknown |
| ThePornDB | ~10,000 | ~5 | ~3.3% have explicit link |

---

## Open Questions

1. **FansDB access:** Need API key to analyze their URL patterns
2. **Duplicate handling:** What if two StashDB performers link to same external URL?
3. **Stale links:** How to handle broken/outdated URLs?
4. **Merge direction:** If JAVStash has more data than StashDB, which takes precedence?
5. **Review queue:** How to present uncertain matches to users?
6. **Name conflicts:** PMVStash may use different name spelling than StashDB

---

## Next Steps

1. [ ] Add FansDB API key to .env
2. [ ] Create unified stash-box client that works with all endpoints
3. [ ] Build URL normalization library
4. [ ] Crawl PMVStash and JAVStash for identity data
5. [ ] Build SQLite identity graph database
6. [ ] Implement conflict resolution rules
7. [ ] Add face matching for performers without URL overlap

---

*This document will be updated as implementation progresses.*
