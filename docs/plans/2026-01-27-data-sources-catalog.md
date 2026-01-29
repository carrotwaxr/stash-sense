# Data Sources Catalog

**Date:** 2026-01-27
**Status:** Research in Progress

---

## Overview

This document catalogs all potential data sources for building the performer identity graph, including API details, rate limits, URL patterns, and scraping strategies.

**Matching Data Available from StashDB:**
- Name
- Aliases (array)
- URLs (array of external links)
- Country
- Birth date
- Images

---

## Tier 1: Stash-Box Endpoints (Authoritative)

These sources have structured APIs and can create new performers.

### 1. StashDB (Primary)

| Property | Value |
|----------|-------|
| **URL** | `https://stashdb.org/graphql` |
| **API Type** | GraphQL |
| **Performers** | ~103,000+ |
| **Images/Performer** | 1-5 (avg ~1.04 with faces) |
| **Rate Limit** | 240/min safe (tested) |
| **Auth** | API key in `ApiKey` header |
| **Cross-Links** | Has `urls` field with external links |

**GraphQL Fields Available:**
```graphql
Performer {
  id, name, disambiguation, aliases, gender, urls,
  birth_date, death_date, age, ethnicity, country,
  eye_color, hair_color, height, cup_size, band_size,
  waist_size, hip_size, breast_type, career_start_year,
  career_end_year, tattoos, piercings, images, deleted,
  scene_count, merged_ids, created, updated
}
```

**Current Implementation:** `stashdb_client.py` - ✅ Updated to capture all identity graph fields:
- `aliases` - Stage names, variations
- `urls` - Grouped by site name (dict[str, list[str]])
- `birth_date` - For matching confidence (YYYY, YYYY-MM, or YYYY-MM-DD)
- `gender` - Prevent cross-gender false matches
- `career_start_year`, `career_end_year` - Disambiguator
- `merged_ids` - Previously merged StashDB entries

**Performer Page URL Pattern:**
```
https://stashdb.org/performers/{uuid}
```

**Search Strategy:** Already building full database via pagination.

---

### 2. ThePornDB

| Property | Value |
|----------|-------|
| **URL** | `https://api.theporndb.net` |
| **API Type** | REST (NOT GraphQL despite Stash config) |
| **Performers** | ~10,000 |
| **Images/Performer** | Multiple (posters + main image) |
| **Rate Limit** | 240/min safe, 275/min max |
| **Auth** | Bearer token in `Authorization` header |
| **Cross-Links** | `extras.links.StashDB` field (~3.3% have it) |

**Key Fields:**
```json
{
  "id": "string",
  "name": "string",
  "full_name": "string",
  "image": "url",
  "face": "url (pre-cropped!)",
  "posters": [{"url": "..."}],
  "extras": {
    "nationality": "string",
    "birthplace_code": "string",
    "links": {
      "StashDB": "https://stashdb.org/performers/{uuid}",
      "Twitter": "...",
      "Instagram": "..."
    }
  }
}
```

**Current Implementation:** `theporndb_client.py` - extracts `stashdb_id` from links

**Performer Page URL Pattern:**
```
https://theporndb.net/performers/{slug}
```

**Cross-Link Coverage:** Only ~3.3% have explicit StashDB links. Most will require name/face matching.

**Search Strategy:**
1. Iterate all performers via pagination
2. For each: check if `extras.links.StashDB` exists → direct link
3. Otherwise: match by name + face recognition

---

### 3. PMVStash

| Property | Value |
|----------|-------|
| **URL** | `https://pmvstash.org/graphql` |
| **API Type** | GraphQL (standard stash-box) |
| **Performers** | ~6,500 |
| **Images/Performer** | 3.7 avg (100% have at least 1) |
| **Rate Limit** | 300/min+ (tested safe) |
| **Auth** | API key in `ApiKey` header |
| **Cross-Links** | Unknown - needs testing |

**Performer Page URL Pattern:**
```
https://pmvstash.org/performers/{uuid}
```

**Search Strategy:** Same as StashDB - use `database_builder.py` with env override.

**TODO:** Test if PMVStash has cross-links to StashDB in performer data.

---

### 4. JAVStash

| Property | Value |
|----------|-------|
| **URL** | `https://javstash.org/graphql` |
| **API Type** | GraphQL (standard stash-box) |
| **Performers** | ~21,700 |
| **Images/Performer** | ~1.0 (major limitation!) |
| **Rate Limit** | 300/min+ (tested safe) |
| **Auth** | API key in `ApiKey` header |
| **Cross-Links** | Unlikely (JAV-specific) |

**Performer Page URL Pattern:**
```
https://javstash.org/performers/{uuid}
```

**Special Considerations:**
- Single image per performer limits face recognition accuracy
- Japanese names: family name first vs romanized Western order
- Many performers won't exist in Western databases
- Need crowd-sourced scene frames to improve embeddings

**Search Strategy:** Same as StashDB, but expect standalone performers (no cross-links).

---

### 5. FansDB

| Property | Value |
|----------|-------|
| **URL** | `https://fansdb.cc/graphql` |
| **API Type** | GraphQL (standard stash-box) |
| **Performers** | Unknown (~50k estimated) |
| **Images/Performer** | Unknown |
| **Rate Limit** | Untested |
| **Auth** | API key in `ApiKey` header |
| **Cross-Links** | Unknown |

**Performer Page URL Pattern:**
```
https://fansdb.cc/performers/{uuid}
```

**TODO:** Test FansDB API and document performer count/coverage.

---

## Tier 2: Reference Sites (Enrichment Only)

These sites provide additional images and cross-reference URLs but cannot create new performers.

### 6. IAFD (Internet Adult Film Database)

| Property | Value |
|----------|-------|
| **URL** | `https://www.iafd.com` |
| **API Type** | HTML scraping |
| **Performers** | ~200,000+ |
| **Images/Performer** | 1-3 typically |
| **Rate Limit** | ~2 req/sec estimated (needs testing) |
| **Auth** | None (public) |
| **Cloudflare** | No |

**Performer Page URL Pattern:**
```
https://www.iafd.com/person.rme/perfid={slug}/{Name}.htm
Example: https://www.iafd.com/person.rme/perfid=aaliyahhadid/Aaliyah-Hadid.htm
```

**Search URL Pattern:**
```
https://www.iafd.com/results.asp?searchtype=comprehensive&searchstring={name}
```

**Available Data:**
- Headshot image(s)
- Aliases
- Birth date, birthplace
- Ethnicity, measurements
- Career years
- External links (Twitter, Instagram, OnlyFans)
- Filmography

**Matching Strategy:**
1. Search by performer name
2. Parse results page for matching entries
3. Fetch performer page, extract image URLs
4. Match by face + name validation

**TODO:** Test scraping with proper user-agent, document exact selectors.

---

### 7. FreeOnes

| Property | Value |
|----------|-------|
| **URL** | `https://www.freeones.com` |
| **API Type** | HTML scraping |
| **Performers** | ~25,000+ |
| **Images/Performer** | Multiple (up to 200 from 20 galleries) |
| **Rate Limit** | ~0.5 req/sec (conservative for bulk) |
| **Auth** | None (public) |
| **Cloudflare** | Sometimes (works for single requests, may need FlareSolverr for bulk) |

**Performer Page URL Pattern:**
```
https://www.freeones.com/{slug}/photos
Example: https://www.freeones.com/aaliyah-hadid/photos
```

**Search URL Pattern:**
```
https://www.freeones.com/babes?q={name}
```

**Available Data:**
- Multiple photos (profile, galleries)
- Bio, measurements
- Social links (Twitter, Instagram, Snapchat, OnlyFans)
- Career info

**Scraping Strategy (from performerImageSearch):**
1. Fetch `/photos` page to get gallery URLs
2. Extract gallery links matching `/{performer}/photos/{gallery-slug}`
3. Drill into each gallery (up to 20)
4. Extract images from CDN: `thumbs.freeones.com`, `ch-thumbs.freeones.com`, `img.freeones.com`

**Note:** Single requests with browser headers work without FlareSolverr. Bulk scraping may trigger Cloudflare challenges.

---

### 8. Babepedia

| Property | Value |
|----------|-------|
| **URL** | `https://www.babepedia.com` |
| **API Type** | HTML scraping |
| **Performers** | ~30,000+ |
| **Images/Performer** | Up to 50 (curated photos) |
| **Rate Limit** | ~1 req/sec estimated |
| **Auth** | None (public) |
| **Cloudflare** | No (verified working with simple requests) |

**Performer Page URL Pattern:**
```
https://www.babepedia.com/babe/{Name_With_Underscores}
Example: https://www.babepedia.com/babe/Aaliyah_Hadid
```

**Gallery Pages:**
```
https://www.babepedia.com/babe/{Name}/gallery
https://www.babepedia.com/babe/{Name}/pics
```

**Available Data:**
- Profile image
- Gallery images (multiple pages)
- Bio, measurements
- External links (IAFD, Twitter, Instagram, OnlyFans)
- Aliases

**Scraping Strategy (from performerImageSearch):**
```python
pattern = r'href="(/pics/[^"]+\.jpg)"'
# Full image: https://www.babepedia.com{match}
# Thumbnail: image_url.replace(".jpg", "_thumb3.jpg")
```

**Matching Strategy:**
1. Construct URL from performer name (replace spaces with underscores)
2. Fetch main page + `/gallery` + `/pics` subpages
3. Extract image URLs matching `/pics/*.jpg` pattern
4. Match by face + name validation

---

### 9. Indexxx

| Property | Value |
|----------|-------|
| **URL** | `https://www.indexxx.com` |
| **API Type** | HTML scraping |
| **Performers** | ~50,000+ |
| **Images/Performer** | 1-2 typically |
| **Rate Limit** | Unknown |
| **Auth** | None (public) |
| **Cloudflare** | Unknown |

**Performer Page URL Pattern:**
```
https://www.indexxx.com/m/{Name-With-Dashes}/
Example: https://www.indexxx.com/m/Aaliyah-Hadid/
```

**Available Data:**
- Profile image
- Award history
- Filmography links
- Some social links

**TODO:** Test accessibility and document structure.

---

### 10. Boobpedia

| Property | Value |
|----------|-------|
| **URL** | `https://www.boobpedia.com` |
| **API Type** | HTML scraping (MediaWiki) |
| **Performers** | ~20,000+ |
| **Images/Performer** | Up to 50 |
| **Rate Limit** | ~1 req/sec estimated |
| **Auth** | None (public) |
| **Cloudflare** | No (verified working with simple requests) |

**Performer Page URL Pattern:**
```
https://www.boobpedia.com/boobs/{Name_With_Underscores}
Example: https://www.boobpedia.com/boobs/Aaliyah_Hadid
```

**Available Data:**
- Infobox image
- Gallery
- Bio from wiki content
- External links

**Scraping Strategy (from performerImageSearch):**
```python
# MediaWiki thumbnail format:
# /wiki/images/thumb/X/XX/Filename.jpg/NNNpx-Filename.jpg
# Full size: /wiki/images/X/XX/Filename.jpg

pattern = r'src="(/wiki/images/thumb/[^"]+)"'
# Skip small icons (16px, 18px, 20px)
# Transform thumb path to full-size path
```

**Matching Strategy:**
1. Construct URL from performer name (replace spaces with underscores)
2. Extract images from wiki page, skipping icons
3. Transform thumbnail URLs to full-size
4. Match by face + name validation

---

## Rate Limit Summary

| Source | Tested Rate | Recommended | Notes |
|--------|-------------|-------------|-------|
| StashDB | 240/min | 240/min | Running 24h+ stable |
| ThePornDB | 275/min | 240/min | Safe margin |
| PMVStash | 300/min | 300/min | Fast endpoint |
| JAVStash | 300/min | 300/min | Slower response times |
| FansDB | Untested | 240/min | Start conservative |
| IAFD | Untested | 120/min | HTML, be respectful |
| FreeOnes | Works (single) | 30/min | May need FlareSolverr for bulk |
| Babepedia | Works (single) | 60/min | No Cloudflare (verified) |
| PornPics | Works (single) | 60/min | Gallery drilling supported |
| Boobpedia | Works (single) | 60/min | No Cloudflare (verified) |
| EliteBabes | Works (single) | 60/min | High-quality images |
| JavDatabase | Works (single) | 60/min | Good JAV supplement |
| Indexxx | Untested | 60/min | HTML scraping |

---

## Cross-Link Matrix

How sources link to each other:

| Source | Links To | Link Location |
|--------|----------|---------------|
| **StashDB** | Twitter, Instagram, IAFD, FreeOnes, OnlyFans | `urls` field |
| **ThePornDB** | StashDB (~3.3%), Twitter, Instagram | `extras.links` |
| **IAFD** | Twitter, Instagram, OnlyFans | Page sidebar |
| **FreeOnes** | Twitter, Instagram, Snapchat, OnlyFans | Profile section |
| **Babepedia** | IAFD, Twitter, Instagram, OnlyFans | Infobox |
| **Indexxx** | Various | Profile links |

**Key Insight:** StashDB's `urls` field is gold - it already contains cross-references that we're not currently capturing.

**UPDATE (2026-01-27):** The `stashdb_client.py` has been updated to capture all identity graph fields.

**URL Site Frequency (Top 50 performers by scene count):**

| Site | Count | Has Images? | Example URL Pattern |
|------|-------|-------------|---------------------|
| IAFD | 51 | Yes | `iafd.com/person.rme/perfid={slug}` |
| Twitter | 47 | No | `twitter.com/{username}` |
| Wikidata | 47 | No | `wikidata.org/wiki/{id}` |
| Studio Profile | 32 | Maybe | Various studio sites |
| OnlyFans | 31 | Yes (paywall) | `onlyfans.com/{username}` |
| AFDB | 30 | Yes | `adultfilmdatabase.com/actor/{slug}` |
| DATA18 | 29 | Yes | `data18.com/name/{slug}` |
| IMDb | 25 | Yes | `imdb.com/name/{id}` |
| Instagram | 25 | Yes (private) | `instagram.com/{username}` |
| dbNaked | 23 | Yes | `dbnaked.com/models/general/{letter}/{name}` |
| Pornhub | 21 | Yes | `pornhub.com/pornstar/{slug}` |
| Wikipedia | 14 | Maybe | `en.wikipedia.org/wiki/{name}` |
| GEVI | 9 | Maybe | `gayeroticvideoindex.com/performer/{id}` |
| ManyVids | 8 | Yes | `manyvids.com/Profile/{id}/{slug}` |
| XVideos | 6 | Yes | `xvideos.com/pornstars/{slug}` |
| Linktree | 5 | No | `linktr.ee/{username}` |
| ThePornDB | 4 | Yes | `theporndb.net/performers/{uuid}` |
| YouTube | 3 | No | `youtube.com/{channel}` |
| Babepedia | 3 | Yes | `babepedia.com/babe/{Name}` |
| FreeOnes | 3 | Yes | `freeones.com/{slug}` |
| Boobpedia | 2 | Yes | `boobpedia.com/boobs/{Name}` |
| Indexxx | 2 | Yes | `indexxx.com/m/{slug}` |
| theNude | 2 | Yes | `thenude.com/{name}_{id}.htm` |
| TikTok | 2 | No | `tiktok.com/@{username}` |

**Key insight:** IAFD, AFDB, DATA18, dbNaked, and Pornhub are high-frequency sources with images that we haven't documented yet.

This means much of the identity graph work is already done - we just need to parse these URLs and use them for matching.

### Scrape Test Results (2026-01-27)

Tested with `api/test_scrape_sources.py` and FlareSolverr at `10.0.0.4:8191`:

| Source | Status | Images | Notes |
|--------|--------|--------|-------|
| **Indexxx** | ✓ Works | 88 | Via FlareSolverr |
| **EliteBabes** | ✓ Works | 44 | High-quality images |
| **FreeOnes** | ✓ Works | 24 | Reliable CDN |
| **Boobpedia** | ✓ Works | 23 | MediaWiki format |
| **PornPics** | ✓ Works | 21 | Gallery drilling |
| **Pornhub** | ✓ Works | 20 | Needs age cookie |
| **JavDatabase** | ✓ Works | 7 | JAV performers only |
| **Babepedia** | ✓ Works | 4 | Via FlareSolverr |
| **IAFD** | ✓ Works | 1 | Via FlareSolverr, hotlink protected |
| **AFDB** | Requires URL | - | URLs include ID suffix from StashDB |
| **DATA18** | Requires URL | - | Use pre-existing StashDB URLs |
| **theNude** | Requires URL | - | URLs use ID only (e.g., `/_44689.htm`) |
| **dbNaked** | Slow/Unreliable | - | Timeouts common |

**Total: 9/13 sources working, 232 images found for test performer**

**Notes:**
- IAFD and Indexxx have hotlink protection - need to download through FlareSolverr session
- AFDB, DATA18, theNude require full URLs from StashDB (can't construct from name)
- FlareSolverr client: `api/flaresolverr_client.py`

**Strategy:** Use StashDB's pre-existing `urls` field to get full URLs for AFDB, DATA18, theNude. Don't try to construct URLs from performer names for these sites.

---

## Implementation Priority

### Phase 1: Stash-Boxes (Structured APIs)
1. **StashDB** - Update client to capture `aliases` and `urls` fields
2. **ThePornDB** - Already implemented, leverage `stashdb_id` links
3. **PMVStash** - Use existing builder with env override
4. **JAVStash** - Use existing builder, accept 1-image limitation
5. **FansDB** - Test and document

### Phase 2: Reference Sites (Enrichment)
1. **IAFD** - Good coverage, no Cloudflare
2. **Babepedia** - Wiki-style, links to IAFD
3. **FreeOnes** - Requires FlareSolverr
4. **Indexxx** - Lower priority
5. **Boobpedia** - Lower priority

---

## Matching Confidence Tiers

| Tier | Signals | Auto-Accept? |
|------|---------|--------------|
| **1: Explicit ID** | ThePornDB has `stashdb_id` | Yes |
| **2: URL Match** | Same Twitter/IAFD/etc URL | Yes |
| **3: Strong Name + Metadata** | Exact name + same country + similar birthdate | Yes (configurable) |
| **4: Fuzzy Name + Face** | Name similarity > 0.85 + face distance < 0.45 | Configurable |
| **5: Face Only** | Face distance < 0.35, name doesn't match | Review queue |
| **6: Name Only** | Name matches but no face match | Review queue |

---

## Curation UI Requirements

Based on the scale (~200-300k performers):

1. **Review Queue** - Show only uncertain matches (Tier 5-6)
2. **Confidence Thresholds** - Configurable per signal type
3. **Bulk Accept/Reject** - For high-confidence batches
4. **Search** - Find performer by name across all sources
5. **Manual Link** - Add cross-reference manually
6. **Statistics Dashboard** - Coverage by source, link density

---

## Open Questions

1. **StashDB URLs field format:** Need to test what external links are commonly present
2. **FansDB coverage:** How many performers? Any cross-links?
3. **Reference site testing:** Need browser-based testing for scraping feasibility
4. **FlareSolverr deployment:** Required for FreeOnes, maybe others
5. **Japanese names:** How to normalize JAVStash names for matching?

---

## Proven Scraping Patterns (from performerImageSearch Plugin)

The existing `performerImageSearch` Stash plugin successfully scrapes most reference sites using simple HTTP requests. Key learnings:

### Why It Works (Single Requests)

The plugin makes **individual requests** per user action, which look like normal browser traffic:

```python
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}
```

### URL Patterns (Verified Working)

| Site | URL Pattern | Name Transform |
|------|-------------|----------------|
| Babepedia | `/babe/{Name_With_Underscores}` | `replace(" ", "_")` |
| Boobpedia | `/boobs/{Name_With_Underscores}` | `replace(" ", "_")` |
| FreeOnes | `/{name-with-hyphens}/photos` | `lower().replace(" ", "-")` |
| PornPics | `/pornstars/{name-with-hyphens}/` | `lower().replace(" ", "-")` |
| EliteBabes | `/model/{name-with-hyphens}/` | `lower().replace(" ", "-")` |
| JavDatabase | `/idols/{name-with-hyphens}/` | `lower().replace(" ", "-")` |

### Image Extraction Patterns

**Babepedia:**
```python
pattern = r'href="(/pics/[^"]+\.jpg)"'
# Thumbnail: {image_url}_thumb3.jpg
```

**FreeOnes:**
```python
# First get gallery URLs from photos page
gallery_pattern = rf'href="(/{performer}/photos/[^"]+)"'
# Then extract images from each gallery
pattern = r'(https://(?:thumbs|ch-thumbs|img)\.freeones\.com/[^"\']+\.(?:jpg|webp|png))'
```

**Boobpedia (MediaWiki):**
```python
# Thumbnails: /wiki/images/thumb/X/XX/File.jpg/NNNpx-File.jpg
# Full size: /wiki/images/X/XX/File.jpg
pattern = r'src="(/wiki/images/thumb/[^"]+)"'
```

**PornPics:**
```python
# Extract gallery set IDs, then drill into each
set_id_pattern = r'/(\d{8})/\d{8}_'
# Image URL: /460/ for thumb, /1280/ for full
```

### Bulk Scraping Considerations

For database building (thousands of performers), we need additional safeguards:

1. **Explicit rate limiting** - 1-2 req/sec for HTML sites
2. **FlareSolverr fallback** - When Cloudflare challenges appear
3. **Resume capability** - Track processed performers
4. **Retry with backoff** - Handle transient failures
5. **Parallel requests** - Only for different domains simultaneously

### FlareSolverr Integration

FlareSolverr is only needed when bulk scraping triggers Cloudflare protection. Configuration:

```yaml
# docker-compose.yml
services:
  flaresolverr:
    image: ghcr.io/flaresolverr/flaresolverr:latest
    ports:
      - "8191:8191"
```

Usage pattern:
```python
# Try direct request first
response = requests.get(url, headers=HEADERS)
if response.status_code == 403 or "cf-ray" in response.headers:
    # Fall back to FlareSolverr
    response = flaresolverr_get(url)
```

---

## Additional Sources (from performerImageSearch)

### 11. PornPics

| Property | Value |
|----------|-------|
| **URL** | `https://www.pornpics.com` |
| **API Type** | HTML scraping |
| **Performers** | Large database |
| **Images/Performer** | Many (gallery-based, up to 200) |
| **Rate Limit** | Unknown |
| **Auth** | None (public) |
| **Cloudflare** | No (works with simple requests) |

**Performer Page URL Pattern:**
```
https://www.pornpics.com/pornstars/{name-with-hyphens}/
```

**Scraping Strategy:**
1. Fetch performer page
2. Extract gallery set IDs from image URLs (8-digit numbers)
3. Drill into each gallery at `/galleries/{set_id}/`
4. Extract full-size images (CDN: cdni.pornpics.com)

### 12. EliteBabes

| Property | Value |
|----------|-------|
| **URL** | `https://www.elitebabes.com` |
| **API Type** | HTML scraping |
| **Performers** | Unknown |
| **Images/Performer** | High-quality photosets |
| **Rate Limit** | Unknown |
| **Auth** | None (public) |
| **Cloudflare** | Unknown |

**Performer Page URL Pattern:**
```
https://www.elitebabes.com/model/{name-with-hyphens}/
```

**Image Sizes:** `_w200`, `_w400`, `_w600`, `_w800`, or no suffix for full size

### 13. JavDatabase

| Property | Value |
|----------|-------|
| **URL** | `https://www.javdatabase.com` |
| **API Type** | HTML scraping |
| **Performers** | JAV idols |
| **Images/Performer** | Multiple (idol images + covers) |
| **Rate Limit** | Unknown |
| **Auth** | None (public) |
| **Cloudflare** | No |

**Performer Page URL Pattern:**
```
https://www.javdatabase.com/idols/{name-with-hyphens}/
```

**Image Types:**
- Idol images: `/idolimages/full/` or `/idolimages/thumb/`
- Movie covers: `/covers/full/` or `/covers/thumb/`
- Vertical promos: `/vertical/`

**Use Case:** Supplement JAVStash's single-image limitation

---

## Test Scripts

Two test scripts are available in `api/`:

### `test_scrape_sources.py`
Tests our ability to scrape images from various sites by performer name.

```bash
python test_scrape_sources.py "Angela White"
```

### `test_url_enrichment.py`
Tests image enrichment using actual StashDB URLs (the practical approach).

```bash
python test_url_enrichment.py 50459d16-787c-47c9-8ce9-a4cac9404324
```

**Example Results (Aaliyah Hadid) with FlareSolverr:**
- StashDB base images: 18
- Additional images from URLs: 185
- **Potential total: 203 (11x enrichment)**

Working sources: FreeOnes (72), Pornhub (67), Indexxx (27), theNude (9), AFDB (4), Babepedia (4), Boobpedia (1), IAFD (1)

---

## Next Steps

1. [x] Update `stashdb_client.py` to capture `aliases` and `urls` fields ✅
2. [x] Create scraping test scripts ✅
3. [ ] Run second pass of StashDB builder to collect new fields (after current build completes)
4. [ ] Test PMVStash and FansDB APIs
5. [ ] Build URL parser to extract IDs from StashDB's pre-built cross-references
6. [ ] Adapt performerImageSearch patterns for bulk scraping
7. [ ] Build curation UI framework
8. [ ] Implement confidence threshold configuration
9. [ ] Add FlareSolverr integration for Cloudflare-protected sites (Babepedia, IAFD, Indexxx)

---

*This document will be updated as testing progresses.*
