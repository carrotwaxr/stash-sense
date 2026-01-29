# Skier AI Ecosystem Analysis

**Date:** 2026-01-27
**Status:** Research Complete

---

## Overview

This document analyzes the Skier233 AI ecosystem for Stash, documenting capabilities, architecture patterns, and learnings that can inform our open-source development.

**Repositories Analyzed:**
- [nsfw_ai_model_server](https://github.com/skier233/nsfw_ai_model_server) (121 stars) - Core ML inference server
- [Stash-AIServer](https://github.com/skier233/Stash-AIServer) (12 stars) - Task orchestration, recommendations, plugin UI
- [AIOverhaul_Plugin_Catalog_Official](https://github.com/skier233/AIOverhaul_Plugin_Catalog_Official) - Plugin marketplace
- [MiVOLO](https://github.com/skier233/MiVOLO) - Age/gender detection fork

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Skier AI Ecosystem                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  Stash (Browser)                                                            │
│  └─ AI Overhaul Plugin (frontend)                                           │
│      ├─ AIButton.tsx (trigger actions)                                      │
│      ├─ InteractionTracker.ts (watch behavior analytics)                    │
│      ├─ RecommendedScenes.tsx (personalized recommendations UI)             │
│      └─ SimilarScenes.tsx (segment similarity UI)                           │
│                         │                                                   │
│                         ▼                                                   │
│  Stash-AIServer (Backend - FastAPI)                                         │
│  ├─ Task Manager (queued processing with priorities)                        │
│  ├─ Plugin System (yaml-defined, pip dependencies, settings)                │
│  ├─ Recommendations Engine (TF-IDF, segment similarity)                     │
│  ├─ Interaction Tracking (watch segments, scene views)                      │
│  └─ Service Registry (remote service abstraction)                           │
│                         │                                                   │
│                         ▼                                                   │
│  nsfw_ai_model_server (ML Inference - separate process)                     │
│  ├─ Pipeline system (configurable model chains)                             │
│  ├─ Video preprocessor (frame extraction with DeFFcode/Decord)              │
│  ├─ Dynamic AI models (hot-swap models at runtime)                          │
│  └─ Custom-trained models (Patreon-gated, 151 tags)                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Technology | Purpose |
|-----------|------------|---------|
| Frontend Plugin | TypeScript/React | UI integration, interaction tracking, recommendations display |
| Stash-AIServer | FastAPI + SQLAlchemy | Task orchestration, plugin management, recommendations logic |
| nsfw_ai_model_server | FastAPI + PyTorch | ML inference, video preprocessing, model management |

---

## Capabilities

### 1. Scene Tagging

**Model Tiers (Patreon-gated):**

| Model | Access | Tags | Specialty |
|-------|--------|------|-----------|
| Vivid Galaxy | Free | 10 | Basic sexual actions |
| Gentler River | $5/mo | 36 | Sexual actions (highest accuracy) |
| Distinctive Haze | VIP | 36 | 80% faster processing |
| Stilted Glade | $5/mo | 50 | BDSM detection |
| Happy Terrain | VIP | 50 | Fast BDSM detection |
| Fearless Terrain | $5/mo | 37 | Body parts identification |
| Electric Smoke | VIP | 37 | Fast body parts |
| Blooming Star | VIP | 28 | Positions (beta) |

**Total: 151 unique tags across paid models, 10 tags free**

**Technical Implementation:**
- Frame extraction at configurable intervals (default 0.5s)
- Image preprocessing: 512px, half precision
- GPU backends: NVIDIA (1080+), AMD/Intel (beta), CPU (slow)
- Video decoding: DeFFcode (GPU accelerated) or Decord fallback

### 2. Personalized Recommendations

Sophisticated recommendation system based on actual viewing behavior.

**Watch Tracking Events:**
```typescript
type InteractionEventType =
  | 'session_start'
  | 'session_end'
  | 'scene_view'
  | 'scene_page_enter'
  | 'scene_page_leave'
  | 'scene_watch_start'
  | 'scene_watch_pause'
  | 'scene_seek'
  | 'scene_watch_progress'
  | 'scene_watch_complete'
  | 'image_view'
  | 'gallery_view'
  | 'library_search';
```

**Data Model:**
```python
class SceneWatch:
    session_id: str
    scene_id: int
    page_entered_at: datetime
    page_left_at: datetime | None
    total_watched_s: float
    watch_percent: float | None

class SceneWatchSegment:
    scene_watch_id: int
    start_s: float
    end_s: float
    watched_s: float
```

**TF-IDF Recommender Algorithm:**
1. Load watch history from plugin telemetry AND Stash's native `play_duration`
2. Fetch AI tag durations for watched scenes
3. Weight tags by: `watched_seconds × tag_duration × repeat_factor`
4. Apply TF-IDF weighting: `tf × log((1 + corpus_size) / (1 + doc_freq))`
5. Rank candidates by weighted tag overlap
6. Return with debug metadata explaining recommendations

**Segment Similarity Recommender:**
- "Find scenes like this one" based on which segments you actually watched
- Uses watched portions (not full scene metadata) to build preference profile
- Returns scenes with similar tag duration distributions

### 3. Plugin Architecture

YAML-based plugin definition:

```yaml
name: skier_aitagging
human_name: Skier AI Tagging
description: AI tagging of images and videos
version: 0.3.2
required_backend: '>=0.5.0'
files:
  - service
depends_on: []
pip_dependencies: []
settings:
  - key: server_url
    label: Remote Service URL
    type: string
    default: "http://localhost:8000"
  - key: path_mappings
    label: Path Mappings
    type: path_map
    default: []
  - key: apply_ai_tagged_tag
    label: Apply AI_Tagged Tag
    type: boolean
    default: true
  - key: tag_suffix
    label: Tag Name Suffix
    type: string
    default: "_AI"
```

**Plugin Capabilities:**
- Register services (remote ML backends)
- Register actions (user-triggered tasks)
- Register recommenders (custom algorithms)
- Define settings with types: `string`, `boolean`, `number`, `path_map`, `select`
- Declare dependencies on other plugins
- Specify pip dependencies for auto-installation

### 4. Task Management

```python
class TaskPriority(Enum):
    high = 1
    normal = 2
    low = 3

class TaskStatus(Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"
```

**Features:**
- Priority queue with high/normal/low levels
- Task history persistence to database
- Parent-child task relationships
- Configurable concurrency limits per service
- WebSocket progress updates to frontend
- Cancellation support with cascade to children

### 5. Service Abstraction

```python
class RemoteServiceBase:
    name = "service_name"
    description = "Service description"
    max_concurrency = 10
    ready_endpoint = "/ready"
    readiness_cache_seconds = 30.0
    failure_backoff_seconds = 60.0
```

Pattern for health checking remote ML backends with caching and backoff.

---

## Patterns Worth Adopting

### 1. Path Mapping System

Essential for Docker/WSL environments where file paths differ between Stash and the AI server.

```yaml
settings:
  - key: path_mappings
    type: path_map
    description: Rewrite stash file paths before calling the ai server
```

**Implementation:** Each mapping defines source prefix, replacement, and slash mode (Windows vs Unix).

### 2. Debug Metadata on Results

Their recommendations include transparency about WHY something was recommended:

```python
debug_meta = {
    "tfidf": {
        "profile_size": len(limited_weights),
        "scene_contributors": [
            {
                "tag_id": tag_id,
                "tag_name": "Cowgirl",
                "duration_s": 45.3,
                "profile_weight": 0.892,
                "partial_score": 40.4,
            }
        ],
        "profile_summary": {
            "history_count": 150,
            "plugin_history_count": 80,
            "stash_history_count": 70,
        }
    }
}
```

**Benefit:** Users can understand and trust the system's decisions.

### 3. Per-Tag Threshold Configuration

From their tag settings template:

```csv
ai_tag_name,stash_tag_name,apply_threshold
Cowgirl,Cowgirl_AI,0.5
Anal,Anal_AI,0.4
Outdoor,Outdoor_AI,0.6
```

**Insight:** Different tags need different confidence thresholds. "Outdoor" is easy to detect (high threshold), while specific positions may need lower thresholds.

### 4. Workflow Status Tags

Pattern for batch processing state:

| Tag | Meaning |
|-----|---------|
| `AI_TagMe` | Queued for processing |
| `AI_Tagged` | Successfully processed |
| `AI_Errored` | Processing failed |

### 5. Watch Segment Tracking

Track which portions of videos users actually watch:

```python
class SceneWatchSegment:
    start_s: float   # Started watching at 2:30
    end_s: float     # Stopped at 5:45
    watched_s: float # Actually watched 3:15
```

**Value:** Recommendations based on what users WATCH, not just what they click.

### 6. Dual History Sources

Their recommender merges data from:
1. Plugin telemetry (detailed segment data)
2. Stash's native `play_duration` field (coarse but available for all users)

```python
# Merge plugin tracking with Stash native data
for stash_entry in stash_history:
    if scene_id in history_by_scene:
        # Combine watched times from both sources
        existing["watched_s"] += stash_entry["watched_s"]
        existing["source"] = "plugin+stash"
```

### 7. Service Health Pattern

```python
class RemoteServiceBase:
    ready_endpoint = "/ready"
    readiness_cache_seconds = 30.0    # Cache health status
    failure_backoff_seconds = 60.0    # Wait before retry after failure
```

### 8. Video Preprocessing Pipeline

```python
class VideoPreprocessorModel:
    image_size = 512
    frame_interval = 0.5  # seconds
    use_half_precision = True

    # Backend selection with fallbacks
    backends = ["deffcode_gpu", "deffcode", "decord"]
```

DeFFcode with GPU acceleration for fast frame extraction, with automatic fallback to CPU.

---

## Comparison: Skier AI vs Stash Sense

| Aspect | Skier AI | Stash Sense |
|--------|----------|-------------|
| **Primary Focus** | Scene tagging + recommendations | Face recognition + performer ID |
| **Core Question** | "What's happening?" | "Who's in this?" |
| **Models** | Custom-trained (Patreon-gated) | Off-the-shelf (CLIP/YOLO) + pre-built database |
| **Data Source** | User's content → trained models | StashDB images → embedding database |
| **Distribution** | Model weights + code | Pre-built database file |
| **Cost** | Free (10 tags) / $5-15/mo (151 tags) | Free |
| **Tag Count** | 151 domain-specific | Unlimited (zero-shot vocabulary) |
| **Recommendations** | Full implementation | Not planned |
| **Performer Recognition** | Not present | Core feature |
| **OCR/Text Extraction** | Not present | Planned |

### What They Have That We Don't

1. **Custom-trained models** - Years of user-contributed training data
2. **Watch behavior analytics** - Tracks exactly which segments users watch
3. **Recommendation engine** - Personalized "For You" feed
4. **Plugin ecosystem** - Third-party extensibility
5. **Production UI polish** - Task dashboard, progress indicators

### What We Have That They Don't

1. **Performer identification** - Face recognition against known database
2. **Cross-stash-box linking** - Match across StashDB/TPDB/JAVStash
3. **Pre-built database** - No model training, download and run
4. **Fully free** - No paid tiers
5. **OCR capability** - Studio watermark detection (planned)

### Complementary, Not Competing

The systems solve different problems:
- **Skier AI:** "This scene contains: Cowgirl, Outdoor, Pool"
- **Stash Sense:** "This scene features: Jane Doe, John Smith"

Users could run both.

---

## Building Free Alternatives

### Scene Tagging (Their Core Feature)

| Their Approach | Our Free Alternative |
|----------------|---------------------|
| Custom 151-tag model | CLIP zero-shot (unlimited tags) |
| Years of training data | Pre-trained on internet-scale data |
| Domain-specific accuracy | General accuracy, prompt engineering needed |
| Patreon subscription | Free |

**CLIP Zero-Shot Strategy:**
```python
# Their approach: Fixed 151 tags from trained model
tags = model.predict(frame)  # Returns: ["Cowgirl", "Pool"]

# Our approach: Define any tags via text prompts
prompts = [
    "a scene with cowgirl position",
    "a scene at a swimming pool",
    "an outdoor scene",
    # Add any tag anytime - no retraining
]
similarities = clip.encode_image(frame) @ clip.encode_text(prompts).T
```

**Expected Coverage:**
- CLIP: 60-70% of common tagging needs
- Their model: Higher accuracy on trained tags, but limited to 151

### Recommendations (If We Ever Build)

Their TF-IDF approach is well-designed but not rocket science:

1. Track what users watch (segments, not just clicks)
2. Weight tags by watch duration
3. Apply standard TF-IDF weighting
4. Rank by weighted overlap

This is implementable without their code - it's a standard information retrieval technique.

### Action/Position Tags (The Hard Part)

This is where their custom training genuinely provides value. Alternatives:

| Approach | Accuracy | Speed | Cost |
|----------|----------|-------|------|
| Their trained models | High | Fast | $5-15/mo |
| VLM (Haven approach) | Medium-High | Very Slow | Free (GPU required) |
| CLIP zero-shot | Medium | Fast | Free |
| Train our own | High (eventually) | Fast | Free (needs labeled data) |

**Recommendation:** Start with CLIP for basics. If user demand exists for action tags, either:
- Integrate VLM support (slow but free)
- Build community labeling pipeline for training data

---

## Implementation Roadmap

### Phase 1: Face Recognition (Current)
Focus remains on performer identification - our unique value.

### Phase 2: Basic Scene Tagging
- CLIP zero-shot for locations/settings
- YOLO for object detection
- PaddleOCR for watermarks/text
- Adopt patterns: path mapping, debug metadata, workflow tags

### Phase 3: Enhanced Tagging (If Demand)
- VLM integration for action tags (optional, slow)
- Community labeling for training data collection
- Per-tag threshold configuration

### Phase 4: Recommendations (If Demand)
- Watch segment tracking
- TF-IDF or similar algorithm
- Dual source (plugin + Stash native data)

---

## Technical Details Reference

### Video Preprocessing

```python
# Their approach - worth adopting
class VideoPreprocessorModel:
    image_size = 512
    frame_interval = 0.5
    use_half_precision = True

    # Backends in priority order
    # 1. deffcode_gpu - NVIDIA GPU accelerated
    # 2. deffcode - CPU with FFmpeg
    # 3. decord - Fallback
```

### Recommender Registration

```python
@recommender(
    id="personalized_tfidf",
    label="Personalized TF-IDF",
    description="Recommends scenes aligned with recently watched tag durations.",
    contexts=[RecContext.global_feed],
    config=[
        {"name": "recent_days", "type": "number", "default": 45, "min": 1, "max": 365},
        {"name": "min_watch_seconds", "type": "number", "default": 30, "min": 0, "max": 7200},
        {"name": "history_limit", "type": "number", "default": 400, "min": 25, "max": 1000},
    ],
    supports_pagination=True,
    exposes_scores=True,
)
async def personalized_tfidf(ctx, request):
    ...
```

### Action Registration

```python
@action(
    id="skier.ai_tag.scene",
    label="AI Tag Scene",
    description="Generate tag suggestions for a scene",
    result_kind="dialog",
    contexts=[ContextRule(pages=["scenes"], selection="single")],
)
async def tag_scene_single(self, ctx, params, task_record):
    ...
```

---

## Conclusion

Skier's ecosystem represents significant engineering effort, particularly in the recommendation system and plugin architecture. The core tagging models are Patreon-gated, but the architectural patterns and algorithms are standard techniques that can be implemented independently.

Our focus on performer identification addresses a gap their system doesn't cover. For scene tagging, CLIP zero-shot provides a free alternative with reasonable accuracy. The sophisticated recommendation features are nice-to-have but not essential for our MVP.

**Key Takeaways:**
1. Adopt their path mapping and debug metadata patterns
2. Use CLIP/YOLO for free scene tagging
3. Their recommendation algorithm is standard TF-IDF - implementable if needed
4. Performer recognition remains our unique differentiator

---

*Analysis completed 2026-01-27*
