# Scene Tagging Strategy

**Date:** 2026-01-26
**Status:** Conceptual

---

## Overview

Scene tagging is fundamentally different from face recognition:

| Aspect | Face Recognition | Scene Tagging |
|--------|------------------|---------------|
| **Data source** | Pre-built database of performer embeddings | Pre-trained models run on user content |
| **Query type** | "Who is this?" (match against known) | "What's in this?" (classify/detect) |
| **Distribution** | Database file (~1GB) | Model weights (~2-5GB) |
| **External data needed** | Yes (performer images) | No (models generalize) |

**Key insight:** We don't need to source "every scene in existence" - models trained on general data can classify any scene. The user's own Stash is the only data source needed at inference time.

---

## Phased Approach

### Phase 1: Zero-Shot Proof of Concept

Use off-the-shelf models with no custom training:

| Model | Task | Download Size |
|-------|------|---------------|
| **CLIP** (ViT-L/14) | Zero-shot classification | ~900MB |
| **YOLOv8** (large) | Object detection | ~140MB |
| **PaddleOCR** | Text/watermark extraction | ~150MB |

**Expected coverage:** 60-70% of common tagging needs

**What will work well:**
- Location tags (indoor, outdoor, pool, beach, car, office)
- Basic object detection (furniture, vehicles)
- Watermark/studio text extraction
- General scene type classification

**What may struggle:**
- Adult-specific positions/acts
- Subtle distinctions (specific furniture types)
- Domain-specific terminology

### Phase 2: Gap Identification

Run Phase 1 models on real user content and identify:

1. **False positives** - Tags that are wrong
2. **False negatives** - Tags that should exist but weren't detected
3. **Missing categories** - Things users want to tag that models don't understand

This produces a prioritized list of gaps to address.

### Phase 3: Targeted Fine-Tuning

For identified gaps, options include:

| Approach | When to Use | Data Needed |
|----------|-------------|-------------|
| **Prompt engineering** | CLIP understands concept but wrong prompt | None |
| **Few-shot examples** | Model close but needs nudging | 5-20 examples |
| **Fine-tuning** | Model fundamentally lacks concept | 100+ labeled examples |
| **Custom classifier** | Highly specialized domain | 1000+ labeled examples |

---

## Models Deep Dive

### CLIP (Contrastive Language-Image Pre-training)

**What it is:** OpenAI model that understands images and text in the same embedding space.

**How it works:**
```python
import clip

model, preprocess = clip.load("ViT-L/14")

# Define candidate labels
labels = ["indoor scene", "outdoor scene", "swimming pool", "beach", "office"]

# Encode image and labels
image_features = model.encode_image(preprocess(image))
text_features = model.encode_text(clip.tokenize(labels))

# Compute similarity
similarities = (image_features @ text_features.T).softmax(dim=-1)
# Returns probability distribution over labels
```

**Strengths:**
- Zero-shot: No training needed, just describe what you want
- Flexible: Add new labels anytime by changing text prompts
- Generalizes well to unseen concepts

**Weaknesses:**
- Less precise than specialized models
- May not understand domain-specific terminology
- Prompt wording matters (needs experimentation)

**Prompt engineering tips:**
- "a photo of a swimming pool" > "pool"
- "an outdoor scene with natural lighting" > "outdoor"
- Multiple prompts per concept, average results

### YOLOv8 (Object Detection)

**What it is:** State-of-the-art real-time object detector.

**Pre-trained classes (COCO):** 80 common objects including:
- Furniture: couch, bed, chair, dining table
- Vehicles: car, motorcycle, bicycle, truck
- Outdoor: bench, umbrella, surfboard
- Electronics: tv, laptop, cell phone

**How it works:**
```python
from ultralytics import YOLO

model = YOLO("yolov8l.pt")
results = model(image)

for box in results[0].boxes:
    class_name = model.names[int(box.cls)]
    confidence = float(box.conf)
    # Returns: "couch" with 0.92 confidence, bounding box coords
```

**Strengths:**
- Very accurate for trained classes
- Fast inference
- Provides bounding boxes (where, not just what)

**Weaknesses:**
- Limited to 80 COCO classes
- No adult-specific objects
- Custom classes need fine-tuning

### PaddleOCR (Text Extraction)

**What it is:** Optical character recognition for extracting text from images.

**Use cases:**
- Studio watermarks
- Title cards
- Scene information overlays

**How it works:**
```python
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang='en')
results = ocr.ocr(image)

for line in results[0]:
    text = line[1][0]
    confidence = line[1][1]
    # Returns: "BangBros" with 0.95 confidence
```

**Studio identification flow:**
1. Extract text from scene frames
2. Match against known studio names
3. Suggest studio tag if confidence high

### DINOv2 (Visual Similarity)

**What it is:** Meta's self-supervised vision transformer for image embeddings.

**Use cases:**
- Scene similarity (find visually similar scenes)
- Studio clustering (scenes with similar sets/lighting)
- Duplicate detection (same scene, different files)

**How it works:**
```python
import torch
from transformers import AutoModel, AutoProcessor

model = AutoModel.from_pretrained("facebook/dinov2-large")
processor = AutoProcessor.from_pretrained("facebook/dinov2-large")

# Get embedding
inputs = processor(images=image, return_tensors="pt")
embedding = model(**inputs).last_hidden_state.mean(dim=1)

# Compare scenes by cosine similarity
similarity = cosine_similarity(embedding1, embedding2)
```

**Strengths:**
- Excellent for "find similar" queries
- No labels needed
- Captures visual style, not just content

**Weaknesses:**
- Doesn't produce labels
- Similarity ≠ same content (style can match across different scenes)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Scene Analysis Sidecar                       │
├─────────────────────────────────────────────────────────────────────┤
│  Existing (Face Recognition)        New (Scene Tagging)             │
│  ├─ RetinaFace (detection)          ├─ CLIP (zero-shot classify)   │
│  ├─ FaceNet512 (embedding)          ├─ YOLOv8 (object detection)   │
│  └─ ArcFace (embedding)             ├─ PaddleOCR (text extraction) │
│                                      ├─ DINOv2 (visual similarity)  │
│                                      └─ VLM* (action tags, optional)│
├─────────────────────────────────────────────────────────────────────┤
│  API Endpoints                                                       │
│  ├─ POST /identify/scene      (existing - face recognition)        │
│  ├─ POST /tag/scene           (new - suggest tags)                 │
│  ├─ POST /detect/objects      (new - object detection)             │
│  ├─ POST /extract/text        (new - OCR)                          │
│  ├─ POST /similar/scenes      (new - visual similarity search)     │
│  └─ POST /tag/actions*        (new - VLM action tags, optional)    │
└─────────────────────────────────────────────────────────────────────┘

* VLM integration is optional, for action/position tags only.
  Can use external endpoints (LM Studio) or defer to Haven VLM Connector.
```

### Tag Suggestion Endpoint

```python
@app.post("/tag/scene")
async def tag_scene(request: TagRequest):
    """
    Analyze scene and suggest tags.

    Input: Scene image or sprite sheet URL
    Output: Suggested tags with confidence scores
    """
    image = await fetch_image(request.image_url)

    suggestions = []

    # CLIP zero-shot classification
    clip_tags = clip_classify(image, TAG_CATEGORIES)
    suggestions.extend(clip_tags)

    # YOLO object detection
    objects = yolo_detect(image)
    suggestions.extend(object_to_tag(obj) for obj in objects)

    # OCR for studio detection
    text = ocr_extract(image)
    if studio := match_studio(text):
        suggestions.append({"tag": f"studio:{studio}", "confidence": 0.9})

    return {"suggestions": dedupe_and_rank(suggestions)}
```

---

## Tag Categories for CLIP

Initial tag categories to test with CLIP zero-shot:

### Location/Setting
```python
LOCATION_PROMPTS = [
    "an indoor scene",
    "an outdoor scene",
    "a scene at a swimming pool",
    "a scene at a beach",
    "a scene in a car or vehicle",
    "a scene in an office",
    "a scene in a hotel room",
    "a scene in a bathroom",
    "a scene in a kitchen",
    "a scene with natural outdoor lighting",
]
```

### Scene Type
```python
SCENE_TYPE_PROMPTS = [
    "a professional studio scene with good lighting",
    "an amateur home video scene",
    "a POV (point of view) scene",
    "a scene shot from multiple camera angles",
]
```

### Lighting/Style
```python
STYLE_PROMPTS = [
    "a scene with bright natural lighting",
    "a scene with dim or dark lighting",
    "a scene with colorful neon lighting",
    "a high production value scene",
    "a low production value amateur scene",
]
```

---

## Training Data Collection (Future)

If zero-shot isn't sufficient for certain categories:

### Option 1: User Opt-In Contribution

```
User tags scene in Stash → Opt-in to contribute →
Label (tag + scene ID) sent to central server →
Aggregated for model training
```

**Privacy consideration:** Only tag labels, never images or scene content.

### Option 2: Active Learning

```
Model suggests tag with low confidence →
User confirms or rejects →
Feedback used to improve model
```

### Option 3: Synthetic Labels

```
Use CLIP to bootstrap labels →
Human review to filter errors →
Train specialized model on filtered data
```

---

## Resource Requirements

### Model Sizes (Download)

| Model | Size | GPU Memory | Use Case |
|-------|------|------------|----------|
| CLIP ViT-L/14 | ~900MB | ~2GB | Settings, locations, scene types |
| YOLOv8-large | ~140MB | ~1GB | Object detection |
| PaddleOCR | ~150MB | ~500MB | Text/watermark extraction |
| DINOv2-large | ~1.2GB | ~3GB | Visual similarity |
| **Core Total** | ~2.4GB | ~6GB | (not all loaded simultaneously) |
| GLM-4.6V-Flash* | ~12GB | ~12GB | Action/position tags (optional) |
| Qwen-VL-8B* | ~16GB | ~16GB | Action/position tags (optional) |

*VLM models are optional, for action tags only. Require separate GPU or external endpoint.

### Inference Speed (Estimated)

| Model | GPU | CPU | Notes |
|-------|-----|-----|-------|
| CLIP | ~50ms | ~500ms | Fast, scalable |
| YOLOv8 | ~30ms | ~300ms | Very fast |
| PaddleOCR | ~100ms | ~1s | Per-frame |
| DINOv2 | ~80ms | ~800ms | Per-frame |
| VLM* | ~1-10s | N/A | **100x slower**, action tags only |

### Processing Time Comparison

For a scene with sprite sheet (20 frames):

| Approach | Time | Notes |
|----------|------|-------|
| CLIP+YOLO only | ~2-5 seconds | Settings, objects, locations |
| CLIP+YOLO+VLM | ~20-200 seconds | Adds action/position tags |

For a 10,000 scene library:

| Approach | Total Time |
|----------|------------|
| CLIP+YOLO | ~6 GPU-hours |
| CLIP+YOLO+VLM | ~60-600 GPU-hours |

**Recommendation:** Use CLIP/YOLO for bulk tagging, VLM only for targeted action tag queries or small batches.

---

## Implementation Priority

1. **CLIP integration** - Biggest bang for buck, most flexible
2. **Tag category design** - Define initial tag vocabulary and prompts
3. **API endpoint** - `/tag/scene` with CLIP backend
4. **YOLOv8 integration** - Object detection for specific items
5. **PaddleOCR integration** - Studio watermark detection
6. **Accuracy evaluation** - Test on real scenes, identify gaps
7. **DINOv2 integration** - Visual similarity (lower priority)
8. **Fine-tuning pipeline** - Only after gaps identified
9. **VLM integration (optional)** - Only if user demand for action tags justifies 100x slowdown. Consider recommending Haven VLM Connector as complement instead of building in.

---

## Open Questions

1. **Tag vocabulary:** What tags do users actually want? Need to survey Stash community.
2. **Prompt optimization:** CLIP is sensitive to prompt wording - needs experimentation.
3. **Multi-frame analysis:** How to combine results across sprite sheet frames? Consider Haven's gap tolerance approach (if detected at 0:10 and 0:50, assume continuous).
4. **Confidence thresholds:** What confidence level to suggest vs auto-apply tags? Haven uses 0.3-0.5 depending on tag category - per-tag thresholds may be needed.
5. **GPU memory management:** Can we load all models simultaneously, or need to swap?
6. **User feedback loop:** How to collect feedback on tag suggestions? Haven's workflow tags (TagMe, Tagged, Errored) pattern worth adopting.
7. **Action tags priority:** Do users actually need action/position tags for MVP, or is face recognition + settings/objects sufficient? Defer VLM integration decision until user feedback.
8. **Haven coexistence:** Should we document Haven as complementary, or build VLM support directly?

---

## Comparison to Patreon-Gated Projects

What they likely have that we'd need to build:

| Feature | Their Approach | Our Approach |
|---------|---------------|--------------|
| Basic tagging | Custom trained | CLIP zero-shot (start here) |
| Position detection | Custom trained on user contributions | CLIP zero-shot → fine-tune if needed |
| Studio detection | Database + OCR | OCR + fuzzy matching |
| Training data | Years of user contributions | Start fresh, build over time |

**Our advantage:** Open source, user owns their data, community-driven improvements.

**Their advantage:** Head start on training data for specialized classifiers.

**Strategy:** Start with zero-shot, match their basic features quickly, then build community contribution pipeline for specialized features.

**Detailed Analysis:** See [Skier AI Ecosystem Analysis](2026-01-27-skier-ai-ecosystem-analysis.md) for comprehensive documentation of the existing Patreon-gated solution, including architecture patterns worth adopting.

---

## Haven VLM Connector Analysis

**Date Added:** 2026-01-27
**Reference:** https://discourse.stashapp.cc/t/haven-vlm-connector/5464

Haven VLM Connector is an existing Stash community plugin that uses Vision-Language Models for scene tagging. This section documents our analysis and learnings.

### How Haven Works

1. **Frame sampling**: Extracts frames at configurable intervals (2-80 seconds)
2. **VLM inference**: Sends frames to OpenAI-compatible endpoints (LM Studio locally, or cloud)
3. **Tag matching**: VLM returns which tags from a predefined list apply to each frame
4. **Aggregation**: Combines frame results with confidence thresholds (0.3-0.5)

**Predefined tags**: 35 action/position tags (Blowjob, Deepthroat, Gangbang, etc.)

**Tested models** (in accuracy order per Haven docs):
- zai-org/glm-4.6v-flash (~12GB VRAM)
- huihui-mistral-small-3.2-24b (~24GB VRAM)
- qwen/qwen3-vl-8b (~16GB VRAM)
- lfm2.5-vl

### Haven Strengths

| Advantage | Details |
|-----------|---------|
| **Contextual understanding** | VLMs understand activities, not just objects. "Doggy style" vs "reverse cowgirl" |
| **Zero training** | Just define tag list in config, VLM generalizes |
| **Natural language** | Can describe nuanced scenarios |
| **Distributed processing** | Built-in multi-endpoint load balancing with weights |
| **Already built** | Working plugin, community supported |

### Haven Weaknesses

| Disadvantage | Details |
|--------------|---------|
| **Slow** | VLM inference is 1-10+ seconds per frame. 30-min video at 2-sec intervals = 900 VLM calls |
| **Resource heavy** | 12-24GB VRAM per model instance |
| **Accuracy unknown** | Claims "superior accuracy" but no published benchmarks |
| **Black box** | No visible prompt templates - hard to debug or tune |
| **Model dependent** | Quality varies wildly between VLMs |
| **No performer ID** | Doesn't solve WHO - only WHAT |

### Head-to-Head Comparison

| Aspect | Haven (VLM) | Our Plan (CLIP/YOLO) |
|--------|-------------|----------------------|
| **Object detection** | VLM describes what it sees | YOLO - fast, accurate, ~10ms |
| **Scene classification** | VLM interprets context | CLIP zero-shot - fast, ~50ms |
| **Action/position tags** | **VLM strength** - understands activities | Gap - would need specialized models |
| **Speed per frame** | ~1-10 seconds | ~50-100ms |
| **VRAM required** | 12-24GB per model | ~2-3GB total |
| **Accuracy** | Unknown, model-dependent | YOLO/CLIP well-benchmarked |

### Processing Time Comparison (10,000 scene library)

| Approach | Calculation | Total Time |
|----------|-------------|------------|
| Haven (2-sec intervals) | 900 frames × 3 sec × 10,000 scenes | **7,500+ GPU-hours** |
| Haven (80-sec intervals) | 22 frames × 3 sec × 10,000 scenes | **183 GPU-hours** |
| CLIP+YOLO | 20 frames × 0.1 sec × 10,000 scenes | **~6 GPU-hours** |

### Learnings to Incorporate

| Pattern from Haven | How to Apply |
|--------------------|--------------|
| **Distributed endpoints** | Support multiple GPU backends with weighted load balancing |
| **Confidence thresholds per tag** | Different tags need different thresholds (0.5 for "outdoor" vs 0.3 for "threesome") |
| **Workflow status tags** | Use VLM_TagMe, VLM_Tagged, VLM_Errored pattern for batch processing state |
| **Gap tolerance** | If tag detected at 0:10 and 0:50, assume continuous (max gap: 30 seconds) |
| **Fallback endpoints** | Support cloud fallback when local GPU unavailable |

### Recommended Hybrid Approach

Based on this analysis, the optimal strategy is:

1. **CLIP** for settings/environment tags (indoor, outdoor, pool, etc.) - fast, accurate
2. **YOLO** for object detection (furniture, props) - fast, well-understood
3. **PaddleOCR** for text/watermark extraction - specialized, fast
4. **VLM (optional)** for action/position tags only - this is where VLMs genuinely excel

This hybrid approach gets ~90% of tagging needs with CLIP/YOLO speed, and optionally uses VLM for the remaining action tags where it provides genuine value over classification models.

### Integration Options

**Option A: Recommend Haven for action tags**
- Document Haven as complementary tool
- Users install both plugins
- Stash Sense does faces + settings/objects
- Haven does action/position tags

**Option B: Build VLM support into Stash Sense**
- Add optional VLM endpoint configuration
- Use VLM only for action tag categories
- Keep CLIP/YOLO for everything else
- Single plugin, unified experience

**Option C: Defer action tags entirely**
- Ship face recognition + CLIP/YOLO first
- Evaluate user demand for action tags
- Decide VLM integration based on feedback

**Recommendation:** Option C for MVP. Action tags are nice-to-have, not core value prop.

---

*This document will be updated as implementation progresses.*
