# Duplicate Detection

Stash Sense finds duplicate scenes that Stash's built-in phash matching misses — different intros/outros, trimming, aspect ratio changes, or watermarks.

## Prerequisites

- **Face recognition results** for scenes (run performer identification first, which requires sprite sheets)
- More face recognition coverage across your library means better duplicate detection accuracy

Duplicate detection relies on face fingerprints — per-scene records of which performers appeared, how often, and in what proportions. Scenes without face recognition results can still be detected as duplicates via Stash-Box ID matching or metadata overlap, but face fingerprints are the strongest non-authoritative signal.

---

## Detection Signals

Stash Sense uses multiple signals to identify duplicates, each with a confidence cap:

| Signal | Max Confidence | How It Works |
|--------|---------------|--------------|
| Stash-Box ID match | 100% | Authoritative — same scene on a Stash-Box endpoint |
| Face fingerprint similarity | 85% | Same performers in same ratios (robust to trimming) |
| Metadata overlap | 60% | Same studio + same performers |

Signals combine with diminishing returns (`primary + secondary × 0.3`) to prevent false confidence inflation. No single signal other than Stash-Box ID reaches 100%.

---

## How It Works

### Candidate Generation

Direct comparison of every scene pair would be O(n²) — impossible for large libraries. Instead, Stash Sense uses a two-phase approach:

1. **Candidate generation** — SQL joins and inverted indices produce O(n) candidate pairs from three sources:
    - Stash-Box ID grouping (same scene on endpoint)
    - Face fingerprint self-join (shared performers)
    - Metadata intersection (same studio AND performer)
2. **Scoring** — Each candidate pair is scored using the signal hierarchy above

This handles libraries with 15,000+ scenes efficiently.

### Face Fingerprints

A face fingerprint records which performers appeared in a scene, how many times each was detected, and what proportion of total faces they represent. Two scenes with the same performers appearing in the same ratios are likely the same scene, even if one is trimmed, has different resolution, or has a watermark.

---

## Reviewing Duplicates

Duplicate candidates appear on the recommendations dashboard with:

- **Confidence score** — Combined signal strength
- **Signal breakdown** — Which signals contributed and how much
- **Scene details** — Thumbnails, titles, studios for both scenes

You can dismiss false positives (soft or permanent dismiss) or take action on confirmed duplicates.
