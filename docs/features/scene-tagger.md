# Scene Tagger

The Scene Stash-Box Tagger searches your untagged scenes across Stash-Box endpoints using file fingerprints to find matching metadata — title, studio, performers, tags, and more.

## Prerequisites

- **Perceptual hashes** generated for scenes (Stash: **Settings > Tasks > Generate > Perceptual Hashes**)
- At least one **Stash-Box endpoint** configured in Stash's **Settings > Metadata Providers**

File fingerprints (MD5, oshash, phash) are how Stash-Box endpoints identify your scenes. Without them, there's nothing to search with.

---

## How It Works

1. Trigger the Scene Stash-Box Tagger from the **Operations** tab or the recommendations dashboard
2. Stash Sense queries each configured Stash-Box endpoint with your scenes' file fingerprints
3. When a match is found, the upstream metadata is presented for review
4. You can accept, modify, or skip each match before it's applied to your library

---

## What Gets Matched

When a scene's fingerprint matches on a Stash-Box endpoint, the following metadata is available:

- **Title and details**
- **Date, director, and code**
- **Studio** assignment
- **Performer lineup** with aliases
- **Tags**
- **URLs**

---

## UX Features

- **Auto-expand search** — Search results are automatically expanded when matches are found, reducing clicks
- **Local entity display** — Matched performers and studios show whether they already exist in your local Stash library
- **Auto-matching with aliases** — When a Stash-Box performer has aliases that match a local performer's name, the match is made automatically

---

## Running the Tagger

The tagger can be run from two places:

- **Operations tab** — Trigger a full scan of unmatched scenes
- **Recommendations dashboard** — Untagged scenes appear as recommendations; click to tag them

The tagger runs as a background job in the [operation queue](../plugin.md#operation-queue) and supports incremental runs — only scenes modified or added since the last run are processed.
