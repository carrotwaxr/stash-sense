# Upstream Changes

Stash Sense detects metadata changes on Stash-Box endpoints and presents per-field merge controls to keep your local Stash library in sync. This covers **performers**, **studios**, and **scenes**.

## Prerequisites

- At least one **Stash-Box endpoint** configured in Stash's **Settings > Metadata Providers**
- Performers, studios, or scenes in your library linked to Stash-Box IDs

---

## How It Works

### 3-Way Diff Engine

For each field, the diff engine compares three states:

- **Upstream** — Current value on the Stash-Box endpoint
- **Local** — Current value in your Stash library
- **Snapshot** — Last-seen upstream value (stored locally by Stash Sense)

This three-way comparison distinguishes intentional local differences from actual upstream changes. If you've deliberately set a different name locally, it won't keep suggesting the upstream name on every sync cycle.

### Change Detection

A field is flagged only when the upstream value has changed since the last snapshot **and** the new upstream value differs from your local value. Fields where upstream hasn't changed, or where your local value already matches upstream, are skipped.

---

## Performer Sync

Detects field changes for performers linked to Stash-Box endpoints.

**Per-field merge controls:**

| Field type | Options |
|------------|---------|
| Name fields | Keep local / Accept upstream / Demote to alias / Add as alias / Custom |
| Aliases | Union checkboxes (pick which aliases to keep) |
| Simple fields | Keep local / Accept upstream / Custom |

---

## Studio Sync

Detects changes for studios linked to Stash-Box endpoints.

**Tracked fields:**

- **Name** — Studio display name
- **URL** — Studio website
- **Parent studio** — Hierarchical studio relationships (resolved via Stash-Box IDs, not name matching)

---

## Scene Sync

Detects metadata changes for scenes linked to Stash-Box endpoints.

**Tracked field categories:**

| Category | Fields |
|----------|--------|
| Core metadata | Title, details, date, director, code, URLs |
| Studio assignment | Studio linked to the scene |
| Performer lineup | Added/removed performers, alias changes |
| Tags | Tag set additions and removals |

Performer lineup changes show which performers were added or removed upstream, including alias detection when an upstream performer's credited name differs from the local alias.

---

## Dismissal

When you review an upstream change and decide not to apply it:

- **Soft dismiss** — Hides the recommendation for now. If the upstream value changes again in a future sync, the recommendation reappears.
- **Permanent dismiss** — Skips this field/entity entirely in future syncs. Use this when your local value is intentionally different.

---

## Dismiss All

The **Dismiss All** button lets you dismiss all currently visible upstream change recommendations at once. This is useful after reviewing a batch and deciding to skip the remaining items. Dismiss All applies a soft dismiss — if upstream changes again, items will reappear.

---

## Endpoint Enable/Disable Toggle

Each Stash-Box endpoint can be individually enabled or disabled for upstream sync analysis. Disabled endpoints are skipped during analysis runs but their configuration is preserved.

Use this to:

- Temporarily pause sync from a noisy endpoint
- Focus on changes from a single endpoint at a time
- Disable endpoints you only use for scene tagging, not upstream sync

Toggle endpoints in the plugin's **Settings** tab under the Stash-Box Endpoints section.

---

## Merge Conflict Improvements

When the same field has been changed both locally and upstream since the last snapshot, an **inline comparison panel** shows:

- The **snapshot** value (what the field was when last synced)
- The **local** value (what you changed it to)
- The **upstream** value (what Stash-Box changed it to)

This side-by-side view makes it easy to understand the conflict and decide which value to keep.

---

## Running Upstream Sync

Upstream analysis runs as a background job in the [operation queue](../plugin.md#operation-queue). It can be:

- **Triggered manually** from the Operations tab
- **Scheduled** to run automatically at a configurable interval
- **Run incrementally** — only entities modified since the last watermark are checked, keeping sync fast after the initial scan
