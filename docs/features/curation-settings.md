# Curation Settings

Two settings apply to every Stash Sense curation feature: an exclusion list of content nothing may touch, and the video formats you would rather not keep. Both are stored in Stash itself, so other tools can read them too.

Edit them in **Stash Sense > Settings > Curation**.

## Never touch (exclusion list)

Pick tags, groups and studios. Ids that are not numbers are ignored. A scene matching any of them is excluded:

- **A tag** excludes scenes with that tag or any of its child tags, at any depth.
- **A group** excludes scenes in that group or any of its sub-groups.
- **A studio** excludes scenes from that studio or any of its child studios, so a network covers all its sites.

For an excluded scene, Stash Sense:

- never fingerprints it (face fingerprints are skipped);
- never creates a recommendation for it, and hides any existing recommendation that names it, including a duplicate pair where only one scene is excluded;
- refuses any action on it: preview, accept, batch accept, revert, merge, delete, update, and scene identification. The refusal names the scene and points here.

Changes apply immediately after **Save**. Every write action and every analysis run reads the list fresh from Stash; the dashboard's lists refresh within a minute. When you take something off the list, the next run of each analysis checks its scenes again in full.

Stash Sense also never sends an excluded scene's fingerprints to a stash-box when looking for matches.

If Stash can't be reached, analysis runs and write actions are refused rather than risk touching an excluded scene. The dashboard keeps showing lists filtered by the last exclusion list it read.

Merging performers is refused when a performer being merged away appears in an excluded scene, because the merge rewrites the performers on those scenes.

## Disfavored formats

Two comma-separated lists: **video codecs** as Stash reports them (for example `hevc`, `av1`, `vp9`) and **containers** (for example `wmv`, `avi`).

They are taste, not quality. When Stash Sense suggests which of a scene's files to keep, a better file still wins on resolution and bitrate. A disfavored format only decides between files that are otherwise equal. With both lists empty, no codec or container is preferred.

## Reading the settings from other tools

The settings live in the `stash-sense` plugin's configuration in Stash. Any tool with a Stash API key can read them:

```graphql
query { configuration { plugins } }
```

`plugins["stash-sense"]` holds these curation keys, empty by default unless noted:

| Key | Contents |
| --- | --- |
| `excludeTagIds` | Tag ids |
| `excludeGroupIds` | Group ids |
| `excludeStudioIds` | Studio ids |
| `disfavoredVideoCodecs` | Lowercase codec names |
| `disfavoredContainers` | Lowercase container names |
| `qualityReportOnly` | Boolean, on by default: [quality flag](quality-flags.md) cards record answers and change nothing |
| `qualityGroupBroken`, `qualityGroupRip`, `qualityGroupBelowStudio`, `qualityGroupFormat` | One group id each: the work-queue group an accepted quality card adds the scene to |
| `qualityPeerPercentile` | Number from 1 to 50, default 10: the below-studio percentile |

To resolve the exclusion list to scenes, query `findScenes` with each list as a `tags`, `groups` or `studios` criterion, using `modifier: INCLUDES` and `depth: -1`, and take the union.

!!! warning "Writing the settings"
    Stash's `configurePlugin` mutation replaces the plugin's entire settings map. Read the current map, change your keys, then write the whole map back. Sending only your keys erases the others, including the sidecar URL.

The same fields also appear under Stash's **Settings > Plugins > Stash Sense** as plain text.
