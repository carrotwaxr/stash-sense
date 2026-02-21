"""
Unified Stash GraphQL Client

Combines functionality from:
- duplicate-performer-finder/stash_client.py
- scene-file-deduper/stash_client.py
- Additional queries for recommendations engine

Supports both sync and async operations.
All async requests go through the rate limiter to prevent overwhelming Stash.
"""

import httpx
from typing import Optional

from rate_limiter import RateLimiter, Priority


class StashClientUnified:
    """
    Unified client for Stash GraphQL API.

    Provides all queries needed for the recommendations engine.
    """

    def __init__(self, url: str, api_key: str = ""):
        self.base_url = url.rstrip("/")
        self.graphql_url = f"{self.base_url}/graphql"
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
        }
        if api_key:
            self.headers["ApiKey"] = api_key

    def _execute_sync(self, query: str, variables: dict | None = None) -> dict:
        """Execute a GraphQL query synchronously."""
        payload = {"query": query}
        if variables:
            payload["variables"] = variables

        with httpx.Client(timeout=60.0) as client:
            response = client.post(self.graphql_url, json=payload, headers=self.headers)
            response.raise_for_status()

            result = response.json()
            if "errors" in result:
                raise RuntimeError(f"GraphQL error: {result['errors']}")

            return result["data"]

    async def _execute(
        self,
        query: str,
        variables: dict | None = None,
        priority: Priority = Priority.NORMAL,
        skip_rate_limit: bool = False,
    ) -> dict:
        """
        Execute a GraphQL query asynchronously.

        Args:
            query: GraphQL query string
            variables: Query variables
            priority: Request priority for rate limiting (default: NORMAL)
            skip_rate_limit: Skip rate limiting (for health checks, etc.)
        """
        payload = {"query": query}
        if variables:
            payload["variables"] = variables

        # Apply rate limiting unless skipped
        if not skip_rate_limit:
            limiter = await RateLimiter.get_instance()
            async with limiter.acquire(priority):
                return await self._do_request(payload)
        else:
            return await self._do_request(payload)

    async def _do_request(self, payload: dict) -> dict:
        """Execute the actual HTTP request."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(self.graphql_url, json=payload, headers=self.headers)

            if not response.is_success:
                # Include response body in error for debugging
                try:
                    body = response.json()
                except Exception:
                    body = response.text  # Response not JSON, use raw text
                raise RuntimeError(
                    f"Stash API error (HTTP {response.status_code}): {body}"
                )

            result = response.json()
            if "errors" in result:
                raise RuntimeError(f"GraphQL error: {result['errors']}")

            return result["data"]

    # ==================== Connection ====================

    async def test_connection(self) -> bool:
        """Test connection to Stash. Returns True if successful."""
        query = "query { systemStatus { databaseSchema databasePath } }"
        await self._execute(query, skip_rate_limit=True)
        return True

    def test_connection_sync(self) -> bool:
        """Test connection synchronously."""
        query = "query { systemStatus { databaseSchema databasePath } }"
        self._execute_sync(query)
        return True

    # ==================== Performers ====================

    async def get_all_performers(self) -> list[dict]:
        """Fetch all performers with stash_ids."""
        query = """
        query AllPerformersWithStashIDs {
          findPerformers(filter: { per_page: -1 }) {
            performers {
              id
              name
              alias_list
              gender
              country
              scene_count
              image_count
              gallery_count
              image_path
              stash_ids {
                endpoint
                stash_id
              }
            }
          }
        }
        """
        data = await self._execute(query)
        return data["findPerformers"]["performers"]

    async def get_performer(self, performer_id: str) -> dict:
        """Get a performer by ID with full details."""
        query = """
        query GetPerformer($id: ID!) {
          findPerformer(id: $id) {
            id
            name
            alias_list
            gender
            country
            scene_count
            image_count
            gallery_count
            image_path
            stash_ids {
              endpoint
              stash_id
            }
          }
        }
        """
        data = await self._execute(query, {"id": performer_id})
        return data["findPerformer"]

    async def merge_performers(self, source_ids: list[str], destination_id: str) -> dict:
        """
        Merge source performers into destination performer.

        Reassigns all scenes/images/galleries and merges aliases.
        """
        # Get destination performer info
        dest = await self.get_performer(destination_id)
        dest_aliases = set(dest.get("alias_list") or [])

        for source_id in source_ids:
            source = await self.get_performer(source_id)
            source_name = source.get("name", "")
            source_aliases = source.get("alias_list") or []

            # Add source name and aliases to destination aliases
            if source_name and source_name != dest["name"]:
                dest_aliases.add(source_name)
            dest_aliases.update(source_aliases)

            # Reassign scenes
            scenes = await self.get_performer_scenes(source_id)
            for scene in scenes:
                current_performer_ids = [p["id"] for p in scene["performers"]]
                new_performer_ids = [pid for pid in current_performer_ids if pid != source_id]
                if destination_id not in new_performer_ids:
                    new_performer_ids.append(destination_id)
                if set(new_performer_ids) != set(current_performer_ids):
                    await self.update_scene_performers(scene["id"], new_performer_ids)

            # Reassign images
            images = await self.get_performer_images(source_id)
            for image in images:
                current_performer_ids = [p["id"] for p in image["performers"]]
                new_performer_ids = [pid for pid in current_performer_ids if pid != source_id]
                if destination_id not in new_performer_ids:
                    new_performer_ids.append(destination_id)
                if set(new_performer_ids) != set(current_performer_ids):
                    await self.update_image_performers(image["id"], new_performer_ids)

            # Reassign galleries
            galleries = await self.get_performer_galleries(source_id)
            for gallery in galleries:
                current_performer_ids = [p["id"] for p in gallery["performers"]]
                new_performer_ids = [pid for pid in current_performer_ids if pid != source_id]
                if destination_id not in new_performer_ids:
                    new_performer_ids.append(destination_id)
                if set(new_performer_ids) != set(current_performer_ids):
                    await self.update_gallery_performers(gallery["id"], new_performer_ids)

        # Update destination performer's aliases
        dest_aliases.discard(dest["name"])
        if dest_aliases:
            await self.update_performer_aliases(destination_id, list(dest_aliases))

        return {"id": destination_id, "name": dest["name"]}

    async def get_performer_scenes(self, performer_id: str) -> list[dict]:
        """Get all scenes for a performer."""
        query = """
        query GetPerformerScenes($id: ID!) {
          findScenes(scene_filter: { performers: { value: [$id], modifier: INCLUDES } }, filter: { per_page: -1 }) {
            scenes {
              id
              performers { id }
            }
          }
        }
        """
        data = await self._execute(query, {"id": performer_id})
        return data["findScenes"]["scenes"]

    async def get_performer_images(self, performer_id: str) -> list[dict]:
        """Get all images for a performer."""
        query = """
        query GetPerformerImages($id: ID!) {
          findImages(image_filter: { performers: { value: [$id], modifier: INCLUDES } }, filter: { per_page: -1 }) {
            images {
              id
              performers { id }
            }
          }
        }
        """
        data = await self._execute(query, {"id": performer_id})
        return data["findImages"]["images"]

    async def get_performer_galleries(self, performer_id: str) -> list[dict]:
        """Get all galleries for a performer."""
        query = """
        query GetPerformerGalleries($id: ID!) {
          findGalleries(gallery_filter: { performers: { value: [$id], modifier: INCLUDES } }, filter: { per_page: -1 }) {
            galleries {
              id
              performers { id }
            }
          }
        }
        """
        data = await self._execute(query, {"id": performer_id})
        return data["findGalleries"]["galleries"]

    async def update_performer_aliases(self, performer_id: str, aliases: list[str]) -> None:
        """Update the aliases for a performer."""
        query = """
        mutation UpdatePerformer($id: ID!, $alias_list: [String!]) {
          performerUpdate(input: { id: $id, alias_list: $alias_list }) {
            id
          }
        }
        """
        await self._execute(query, {"id": performer_id, "alias_list": aliases}, priority=Priority.CRITICAL)

    async def get_performers_for_endpoint(self, endpoint: str) -> list[dict]:
        """Query local Stash for all performers linked to a specific stash-box endpoint."""
        query = """
        query PerformersForEndpoint($performer_filter: PerformerFilterType) {
          findPerformers(performer_filter: $performer_filter, filter: { per_page: -1 }) {
            performers {
              id
              name
              disambiguation
              alias_list
              gender
              birthdate
              death_date
              ethnicity
              country
              eye_color
              hair_color
              height_cm
              measurements
              fake_tits
              career_length
              tattoos
              piercings
              details
              urls
              favorite
              image_path
              stash_ids {
                endpoint
                stash_id
              }
            }
          }
        }
        """
        variables = {
            "performer_filter": {
                "stash_id_endpoint": {
                    "endpoint": endpoint,
                    "modifier": "NOT_NULL",
                }
            }
        }
        data = await self._execute(query, variables)
        return data["findPerformers"]["performers"]

    async def update_performer(self, performer_id: str, **fields) -> dict:
        """
        Generic performer update via PerformerUpdateInput mutation.

        Args:
            performer_id: The ID of the performer to update.
            **fields: Arbitrary performer fields to update (e.g. name, height_cm, country).

        Returns:
            The performerUpdate result dict.
        """
        query = """
        mutation PerformerUpdate($input: PerformerUpdateInput!) {
          performerUpdate(input: $input) {
            id
          }
        }
        """
        input_dict = {"id": performer_id, **fields}
        data = await self._execute(query, {"input": input_dict}, priority=Priority.CRITICAL)
        return data["performerUpdate"]

    async def create_performer(self, **fields) -> dict:
        """Create a new performer in Stash."""
        query = """
        mutation PerformerCreate($input: PerformerCreateInput!) {
          performerCreate(input: $input) {
            id
            name
          }
        }
        """
        data = await self._execute(query, {"input": fields}, priority=Priority.CRITICAL)
        return data["performerCreate"]

    # ==================== Scenes ====================

    async def get_multi_file_scenes(self, exclude_tag_ids: list[str] | None = None) -> list[dict]:
        """Fetch all scenes with more than one file."""
        query = """
        query MultiFileScenes($scene_filter: SceneFilterType) {
          findScenes(scene_filter: $scene_filter, filter: { per_page: -1 }) {
            scenes {
              id
              title
              files {
                id
                path
                basename
                size
                duration
                video_codec
                audio_codec
                width
                height
                frame_rate
                bit_rate
              }
              performers {
                id
                name
              }
              studio {
                id
                name
              }
              tags {
                id
                name
              }
            }
          }
        }
        """
        scene_filter: dict = {"file_count": {"value": 1, "modifier": "GREATER_THAN"}}
        if exclude_tag_ids:
            scene_filter["tags"] = {"value": exclude_tag_ids, "modifier": "EXCLUDES"}

        data = await self._execute(query, {"scene_filter": scene_filter})
        return data["findScenes"]["scenes"]

    async def get_scenes_with_sprites(
        self,
        updated_after: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[dict], int]:
        """
        Get scenes that have sprites generated.
        Returns (scenes, total_count).
        """
        query = """
        query ScenesWithSprites($filter: FindFilterType, $scene_filter: SceneFilterType) {
          findScenes(filter: $filter, scene_filter: $scene_filter) {
            count
            scenes {
              id
              title
              updated_at
              paths {
                sprite
              }
              performers {
                id
                name
                stash_ids {
                  endpoint
                  stash_id
                }
              }
              scene_markers {
                id
              }
            }
          }
        }
        """
        filter_input = {"per_page": limit, "page": (offset // limit) + 1}
        scene_filter = {}

        if updated_after:
            scene_filter["updated_at"] = {"value": updated_after, "modifier": "GREATER_THAN"}

        data = await self._execute(query, {"filter": filter_input, "scene_filter": scene_filter})
        return data["findScenes"]["scenes"], data["findScenes"]["count"]

    async def update_scene_performers(self, scene_id: str, performer_ids: list[str]) -> None:
        """Update the performers for a scene."""
        query = """
        mutation UpdateScene($id: ID!, $performer_ids: [ID!]) {
          sceneUpdate(input: { id: $id, performer_ids: $performer_ids }) {
            id
          }
        }
        """
        await self._execute(query, {"id": scene_id, "performer_ids": performer_ids}, priority=Priority.CRITICAL)

    async def update_image_performers(self, image_id: str, performer_ids: list[str]) -> None:
        """Update the performers for an image."""
        query = """
        mutation UpdateImage($id: ID!, $performer_ids: [ID!]) {
          imageUpdate(input: { id: $id, performer_ids: $performer_ids }) {
            id
          }
        }
        """
        await self._execute(query, {"id": image_id, "performer_ids": performer_ids}, priority=Priority.CRITICAL)

    async def update_gallery_performers(self, gallery_id: str, performer_ids: list[str]) -> None:
        """Update the performers for a gallery."""
        query = """
        mutation UpdateGallery($id: ID!, $performer_ids: [ID!]) {
          galleryUpdate(input: { id: $id, performer_ids: $performer_ids }) {
            id
          }
        }
        """
        await self._execute(query, {"id": gallery_id, "performer_ids": performer_ids}, priority=Priority.CRITICAL)

    async def get_scenes_for_endpoint(self, endpoint: str) -> list[dict]:
        """Query local Stash for all scenes linked to a specific stash-box endpoint."""
        query = """
        query ScenesForEndpoint($scene_filter: SceneFilterType) {
          findScenes(scene_filter: $scene_filter, filter: { per_page: -1 }) {
            scenes {
              id
              title
              date
              details
              director
              code
              urls
              studio {
                id
                name
                stash_ids {
                  endpoint
                  stash_id
                }
              }
              performers {
                id
                name
                stash_ids {
                  endpoint
                  stash_id
                }
              }
              tags {
                id
                name
                stash_ids {
                  endpoint
                  stash_id
                }
              }
              stash_ids {
                endpoint
                stash_id
              }
            }
          }
        }
        """
        variables = {
            "scene_filter": {
                "stash_id_endpoint": {
                    "endpoint": endpoint,
                    "modifier": "NOT_NULL",
                }
            }
        }
        data = await self._execute(query, variables)
        return data["findScenes"]["scenes"]

    async def update_scene(self, scene_id: str, **fields) -> dict:
        """Generic scene update via SceneUpdateInput mutation."""
        query = """
        mutation SceneUpdate($input: SceneUpdateInput!) {
          sceneUpdate(input: $input) {
            id
          }
        }
        """
        input_dict = {"id": scene_id, **fields}
        data = await self._execute(query, {"input": input_dict}, priority=Priority.CRITICAL)
        return data["sceneUpdate"]

    # ==================== Files ====================

    async def set_scene_primary_file(self, scene_id: str, file_id: str) -> None:
        """Set the primary file for a scene."""
        query = """
        mutation SetPrimaryFile($id: ID!, $primary_file_id: ID!) {
          sceneUpdate(input: { id: $id, primary_file_id: $primary_file_id }) {
            id
          }
        }
        """
        await self._execute(query, {"id": scene_id, "primary_file_id": file_id}, priority=Priority.CRITICAL)

    async def delete_files(self, file_ids: list[str]) -> bool:
        """Delete files by ID. Returns True if successful."""
        query = """
        mutation DeleteFiles($ids: [ID!]!) {
          deleteFiles(ids: $ids)
        }
        """
        data = await self._execute(query, {"ids": file_ids}, priority=Priority.CRITICAL)
        return data["deleteFiles"]

    async def delete_scene_files(
        self,
        scene_id: str,
        file_ids_to_delete: list[str],
        keep_file_id: str,
        all_file_ids: list[str],
    ) -> bool:
        """
        Delete specified files from a scene, handling primary file logic.
        """
        primary_file_id = all_file_ids[0] if all_file_ids else None

        # If we're deleting the primary file, set the keep file as primary first
        if primary_file_id in file_ids_to_delete:
            await self.set_scene_primary_file(scene_id, keep_file_id)

        return await self.delete_files(file_ids_to_delete)

    # ==================== Tags ====================

    async def get_all_tags(self) -> list[dict]:
        """Fetch all tags."""
        query = """
        query AllTags {
          allTags {
            id
            name
          }
        }
        """
        data = await self._execute(query)
        return data["allTags"]

    async def get_tags_for_endpoint(self, endpoint: str) -> list[dict]:
        """Query local Stash for all tags linked to a specific stash-box endpoint."""
        query = """
        query TagsForEndpoint($tag_filter: TagFilterType) {
          findTags(tag_filter: $tag_filter, filter: { per_page: -1 }) {
            tags {
              id
              name
              description
              aliases
              stash_ids {
                endpoint
                stash_id
              }
            }
          }
        }
        """
        variables = {
            "tag_filter": {
                "stash_id_endpoint": {
                    "endpoint": endpoint,
                    "modifier": "NOT_NULL",
                }
            }
        }
        data = await self._execute(query, variables)
        return data["findTags"]["tags"]

    async def update_tag(self, tag_id: str, **fields) -> dict:
        """
        Generic tag update via TagUpdateInput mutation.

        Args:
            tag_id: The ID of the tag to update.
            **fields: Arbitrary tag fields to update (e.g. name, description, aliases).

        Returns:
            The tagUpdate result dict.
        """
        query = """
        mutation TagUpdate($input: TagUpdateInput!) {
          tagUpdate(input: $input) {
            id
          }
        }
        """
        input_dict = {"id": tag_id, **fields}
        data = await self._execute(query, {"input": input_dict}, priority=Priority.CRITICAL)
        return data["tagUpdate"]

    async def create_tag(self, **fields) -> dict:
        """Create a new tag in Stash."""
        query = """
        mutation TagCreate($input: TagCreateInput!) {
          tagCreate(input: $input) {
            id
            name
          }
        }
        """
        data = await self._execute(query, {"input": fields}, priority=Priority.CRITICAL)
        return data["tagCreate"]

    # ==================== Studios ====================

    async def get_studios_for_endpoint(self, endpoint: str) -> list[dict]:
        """Query local Stash for all studios linked to a specific stash-box endpoint."""
        query = """
        query StudiosForEndpoint($studio_filter: StudioFilterType) {
          findStudios(studio_filter: $studio_filter, filter: { per_page: -1 }) {
            studios {
              id
              name
              url
              parent_studio {
                id
                name
                stash_ids {
                  endpoint
                  stash_id
                }
              }
              stash_ids {
                endpoint
                stash_id
              }
            }
          }
        }
        """
        variables = {
            "studio_filter": {
                "stash_id_endpoint": {
                    "endpoint": endpoint,
                    "modifier": "NOT_NULL",
                }
            }
        }
        data = await self._execute(query, variables)
        return data["findStudios"]["studios"]

    async def update_studio(self, studio_id: str, **fields) -> dict:
        """Generic studio update via StudioUpdateInput mutation."""
        query = """
        mutation StudioUpdate($input: StudioUpdateInput!) {
          studioUpdate(input: $input) {
            id
          }
        }
        """
        input_dict = {"id": studio_id, **fields}
        data = await self._execute(query, {"input": input_dict}, priority=Priority.CRITICAL)
        return data["studioUpdate"]

    async def create_studio(
        self,
        name: str,
        stash_ids: list[dict],
        url: str | None = None,
        parent_id: str | None = None,
    ) -> dict:
        """Create a new studio in Stash."""
        query = """
        mutation StudioCreate($input: StudioCreateInput!) {
          studioCreate(input: $input) {
            id
            name
          }
        }
        """
        input_dict: dict = {"name": name, "stash_ids": stash_ids}
        if url:
            input_dict["url"] = url
        if parent_id:
            input_dict["parent_id"] = parent_id
        data = await self._execute(query, {"input": input_dict}, priority=Priority.CRITICAL)
        return data["studioCreate"]

    # ==================== Configuration ====================

    async def get_stashbox_connections(self) -> list[dict]:
        """Get configured stash-box connections."""
        query = """
        query StashBoxConnections {
          configuration {
            general {
              stashBoxes {
                endpoint
                api_key
                name
                max_requests_per_minute
              }
            }
          }
        }
        """
        data = await self._execute(query)
        return data["configuration"]["general"]["stashBoxes"]

    # ==================== Scene Query Methods ====================

    async def get_scenes_for_fingerprinting(
        self,
        updated_after: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[dict], int]:
        """
        Get scenes with full metadata for fingerprinting.
        Returns (scenes, total_count).
        """
        query = """
        query ScenesForFingerprinting($filter: FindFilterType, $scene_filter: SceneFilterType) {
          findScenes(filter: $filter, scene_filter: $scene_filter) {
            count
            scenes {
              id
              title
              date
              updated_at
              studio {
                id
                name
              }
              performers {
                id
                name
              }
              files {
                duration
              }
              stash_ids {
                endpoint
                stash_id
              }
            }
          }
        }
        """
        filter_input = {"per_page": limit, "page": (offset // limit) + 1}
        scene_filter = {}

        if updated_after:
            scene_filter["updated_at"] = {"value": updated_after, "modifier": "GREATER_THAN"}

        data = await self._execute(query, {"filter": filter_input, "scene_filter": scene_filter})
        return data["findScenes"]["scenes"], data["findScenes"]["count"]

    async def get_scenes_with_fingerprints(
        self,
        updated_after: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[dict], int]:
        """
        Get scenes with file fingerprint hashes for stash-box matching.
        Returns (scenes, total_count).

        Each scene includes files[].fingerprints with type (md5/oshash/phash)
        and value, plus stash_ids showing which endpoints are already linked.
        """
        query = """
        query ScenesWithFingerprints($filter: FindFilterType, $scene_filter: SceneFilterType) {
          findScenes(filter: $filter, scene_filter: $scene_filter) {
            count
            scenes {
              id
              title
              updated_at
              files {
                id
                duration
                fingerprints {
                  type
                  value
                }
              }
              stash_ids {
                endpoint
                stash_id
              }
            }
          }
        }
        """
        filter_input = {"per_page": limit, "page": (offset // limit) + 1}
        scene_filter = {}

        if updated_after:
            scene_filter["updated_at"] = {"value": updated_after, "modifier": "GREATER_THAN"}

        data = await self._execute(query, {"filter": filter_input, "scene_filter": scene_filter})
        return data["findScenes"]["scenes"], data["findScenes"]["count"]

    async def get_scene_stream_url(self, scene_id: str) -> Optional[str]:
        """Get the stream URL for a scene."""
        query = """
        query SceneStream($id: ID!) {
          findScene(id: $id) {
            id
            sceneStreams {
              url
              label
            }
          }
        }
        """
        data = await self._execute(query, {"id": scene_id})
        scene = data.get("findScene")
        if not scene or not scene.get("sceneStreams"):
            return None

        # Return first available stream
        streams = scene["sceneStreams"]
        return streams[0]["url"] if streams else None

    async def get_scene_by_id(self, scene_id: str) -> Optional[dict]:
        """Get a scene by ID with full metadata."""
        query = """
        query GetScene($id: ID!) {
          findScene(id: $id) {
            id
            title
            date
            updated_at
            studio {
              id
              name
            }
            performers {
              id
              name
            }
            files {
              duration
            }
            stash_ids {
              endpoint
              stash_id
            }
          }
        }
        """
        data = await self._execute(query, {"id": scene_id})
        return data.get("findScene")

    # ==================== Images ====================

    async def get_image_by_id(self, image_id: str) -> Optional[dict]:
        """Get an image by ID with paths and performers."""
        query = """
        query GetImage($id: ID!) {
          findImage(id: $id) {
            id
            title
            paths {
              image
              thumbnail
            }
            performers {
              id
              name
              image_path
            }
          }
        }
        """
        data = await self._execute(query, {"id": image_id})
        return data.get("findImage")

    # ==================== Galleries ====================

    async def get_gallery_by_id(self, gallery_id: str) -> Optional[dict]:
        """Get a gallery by ID with all images and performers."""
        # Gallery metadata
        gallery_query = """
        query GetGallery($id: ID!) {
          findGallery(id: $id) {
            id
            title
            image_count
            performers {
              id
              name
              image_path
            }
          }
        }
        """
        data = await self._execute(gallery_query, {"id": gallery_id})
        gallery = data.get("findGallery")
        if not gallery:
            return None

        # Fetch all images in this gallery via findImages with gallery filter
        image_count = gallery.get("image_count", 0)
        images_query = """
        query GetGalleryImages($gallery_id: [ID!]!, $per_page: Int!) {
          findImages(
            image_filter: { galleries: { value: $gallery_id, modifier: INCLUDES } }
            filter: { per_page: $per_page }
          ) {
            images {
              id
              title
              paths {
                image
                thumbnail
              }
              performers {
                id
                name
              }
            }
          }
        }
        """
        images_data = await self._execute(
            images_query,
            {"gallery_id": [gallery_id], "per_page": max(image_count, 100)},
        )
        gallery["images"] = images_data.get("findImages", {}).get("images", [])
        return gallery
