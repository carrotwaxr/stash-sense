"""Client for interacting with local Stash instance."""
import requests
from typing import Iterator, Optional
from dataclasses import dataclass

@dataclass
class Performer:
    """Performer data from Stash."""
    id: str
    name: str
    image_url: Optional[str]
    stashdb_id: Optional[str]
    scene_count: int
    image_count: int
    gallery_count: int

class StashClient:
    """Client for the local Stash GraphQL API."""

    def __init__(self, url: str, api_key: str):
        self.url = url.rstrip("/")
        self.graphql_url = f"{self.url}/graphql"
        self.headers = {
            "ApiKey": api_key,
            "Content-Type": "application/json",
        }

    def _query(self, query: str, variables: dict = None) -> dict:
        """Execute a GraphQL query."""
        payload = {"query": query}
        if variables:
            payload["variables"] = variables
        response = requests.post(self.graphql_url, json=payload, headers=self.headers)
        response.raise_for_status()
        result = response.json()
        if "errors" in result:
            raise Exception(f"GraphQL errors: {result['errors']}")
        return result["data"]

    def get_stats(self) -> dict:
        """Get basic stats from Stash."""
        query = """
        query {
            stats {
                performer_count
                scene_count
                image_count
                gallery_count
            }
        }
        """
        return self._query(query)["stats"]

    def get_performer_count(self) -> int:
        """Get total number of performers."""
        query = """
        query {
            findPerformers(filter: {per_page: 0}) {
                count
            }
        }
        """
        return self._query(query)["findPerformers"]["count"]

    def get_performers(
        self,
        page: int = 1,
        per_page: int = 100,
        with_stashdb_id: bool = False,
    ) -> tuple[int, list[Performer]]:
        """
        Get performers with pagination.

        Returns: (total_count, list of Performer objects)
        """
        # Build filter
        performer_filter = {}
        if with_stashdb_id:
            performer_filter["stash_id_endpoint"] = {
                "endpoint": "https://stashdb.org/graphql",
                "modifier": "NOT_NULL"
            }

        query = """
        query FindPerformers($filter: FindFilterType, $performer_filter: PerformerFilterType) {
            findPerformers(filter: $filter, performer_filter: $performer_filter) {
                count
                performers {
                    id
                    name
                    image_path
                    stash_ids {
                        endpoint
                        stash_id
                    }
                    scene_count
                    image_count
                    gallery_count
                }
            }
        }
        """
        variables = {
            "filter": {
                "page": page,
                "per_page": per_page,
            },
            "performer_filter": performer_filter if performer_filter else None,
        }

        result = self._query(query, variables)["findPerformers"]

        performers = []
        for p in result["performers"]:
            # Extract StashDB ID if present
            stashdb_id = None
            for stash_id in p.get("stash_ids", []):
                if stash_id["endpoint"] == "https://stashdb.org/graphql":
                    stashdb_id = stash_id["stash_id"]
                    break

            performers.append(Performer(
                id=p["id"],
                name=p["name"],
                image_url=p.get("image_path"),
                stashdb_id=stashdb_id,
                scene_count=p.get("scene_count", 0),
                image_count=p.get("image_count", 0),
                gallery_count=p.get("gallery_count", 0),
            ))

        return result["count"], performers

    def iter_performers(
        self,
        per_page: int = 100,
        with_stashdb_id: bool = False,
    ) -> Iterator[Performer]:
        """Iterate through all performers."""
        page = 1
        while True:
            count, performers = self.get_performers(
                page=page,
                per_page=per_page,
                with_stashdb_id=with_stashdb_id,
            )
            if not performers:
                break
            yield from performers
            if page * per_page >= count:
                break
            page += 1

    def download_performer_image(self, performer: Performer) -> Optional[bytes]:
        """Download a performer's profile image."""
        if not performer.image_url:
            return None
        try:
            response = requests.get(
                performer.image_url,
                headers={"ApiKey": self.headers["ApiKey"]},
                timeout=30,
            )
            response.raise_for_status()
            return response.content
        except Exception as e:
            print(f"Failed to download image for {performer.name}: {e}")
            return None

    def get_performer_scenes(self, performer_id: str, limit: int = 10) -> list[dict]:
        """Get scenes for a performer."""
        query = """
        query FindScenes($filter: FindFilterType, $scene_filter: SceneFilterType) {
            findScenes(filter: $filter, scene_filter: $scene_filter) {
                scenes {
                    id
                    title
                    paths {
                        screenshot
                        sprite
                    }
                }
            }
        }
        """
        variables = {
            "filter": {"per_page": limit},
            "scene_filter": {
                "performers": {
                    "value": [performer_id],
                    "modifier": "INCLUDES",
                }
            },
        }
        return self._query(query, variables)["findScenes"]["scenes"]


if __name__ == "__main__":
    # Quick test
    import os
    from dotenv import load_dotenv
    load_dotenv()

    client = StashClient(
        url=os.environ["STASH_URL"],
        api_key=os.environ["STASH_API_KEY"],
    )

    print("Stats:", client.get_stats())
    print(f"\nTotal performers: {client.get_performer_count()}")

    count, performers = client.get_performers(per_page=5)
    print(f"\nFirst 5 performers (of {count}):")
    for p in performers:
        print(f"  - {p.name} (ID: {p.id}, StashDB: {p.stashdb_id})")
