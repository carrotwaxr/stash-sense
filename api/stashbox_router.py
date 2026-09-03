"""StashBox and Stash performer management API endpoints.

Provides routes for fetching performer data from StashBox, searching
local Stash performers, creating performers from StashBox data, and
linking performers to scenes, images and galleries.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator

from stashbox_utils import _get_stashbox_client, _get_endpoint_url

logger = logging.getLogger(__name__)

router = APIRouter(tags=["stashbox"])

# Module-level globals set by init
_stash_url = ""
_stash_api_key = ""


def init_stashbox_router(stash_url: str, stash_api_key: str):
    """Initialize the stashbox router with runtime dependencies."""
    global _stash_url, _stash_api_key
    _stash_url = stash_url
    _stash_api_key = stash_api_key


# ==================== Pydantic Models ====================


class StashBoxPerformerResponse(BaseModel):
    """Performer data from StashBox, mapped to Stash's PerformerCreateInput field names."""
    name: str
    disambiguation: Optional[str] = None
    alias_list: list[str] = []
    gender: Optional[str] = None
    birthdate: Optional[str] = None
    death_date: Optional[str] = None
    ethnicity: Optional[str] = None
    country: Optional[str] = None
    eye_color: Optional[str] = None
    hair_color: Optional[str] = None
    height_cm: Optional[int] = None
    measurements: Optional[str] = None
    fake_tits: Optional[str] = None
    career_length: Optional[str] = None
    tattoos: Optional[str] = None
    piercings: Optional[str] = None
    details: Optional[str] = None
    urls: list[str] = []
    image: Optional[str] = None
    weight: Optional[int] = None
    stash_ids: list[dict] = []


class SearchPerformersRequest(BaseModel):
    query: str


class SearchPerformerResult(BaseModel):
    id: str
    name: str
    disambiguation: Optional[str] = None
    alias_list: list[str] = []
    image_path: Optional[str] = None


# Identify results can be applied to a scene, an image or a gallery.
# Requests name exactly one of these targets.
_TARGET_FIELDS = ("scene_id", "image_id", "gallery_id")


def _require_single_target(request: BaseModel) -> None:
    provided = [field for field in _TARGET_FIELDS if getattr(request, field, None)]
    if len(provided) != 1:
        raise ValueError("Exactly one of scene_id, image_id or gallery_id is required")


class CreatePerformerRequest(BaseModel):
    """Create a performer from StashBox data and add it to one scene, image or gallery."""
    endpoint: str
    stashdb_id: str
    scene_id: Optional[str] = None
    image_id: Optional[str] = None
    gallery_id: Optional[str] = None

    @model_validator(mode="after")
    def _validate_target(self):
        _require_single_target(self)
        return self


class CreatePerformerResponse(BaseModel):
    performer_id: str
    name: str
    success: bool
    # False when a performer with this stash-box ID already existed and was linked instead
    created: bool = True


class LinkPerformerRequest(BaseModel):
    """Add an existing Stash performer to one scene, image or gallery."""
    performer_id: str
    scene_id: Optional[str] = None
    image_id: Optional[str] = None
    gallery_id: Optional[str] = None
    stash_ids: list[dict] = []
    update_metadata: bool = False

    @model_validator(mode="after")
    def _validate_target(self):
        _require_single_target(self)
        return self


class LinkPerformerResponse(BaseModel):
    success: bool


# ==================== Helper Functions ====================


def _map_stashbox_to_stash(performer: dict, endpoint_url: str, stashdb_id: str) -> StashBoxPerformerResponse:
    """Map StashBox performer fields to Stash's PerformerCreateInput field names."""
    # breast_type mapping
    breast_type = performer.get("breast_type")
    fake_tits = None
    if breast_type:
        fake_tits = {"NATURAL": "Natural", "FAKE": "Augmented"}.get(breast_type.upper())

    # measurements: "{band}{cup}-{waist}-{hip}"
    cup = performer.get("cup_size")
    band = performer.get("band_size")
    waist = performer.get("waist_size")
    hip = performer.get("hip_size")
    measurements = None
    if any([cup, band, waist, hip]):
        bust = f"{band or ''}{cup or ''}"
        parts = [bust, str(waist) if waist else "", str(hip) if hip else ""]
        measurements = "-".join(parts).strip("-") or None

    # career_length: "YYYY-YYYY" or "YYYY-"
    start = performer.get("career_start_year")
    end = performer.get("career_end_year")
    career_length = None
    if start:
        career_length = f"{start}-{end}" if end else f"{start}-"

    # tattoos/piercings: "location: description" semicolon-separated
    def _format_body_mods(mods):
        if not mods:
            return None
        parts = []
        for mod in mods:
            loc = mod.get("location", "")
            desc = mod.get("description", "")
            if loc and desc:
                parts.append(f"{loc}: {desc}")
            elif loc:
                parts.append(loc)
            elif desc:
                parts.append(desc)
        return "; ".join(parts) if parts else None

    # urls: extract URL strings
    url_list = [u.get("url") for u in (performer.get("urls") or []) if u.get("url")]

    # image: first image URL
    images = performer.get("images") or []
    image_url = images[0].get("url") if images else None

    return StashBoxPerformerResponse(
        name=performer.get("name", ""),
        disambiguation=performer.get("disambiguation"),
        alias_list=performer.get("aliases") or [],
        gender=performer.get("gender"),
        birthdate=performer.get("birth_date"),
        death_date=performer.get("death_date"),
        ethnicity=performer.get("ethnicity"),
        country=performer.get("country"),
        eye_color=performer.get("eye_color"),
        hair_color=performer.get("hair_color"),
        height_cm=performer.get("height"),
        measurements=measurements,
        fake_tits=fake_tits,
        career_length=career_length,
        tattoos=_format_body_mods(performer.get("tattoos")),
        piercings=_format_body_mods(performer.get("piercings")),
        urls=url_list,
        image=image_url,
        stash_ids=[{"endpoint": endpoint_url, "stash_id": stashdb_id}],
    )


def _get_target(request: BaseModel) -> tuple[str, str]:
    """Return ("scene" | "image" | "gallery", id) for the request's single target."""
    for field in _TARGET_FIELDS:
        value = getattr(request, field, None)
        if value:
            return field[: -len("_id")], str(value)
    raise HTTPException(status_code=400, detail="One of scene_id, image_id or gallery_id is required")


# Stash GraphQL query name and StashClientUnified update method per target type
_TARGET_STASH_OPS = {
    "scene": ("findScene", "update_scene_performers"),
    "image": ("findImage", "update_image_performers"),
    "gallery": ("findGallery", "update_gallery_performers"),
}


async def _add_performer_to_target(stash_client, target_type: str, target_id: str, performer_id: str) -> bool:
    """Append a performer to a scene/image/gallery. Returns False if it was already there."""
    find_query, update_method = _TARGET_STASH_OPS[target_type]
    query = f"""
    query GetTargetPerformers($id: ID!) {{
        {find_query}(id: $id) {{ performers {{ id }} }}
    }}
    """
    data = await stash_client._execute(query, {"id": target_id})
    target = data.get(find_query)
    if not target:
        raise HTTPException(status_code=404, detail=f"{target_type.capitalize()} {target_id} not found")

    current_ids = [p["id"] for p in (target.get("performers") or [])]
    if performer_id in current_ids:
        return False
    current_ids.append(performer_id)
    await getattr(stash_client, update_method)(target_id, current_ids)
    return True


async def _find_performer_by_stash_id(stash_client, endpoint_url: str, stashdb_id: str) -> Optional[dict]:
    """Find a local performer already carrying this stash-box ID, if any."""
    query = """
    query FindPerformerByStashId($endpoint: String!, $stash_id: String!) {
        findPerformers(performer_filter: {
            stash_id_endpoint: { endpoint: $endpoint, stash_id: $stash_id, modifier: EQUALS }
        }) {
            performers { id name }
        }
    }
    """
    data = await stash_client._execute(query, {"endpoint": endpoint_url, "stash_id": stashdb_id})
    performers = (data.get("findPerformers") or {}).get("performers") or []
    return performers[0] if performers else None


# ==================== Route Handlers ====================


@router.get("/stashbox/performer/{endpoint}/{stashdb_id}", response_model=StashBoxPerformerResponse)
async def get_stashbox_performer(endpoint: str, stashdb_id: str):
    """Get performer from StashBox, mapped to Stash field names."""
    client = _get_stashbox_client(endpoint)
    if not client:
        raise HTTPException(status_code=400, detail=f"Unknown or unconfigured endpoint: {endpoint}")

    performer = await client.get_performer(stashdb_id)
    if not performer:
        raise HTTPException(status_code=404, detail="Performer not found on StashBox")

    endpoint_url = _get_endpoint_url(endpoint)
    if not endpoint_url:
        raise HTTPException(status_code=400, detail=f"Unknown endpoint: {endpoint}")
    return _map_stashbox_to_stash(performer, endpoint_url, stashdb_id)


@router.post("/stash/search-performers", response_model=list[SearchPerformerResult])
async def search_stash_performers(request: SearchPerformersRequest):
    """Search performers in local Stash library."""
    if not _stash_url:
        raise HTTPException(status_code=400, detail="STASH_URL not configured")

    from stash_client_unified import StashClientUnified
    stash_client = StashClientUnified(_stash_url, _stash_api_key)

    query = """
    query FindPerformers($filter: FindFilterType) {
        findPerformers(filter: $filter) {
            performers {
                id name disambiguation alias_list image_path
            }
        }
    }
    """
    data = await stash_client._execute(query, {
        "filter": {"q": request.query, "per_page": 10, "sort": "name", "direction": "ASC"}
    })
    performers = data.get("findPerformers", {}).get("performers", [])
    return [SearchPerformerResult(**p) for p in performers]


@router.post("/stash/create-performer", response_model=CreatePerformerResponse)
async def create_performer_from_stashbox(request: CreatePerformerRequest):
    """Create a performer in Stash from StashBox data, then add it to a scene, image or gallery.

    If the library already has a performer with this stash-box ID, that performer is
    linked instead and ``created`` is False in the response.
    """
    if not _stash_url:
        raise HTTPException(status_code=400, detail="STASH_URL not configured")

    client = _get_stashbox_client(request.endpoint)
    if not client:
        raise HTTPException(status_code=400, detail=f"Unknown or unconfigured endpoint: {request.endpoint}")

    endpoint_url = _get_endpoint_url(request.endpoint)
    if not endpoint_url:
        raise HTTPException(status_code=400, detail=f"Unknown endpoint: {request.endpoint}")

    target_type, target_id = _get_target(request)

    from stash_client_unified import StashClientUnified
    stash_client = StashClientUnified(_stash_url, _stash_api_key)

    # 1. Reuse an existing performer that already carries this stash-box ID
    existing = await _find_performer_by_stash_id(stash_client, endpoint_url, request.stashdb_id)
    if existing:
        logger.info(
            "Performer %s (#%s) already linked to %s; adding to %s %s instead of creating a duplicate",
            existing["name"], existing["id"], request.stashdb_id, target_type, target_id,
        )
        await _add_performer_to_target(stash_client, target_type, target_id, existing["id"])
        return CreatePerformerResponse(
            performer_id=existing["id"],
            name=existing["name"],
            success=True,
            created=False,
        )

    # 2. Fetch performer data from StashBox
    performer = await client.get_performer(request.stashdb_id)
    if not performer:
        raise HTTPException(status_code=404, detail="Performer not found on StashBox")

    mapped = _map_stashbox_to_stash(performer, endpoint_url, request.stashdb_id)

    # 3. Create performer in Stash
    create_input = {
        "name": mapped.name,
        "stash_ids": mapped.stash_ids,
    }
    # Add optional fields
    if mapped.disambiguation:
        create_input["disambiguation"] = mapped.disambiguation
    if mapped.alias_list:
        create_input["alias_list"] = mapped.alias_list
    if mapped.gender:
        create_input["gender"] = mapped.gender
    if mapped.birthdate:
        create_input["birthdate"] = mapped.birthdate
    if mapped.death_date:
        create_input["death_date"] = mapped.death_date
    if mapped.ethnicity:
        create_input["ethnicity"] = mapped.ethnicity
    if mapped.country:
        create_input["country"] = mapped.country
    if mapped.eye_color:
        create_input["eye_color"] = mapped.eye_color
    if mapped.hair_color:
        create_input["hair_color"] = mapped.hair_color
    if mapped.height_cm is not None:
        create_input["height_cm"] = mapped.height_cm
    if mapped.measurements:
        create_input["measurements"] = mapped.measurements
    if mapped.fake_tits:
        create_input["fake_tits"] = mapped.fake_tits
    if mapped.career_length:
        create_input["career_length"] = mapped.career_length
    if mapped.tattoos:
        create_input["tattoos"] = mapped.tattoos
    if mapped.piercings:
        create_input["piercings"] = mapped.piercings
    if mapped.image:
        create_input["image"] = mapped.image
    if mapped.urls:
        create_input["url"] = mapped.urls[0]  # Stash only accepts single URL on create

    create_query = """
    mutation PerformerCreate($input: PerformerCreateInput!) {
        performerCreate(input: $input) {
            id name
        }
    }
    """
    from rate_limiter import Priority
    data = await stash_client._execute(create_query, {"input": create_input}, priority=Priority.CRITICAL)
    new_performer = data["performerCreate"]

    # 4. Add performer to the target
    await _add_performer_to_target(stash_client, target_type, target_id, new_performer["id"])

    return CreatePerformerResponse(
        performer_id=new_performer["id"],
        name=new_performer["name"],
        success=True,
        created=True,
    )


@router.post("/stash/link-performer", response_model=LinkPerformerResponse)
async def link_performer(request: LinkPerformerRequest):
    """Add an existing Stash performer to a scene, image or gallery. Optionally update stash_ids."""
    if not _stash_url:
        raise HTTPException(status_code=400, detail="STASH_URL not configured")

    target_type, target_id = _get_target(request)

    from stash_client_unified import StashClientUnified
    stash_client = StashClientUnified(_stash_url, _stash_api_key)

    await _add_performer_to_target(stash_client, target_type, target_id, request.performer_id)

    # Optionally update performer's stash_ids
    if request.update_metadata and request.stash_ids:
        # Get current stash_ids
        perf_query = """
        query GetPerformer($id: ID!) {
            findPerformer(id: $id) { stash_ids { endpoint stash_id } }
        }
        """
        perf_data = await stash_client._execute(perf_query, {"id": request.performer_id})
        current_stash_ids = perf_data["findPerformer"]["stash_ids"]

        # Merge new stash_ids (avoid duplicates)
        existing = {(s["endpoint"], s["stash_id"]) for s in current_stash_ids}
        merged = list(current_stash_ids)
        for new_sid in request.stash_ids:
            key = (new_sid["endpoint"], new_sid["stash_id"])
            if key not in existing:
                merged.append(new_sid)

        if len(merged) > len(current_stash_ids):
            await stash_client.update_performer(request.performer_id, stash_ids=merged)

    return LinkPerformerResponse(success=True)
