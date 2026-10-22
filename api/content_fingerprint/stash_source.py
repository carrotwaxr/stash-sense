"""Which files to fingerprint: from Stash's GraphQL API, or from a JSON-lines list exported elsewhere.

Stash reports paths as its container sees them (e.g. ``/data/...``); ``--mount-map /data=/mnt/stash`` turns them into
paths this machine can open.
"""

import json
import urllib.request
from dataclasses import dataclass, field


@dataclass
class FileRecord:
    oshash: str
    size: int
    path: str                 # as this machine reaches it
    md5: str | None = None
    duration: float | None = None
    scene_ids: list = field(default_factory=list)


def parse_mount_map(specs):
    """["/data=/mnt/stash", ...] -> [(container prefix, local prefix)], longest prefix first."""
    pairs = []
    for spec in specs or []:
        src, sep, dst = spec.partition("=")
        if not sep or not src or not dst:
            raise ValueError(f"mount map {spec!r} is not SRC=DST")
        pairs.append((src.rstrip("/"), dst.rstrip("/")))
    return sorted(pairs, key=lambda p: -len(p[0]))


def map_path(path, mapping):
    for src, dst in mapping:
        if path == src or path.startswith(src + "/"):
            return dst + path[len(src):]
    return path


_QUERY = """query($f: FindFilterType, $sf: SceneFilterType, $ids: [ID!]) {
  findScenes(filter: $f, scene_filter: $sf, ids: $ids) { count scenes { id files {
    path size duration fingerprints { type value } } } } }"""


def _gql(url, api_key, variables):
    req = urllib.request.Request(url.rstrip("/") + "/graphql", data=json.dumps({"query": _QUERY, "variables": variables}).encode(),
                                 headers={"Content-Type": "application/json", **({"ApiKey": api_key} if api_key else {})})
    with urllib.request.urlopen(req, timeout=120) as r:
        body = json.load(r)
    if body.get("errors"):
        raise RuntimeError(f"Stash GraphQL error: {body['errors'][0].get('message')}")
    return body["data"]["findScenes"]


def from_stash(url, api_key, mapping, group_ids=None, scene_ids=None, per_page=200):
    """FileRecords for every file of the selected scenes (all scenes when nothing is selected), one per content."""
    sf = {"groups": {"value": [str(g) for g in group_ids], "modifier": "INCLUDES"}} if group_ids else None
    seen, page = {}, 1
    while True:
        res = _gql(url, api_key, {"f": {"page": page, "per_page": per_page, "sort": "id", "direction": "ASC"},
                                  "sf": sf, "ids": [str(i) for i in scene_ids] if scene_ids else None})
        for scene in res["scenes"]:
            for f in scene["files"]:
                fps = {x["type"]: x["value"] for x in f["fingerprints"]}
                if "oshash" not in fps:
                    continue
                key = (fps["oshash"], f["size"])
                if key in seen:
                    seen[key].scene_ids.append(scene["id"])
                    continue
                seen[key] = FileRecord(fps["oshash"], int(f["size"]), map_path(f["path"], mapping), fps.get("md5"),
                                       f.get("duration"), [scene["id"]])
        if page * per_page >= res["count"]:
            break
        page += 1
    return list(seen.values())


def from_list(path, mapping):
    """FileRecords from a JSON-lines file of {oshash, size, path, md5?, duration?, scene_ids?}."""
    out = []
    with open(path) as fh:
        for line in fh:
            if line.strip():
                d = json.loads(line)
                out.append(FileRecord(d["oshash"], int(d["size"]), map_path(d["path"], mapping), d.get("md5"),
                                      d.get("duration"), d.get("scene_ids", [])))
    return out
