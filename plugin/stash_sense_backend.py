#!/usr/bin/env python3
"""Backend for Stash Sense plugin.

Proxies requests to the Stash Sense sidecar API to bypass browser CSP restrictions.
"""
import json
import sys
import requests


def main():
    """Handle plugin operations from Stash."""
    # Read input from stdin
    input_data = json.load(sys.stdin)

    # Get operation mode and args
    args = input_data.get("args", {})
    mode = args.get("mode", "")


    # Get sidecar URL from args (passed by JS which reads settings)
    sidecar_url = args.get("sidecar_url", "http://localhost:5000").rstrip("/")

    if mode == "health":
        result = health_check(sidecar_url)
    elif mode == "identify_scene":
        scene_id = args.get("scene_id")
        result = identify_scene(
            sidecar_url, scene_id,
            num_frames=args.get("max_frames") or args.get("num_frames"),
            top_k=args.get("top_k"),
            max_distance=args.get("max_distance"),
            min_face_size=args.get("min_face_size"),
            scene_performer_stashdb_ids=args.get("scene_performer_stashdb_ids", []),
        )
    elif mode == "identify_image":
        image_id = args.get("image_id")
        result = identify_image(sidecar_url, image_id)
    elif mode == "identify_gallery":
        gallery_id = args.get("gallery_id")
        result = identify_gallery(sidecar_url, gallery_id)
    elif mode == "database_info":
        result = database_info(sidecar_url)
    elif mode == "db_check_update":
        result = sidecar_get(sidecar_url, "/database/check-update")
    elif mode == "db_update":
        result = sidecar_post(sidecar_url, "/database/update", timeout=10)
    elif mode == "db_update_status":
        result = sidecar_get(sidecar_url, "/database/update/status")
    elif mode == "search_performers":
        query = args.get("query", "")
        result = sidecar_post(sidecar_url, "/stash/search-performers", {"query": query})
    elif mode == "create_performer_from_stashbox":
        result = sidecar_post(sidecar_url, "/stash/create-performer", {
            "scene_id": str(args.get("scene_id", "")),
            "endpoint": args.get("endpoint", ""),
            "stashdb_id": args.get("stashdb_id", ""),
        }, timeout=30)
    elif mode == "link_performer_stashbox":
        result = sidecar_post(sidecar_url, "/stash/link-performer", {
            "scene_id": str(args.get("scene_id", "")),
            "performer_id": str(args.get("performer_id", "")),
            "stash_ids": args.get("stash_ids", []),
            "update_metadata": args.get("update_metadata", False),
        })
    elif mode.startswith("settings_") or mode == "system_info":
        result = handle_settings(mode, args, sidecar_url)
        if result is None:
            result = {"error": f"Unknown settings mode: {mode}"}
    elif mode.startswith("queue_"):
        result = handle_queue(mode, args, sidecar_url)
    elif mode.startswith("rec_") or mode.startswith("fp_") or mode.startswith("user_"):
        result = handle_recommendations(mode, args, sidecar_url)
        if result is None:
            result = {"error": f"Unknown recommendations mode: {mode}"}
    else:
        result = {"error": f"Unknown mode: {mode}"}

    # Output result
    output = {"output": result}
    print(json.dumps(output))


def log(message):
    """Log a message to Stash."""
    print(json.dumps({"log": f"[Stash Sense] {message}"}), file=sys.stderr)


def health_check(sidecar_url):
    """Check sidecar health."""
    try:
        response = requests.get(f"{sidecar_url}/health", timeout=10)
        if response.ok:
            return response.json()
        return {"error": f"Health check failed: HTTP {response.status_code}"}
    except requests.ConnectionError:
        return {"error": "Connection refused - is Stash Sense running?"}
    except requests.Timeout:
        return {"error": "Connection timed out"}
    except requests.RequestException as e:
        return {"error": f"Connection failed: {e}"}


def database_info(sidecar_url):
    """Get database info from sidecar."""
    try:
        response = requests.get(f"{sidecar_url}/database/info", timeout=10)
        if response.ok:
            return response.json()
        return {"error": f"Failed to get database info: HTTP {response.status_code}"}
    except requests.RequestException as e:
        return {"error": f"Request failed: {e}"}


def identify_scene(sidecar_url, scene_id, num_frames=None, top_k=None, max_distance=None,
                    min_face_size=None, scene_performer_stashdb_ids=None):
    """Identify performers in a scene. Params default to sidecar's face_config when omitted."""
    if not scene_id:
        return {"error": "No scene_id provided"}

    try:
        payload = {"scene_id": str(scene_id)}
        if num_frames is not None:
            payload["num_frames"] = int(num_frames)
        if top_k is not None:
            payload["top_k"] = int(top_k)
        if max_distance is not None:
            payload["max_distance"] = float(max_distance)
        if min_face_size is not None:
            payload["min_face_size"] = int(min_face_size)
        if scene_performer_stashdb_ids:
            payload["scene_performer_stashdb_ids"] = scene_performer_stashdb_ids

        log(f"Identifying scene {scene_id}")

        response = requests.post(
            f"{sidecar_url}/identify/scene",
            json=payload,
            timeout=180,  # ffmpeg extraction can take a while
        )

        if response.ok:
            result = response.json()
            log(f"Scene {scene_id}: {result.get('faces_detected', 0)} faces, {len(result.get('persons', []))} persons")
            return result

        # Try to extract error detail from response
        try:
            error_detail = response.json().get("detail", response.text)
        except Exception:
            error_detail = response.text or f"HTTP {response.status_code}"

        return {"error": f"Identification failed: {error_detail}"}

    except requests.ConnectionError:
        return {"error": "Connection refused - is Stash Sense running?"}
    except requests.Timeout:
        return {"error": "Request timed out - scene may be too long or sidecar is overloaded"}
    except requests.RequestException as e:
        return {"error": f"Request failed: {e}"}


def identify_image(sidecar_url, image_id):
    """Identify performers in a single image."""
    if not image_id:
        return {"error": "No image_id provided"}

    try:
        payload = {"image_id": str(image_id)}
        log(f"Identifying image {image_id}")

        response = requests.post(
            f"{sidecar_url}/identify/image",
            json=payload,
            timeout=30,
        )

        if response.ok:
            result = response.json()
            log(f"Image {image_id}: {result.get('face_count', 0)} faces")
            return result

        try:
            error_detail = response.json().get("detail", response.text)
        except Exception:
            error_detail = response.text or f"HTTP {response.status_code}"
        return {"error": f"Identification failed: {error_detail}"}

    except requests.ConnectionError:
        return {"error": "Connection refused - is Stash Sense running?"}
    except requests.Timeout:
        return {"error": "Request timed out"}
    except requests.RequestException as e:
        return {"error": f"Request failed: {e}"}


def identify_gallery(sidecar_url, gallery_id):
    """Identify performers across an entire gallery."""
    if not gallery_id:
        return {"error": "No gallery_id provided"}

    try:
        payload = {"gallery_id": str(gallery_id)}
        log(f"Identifying gallery {gallery_id}")

        response = requests.post(
            f"{sidecar_url}/identify/gallery",
            json=payload,
            timeout=300,
        )

        if response.ok:
            result = response.json()
            log(f"Gallery {gallery_id}: {result.get('images_processed', 0)} images, "
                f"{len(result.get('performers', []))} performers")
            return result

        try:
            error_detail = response.json().get("detail", response.text)
        except Exception:
            error_detail = response.text or f"HTTP {response.status_code}"
        return {"error": f"Identification failed: {error_detail}"}

    except requests.ConnectionError:
        return {"error": "Connection refused - is Stash Sense running?"}
    except requests.Timeout:
        return {"error": "Gallery identification timed out - gallery may be too large"}
    except requests.RequestException as e:
        return {"error": f"Request failed: {e}"}


# ==================== Recommendations API Proxy ====================

def sidecar_get(sidecar_url, endpoint, timeout=30):
    """GET request to sidecar."""
    try:
        response = requests.get(f"{sidecar_url}{endpoint}", timeout=timeout)
        if response.ok:
            return response.json()
        try:
            error_detail = response.json().get("detail", response.text)
        except Exception:
            error_detail = response.text or f"HTTP {response.status_code}"
        return {"error": error_detail}
    except requests.ConnectionError:
        return {"error": "Connection refused - is Stash Sense running?"}
    except requests.Timeout:
        return {"error": "Request timed out"}
    except requests.RequestException as e:
        return {"error": f"Request failed: {e}"}


def sidecar_post(sidecar_url, endpoint, data=None, timeout=60):
    """POST request to sidecar."""
    try:
        response = requests.post(
            f"{sidecar_url}{endpoint}",
            json=data,
            timeout=timeout,
        )
        if response.ok:
            return response.json()
        try:
            error_detail = response.json().get("detail", response.text)
        except Exception:
            error_detail = response.text or f"HTTP {response.status_code}"
        return {"error": error_detail}
    except requests.ConnectionError:
        return {"error": "Connection refused - is Stash Sense running?"}
    except requests.Timeout:
        return {"error": "Request timed out"}
    except requests.RequestException as e:
        return {"error": f"Request failed: {e}"}


def sidecar_put(sidecar_url, endpoint, data=None, timeout=30):
    """PUT request to sidecar."""
    try:
        response = requests.put(
            f"{sidecar_url}{endpoint}",
            json=data,
            timeout=timeout,
        )
        if response.ok:
            return response.json()
        try:
            error_detail = response.json().get("detail", response.text)
        except Exception:
            error_detail = response.text or f"HTTP {response.status_code}"
        return {"error": error_detail}
    except requests.ConnectionError:
        return {"error": "Connection refused - is Stash Sense running?"}
    except requests.Timeout:
        return {"error": "Request timed out"}
    except requests.RequestException as e:
        return {"error": f"Request failed: {e}"}


def sidecar_delete(sidecar_url, endpoint, timeout=30):
    """DELETE request to sidecar."""
    try:
        response = requests.delete(f"{sidecar_url}{endpoint}", timeout=timeout)
        if response.ok:
            return response.json()
        try:
            error_detail = response.json().get("detail", response.text)
        except Exception:
            error_detail = response.text or f"HTTP {response.status_code}"
        return {"error": error_detail}
    except requests.ConnectionError:
        return {"error": "Connection refused - is Stash Sense running?"}
    except requests.Timeout:
        return {"error": "Request timed out"}
    except requests.RequestException as e:
        return {"error": f"Request failed: {e}"}


# ==================== Queue API Proxy ====================

def handle_queue(mode, args, sidecar_url):
    """Handle queue-related proxy operations."""
    if mode == "queue_list":
        qs = ""
        if args.get("status"):
            qs += f"?status={args['status']}"
        return sidecar_get(sidecar_url, f"/queue{qs}")
    elif mode == "queue_status":
        return sidecar_get(sidecar_url, "/queue/status")
    elif mode == "queue_types":
        return sidecar_get(sidecar_url, "/queue/types")
    elif mode == "queue_submit":
        return sidecar_post(sidecar_url, "/queue", {
            "type": args["type"],
            "triggered_by": args.get("triggered_by", "user"),
        })
    elif mode == "queue_cancel":
        return sidecar_delete(sidecar_url, f"/queue/{args['job_id']}")
    elif mode == "queue_stop":
        return sidecar_post(sidecar_url, f"/queue/{args['job_id']}/stop")
    elif mode == "queue_retry":
        return sidecar_post(sidecar_url, f"/queue/{args['job_id']}/retry")
    elif mode == "queue_clear_history":
        return sidecar_delete(sidecar_url, "/queue/history")
    elif mode == "queue_schedules":
        return sidecar_get(sidecar_url, "/queue/schedules")
    elif mode == "queue_update_schedule":
        return sidecar_put(sidecar_url, f"/queue/schedules/{args['type']}", {
            "enabled": args["enabled"],
            "interval_hours": args["interval_hours"],
        })
    return {"error": f"Unknown queue mode: {mode}"}


def rec_counts(sidecar_url):
    """Get recommendation counts."""
    return sidecar_get(sidecar_url, "/recommendations/counts")


def rec_list(sidecar_url, status=None, rec_type=None, limit=100, offset=0):
    """List recommendations."""
    params = []
    if status:
        params.append(f"status={status}")
    if rec_type:
        params.append(f"type={rec_type}")
    params.append(f"limit={limit}")
    params.append(f"offset={offset}")
    query = "?" + "&".join(params) if params else ""
    return sidecar_get(sidecar_url, f"/recommendations{query}")


def rec_get(sidecar_url, rec_id):
    """Get single recommendation."""
    return sidecar_get(sidecar_url, f"/recommendations/{rec_id}")


def rec_resolve(sidecar_url, rec_id, action, details=None):
    """Resolve a recommendation."""
    data = {"action": action}
    if details:
        data["details"] = details
    return sidecar_post(sidecar_url, f"/recommendations/{rec_id}/resolve", data)


def rec_dismiss(sidecar_url, rec_id, reason=None):
    """Dismiss a recommendation."""
    data = {"reason": reason} if reason else {}
    return sidecar_post(sidecar_url, f"/recommendations/{rec_id}/dismiss", data)


def rec_analysis_types(sidecar_url):
    """Get analysis types."""
    return sidecar_get(sidecar_url, "/recommendations/analysis/types")


def rec_run_analysis(sidecar_url, analysis_type):
    """Run an analysis."""
    return sidecar_post(sidecar_url, f"/recommendations/analysis/{analysis_type}/run")


def rec_analysis_runs(sidecar_url, analysis_type=None, limit=20):
    """Get recent analysis runs."""
    params = [f"limit={limit}"]
    if analysis_type:
        params.append(f"type={analysis_type}")
    query = "?" + "&".join(params)
    return sidecar_get(sidecar_url, f"/recommendations/analysis/runs{query}")


def rec_stash_status(sidecar_url):
    """Get Stash connection status."""
    return sidecar_get(sidecar_url, "/recommendations/stash/status")


def rec_merge_performers(sidecar_url, destination_id, source_ids):
    """Execute performer merge."""
    data = {
        "destination_id": destination_id,
        "source_ids": source_ids,
    }
    return sidecar_post(sidecar_url, "/recommendations/actions/merge-performers", data, timeout=120)


def rec_delete_files(sidecar_url, scene_id, file_ids_to_delete, keep_file_id, all_file_ids):
    """Delete scene files."""
    data = {
        "scene_id": scene_id,
        "file_ids_to_delete": file_ids_to_delete,
        "keep_file_id": keep_file_id,
        "all_file_ids": all_file_ids,
    }
    return sidecar_post(sidecar_url, "/recommendations/actions/delete-scene-files", data, timeout=120)


# ==================== Fingerprint Operations ====================

def fp_status(sidecar_url):
    """Get fingerprint status and coverage."""
    return sidecar_get(sidecar_url, "/recommendations/fingerprints/status")


def fp_generate(sidecar_url, refresh_outdated=True, num_frames=None, min_face_size=None, max_distance=None):
    """Start fingerprint generation. Params default to sidecar's face_config when omitted."""
    data = {"refresh_outdated": refresh_outdated}
    if num_frames is not None:
        data["num_frames"] = int(num_frames)
    if min_face_size is not None:
        data["min_face_size"] = int(min_face_size)
    if max_distance is not None:
        data["max_distance"] = float(max_distance)
    return sidecar_post(sidecar_url, "/recommendations/fingerprints/generate", data)


def fp_progress(sidecar_url):
    """Get fingerprint generation progress."""
    return sidecar_get(sidecar_url, "/recommendations/fingerprints/progress")


def fp_stop(sidecar_url):
    """Stop fingerprint generation."""
    return sidecar_post(sidecar_url, "/recommendations/fingerprints/stop", {})


def handle_recommendations(mode, args, sidecar_url):
    """Handle recommendations-related operations."""
    if mode == "rec_counts":
        return rec_counts(sidecar_url)

    elif mode == "rec_list":
        return rec_list(
            sidecar_url,
            status=args.get("status"),
            rec_type=args.get("type"),
            limit=int(args.get("limit", 100)),
            offset=int(args.get("offset", 0)),
        )

    elif mode == "rec_get":
        rec_id = args.get("rec_id")
        if not rec_id:
            return {"error": "No rec_id provided"}
        return rec_get(sidecar_url, rec_id)

    elif mode == "rec_resolve":
        rec_id = args.get("rec_id")
        action = args.get("action")
        if not rec_id or not action:
            return {"error": "rec_id and action required"}
        return rec_resolve(sidecar_url, rec_id, action, args.get("details"))

    elif mode == "rec_dismiss":
        rec_id = args.get("rec_id")
        if not rec_id:
            return {"error": "No rec_id provided"}
        return rec_dismiss(sidecar_url, rec_id, args.get("reason"))

    elif mode == "rec_analysis_types":
        return rec_analysis_types(sidecar_url)

    elif mode == "rec_run_analysis":
        analysis_type = args.get("analysis_type")
        if not analysis_type:
            return {"error": "No analysis_type provided"}
        return rec_run_analysis(sidecar_url, analysis_type)

    elif mode == "rec_analysis_runs":
        return rec_analysis_runs(
            sidecar_url,
            analysis_type=args.get("analysis_type"),
            limit=int(args.get("limit", 20)),
        )

    elif mode == "rec_stash_status":
        return rec_stash_status(sidecar_url)

    elif mode == "rec_merge_performers":
        destination_id = args.get("destination_id")
        source_ids = args.get("source_ids", [])
        if not destination_id or not source_ids:
            return {"error": "destination_id and source_ids required"}
        return rec_merge_performers(sidecar_url, destination_id, source_ids)

    elif mode == "rec_delete_files":
        scene_id = args.get("scene_id")
        file_ids_to_delete = args.get("file_ids_to_delete", [])
        keep_file_id = args.get("keep_file_id")
        all_file_ids = args.get("all_file_ids", [])
        if not scene_id or not keep_file_id:
            return {"error": "scene_id and keep_file_id required"}
        return rec_delete_files(sidecar_url, scene_id, file_ids_to_delete, keep_file_id, all_file_ids)

    elif mode == "rec_update_performer":
        performer_id = args.get("performer_id")
        fields = args.get("fields", {})
        if not performer_id:
            return {"error": "performer_id required"}
        return sidecar_post(
            sidecar_url,
            "/recommendations/actions/update-performer",
            {"performer_id": performer_id, "fields": fields},
            timeout=60,
        )

    elif mode == "rec_update_tag":
        tag_id = args.get("tag_id")
        fields = args.get("fields", {})
        if not tag_id:
            return {"error": "tag_id required"}
        return sidecar_post(
            sidecar_url,
            "/recommendations/actions/update-tag",
            {"tag_id": tag_id, "fields": fields},
            timeout=60,
        )

    elif mode == "rec_dismiss_upstream":
        rec_id = args.get("rec_id")
        if not rec_id:
            return {"error": "No rec_id provided"}
        return sidecar_post(
            sidecar_url,
            f"/recommendations/{rec_id}/dismiss-upstream",
            {"reason": args.get("reason"), "permanent": args.get("permanent", False)},
        )

    elif mode == "rec_get_field_config":
        import base64
        endpoint = args.get("endpoint", "")
        endpoint_b64 = base64.b64encode(endpoint.encode()).decode()
        return sidecar_get(sidecar_url, f"/recommendations/upstream/field-config/{endpoint_b64}")

    elif mode == "rec_set_field_config":
        import base64
        endpoint = args.get("endpoint", "")
        endpoint_b64 = base64.b64encode(endpoint.encode()).decode()
        return sidecar_post(
            sidecar_url,
            f"/recommendations/upstream/field-config/{endpoint_b64}",
            args.get("field_configs", {}),
        )

    elif mode == "user_get_all_settings":
        return sidecar_get(sidecar_url, "/recommendations/settings")

    elif mode == "user_get_setting":
        key = args.get("key", "")
        return sidecar_get(sidecar_url, f"/recommendations/settings/{key}")

    elif mode == "user_set_setting":
        key = args.get("key", "")
        value = args.get("value")
        return sidecar_post(
            sidecar_url,
            f"/recommendations/settings/{key}",
            {"value": value},
        )

    # Fingerprint operations
    elif mode == "fp_status":
        return fp_status(sidecar_url)

    elif mode == "fp_generate":
        return fp_generate(
            sidecar_url,
            refresh_outdated=args.get("refresh_outdated", True),
            num_frames=args.get("num_frames"),
            min_face_size=args.get("min_face_size"),
            max_distance=args.get("max_distance"),
        )

    elif mode == "fp_progress":
        return fp_progress(sidecar_url)

    elif mode == "fp_stop":
        return fp_stop(sidecar_url)

    return None


# ==================== Settings API Proxy ====================

def handle_settings(mode, args, sidecar_url):
    """Handle settings-related operations."""
    if mode == "settings_get_all":
        return sidecar_get(sidecar_url, "/settings")

    elif mode == "settings_get":
        key = args.get("key", "")
        return sidecar_get(sidecar_url, f"/settings/{key}")

    elif mode == "settings_update":
        key = args.get("key", "")
        value = args.get("value")
        return sidecar_put(sidecar_url, f"/settings/{key}", {"value": value})

    elif mode == "settings_update_bulk":
        settings = args.get("settings", {})
        return sidecar_put(sidecar_url, "/settings", {"settings": settings})

    elif mode == "settings_reset":
        key = args.get("key", "")
        return sidecar_delete(sidecar_url, f"/settings/{key}")

    elif mode == "system_info":
        return sidecar_get(sidecar_url, "/system/info")

    return None


if __name__ == "__main__":
    main()
