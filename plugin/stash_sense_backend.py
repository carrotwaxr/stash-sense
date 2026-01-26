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

    log(f"Mode: {mode}, Args: {list(args.keys())}")

    # Get sidecar URL from args (passed by JS which reads settings)
    sidecar_url = args.get("sidecar_url", "http://localhost:5000").rstrip("/")

    if mode == "health":
        result = health_check(sidecar_url)
    elif mode == "identify_scene":
        scene_id = args.get("scene_id")
        max_frames = int(args.get("max_frames", 20))
        top_k = int(args.get("top_k", 5))
        max_distance = float(args.get("max_distance", 0.5))

        result = identify_scene(
            sidecar_url, scene_id, max_frames, top_k, max_distance
        )
    elif mode == "database_info":
        result = database_info(sidecar_url)
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


def identify_scene(sidecar_url, scene_id, max_frames, top_k, max_distance):
    """Identify performers in a scene."""
    if not scene_id:
        return {"error": "No scene_id provided"}

    try:
        payload = {
            "scene_id": str(scene_id),
            "max_frames": max_frames,
            "top_k": top_k,
            "max_distance": max_distance,
        }

        log(f"Identifying scene {scene_id} with max_distance={max_distance}")

        response = requests.post(
            f"{sidecar_url}/identify/scene",
            json=payload,
            timeout=120,  # Scene analysis can take a while
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


if __name__ == "__main__":
    main()
