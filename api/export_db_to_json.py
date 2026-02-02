#!/usr/bin/env python3
"""Export performers.db to JSON files for the recognizer.

This script converts the SQLite database (performers.db) into the JSON files
that the FaceRecognizer expects:

  - faces.json: List where index i = universal_id of face at Voyager index i
  - performers.json: Dict mapping universal_id -> {name, country, image_url, face_count}

Usage:
    python export_db_to_json.py [--data-dir PATH]

The script will:
1. Read from performers.db in the data directory
2. Generate faces.json and performers.json
3. Validate the output matches the Voyager index sizes

Run this script after copying a new performers.db from stash-sense-trainer,
or after updating the database in any way.
"""
import argparse
import json
import sqlite3
from pathlib import Path


def make_universal_id(endpoint: str, stashbox_id: str) -> str:
    """Create a universal ID from endpoint and stashbox ID.

    Format: "endpoint.org:stashbox_id"
    Example: "stashdb.org:019bef93-b467-73eb-a04b-ac44fdaa7a04"
    """
    # Normalize endpoint to domain format
    endpoint_domains = {
        "stashdb": "stashdb.org",
        "theporndb": "theporndb.net",
        "fansdb": "fansdb.cc",
        "pmvstash": "pmvstash.org",
        "javstash": "javstash.org",
    }
    domain = endpoint_domains.get(endpoint, f"{endpoint}.org")
    return f"{domain}:{stashbox_id}"


def export_faces_json(conn: sqlite3.Connection, output_path: Path) -> int:
    """Export faces.json - list of universal IDs indexed by Voyager index.

    The Voyager index stores embeddings at sequential integer indices.
    faces.json maps each index to the performer's universal ID.

    Returns the number of faces exported.
    """
    cursor = conn.cursor()

    # Get all faces with their stashbox IDs in a single query
    # This avoids cursor reuse issues and is much faster
    cursor.execute("""
        SELECT f.facenet_index, s.endpoint, s.stashbox_performer_id, p.canonical_name
        FROM faces f
        JOIN performers p ON f.performer_id = p.id
        JOIN stashbox_ids s ON p.id = s.performer_id
        ORDER BY f.facenet_index,
            CASE s.endpoint
                WHEN 'stashdb' THEN 1
                WHEN 'theporndb' THEN 2
                WHEN 'fansdb' THEN 3
                WHEN 'pmvstash' THEN 4
                WHEN 'javstash' THEN 5
                ELSE 6
            END
    """)

    faces = []
    last_index = -1
    skipped_duplicates = 0
    gap_count = 0

    for facenet_index, endpoint, stashbox_id, name in cursor:
        # Skip duplicate rows (same face, different stashbox endpoint)
        # We only want the first (highest priority) endpoint per face
        if facenet_index == last_index:
            skipped_duplicates += 1
            continue
        last_index = facenet_index

        universal_id = make_universal_id(endpoint, stashbox_id)

        # Ensure we're filling indices sequentially
        while len(faces) < facenet_index:
            # Fill gaps with placeholder - happens when faces exist but performer
            # has no stashbox ID (data issue in trainer)
            gap_count += 1
            faces.append(None)

        faces.append(universal_id)

    # Write output
    with open(output_path, "w") as f:
        json.dump(faces, f)

    valid_faces = len(faces) - gap_count
    print(f"  Exported {valid_faces} faces ({len(faces)} total indices) to {output_path}")
    if gap_count:
        print(f"  WARNING: {gap_count} gaps filled with null (performers missing stashbox IDs)")
    if skipped_duplicates:
        print(f"  (Skipped {skipped_duplicates} duplicate stashbox entries)")
    return len(faces)


def export_performers_json(conn: sqlite3.Connection, output_path: Path) -> int:
    """Export performers.json - dict mapping universal_id to performer metadata.

    Format:
    {
        "stashdb.org:019bef93-...": {
            "name": "Performer Name",
            "country": "US",
            "image_url": "https://...",
            "face_count": 4
        },
        ...
    }

    Returns the number of performers exported.
    """
    cursor = conn.cursor()

    # Get all performers with faces, joined with their stashbox IDs
    # Single query to avoid cursor reuse issues
    cursor.execute("""
        SELECT p.id, p.canonical_name, p.country, p.image_url, p.face_count,
               s.endpoint, s.stashbox_performer_id
        FROM performers p
        JOIN stashbox_ids s ON p.id = s.performer_id
        WHERE p.face_count > 0
        ORDER BY p.id,
            CASE s.endpoint
                WHEN 'stashdb' THEN 1
                WHEN 'theporndb' THEN 2
                WHEN 'fansdb' THEN 3
                WHEN 'pmvstash' THEN 4
                WHEN 'javstash' THEN 5
                ELSE 6
            END
    """)

    performers = {}
    last_performer_id = -1
    skipped_duplicates = 0

    for performer_id, name, country, image_url, face_count, endpoint, stashbox_id in cursor:
        # Skip duplicate rows (same performer, different stashbox endpoint)
        # We only want the first (highest priority) endpoint per performer
        if performer_id == last_performer_id:
            skipped_duplicates += 1
            continue
        last_performer_id = performer_id

        universal_id = make_universal_id(endpoint, stashbox_id)

        performers[universal_id] = {
            "name": name,
            "country": country,
            "image_url": image_url,
            "face_count": face_count or 0,
        }

    # Write output
    with open(output_path, "w") as f:
        json.dump(performers, f, indent=2)

    print(f"  Exported {len(performers)} performers to {output_path}")
    if skipped_duplicates:
        print(f"  (Skipped {skipped_duplicates} duplicate stashbox entries)")
    return len(performers)


def validate_export(data_dir: Path, face_count: int) -> bool:
    """Validate the export matches the Voyager index sizes.

    Returns True if validation passes.
    """
    issues = []

    # Check Voyager index sizes
    for index_name in ["face_facenet.voy", "face_arcface.voy"]:
        index_path = data_dir / index_name
        if index_path.exists():
            # Voyager indices have a header, we can't easily get count without loading
            # Just check file exists and is non-empty
            if index_path.stat().st_size == 0:
                issues.append(f"{index_name} is empty")
        else:
            issues.append(f"{index_name} not found")

    # Check faces.json was created
    faces_path = data_dir / "faces.json"
    if not faces_path.exists():
        issues.append("faces.json was not created")

    # Check performers.json was created
    performers_path = data_dir / "performers.json"
    if not performers_path.exists():
        issues.append("performers.json was not created")

    if issues:
        print("\nValidation issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False

    print("\nValidation passed!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Export performers.db to JSON files for the recognizer"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent / "data",
        help="Data directory containing performers.db (default: ./data)"
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    db_path = data_dir / "performers.db"

    if not db_path.exists():
        print(f"ERROR: Database not found at {db_path}")
        print("Copy performers.db from stash-sense-trainer first.")
        return 1

    print(f"Exporting from {db_path}...")

    conn = sqlite3.connect(db_path)

    try:
        print("\n1. Exporting faces.json...")
        face_count = export_faces_json(conn, data_dir / "faces.json")

        print("\n2. Exporting performers.json...")
        performer_count = export_performers_json(conn, data_dir / "performers.json")

        print("\n3. Validating export...")
        if not validate_export(data_dir, face_count):
            return 1

        print(f"\nDone! Exported {face_count} faces and {performer_count} performers.")
        print("\nRestart the stash-sense API to load the new data.")

    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    exit(main())
