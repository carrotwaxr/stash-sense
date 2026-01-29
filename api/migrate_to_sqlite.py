#!/usr/bin/env python3
"""
Migration Script: JSON to SQLite

Migrates existing JSON-based performer data to the new SQLite database.
Preserves all existing data while adding support for URLs, aliases, etc.

Usage:
    # Migrate from a data directory
    python migrate_to_sqlite.py --input ./data --output ./data/performers.db

    # Migrate from snapshot
    python migrate_to_sqlite.py --input ./data-snapshot-20260127-1653 --output ./data-snapshot-20260127-1653/performers.db

    # Include URL data if available
    python migrate_to_sqlite.py --input ./data --output ./data/performers.db --urls ./data/urls.json
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

from database import PerformerDatabase


def load_json(path: Path) -> dict:
    """Load a JSON file."""
    with open(path) as f:
        return json.load(f)


def migrate_performers(
    db: PerformerDatabase,
    performers_json: dict,
    faces_json: dict,
    urls_json: dict | None = None,
    verbose: bool = True,
) -> dict:
    """
    Migrate performers from JSON to SQLite.

    Args:
        db: Target database
        performers_json: Contents of performers.json
        faces_json: Contents of faces.json
        urls_json: Optional contents of urls.json (from url_fetcher.py)
        verbose: Print progress

    Returns:
        Migration statistics
    """
    stats = {
        'performers_migrated': 0,
        'faces_migrated': 0,
        'urls_migrated': 0,
        'aliases_migrated': 0,
        'errors': 0,
    }

    # Build face index -> performer mapping from faces.json
    # faces.json format: ["stashdb.org:uuid", "stashdb.org:uuid", ...] (list, index = face index)
    face_to_performer = {}
    for face_idx, performer_id in enumerate(faces_json):
        face_to_performer[face_idx] = performer_id

    # Build URL data lookup if available
    url_lookup = {}
    if urls_json:
        for uid, url_data in urls_json.items():
            url_lookup[uid] = url_data

    total = len(performers_json)
    if verbose:
        print(f"Migrating {total} performers...")

    for i, (universal_id, performer_data) in enumerate(performers_json.items()):
        if verbose and i % 1000 == 0:
            print(f"  Progress: {i}/{total} ({i/total*100:.1f}%)")

        try:
            # Parse universal ID (format: "stashdb.org:uuid")
            parts = universal_id.split(":", 1)
            if len(parts) != 2:
                print(f"  Warning: Invalid universal_id format: {universal_id}")
                stats['errors'] += 1
                continue

            endpoint_domain, stashbox_id = parts
            # Convert domain to endpoint name
            endpoint_map = {
                "stashdb.org": "stashdb",
                "theporndb.net": "theporndb",
                "pmvstash.org": "pmvstash",
                "javstash.org": "javstash",
                "fansdb.cc": "fansdb",
            }
            endpoint = endpoint_map.get(endpoint_domain, endpoint_domain)

            # Get additional data from URL fetch if available
            url_data = url_lookup.get(universal_id, {})

            # Create performer
            performer_id = db.add_performer(
                canonical_name=performer_data.get('name', 'Unknown'),
                gender=url_data.get('gender'),
                country=performer_data.get('country'),
                birth_date=url_data.get('birth_date'),
                career_start_year=url_data.get('career_start_year'),
                career_end_year=url_data.get('career_end_year'),
                face_count=performer_data.get('face_count', 0),
                image_url=performer_data.get('image_url'),
            )

            # Add stash-box ID
            db.add_stashbox_id(performer_id, endpoint, stashbox_id)

            # Add merged IDs if available
            for merged_id in url_data.get('merged_ids', []):
                db.add_merged_id(performer_id, merged_id, endpoint)

            # Add URLs from url_data
            if url_data.get('urls'):
                for site_name, urls in url_data['urls'].items():
                    for url in urls:
                        if db.add_url(performer_id, url, endpoint):
                            stats['urls_migrated'] += 1

            # Add aliases
            for alias in url_data.get('aliases', []):
                if db.add_alias(performer_id, alias, endpoint):
                    stats['aliases_migrated'] += 1

            stats['performers_migrated'] += 1

        except Exception as e:
            print(f"  Error migrating {universal_id}: {e}")
            stats['errors'] += 1

    # Now migrate faces
    if verbose:
        print(f"Migrating {len(faces_json)} faces...")

    for face_idx, universal_id in enumerate(faces_json):
        if not universal_id:
            continue

        # Find the performer in the database
        parts = universal_id.split(":", 1)
        if len(parts) != 2:
            continue

        endpoint_domain, stashbox_id = parts
        endpoint_map = {
            "stashdb.org": "stashdb",
            "theporndb.net": "theporndb",
            "pmvstash.org": "pmvstash",
            "javstash.org": "javstash",
            "fansdb.cc": "fansdb",
        }
        endpoint = endpoint_map.get(endpoint_domain, endpoint_domain)

        performer = db.get_performer_by_stashbox_id(endpoint, stashbox_id)
        if performer:
            try:
                # Get image_url from performers.json
                performer_data = performers_json.get(universal_id, {})
                image_url = performer_data.get('image_url', '')

                db.add_face(
                    performer_id=performer.id,
                    facenet_index=face_idx,
                    arcface_index=face_idx,  # Same index for both in current implementation
                    image_url=image_url,
                    source_endpoint=endpoint,
                )
                stats['faces_migrated'] += 1
            except Exception as e:
                # Duplicate or other error
                pass

    return stats


def main():
    parser = argparse.ArgumentParser(description="Migrate JSON data to SQLite")
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input directory containing performers.json and faces.json",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output SQLite database path",
    )
    parser.add_argument(
        "--urls",
        help="Optional urls.json file (from url_fetcher.py)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output)

    # Load input files
    performers_path = input_dir / "performers.json"
    faces_path = input_dir / "faces.json"

    if not performers_path.exists():
        print(f"Error: {performers_path} not found")
        return 1

    if not faces_path.exists():
        print(f"Error: {faces_path} not found")
        return 1

    print(f"Loading performers from {performers_path}...")
    performers_json = load_json(performers_path)

    print(f"Loading faces from {faces_path}...")
    faces_json = load_json(faces_path)

    urls_json = None
    if args.urls:
        urls_path = Path(args.urls)
        if urls_path.exists():
            print(f"Loading URLs from {urls_path}...")
            urls_json = load_json(urls_path)
        else:
            print(f"Warning: URLs file not found: {urls_path}")

    # Create or overwrite database
    if output_path.exists():
        print(f"Removing existing database: {output_path}")
        output_path.unlink()

    print(f"Creating database: {output_path}")
    db = PerformerDatabase(output_path)

    # Run migration
    print("Starting migration...")
    start_time = datetime.now()

    stats = migrate_performers(
        db,
        performers_json,
        faces_json,
        urls_json,
        verbose=not args.quiet,
    )

    elapsed = (datetime.now() - start_time).total_seconds()

    # Print results
    print("\n" + "=" * 60)
    print("Migration Complete!")
    print("=" * 60)
    print(f"Performers migrated: {stats['performers_migrated']}")
    print(f"Faces migrated: {stats['faces_migrated']}")
    print(f"URLs migrated: {stats['urls_migrated']}")
    print(f"Aliases migrated: {stats['aliases_migrated']}")
    print(f"Errors: {stats['errors']}")
    print(f"Time: {elapsed:.1f}s")

    # Print database stats
    print("\nDatabase Statistics:")
    db_stats = db.get_stats()
    print(f"  Total performers: {db_stats['performer_count']}")
    print(f"  With faces: {db_stats['performers_with_faces']}")
    print(f"  Total faces: {db_stats['total_faces']}")
    print(f"  Total URLs: {db_stats['total_urls']}")
    print(f"  Total aliases: {db_stats['total_aliases']}")

    if db_stats['url_site_counts']:
        print("\n  URLs by site:")
        for site, count in list(db_stats['url_site_counts'].items())[:10]:
            print(f"    {site}: {count}")

    print(f"\nDatabase saved to: {output_path}")
    print(f"Database size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    return 0


if __name__ == "__main__":
    exit(main())
