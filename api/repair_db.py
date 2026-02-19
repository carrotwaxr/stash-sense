#!/usr/bin/env python3
"""Repair database integrity issues.

Fixes:
1. Performers with faces but no stashbox_ids - extracts from external_urls
2. Performers with incorrect face_count - recalculates from faces table

Usage:
    python repair_db.py [--data-dir PATH] [--dry-run]
"""
import argparse
import re
import sqlite3
from pathlib import Path


# Pattern to extract stashdb performer ID from URL
STASHDB_URL_PATTERN = re.compile(r'https://stashdb\.org/performers/([a-f0-9-]+)')
FANSDB_URL_PATTERN = re.compile(r'https://fansdb\.cc/performers/([a-f0-9-]+)')


def find_orphan_performers(conn: sqlite3.Connection) -> list[dict]:
    """Find performers with faces but no stashbox_ids."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT p.id, p.canonical_name,
               (SELECT COUNT(*) FROM faces f WHERE f.performer_id = p.id) as actual_faces,
               (SELECT GROUP_CONCAT(url, '|') FROM external_urls WHERE performer_id = p.id) as urls
        FROM performers p
        WHERE EXISTS (SELECT 1 FROM faces f WHERE f.performer_id = p.id)
        AND NOT EXISTS (SELECT 1 FROM stashbox_ids s WHERE s.performer_id = p.id)
    """)

    orphans = []
    for row in cursor:
        orphans.append({
            'id': row[0],
            'name': row[1],
            'face_count': row[2],
            'urls': row[3].split('|') if row[3] else []
        })
    return orphans


def extract_stashbox_id(urls: list[str]) -> tuple[str, str] | None:
    """Extract stashbox ID from external URLs.

    Returns (endpoint, stashbox_id) tuple or None.
    Prefers stashdb over fansdb.
    """
    for url in urls:
        match = STASHDB_URL_PATTERN.match(url)
        if match:
            return ('stashdb', match.group(1))

    for url in urls:
        match = FANSDB_URL_PATTERN.match(url)
        if match:
            return ('fansdb', match.group(1))

    return None


def fix_missing_stashbox_ids(conn: sqlite3.Connection, dry_run: bool = False) -> int:
    """Add missing stashbox_ids from external_urls, or merge faces to canonical performer.

    When a stashbox_id already exists for another performer, we merge the faces
    to that canonical performer instead.

    Returns number of fixes applied.
    """
    orphans = find_orphan_performers(conn)

    if not orphans:
        print("  No orphan performers found")
        return 0

    print(f"  Found {len(orphans)} performers with faces but no stashbox_ids")

    fixed = 0
    merged = 0
    unfixable = []

    cursor = conn.cursor()
    for orphan in orphans:
        stashbox_info = extract_stashbox_id(orphan['urls'])

        if stashbox_info:
            endpoint, stashbox_id = stashbox_info

            # Check if this stashbox_id already exists
            cursor.execute(
                "SELECT performer_id FROM stashbox_ids WHERE endpoint = ? AND stashbox_performer_id = ?",
                (endpoint, stashbox_id)
            )
            existing = cursor.fetchone()

            if existing:
                canonical_id = existing[0]
                print(f"    {orphan['name']} (id={orphan['id']}): merging {orphan['face_count']} faces to performer {canonical_id}")

                if not dry_run:
                    # Move faces to canonical performer
                    cursor.execute(
                        "UPDATE faces SET performer_id = ? WHERE performer_id = ?",
                        (canonical_id, orphan['id'])
                    )
                    # Delete the duplicate performer (cascades to aliases, external_urls)
                    cursor.execute("DELETE FROM performers WHERE id = ?", (orphan['id'],))
                    merged += 1
            else:
                print(f"    {orphan['name']}: adding {endpoint}:{stashbox_id}")

                if not dry_run:
                    cursor.execute(
                        "INSERT INTO stashbox_ids (performer_id, endpoint, stashbox_performer_id) VALUES (?, ?, ?)",
                        (orphan['id'], endpoint, stashbox_id)
                    )
                    fixed += 1
        else:
            unfixable.append(orphan)

    if unfixable:
        print(f"\n  WARNING: {len(unfixable)} performers cannot be auto-fixed (no stashdb/fansdb URL):")
        for orphan in unfixable:
            print(f"    - {orphan['name']} (id={orphan['id']}, {orphan['face_count']} faces)")
        print("  These faces will map to null in faces.json (recognizer will skip them)")

    if not dry_run:
        conn.commit()

    print(f"\n  Added {fixed} stashbox_ids, merged {merged} duplicate performers")
    return fixed + merged


def fix_face_counts(conn: sqlite3.Connection, dry_run: bool = False) -> int:
    """Update face_count to match actual faces in faces table.

    Returns number of fixes applied.
    """
    cursor = conn.cursor()

    # Find mismatches
    cursor.execute("""
        SELECT p.id, p.canonical_name, p.face_count as stored,
               (SELECT COUNT(*) FROM faces f WHERE f.performer_id = p.id) as actual
        FROM performers p
        WHERE p.face_count != (SELECT COUNT(*) FROM faces f WHERE f.performer_id = p.id)
    """)

    mismatches = cursor.fetchall()

    if not mismatches:
        print("  No face_count mismatches found")
        return 0

    print(f"  Found {len(mismatches)} performers with incorrect face_count")

    for performer_id, name, stored, actual in mismatches[:10]:
        print(f"    {name}: {stored} -> {actual}")

    if len(mismatches) > 10:
        print(f"    ... and {len(mismatches) - 10} more")

    if not dry_run:
        cursor.execute("""
            UPDATE performers
            SET face_count = (SELECT COUNT(*) FROM faces f WHERE f.performer_id = performers.id)
            WHERE face_count != (SELECT COUNT(*) FROM faces f WHERE f.performer_id = performers.id)
        """)
        conn.commit()

    return len(mismatches)


def analyze_index_gaps(conn: sqlite3.Connection):
    """Analyze gaps in face indices."""
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            MIN(facenet_index) as min_idx,
            MAX(facenet_index) as max_idx,
            COUNT(*) as actual_faces
        FROM faces
    """)

    row = cursor.fetchone()
    min_idx, max_idx, actual = row
    expected = max_idx - min_idx + 1
    gaps = expected - actual

    print(f"\n  Index range: {min_idx} to {max_idx}")
    print(f"  Actual faces: {actual}")
    print(f"  Expected if contiguous: {expected}")
    print(f"  Gaps: {gaps} ({gaps/expected*100:.1f}%)")

    if gaps > 0:
        # Find gap locations
        cursor.execute("""
            WITH indexed_faces AS (
                SELECT facenet_index, LAG(facenet_index) OVER (ORDER BY facenet_index) as prev_idx
                FROM faces
                ORDER BY facenet_index
            )
            SELECT facenet_index as after_idx, prev_idx as before_idx, facenet_index - prev_idx - 1 as gap_size
            FROM indexed_faces
            WHERE facenet_index - prev_idx > 1
            ORDER BY gap_size DESC
            LIMIT 5
        """)

        print("\n  Largest gaps:")
        for after_idx, before_idx, gap_size in cursor:
            print(f"    Gap of {gap_size} indices between {before_idx} and {after_idx}")


def main():
    parser = argparse.ArgumentParser(description="Repair database integrity issues")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent / "data",
        help="Data directory containing performers.db"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be fixed without making changes"
    )
    args = parser.parse_args()

    db_path = args.data_dir / "performers.db"

    if not db_path.exists():
        print(f"ERROR: Database not found at {db_path}")
        return 1

    print(f"{'[DRY RUN] ' if args.dry_run else ''}Repairing {db_path}...")

    conn = sqlite3.connect(db_path)

    try:
        print("\n1. Fixing missing stashbox_ids...")
        stashbox_fixes = fix_missing_stashbox_ids(conn, args.dry_run)

        print("\n2. Fixing face_count mismatches...")
        count_fixes = fix_face_counts(conn, args.dry_run)

        print("\n3. Analyzing index gaps...")
        analyze_index_gaps(conn)

        print(f"\n{'[DRY RUN] Would have made' if args.dry_run else 'Made'} {stashbox_fixes + count_fixes} fixes")

        if not args.dry_run and (stashbox_fixes > 0 or count_fixes > 0):
            print("\nRe-run export_db_to_json.py to regenerate JSON files.")

    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    exit(main())
