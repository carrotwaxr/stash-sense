#!/usr/bin/env python3
"""
Metadata Refresh Tool

Updates performer metadata in the SQLite database from source APIs.
Can refresh URLs, aliases, and other metadata without rebuilding embeddings.

Usage:
    # Refresh all performers (batch mode)
    python metadata_refresh.py --database ./data/performers.db --all

    # Refresh specific performer by StashDB ID
    python metadata_refresh.py --database ./data/performers.db --performer abc12345-...

    # Refresh performers missing URLs
    python metadata_refresh.py --database ./data/performers.db --missing-urls

    # Resume interrupted refresh
    python metadata_refresh.py --database ./data/performers.db --resume
"""

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from database import PerformerDatabase
from stashdb_client import StashDBClient

load_dotenv()


@dataclass
class RefreshStats:
    """Statistics for a refresh operation."""
    total: int = 0
    processed: int = 0
    updated: int = 0
    urls_added: int = 0
    aliases_added: int = 0
    merged_ids_added: int = 0
    errors: int = 0
    skipped: int = 0


class MetadataRefresher:
    """Refreshes performer metadata from source APIs."""

    def __init__(
        self,
        db_path: str,
        progress_path: Optional[str] = None,
        rate_limit: float = 0.25,
    ):
        self.db = PerformerDatabase(db_path)
        self.progress_path = Path(progress_path or str(db_path).replace('.db', '_refresh_progress.json'))
        self.rate_limit = rate_limit

        # Initialize StashDB client
        stashdb_url = os.getenv("STASHDB_URL", "https://stashdb.org/graphql")
        stashdb_key = os.getenv("STASHDB_API_KEY", "")
        if not stashdb_key:
            raise ValueError("STASHDB_API_KEY environment variable required")
        self.client = StashDBClient(stashdb_url, stashdb_key, rate_limit_delay=rate_limit)

        self.stats = RefreshStats()
        self.processed_ids: set[str] = set()

    def load_progress(self) -> bool:
        """Load previous progress if resuming. Returns True if progress was loaded."""
        if self.progress_path.exists():
            print(f"Loading progress from {self.progress_path}")
            with open(self.progress_path) as f:
                progress = json.load(f)
            self.processed_ids = set(progress.get('processed_ids', []))
            self.stats = RefreshStats(**progress.get('stats', {}))
            print(f"Resuming: {len(self.processed_ids)} already processed")
            return True
        return False

    def save_progress(self):
        """Save current progress."""
        progress = {
            'processed_ids': list(self.processed_ids),
            'stats': {
                'total': self.stats.total,
                'processed': self.stats.processed,
                'updated': self.stats.updated,
                'urls_added': self.stats.urls_added,
                'aliases_added': self.stats.aliases_added,
                'merged_ids_added': self.stats.merged_ids_added,
                'errors': self.stats.errors,
                'skipped': self.stats.skipped,
            },
            'saved_at': datetime.now().isoformat(),
        }
        with open(self.progress_path, 'w') as f:
            json.dump(progress, f)

    def clear_progress(self):
        """Clear progress file to start fresh."""
        if self.progress_path.exists():
            self.progress_path.unlink()
            print(f"Cleared progress file: {self.progress_path}")

    def get_performers_to_refresh(
        self,
        missing_urls_only: bool = False,
        performer_id: Optional[str] = None,
    ) -> list[tuple[int, str]]:
        """
        Get list of (performer_id, stashbox_performer_id) tuples to refresh.

        Args:
            missing_urls_only: Only refresh performers without any URLs
            performer_id: Specific StashDB performer ID to refresh

        Returns:
            List of (internal_id, stashbox_performer_id) tuples
        """
        with self.db._connection() as conn:
            cursor = conn.cursor()

            if performer_id:
                # Specific performer
                cursor.execute("""
                    SELECT p.id, s.stashbox_performer_id
                    FROM performers p
                    JOIN stashbox_ids s ON p.id = s.performer_id
                    WHERE s.stashbox_performer_id = ? AND s.endpoint = 'stashdb'
                """, (performer_id,))
            elif missing_urls_only:
                # Performers without any URLs
                cursor.execute("""
                    SELECT p.id, s.stashbox_performer_id
                    FROM performers p
                    JOIN stashbox_ids s ON p.id = s.performer_id
                    LEFT JOIN external_urls u ON p.id = u.performer_id
                    WHERE s.endpoint = 'stashdb' AND u.id IS NULL
                """)
            else:
                # All performers with StashDB IDs
                cursor.execute("""
                    SELECT p.id, s.stashbox_performer_id
                    FROM performers p
                    JOIN stashbox_ids s ON p.id = s.performer_id
                    WHERE s.endpoint = 'stashdb'
                """)

            return cursor.fetchall()

    def refresh_performer(self, internal_id: int, stashbox_performer_id: str) -> bool:
        """
        Refresh metadata for a single performer.

        Returns:
            True if any data was updated
        """
        try:
            performer = self.client.get_performer(stashbox_performer_id)
            if not performer:
                self.stats.errors += 1
                return False

            updated = False

            # Update core fields if changed
            db_performer = self.db.get_performer_by_stashbox_id('stashdb', stashbox_performer_id)
            if db_performer:
                # Update performer record with new metadata including identity fields
                with self.db._connection() as conn:
                    cursor = conn.cursor()

                    cursor.execute("""
                        UPDATE performers SET
                            disambiguation = COALESCE(?, disambiguation),
                            gender = COALESCE(?, gender),
                            country = COALESCE(?, country),
                            ethnicity = COALESCE(?, ethnicity),
                            birth_date = COALESCE(?, birth_date),
                            death_date = COALESCE(?, death_date),
                            height_cm = COALESCE(?, height_cm),
                            eye_color = COALESCE(?, eye_color),
                            hair_color = COALESCE(?, hair_color),
                            career_start_year = COALESCE(?, career_start_year),
                            career_end_year = COALESCE(?, career_end_year),
                            scene_count = COALESCE(?, scene_count),
                            stashdb_updated_at = COALESCE(?, stashdb_updated_at),
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                    """, (
                        performer.disambiguation,
                        performer.gender,
                        performer.country,
                        performer.ethnicity,
                        performer.birth_date,
                        performer.death_date,
                        performer.height_cm,
                        performer.eye_color,
                        performer.hair_color,
                        performer.career_start_year,
                        performer.career_end_year,
                        performer.scene_count,
                        performer.updated,
                        internal_id,
                    ))
                    conn.commit()

                    if cursor.rowcount > 0:
                        updated = True

            # Add URLs
            if performer.urls:
                for site_name, urls in performer.urls.items():
                    for url in urls:
                        if self.db.add_url(internal_id, url, 'stashdb'):
                            self.stats.urls_added += 1
                            updated = True

            # Add aliases
            for alias in (performer.aliases or []):
                if self.db.add_alias(internal_id, alias, 'stashdb'):
                    self.stats.aliases_added += 1
                    updated = True

            # Add merged IDs
            for merged_id in (performer.merged_ids or []):
                if self.db.add_merged_id(internal_id, merged_id, 'stashdb'):
                    self.stats.merged_ids_added += 1
                    updated = True

            # Add tattoos
            for tattoo in (performer.tattoos or []):
                if self.db.add_tattoo(
                    internal_id,
                    location=tattoo.get('location'),
                    description=tattoo.get('description'),
                    source_endpoint='stashdb'
                ):
                    updated = True

            # Add piercings
            for piercing in (performer.piercings or []):
                if self.db.add_piercing(
                    internal_id,
                    location=piercing.get('location'),
                    description=piercing.get('description'),
                    source_endpoint='stashdb'
                ):
                    updated = True

            if updated:
                self.stats.updated += 1

            return updated

        except Exception as e:
            print(f"  Error refreshing {stashbox_performer_id}: {e}")
            self.stats.errors += 1
            return False

    def run(
        self,
        missing_urls_only: bool = False,
        performer_id: Optional[str] = None,
        max_performers: Optional[int] = None,
        resume: bool = False,
    ):
        """
        Run the metadata refresh.

        Args:
            missing_urls_only: Only refresh performers missing URLs
            performer_id: Refresh specific performer only
            max_performers: Maximum number to process
            resume: Continue from previous progress
        """
        if resume:
            self.load_progress()
        else:
            self.clear_progress()

        # Get performers to refresh
        performers = self.get_performers_to_refresh(
            missing_urls_only=missing_urls_only,
            performer_id=performer_id,
        )

        self.stats.total = len(performers)
        print(f"Found {len(performers)} performers to refresh")

        # Filter out already processed
        remaining = [
            (pid, sid) for pid, sid in performers
            if sid not in self.processed_ids
        ]

        if max_performers:
            remaining = remaining[:max_performers]

        print(f"Processing {len(remaining)} performers...")

        for i, (internal_id, stashbox_performer_id) in enumerate(remaining):
            # Progress
            if i % 100 == 0:
                processed = len(self.processed_ids) + i
                pct = processed / self.stats.total * 100 if self.stats.total > 0 else 0
                print(f"Progress: {processed}/{self.stats.total} ({pct:.1f}%)")
                print(f"  Updated: {self.stats.updated}, URLs: {self.stats.urls_added}, "
                      f"Aliases: {self.stats.aliases_added}, Errors: {self.stats.errors}")

            # Refresh
            self.refresh_performer(internal_id, stashbox_performer_id)
            self.processed_ids.add(stashbox_performer_id)
            self.stats.processed += 1

            # Rate limit
            time.sleep(self.rate_limit)

            # Checkpoint every 500
            if (i + 1) % 500 == 0:
                print("  Saving checkpoint...")
                self.save_progress()

        # Final save
        print("Saving final results...")
        self.save_progress()

        self._print_summary()

    def _print_summary(self):
        """Print final summary."""
        print("\n" + "=" * 60)
        print("Metadata Refresh Complete!")
        print("=" * 60)
        print(f"Total performers: {self.stats.total}")
        print(f"Processed: {self.stats.processed}")
        print(f"Updated: {self.stats.updated}")
        print(f"URLs added: {self.stats.urls_added}")
        print(f"Aliases added: {self.stats.aliases_added}")
        print(f"Merged IDs added: {self.stats.merged_ids_added}")
        print(f"Errors: {self.stats.errors}")

        # Print database stats
        print("\nDatabase Statistics:")
        db_stats = self.db.get_stats()
        print(f"  Total performers: {db_stats['performer_count']}")
        print(f"  Total URLs: {db_stats['total_urls']}")
        print(f"  Total aliases: {db_stats['total_aliases']}")

        if db_stats['url_site_counts']:
            print("\n  URLs by site:")
            for site, count in list(db_stats['url_site_counts'].items())[:10]:
                print(f"    {site}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Refresh performer metadata from source APIs")
    parser.add_argument(
        "--database", "-d",
        required=True,
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Refresh all performers",
    )
    parser.add_argument(
        "--missing-urls",
        action="store_true",
        help="Only refresh performers missing URLs",
    )
    parser.add_argument(
        "--performer",
        help="Refresh specific performer by StashDB ID",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=0.25,
        help="Seconds between API requests (default: 0.25)",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="Maximum performers to process",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint",
    )

    args = parser.parse_args()

    if not (args.all or args.missing_urls or args.performer):
        parser.error("Must specify --all, --missing-urls, or --performer")

    if not Path(args.database).exists():
        parser.error(f"Database not found: {args.database}")

    refresher = MetadataRefresher(
        db_path=args.database,
        rate_limit=args.rate_limit,
    )

    refresher.run(
        missing_urls_only=args.missing_urls,
        performer_id=args.performer,
        max_performers=args.max,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
