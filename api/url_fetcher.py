#!/usr/bin/env python3
"""
Supplementary URL Fetcher

Fetches URL and alias data for performers that already exist in the database.
Runs independently of the main database builder - can run in parallel.

This allows us to build the URL index for identity graph linking without
waiting for a full database rebuild.

Usage:
    python url_fetcher.py --input data-snapshot-20260127-1653/performers.json --output urls.json
    python url_fetcher.py --resume  # Continue from last checkpoint
"""

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from stashdb_client import StashDBClient

# Load environment variables
load_dotenv()


@dataclass
class PerformerURLData:
    """URL and identity data for a performer."""
    stashdb_id: str
    name: str
    aliases: list[str]
    urls: dict[str, list[str]]  # site_name -> list of URLs
    gender: Optional[str]
    country: Optional[str]
    birth_date: Optional[str]
    career_start_year: Optional[int]
    career_end_year: Optional[int]
    merged_ids: list[str]
    fetched_at: str


class URLFetcher:
    """Fetches URL data for existing performers."""

    def __init__(
        self,
        input_path: str,
        output_path: str,
        progress_path: Optional[str] = None,
        rate_limit: float = 0.25,  # seconds between requests
    ):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.progress_path = Path(progress_path or output_path.replace('.json', '_progress.json'))
        self.rate_limit = rate_limit

        # Initialize StashDB client from environment
        stashdb_url = os.getenv("STASHDB_URL", "https://stashdb.org/graphql")
        stashdb_key = os.getenv("STASHDB_API_KEY", "")
        if not stashdb_key:
            raise ValueError("STASHDB_API_KEY environment variable required")
        self.client = StashDBClient(stashdb_url, stashdb_key, rate_limit_delay=rate_limit)

        self.url_data: dict[str, PerformerURLData] = {}
        self.processed_ids: set[str] = set()
        self.stats = {
            'total': 0,
            'processed': 0,
            'with_urls': 0,
            'with_aliases': 0,
            'errors': 0,
        }

    def load_performers(self) -> list[str]:
        """Load performer IDs from input file."""
        print(f"Loading performers from {self.input_path}")
        with open(self.input_path) as f:
            performers = json.load(f)

        # Extract StashDB IDs from universal IDs
        stashdb_ids = []
        for uid in performers.keys():
            # Format: "stashdb.org:uuid"
            if uid.startswith("stashdb.org:"):
                stashdb_id = uid.split(":", 1)[1]
                stashdb_ids.append(stashdb_id)

        print(f"Found {len(stashdb_ids)} StashDB performers")
        self.stats['total'] = len(stashdb_ids)
        return stashdb_ids

    def load_progress(self):
        """Load previous progress if resuming."""
        if self.progress_path.exists():
            print(f"Loading progress from {self.progress_path}")
            with open(self.progress_path) as f:
                progress = json.load(f)
            self.processed_ids = set(progress.get('processed_ids', []))
            self.stats = progress.get('stats', self.stats)
            print(f"Resuming: {len(self.processed_ids)} already processed")

        if self.output_path.exists():
            print(f"Loading existing URL data from {self.output_path}")
            with open(self.output_path) as f:
                data = json.load(f)
            for uid, pdata in data.items():
                self.url_data[uid] = PerformerURLData(**pdata)

    def save_progress(self):
        """Save current progress."""
        progress = {
            'processed_ids': list(self.processed_ids),
            'stats': self.stats,
            'saved_at': datetime.now().isoformat(),
        }
        with open(self.progress_path, 'w') as f:
            json.dump(progress, f)

        # Save URL data
        url_data_dict = {uid: asdict(pdata) for uid, pdata in self.url_data.items()}
        with open(self.output_path, 'w') as f:
            json.dump(url_data_dict, f, indent=2)

    def fetch_performer_urls(self, stashdb_id: str) -> Optional[PerformerURLData]:
        """Fetch URL data for a single performer."""
        try:
            performer = self.client.get_performer(stashdb_id)
            if not performer:
                return None

            return PerformerURLData(
                stashdb_id=stashdb_id,
                name=performer.name,
                aliases=performer.aliases or [],
                urls=performer.urls or {},
                gender=performer.gender,
                country=performer.country,
                birth_date=performer.birth_date,
                career_start_year=performer.career_start_year,
                career_end_year=performer.career_end_year,
                merged_ids=performer.merged_ids or [],
                fetched_at=datetime.now().isoformat(),
            )
        except Exception as e:
            print(f"  Error fetching {stashdb_id}: {e}")
            self.stats['errors'] += 1
            return None

    def run(self, max_performers: Optional[int] = None):
        """Run the URL fetcher."""
        stashdb_ids = self.load_performers()
        self.load_progress()

        # Filter out already processed
        remaining = [sid for sid in stashdb_ids if sid not in self.processed_ids]
        if max_performers:
            remaining = remaining[:max_performers]

        print(f"Processing {len(remaining)} performers...")

        for i, stashdb_id in enumerate(remaining):
            # Progress
            if i % 100 == 0:
                pct = (len(self.processed_ids) + i) / self.stats['total'] * 100
                print(f"Progress: {len(self.processed_ids) + i}/{self.stats['total']} ({pct:.1f}%)")
                print(f"  URLs: {self.stats['with_urls']}, Aliases: {self.stats['with_aliases']}, Errors: {self.stats['errors']}")

            # Fetch
            url_data = self.fetch_performer_urls(stashdb_id)
            if url_data:
                uid = f"stashdb.org:{stashdb_id}"
                self.url_data[uid] = url_data
                self.stats['processed'] += 1

                if url_data.urls:
                    self.stats['with_urls'] += 1
                if url_data.aliases:
                    self.stats['with_aliases'] += 1

            self.processed_ids.add(stashdb_id)

            # Rate limit
            time.sleep(self.rate_limit)

            # Checkpoint every 500
            if (i + 1) % 500 == 0:
                print("  Saving checkpoint...")
                self.save_progress()

        # Final save
        print("Saving final results...")
        self.save_progress()

        print("\n=== Final Stats ===")
        print(f"Total: {self.stats['total']}")
        print(f"Processed: {self.stats['processed']}")
        print(f"With URLs: {self.stats['with_urls']}")
        print(f"With aliases: {self.stats['with_aliases']}")
        print(f"Errors: {self.stats['errors']}")


def main():
    parser = argparse.ArgumentParser(description="Fetch URL data for existing performers")
    parser.add_argument(
        "--input",
        default="data-snapshot-20260127-1653/performers.json",
        help="Input performers.json file",
    )
    parser.add_argument(
        "--output",
        default="data-snapshot-20260127-1653/urls.json",
        help="Output URLs JSON file",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=0.25,
        help="Seconds between requests (default: 0.25 = 240/min)",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="Maximum performers to process (for testing)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint",
    )

    args = parser.parse_args()

    fetcher = URLFetcher(
        input_path=args.input,
        output_path=args.output,
        rate_limit=args.rate_limit,
    )

    if args.resume:
        fetcher.load_progress()

    fetcher.run(max_performers=args.max)


if __name__ == "__main__":
    main()
