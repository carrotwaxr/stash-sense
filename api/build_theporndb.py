#!/usr/bin/env python3
"""Build face recognition database from ThePornDB.

This is a wrapper around database_builder.py that uses the ThePornDB REST API
instead of the StashDB GraphQL API.

Usage:
    # Set your API key
    export THEPORNDB_API_KEY="your-api-key"

    # Build database (outputs to ./data-theporndb by default)
    python build_theporndb.py --max-performers 1000

    # Resume interrupted build
    python build_theporndb.py --resume

    # Full build (~10,000 performers)
    python build_theporndb.py

Environment variables:
    THEPORNDB_API_KEY    Your ThePornDB API key (required)
    THEPORNDB_RATE_LIMIT Rate limit delay in seconds (default: 0.25 = 240/min)
"""
import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from theporndb_client import ThePornDBClient
from database_builder import DatabaseBuilder, BuilderConfig, DatabaseConfig
from config import get_stashbox_shortname


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Build face recognition database from ThePornDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--max-performers", type=int, default=None,
                        help="Maximum number of performers to process (default: all ~10,000)")
    parser.add_argument("--max-images", type=int, default=5,
                        help="Maximum images per performer (default: 5)")
    parser.add_argument("--rate-limit", type=float, default=None,
                        help="Delay between requests in seconds (default: 0.25 = 240/min)")
    parser.add_argument("--output", type=str, default="./data-theporndb",
                        help="Output directory for database files (default: ./data-theporndb)")
    parser.add_argument("--version", type=str, default=None,
                        help="Database version string (default: YYYY.MM.DD)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from previous build")
    parser.add_argument("--completeness-threshold", type=int, default=5,
                        help="Minimum faces for a performer to be 'complete' (default: 5)")
    parser.add_argument("--use-face-urls", action="store_true",
                        help="Prefer ThePornDB's pre-cropped face images (faster but may be lower quality)")
    args = parser.parse_args()

    # Get API key
    api_key = os.environ.get("THEPORNDB_API_KEY")
    if not api_key:
        print("Error: THEPORNDB_API_KEY environment variable not set")
        print("Get an API key from https://theporndb.net/")
        sys.exit(1)

    # Rate limit
    rate_limit = args.rate_limit
    if rate_limit is None:
        rate_limit = float(os.environ.get("THEPORNDB_RATE_LIMIT", "0.25"))

    print("=" * 60)
    print("ThePornDB Database Builder")
    print("=" * 60)
    print(f"  API endpoint: https://api.theporndb.net")
    print(f"  Rate limit: {60/rate_limit:.0f} requests/min ({rate_limit}s delay)")
    print(f"  Output directory: {args.output}")
    print(f"  Max performers: {args.max_performers or 'all (~10,000)'}")
    print(f"  Max images per performer: {args.max_images}")
    print(f"  Resume mode: {'ON' if args.resume else 'OFF'}")
    print()

    # Initialize client
    client = ThePornDBClient(
        api_key=api_key,
        rate_limit_delay=rate_limit,
    )

    # Monkey-patch the URL for stashbox name detection
    # (database_builder uses this to create universal IDs)
    client.url = "https://theporndb.net/graphql"  # For get_stashbox_shortname

    # Configure
    db_config = DatabaseConfig(data_dir=Path(args.output))
    builder_config = BuilderConfig(
        max_performers=args.max_performers,
        max_images_per_performer=args.max_images,
        version=args.version,
        completeness_threshold=args.completeness_threshold,
    )

    # Build
    builder = DatabaseBuilder(
        db_config,
        builder_config,
        client,
        resume=args.resume,
    )
    builder.build_from_stashdb()
    builder.save()

    print("\n" + "=" * 60)
    print("Build complete!")
    print(f"Database saved to: {args.output}")


if __name__ == "__main__":
    main()
