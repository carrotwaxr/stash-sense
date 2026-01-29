#!/usr/bin/env python3
"""
Multi-source enrichment builder CLI.

Usage:
    python enrichment_builder.py --sources stashdb
    python enrichment_builder.py --sources stashdb,theporndb
    python enrichment_builder.py --dry-run --sources stashdb
    python enrichment_builder.py --status
"""
import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from database import PerformerDatabase
from enrichment_config import EnrichmentConfig
from enrichment_coordinator import EnrichmentCoordinator
from stashdb_client import StashDBClient
from theporndb_client import ThePornDBClient
from babepedia_client import BabepediaScraper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


def create_scrapers(config: EnrichmentConfig, sources: list[str]) -> list:
    """Create scraper instances for enabled sources."""
    scrapers = []

    for source_name in sources:
        source_config = config.get_source(source_name)
        if not source_config:
            logger.warning(f"Unknown source: {source_name}")
            continue

        if source_name == "stashdb":
            api_key = os.environ.get("STASHDB_API_KEY", "")
            if not api_key:
                logger.warning("STASHDB_API_KEY not set, skipping stashdb")
                continue

            scrapers.append(StashDBClient(
                url=source_config.url or os.environ.get("STASHDB_URL", "https://stashdb.org/graphql"),
                api_key=api_key,
                rate_limit_delay=60 / source_config.rate_limit,  # Convert from req/min to delay
            ))
            logger.info(f"Created StashDB scraper (rate: {source_config.rate_limit} req/min)")

        elif source_name == "theporndb":
            api_key = os.environ.get("THEPORNDB_API_KEY", "")
            if not api_key:
                logger.warning("THEPORNDB_API_KEY not set, skipping theporndb")
                continue

            scrapers.append(ThePornDBClient(
                api_key=api_key,
                rate_limit_delay=60 / source_config.rate_limit,
            ))
            logger.info(f"Created ThePornDB scraper (rate: {source_config.rate_limit} req/min)")

        elif source_name == "babepedia":
            flaresolverr_url = os.environ.get("FLARESOLVERR_URL", "http://10.0.0.4:8191")
            scraper = BabepediaScraper(
                flaresolverr_url=flaresolverr_url,
                rate_limit_delay=60 / source_config.rate_limit,
            )
            if not scraper.flaresolverr.is_available():
                logger.warning(f"FlareSolverr not available at {flaresolverr_url}, skipping babepedia")
                continue

            scrapers.append(scraper)
            logger.info(f"Created Babepedia scraper (rate: {source_config.rate_limit} req/min)")

    return scrapers


def show_status(db: PerformerDatabase):
    """Show current enrichment progress."""
    print("\n=== Enrichment Progress ===\n")

    # Database stats
    stats = db.get_stats()
    print("Database:")
    print(f"  Performers: {stats['performer_count']:,}")
    print(f"  Faces: {stats['total_faces']:,}")
    print(f"  Performers with faces: {stats['performers_with_faces']:,}")
    print(f"  Avg faces/performer: {stats['total_faces'] / max(1, stats['performers_with_faces']):.2f}")
    print()

    # Per-source progress
    progress = db.get_all_scrape_progress()
    if not progress:
        print("No scraping progress recorded yet.")
    else:
        print("Scraper Progress:")
        for source, data in progress.items():
            print(f"\n  {source}:")
            print(f"    Last ID: {data['last_processed_id'][:8]}..." if data['last_processed_id'] else "    Last ID: None")
            print(f"    Performers: {data['performers_processed']:,}")
            print(f"    Faces added: {data['faces_added']:,}")
            print(f"    Errors: {data['errors']}")
            print(f"    Last run: {data['last_processed_time']}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Multi-source enrichment builder for face recognition database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --sources stashdb              # Run StashDB enrichment
  %(prog)s --status                       # Show current progress
  %(prog)s --dry-run --sources stashdb    # Test without writing
  %(prog)s --clear-progress stashdb       # Reset progress for source
        """,
    )

    parser.add_argument(
        "--sources",
        type=str,
        default="stashdb",
        help="Comma-separated list of sources to run (default: stashdb)",
    )
    parser.add_argument(
        "--disable-source",
        type=str,
        action="append",
        default=[],
        dest="disabled_sources",
        help="Disable a source (can be used multiple times)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "sources.yaml",
        help="Path to sources.yaml config file",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path(__file__).parent / "data" / "performers.db",
        help="Path to performers database",
    )
    parser.add_argument(
        "--max-faces-per-source",
        type=int,
        default=5,
        help="Maximum faces per source per performer (default: 5)",
    )
    parser.add_argument(
        "--max-faces-total",
        type=int,
        default=20,
        help="Maximum total faces per performer (default: 20)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without writing to database",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current progress and exit",
    )
    parser.add_argument(
        "--clear-progress",
        type=str,
        metavar="SOURCE",
        help="Clear progress for a source and exit",
    )

    args = parser.parse_args()

    # Load config
    config_path = args.config if args.config.exists() else None
    cli_sources = args.sources.split(",") if args.sources else []

    config = EnrichmentConfig(
        config_path=config_path,
        cli_sources=cli_sources,
        cli_disabled_sources=args.disabled_sources,
    )

    # Open database
    db = PerformerDatabase(args.database)

    # Status mode
    if args.status:
        show_status(db)
        return

    # Clear progress mode
    if args.clear_progress:
        db.clear_scrape_progress(args.clear_progress)
        print(f"Cleared progress for: {args.clear_progress}")
        return

    # Get sources to run
    sources = [s.strip() for s in cli_sources if s.strip() not in args.disabled_sources]

    if not sources:
        logger.error("No sources specified! Use --sources stashdb")
        sys.exit(1)

    logger.info(f"Enabled sources: {sources}")

    # Create scrapers
    scrapers = create_scrapers(config, sources)

    if not scrapers:
        logger.error("No scrapers created! Check API keys and source configuration.")
        sys.exit(1)

    # Create coordinator
    coordinator = EnrichmentCoordinator(
        database=db,
        scrapers=scrapers,
        max_faces_per_source=args.max_faces_per_source,
        max_faces_total=args.max_faces_total,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        logger.info("DRY RUN - no changes will be written to database")

    # Run
    try:
        asyncio.run(coordinator.run())
        print("\n=== Enrichment Complete ===")
        print(f"Performers processed: {coordinator.stats.performers_processed:,}")
        print(f"Errors: {coordinator.stats.errors}")
        for source, count in coordinator.stats.by_source.items():
            print(f"  {source}: {count:,}")
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main()
