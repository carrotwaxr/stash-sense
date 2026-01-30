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
import signal
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from database import PerformerDatabase
from enrichment_config import EnrichmentConfig
from enrichment_coordinator import EnrichmentCoordinator, ReferenceSiteMode
from stashdb_client import StashDBClient
from stashbox_clients import PMVStashClient, JAVStashClient, FansDBClient
from theporndb_client import ThePornDBClient
from babepedia_client import BabepediaScraper
from boobpedia_client import BoobpediaScraper
from iafd_client import IAFDScraper
from indexxx_client import IndexxxScraper
from freeones_client import FreeOnesScraper

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

        elif source_name == "pmvstash":
            api_key = os.environ.get("PMVSTASH_API_KEY", "")
            if not api_key:
                logger.warning("PMVSTASH_API_KEY not set, skipping pmvstash")
                continue

            scrapers.append(PMVStashClient(
                url=source_config.url,
                api_key=api_key,
                rate_limit_delay=60 / source_config.rate_limit,
            ))
            logger.info(f"Created PMVStash scraper (rate: {source_config.rate_limit} req/min)")

        elif source_name == "javstash":
            api_key = os.environ.get("JAVSTASH_API_KEY", "")
            if not api_key:
                logger.warning("JAVSTASH_API_KEY not set, skipping javstash")
                continue

            scrapers.append(JAVStashClient(
                url=source_config.url,
                api_key=api_key,
                rate_limit_delay=60 / source_config.rate_limit,
            ))
            logger.info(f"Created JAVStash scraper (rate: {source_config.rate_limit} req/min)")

        elif source_name == "fansdb":
            api_key = os.environ.get("FANSDB_API_KEY", "")
            if not api_key:
                logger.warning("FANSDB_API_KEY not set, skipping fansdb")
                continue

            scrapers.append(FansDBClient(
                url=source_config.url,
                api_key=api_key,
                rate_limit_delay=60 / source_config.rate_limit,
            ))
            logger.info(f"Created FansDB scraper (rate: {source_config.rate_limit} req/min)")

        elif source_name == "iafd":
            flaresolverr_url = os.environ.get("FLARESOLVERR_URL", "http://10.0.0.4:8191")
            scraper = IAFDScraper(
                flaresolverr_url=flaresolverr_url,
                rate_limit_delay=60 / source_config.rate_limit,
            )
            if not scraper.flaresolverr.is_available():
                logger.warning(f"FlareSolverr not available at {flaresolverr_url}, skipping iafd")
                continue

            scrapers.append(scraper)
            logger.info(f"Created IAFD scraper (rate: {source_config.rate_limit} req/min)")

        elif source_name == "freeones":
            flaresolverr_url = os.environ.get("FLARESOLVERR_URL", "http://10.0.0.4:8191")
            scraper = FreeOnesScraper(
                flaresolverr_url=flaresolverr_url,
                rate_limit_delay=60 / source_config.rate_limit,
            )
            scrapers.append(scraper)
            logger.info(f"Created FreeOnes scraper (rate: {source_config.rate_limit} req/min)")

        elif source_name == "indexxx":
            chrome_cdp_url = os.environ.get("CHROME_CDP_URL", "http://10.0.0.4:9222")
            scraper = IndexxxScraper(
                chrome_cdp_url=chrome_cdp_url,
                rate_limit_delay=60 / source_config.rate_limit,
            )
            scrapers.append(scraper)
            logger.info(f"Created Indexxx scraper (Chrome CDP: {chrome_cdp_url}, rate: {source_config.rate_limit} req/min)")

        elif source_name == "boobpedia":
            scraper = BoobpediaScraper(
                rate_limit_delay=60 / source_config.rate_limit,
            )
            scrapers.append(scraper)
            logger.info(f"Created Boobpedia scraper (rate: {source_config.rate_limit} req/min)")

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
    parser.add_argument(
        "--enable-faces",
        action="store_true",
        help="Enable face detection and embedding (requires GPU, slower)",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=1000,
        help="Save indices every N faces (default: 1000)",
    )
    parser.add_argument(
        "--reference-mode",
        choices=["url", "name"],
        default="url",
        help="How to find performers for reference sites: 'url' (lookup by existing URLs) or 'name' (try all by name)",
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

    # Build trust levels from config
    source_trust_levels = {}
    for source_name in sources:
        try:
            source_config = config.get_source(source_name)
            source_trust_levels[source_name] = source_config.trust_level
        except KeyError:
            pass

    # Determine data directory for face processing
    data_dir = args.database.parent if args.enable_faces else None

    # Parse reference mode
    reference_mode = ReferenceSiteMode.URL_LOOKUP
    if args.reference_mode == "name":
        reference_mode = ReferenceSiteMode.NAME_LOOKUP

    # Create coordinator
    coordinator = EnrichmentCoordinator(
        database=db,
        scrapers=scrapers,
        data_dir=data_dir,
        max_faces_per_source=args.max_faces_per_source,
        max_faces_total=args.max_faces_total,
        dry_run=args.dry_run,
        enable_face_processing=args.enable_faces,
        source_trust_levels=source_trust_levels,
        reference_site_mode=reference_mode,
    )

    if args.dry_run:
        logger.info("DRY RUN - no changes will be written to database")

    if args.enable_faces:
        logger.info(f"Face processing ENABLED - using data dir: {data_dir}")
        logger.info(f"Trust levels: {source_trust_levels}")

    # Set up signal handler for graceful shutdown
    def signal_handler(signum, frame):
        logger.info("Received interrupt signal, requesting graceful shutdown...")
        coordinator.request_shutdown()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run
    try:
        asyncio.run(coordinator.run())
    except KeyboardInterrupt:
        pass  # Already handled by signal handler

    print("\n=== Enrichment Complete ===")
    print(f"Performers processed: {coordinator.stats.performers_processed:,}")
    print(f"Images processed: {coordinator.stats.images_processed:,}")
    print(f"Faces added: {coordinator.stats.faces_added:,}")
    print(f"Faces rejected: {coordinator.stats.faces_rejected:,}")
    print(f"Errors: {coordinator.stats.errors}")
    for source, count in coordinator.stats.by_source.items():
        print(f"  {source}: {count:,}")


if __name__ == "__main__":
    main()
