#!/usr/bin/env python3
"""Display enrichment progress with progress bars and completion percentages."""

import sqlite3
import sys
from pathlib import Path

# Source configuration - defines how to count total for each source
# Keys must match the source names in scrape_progress table
SOURCE_CONFIG = {
    # Stash-boxes (count from stashbox_ids table)
    "stashdb": {"type": "stashbox", "endpoint": "stashdb"},
    "theporndb": {"type": "stashbox", "endpoint": "theporndb"},
    "pmvstash": {"type": "stashbox", "endpoint": "pmvstash"},
    "javstash": {"type": "stashbox", "endpoint": "javstash"},
    "fansdb": {"type": "stashbox", "endpoint": "fansdb"},
    # URL lookup sources (progress key = source:url)
    "iafd:url": {"type": "url_lookup", "site": "iafd"},
    "thenude:url": {"type": "url_lookup", "site": "thenude"},
    "afdb:url": {"type": "url_lookup", "site": "adultfilmdatabase"},
    "pornpics:url": {"type": "url_lookup", "site": "pornpics"},
    "boobpedia:url": {"type": "url_lookup", "site": "boobpedia"},
    "indexxx:url": {"type": "url_lookup", "site": "indexxx"},
    "babepedia:url": {"type": "url_lookup", "site": "babepedia"},
    "freeones:url": {"type": "url_lookup", "site": "freeones"},
    "elitebabes:url": {"type": "url_lookup", "site": "elitebabes"},
    "javdatabase:url": {"type": "url_lookup", "site": "javdatabase"},
    # Name lookup sources (progress key = source, no suffix)
    "babepedia": {"type": "name_lookup", "gender": "FEMALE"},
    "freeones": {"type": "name_lookup", "gender": "FEMALE"},
    "boobpedia": {"type": "name_lookup", "gender": "FEMALE"},
    "elitebabes": {"type": "name_lookup", "gender": "FEMALE"},
    "javdatabase": {"type": "name_lookup", "gender": "FEMALE"},
    "indexxx": {"type": "name_lookup", "gender": None},
    # Legacy entries (old format without mode suffix)
    "iafd": {"type": "name_lookup", "gender": None},
}

# Colors
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
DIM = "\033[2m"
RESET = "\033[0m"
BOLD = "\033[1m"


def get_total_for_source(conn: sqlite3.Connection, source: str) -> int:
    """Get total performers to process for a source."""
    config = SOURCE_CONFIG.get(source)
    if not config:
        return 0

    if config["type"] == "stashbox":
        # Count actual performers from this stashbox endpoint
        endpoint = config["endpoint"]
        cur = conn.execute(
            "SELECT COUNT(*) FROM stashbox_ids WHERE endpoint = ?",
            (endpoint,),
        )
        return cur.fetchone()[0]

    if config["type"] == "url_lookup":
        site = config["site"]
        cur = conn.execute(
            """
            SELECT COUNT(DISTINCT p.id)
            FROM performers p
            JOIN external_urls u ON p.id = u.performer_id
            WHERE u.url LIKE ?
            """,
            (f"%{site}%",),
        )
        return cur.fetchone()[0]

    if config["type"] == "name_lookup":
        gender = config.get("gender")
        if gender:
            cur = conn.execute(
                "SELECT COUNT(*) FROM performers WHERE gender = ?",
                (gender,),
            )
        else:
            cur = conn.execute("SELECT COUNT(*) FROM performers")
        return cur.fetchone()[0]

    return 0


def progress_bar(current: int, total: int, width: int = 30) -> str:
    """Generate a text progress bar."""
    if total == 0:
        return f"[{'?' * width}]"

    pct = min(current / total, 1.0)
    filled = int(width * pct)
    empty = width - filled

    bar = "█" * filled + "░" * empty
    return f"[{bar}]"


def format_number(n: int) -> str:
    """Format number with commas."""
    return f"{n:,}"


def main():
    db_path = Path("data/performers.db")
    if not db_path.exists():
        print(f"{RED}Database not found: {db_path}{RESET}")
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Get scrape progress
    cur = conn.execute(
        """
        SELECT source, performers_processed, faces_added, last_processed_time
        FROM scrape_progress
        ORDER BY performers_processed DESC
        """
    )
    progress_rows = cur.fetchall()

    if not progress_rows:
        print(f"{YELLOW}No enrichment progress recorded yet.{RESET}")
        print(f"\nRun: {CYAN}make enrich{RESET} to start enrichment")
        return

    # Calculate totals
    total_processed = sum(r["performers_processed"] for r in progress_rows)
    total_faces = sum(r["faces_added"] for r in progress_rows)

    print(f"\n{BOLD}{CYAN}═══════════════════════════════════════════════════════════════{RESET}")
    print(f"{BOLD}{CYAN}  Enrichment Progress{RESET}")
    print(f"{BOLD}{CYAN}═══════════════════════════════════════════════════════════════{RESET}\n")

    # Overall summary
    print(f"  {BOLD}Total:{RESET} {GREEN}{format_number(total_processed)}{RESET} performers, "
          f"{GREEN}{format_number(total_faces)}{RESET} faces added\n")

    # Per-source progress
    for row in progress_rows:
        source = row["source"]
        processed = row["performers_processed"]
        faces = row["faces_added"]

        # Get total for this source
        total = get_total_for_source(conn, source)

        if total > 0:
            pct = (processed / total) * 100
            bar = progress_bar(processed, total)
            pct_str = f"{pct:5.1f}%"
            progress_str = f"{format_number(processed):>8} / {format_number(total):<8}"
        else:
            bar = progress_bar(processed, processed)  # Full bar if we don't know total
            pct_str = "  ???"
            progress_str = f"{format_number(processed):>8}"

        # Colorize based on completion
        if total > 0 and pct >= 100:
            status_color = GREEN
            status = "✓"
        elif total > 0 and pct > 0:
            status_color = YELLOW
            status = "⋯"
        else:
            status_color = DIM
            status = "○"

        # Source display name (show mode if it has a suffix)
        if ":url" in source:
            display_source = source.replace(":url", "") + " (url)"
        elif ":name" in source:
            display_source = source.replace(":name", "") + " (name)"
        else:
            display_source = source

        print(f"  {status_color}{status}{RESET} {display_source:15} {bar} {pct_str}  "
              f"{DIM}{progress_str}{RESET}  "
              f"{CYAN}+{format_number(faces)} faces{RESET}")

    print(f"\n{DIM}  Last update times vary by source. Run 'make enrich' to continue.{RESET}\n")

    conn.close()


if __name__ == "__main__":
    main()
