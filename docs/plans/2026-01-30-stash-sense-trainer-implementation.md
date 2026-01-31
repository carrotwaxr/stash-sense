# Stash Sense Trainer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extract enrichment/database building into a separate private repo with React dashboard, Docker support, and unRAID template.

**Architecture:** New private repo `stash-sense-trainer` containing all scrapers, enrichment code, and a FastAPI+React dashboard. Public repo `stash-face-recognition` retains only runtime sidecar and plugin code.

**Tech Stack:** Python 3.11, FastAPI, React 18, Vite, Tailwind CSS, Docker with NVIDIA runtime

---

## Prerequisites

Before starting:
1. GitHub account with ability to create private repos
2. Current working directory: `/home/carrot/code/stash-face-recognition`
3. Node.js 18+ and npm installed
4. Python 3.11 with venv

---

## Task 1: Create New Private Repository

**Files:**
- Create: `~/code/stash-sense-trainer/` (new directory)
- Create: `~/code/stash-sense-trainer/.gitignore`
- Create: `~/code/stash-sense-trainer/README.md`

**Step 1: Create directory and initialize git**

```bash
mkdir -p ~/code/stash-sense-trainer
cd ~/code/stash-sense-trainer
git init
```

**Step 2: Create .gitignore**

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
.venv/
venv/
.env

# Node
node_modules/
web/dist/

# Data
*.db
*.voy
data/
runs.json

# IDE
.idea/
.vscode/
*.swp

# Docker
.docker/

# OS
.DS_Store
Thumbs.db
```

**Step 3: Create README.md**

```markdown
# Stash Sense Trainer

Private database builder for Stash Sense face recognition.

## Features

- Multi-source performer scraping (StashDB, ThePornDB, reference sites)
- Face detection and embedding generation
- React dashboard for monitoring and control
- Docker support with GPU acceleration

## Quick Start

```bash
docker-compose up -d
```

Then visit http://localhost:8080

## Environment Variables

See `docker-compose.yml` for all available configuration options.
```

**Step 4: Initial commit**

```bash
git add .gitignore README.md
git commit -m "Initial commit: repo structure"
```

---

## Task 2: Copy Trainer Files from Source Repo

**Files:**
- Copy: All scraper and enrichment files from `stash-face-recognition/api/`

**Step 1: Create api directory structure**

```bash
cd ~/code/stash-sense-trainer
mkdir -p api
```

**Step 2: Copy trainer-specific files**

```bash
# Core enrichment
cp ~/code/stash-face-recognition/api/enrichment_builder.py api/
cp ~/code/stash-face-recognition/api/enrichment_coordinator.py api/
cp ~/code/stash-face-recognition/api/enrichment_config.py api/
cp ~/code/stash-face-recognition/api/sources.yaml api/

# Database (full version)
cp ~/code/stash-face-recognition/api/database.py api/

# Scrapers - stash-boxes
cp ~/code/stash-face-recognition/api/stashdb_client.py api/
cp ~/code/stash-face-recognition/api/theporndb_client.py api/
cp ~/code/stash-face-recognition/api/stashbox_clients.py api/

# Scrapers - reference sites
cp ~/code/stash-face-recognition/api/babepedia_client.py api/
cp ~/code/stash-face-recognition/api/boobpedia_client.py api/
cp ~/code/stash-face-recognition/api/iafd_client.py api/
cp ~/code/stash-face-recognition/api/freeones_client.py api/
cp ~/code/stash-face-recognition/api/indexxx_client.py api/
cp ~/code/stash-face-recognition/api/thenude_client.py api/
cp ~/code/stash-face-recognition/api/afdb_client.py api/
cp ~/code/stash-face-recognition/api/pornpics_client.py api/
cp ~/code/stash-face-recognition/api/elitebabes_client.py api/
cp ~/code/stash-face-recognition/api/javdatabase_client.py api/

# Scraper infrastructure
cp ~/code/stash-face-recognition/api/base_scraper.py api/
cp ~/code/stash-face-recognition/api/flaresolverr_client.py api/
cp ~/code/stash-face-recognition/api/url_normalizer.py api/
cp ~/code/stash-face-recognition/api/url_fetcher.py api/

# Face processing
cp ~/code/stash-face-recognition/api/face_processor.py api/
cp ~/code/stash-face-recognition/api/face_validator.py api/
cp ~/code/stash-face-recognition/api/quality_filters.py api/
cp ~/code/stash-face-recognition/api/index_manager.py api/
cp ~/code/stash-face-recognition/api/write_queue.py api/

# Shared files (add sync comment later)
cp ~/code/stash-face-recognition/api/embeddings.py api/
cp ~/code/stash-face-recognition/api/models.py api/

# Database builder (legacy, might be useful)
cp ~/code/stash-face-recognition/api/database_builder.py api/
cp ~/code/stash-face-recognition/api/metadata_refresh.py api/

# Config
cp ~/code/stash-face-recognition/api/config.py api/
```

**Step 3: Create api/__init__.py**

```bash
echo '"""Stash Sense Trainer API."""' > api/__init__.py
```

**Step 4: Commit**

```bash
git add api/
git commit -m "feat: copy trainer files from source repo"
```

---

## Task 3: Add Sync Comments to Shared Files

**Files:**
- Modify: `api/embeddings.py` (add header comment)
- Modify: `api/models.py` (add header comment)

**Step 1: Add sync comment to embeddings.py**

Add this comment at the very top of `api/embeddings.py`:

```python
"""
Face detection and embedding generation.

NOTE: This file is shared between stash-sense (sidecar) and stash-sense-trainer.
If you modify this file, copy changes to the other repo.
Trainer repo is source of truth.
"""
```

**Step 2: Add sync comment to models.py**

Add this comment at the very top of `api/models.py`:

```python
"""
Data models for face recognition.

NOTE: This file is shared between stash-sense (sidecar) and stash-sense-trainer.
If you modify this file, copy changes to the other repo.
Trainer repo is source of truth.
"""
```

**Step 3: Commit**

```bash
git add api/embeddings.py api/models.py
git commit -m "docs: add sync comments to shared files"
```

---

## Task 4: Copy and Organize Documentation

**Files:**
- Create: `docs/` directory structure
- Copy: Trainer-relevant docs

**Step 1: Create docs structure**

```bash
cd ~/code/stash-sense-trainer
mkdir -p docs/reference
```

**Step 2: Copy trainer documentation**

```bash
# Core reference docs
cp ~/code/stash-face-recognition/docs/plans/2026-01-27-data-sources-catalog.md docs/reference/data-sources.md
cp ~/code/stash-face-recognition/docs/plans/2026-01-27-performer-identity-graph.md docs/reference/identity-graph.md
cp ~/code/stash-face-recognition/docs/plans/2026-01-26-reference-site-enrichment.md docs/reference/reference-sites.md
cp ~/code/stash-face-recognition/docs/plans/2026-01-29-multi-source-enrichment-design.md docs/reference/enrichment-design.md
cp ~/code/stash-face-recognition/docs/plans/2026-01-26-concurrent-scraping-architecture.md docs/reference/scraping-architecture.md
cp ~/code/stash-face-recognition/docs/stash-box-endpoints.md docs/reference/stash-box-endpoints.md
cp ~/code/stash-face-recognition/docs/url-domains-analysis.md docs/reference/url-domains.md

# Design doc
cp ~/code/stash-face-recognition/docs/plans/2026-01-30-stash-sense-trainer-design.md docs/design.md
```

**Step 3: Create SESSION-CONTEXT.md for trainer repo**

Create `docs/SESSION-CONTEXT.md`:

```markdown
# Stash Sense Trainer: Session Context

**Purpose:** Quick orientation for Claude sessions working on this project.
**Last Updated:** 2026-01-30

---

## Quick Reference

| What | Command |
|------|---------|
| Run trainer (dev) | `cd api && python enrichment_builder.py --status` |
| Start dashboard | `cd api && uvicorn main:app --reload --port 8080` |
| Build Docker | `docker build -t stash-sense-trainer .` |
| Run Docker | `docker run --gpus all -p 8080:8080 -v ./data:/data stash-sense-trainer` |

## What This Project Does

**Stash Sense Trainer** builds the performer face recognition database by:
1. Scraping performer images from StashDB, ThePornDB, and reference sites
2. Detecting faces and generating embeddings
3. Saving versioned databases for the sidecar to consume

**This is a PRIVATE repo** - not for public distribution.

## Reference Documentation

| Document | Purpose |
|----------|---------|
| [design.md](design.md) | Full architecture and design decisions |
| [reference/data-sources.md](reference/data-sources.md) | All data sources with rate limits and URL patterns |
| [reference/enrichment-design.md](reference/enrichment-design.md) | Multi-source enrichment architecture |
| [reference/scraping-architecture.md](reference/scraping-architecture.md) | Concurrent scraping coordination |
| [reference/identity-graph.md](reference/identity-graph.md) | Cross-source performer linking |

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `STASHDB_API_KEY` | Yes | StashDB API key |
| `THEPORNDB_API_KEY` | No | ThePornDB API key |
| `PMVSTASH_API_KEY` | No | PMVStash API key |
| `JAVSTASH_API_KEY` | No | JAVStash API key |
| `FANSDB_API_KEY` | No | FansDB API key |
| `FLARESOLVERR_URL` | No | FlareSolverr for Cloudflare bypass |
| `CHROME_CDP_URL` | No | Chrome DevTools for Indexxx |
| `DATA_DIR` | No | Output directory (default: /data) |

## External Dependencies

- **FlareSolverr** - Required for Babepedia, IAFD, FreeOnes scrapers
- **Chrome CDP** - Required for Indexxx scraper (currently blocked)

---

*This document should be updated as the project progresses.*
```

**Step 4: Commit**

```bash
git add docs/
git commit -m "docs: add trainer documentation and SESSION-CONTEXT"
```

---

## Task 5: Create requirements.txt

**Files:**
- Create: `requirements.txt`

**Step 1: Create requirements.txt**

```txt
# FastAPI and server
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
python-multipart>=0.0.6
aiofiles>=23.2.1
sse-starlette>=1.8.2

# HTTP clients
httpx>=0.26.0
requests>=2.31.0
beautifulsoup4>=4.12.0
lxml>=5.1.0

# Database
aiosqlite>=0.19.0

# ML/Face detection
deepface>=0.0.89
tensorflow>=2.15.0
onnxruntime-gpu>=1.17.0
voyager>=2.0.0
opencv-python>=4.9.0
numpy>=1.26.0
Pillow>=10.2.0

# Browser automation (for Indexxx)
nodriver>=0.32

# Config and utilities
python-dotenv>=1.0.0
pyyaml>=6.0.1
```

**Step 2: Commit**

```bash
git add requirements.txt
git commit -m "feat: add requirements.txt"
```

---

## Task 6: Create FastAPI Dashboard Backend

**Files:**
- Create: `api/main.py` (new dashboard API)
- Create: `api/build_runner.py` (background task runner)
- Create: `api/run_history.py` (run persistence)

**Step 1: Create api/run_history.py**

```python
"""Run history persistence."""
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict, field


@dataclass
class RunStats:
    performers_processed: int = 0
    images_processed: int = 0
    faces_added: int = 0
    faces_rejected: int = 0
    errors: int = 0
    by_source: dict = field(default_factory=dict)


@dataclass
class Run:
    id: str
    started_at: str
    completed_at: Optional[str] = None
    status: str = "running"  # running, completed, failed, cancelled
    params: dict = field(default_factory=dict)
    stats: RunStats = field(default_factory=RunStats)
    output_db: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["stats"] = asdict(self.stats)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Run":
        stats_data = d.pop("stats", {})
        stats = RunStats(**stats_data) if stats_data else RunStats()
        return cls(**d, stats=stats)


class RunHistory:
    """Manages run history persistence."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.history_file = self.data_dir / "runs.json"
        self._runs: list[Run] = []
        self._load()

    def _load(self):
        """Load run history from disk."""
        if self.history_file.exists():
            try:
                data = json.loads(self.history_file.read_text())
                self._runs = [Run.from_dict(r) for r in data]
            except (json.JSONDecodeError, KeyError):
                self._runs = []

    def _save(self):
        """Save run history to disk."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        data = [r.to_dict() for r in self._runs]
        self.history_file.write_text(json.dumps(data, indent=2))

    def create_run(self, params: dict) -> Run:
        """Create a new run."""
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run = Run(
            id=run_id,
            started_at=datetime.now().isoformat(),
            params=params,
        )
        self._runs.insert(0, run)
        self._save()
        return run

    def update_run(self, run_id: str, **kwargs):
        """Update a run's fields."""
        for run in self._runs:
            if run.id == run_id:
                for key, value in kwargs.items():
                    if hasattr(run, key):
                        setattr(run, key, value)
                self._save()
                return

    def complete_run(self, run_id: str, stats: RunStats, output_db: str):
        """Mark a run as completed."""
        self.update_run(
            run_id,
            status="completed",
            completed_at=datetime.now().isoformat(),
            stats=stats,
            output_db=output_db,
        )

    def fail_run(self, run_id: str, error_message: str):
        """Mark a run as failed."""
        self.update_run(
            run_id,
            status="failed",
            completed_at=datetime.now().isoformat(),
            error_message=error_message,
        )

    def get_run(self, run_id: str) -> Optional[Run]:
        """Get a run by ID."""
        for run in self._runs:
            if run.id == run_id:
                return run
        return None

    def get_runs(self, limit: int = 50) -> list[Run]:
        """Get recent runs."""
        return self._runs[:limit]

    def get_latest_completed(self) -> Optional[Run]:
        """Get the most recent completed run."""
        for run in self._runs:
            if run.status == "completed":
                return run
        return None
```

**Step 2: Create api/build_runner.py**

```python
"""Background build task runner."""
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass, field

from enrichment_config import EnrichmentConfig
from enrichment_coordinator import EnrichmentCoordinator, ReferenceSiteMode
from run_history import RunHistory, Run, RunStats

logger = logging.getLogger(__name__)


@dataclass
class BuildProgress:
    """Current build progress."""
    running: bool = False
    run_id: Optional[str] = None
    current_source: Optional[str] = None
    performers_processed: int = 0
    performers_total: int = 0
    faces_added: int = 0
    errors: int = 0
    by_source: dict = field(default_factory=dict)


class BuildRunner:
    """Manages background build execution."""

    def __init__(self, data_dir: Path, config_path: Path):
        self.data_dir = Path(data_dir)
        self.config_path = Path(config_path)
        self.history = RunHistory(data_dir)
        self.progress = BuildProgress()
        self._task: Optional[asyncio.Task] = None
        self._coordinator: Optional[EnrichmentCoordinator] = None
        self._log_callbacks: list[Callable[[str], None]] = []

    def add_log_callback(self, callback: Callable[[str], None]):
        """Add a callback for log messages."""
        self._log_callbacks.append(callback)

    def remove_log_callback(self, callback: Callable[[str], None]):
        """Remove a log callback."""
        if callback in self._log_callbacks:
            self._log_callbacks.remove(callback)

    def _log(self, message: str):
        """Send log message to callbacks."""
        for callback in self._log_callbacks:
            try:
                callback(message)
            except Exception:
                pass

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    def get_progress(self) -> BuildProgress:
        """Get current build progress."""
        if self._coordinator and self.is_running:
            stats = self._coordinator.stats
            self.progress.performers_processed = stats.performers_processed
            self.progress.faces_added = stats.faces_added
            self.progress.errors = stats.errors
            self.progress.by_source = dict(stats.by_source)
        return self.progress

    async def start_build(self, params: dict) -> Run:
        """Start a new build."""
        if self.is_running:
            raise RuntimeError("Build already in progress")

        # Create run record
        run = self.history.create_run(params)
        self.progress = BuildProgress(running=True, run_id=run.id)

        # Start background task
        self._task = asyncio.create_task(self._run_build(run, params))
        return run

    async def stop_build(self):
        """Request graceful shutdown of current build."""
        if self._coordinator:
            self._coordinator.request_shutdown()
            self._log("Shutdown requested, waiting for current operations...")

    async def _run_build(self, run: Run, params: dict):
        """Execute the build in background."""
        try:
            self._log(f"Starting build {run.id}")

            # Import here to avoid circular imports
            from enrichment_builder import create_scrapers

            # Load config
            config = EnrichmentConfig(
                config_path=self.config_path,
                cli_sources=params.get("sources", []),
            )

            # Create scrapers
            scrapers = create_scrapers(config, params.get("sources", []))
            if not scrapers:
                raise RuntimeError("No scrapers created - check API keys")

            # Build trust levels
            source_trust_levels = {}
            for source_name in params.get("sources", []):
                try:
                    source_config = config.get_source(source_name)
                    source_trust_levels[source_name] = source_config.trust_level
                except KeyError:
                    pass

            # Determine reference mode
            reference_mode = ReferenceSiteMode.URL_LOOKUP
            if params.get("reference_mode") == "name":
                reference_mode = ReferenceSiteMode.NAME_LOOKUP

            # Import database
            from database import PerformerDatabase
            db_path = self.data_dir / "performers.db"
            db = PerformerDatabase(db_path)

            # Create coordinator
            self._coordinator = EnrichmentCoordinator(
                database=db,
                scrapers=scrapers,
                data_dir=self.data_dir if params.get("enable_faces") else None,
                max_faces_per_source=params.get("max_faces_per_source", 5),
                max_faces_total=params.get("max_faces_total", 20),
                dry_run=False,
                enable_face_processing=params.get("enable_faces", True),
                source_trust_levels=source_trust_levels,
                reference_site_mode=reference_mode,
            )

            self._log(f"Running with sources: {params.get('sources', [])}")

            # Run
            await self._coordinator.run()

            # Create timestamped output
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_name = f"performers_{timestamp}.db"
            output_path = self.data_dir / output_name

            # Copy database to versioned file
            import shutil
            shutil.copy(db_path, output_path)

            # Complete run
            stats = RunStats(
                performers_processed=self._coordinator.stats.performers_processed,
                images_processed=self._coordinator.stats.images_processed,
                faces_added=self._coordinator.stats.faces_added,
                faces_rejected=self._coordinator.stats.faces_rejected,
                errors=self._coordinator.stats.errors,
                by_source=dict(self._coordinator.stats.by_source),
            )
            self.history.complete_run(run.id, stats, output_name)
            self._log(f"Build complete: {output_name}")

        except Exception as e:
            logger.exception("Build failed")
            self.history.fail_run(run.id, str(e))
            self._log(f"Build failed: {e}")

        finally:
            self.progress.running = False
            self._coordinator = None
```

**Step 3: Create api/main.py**

```python
"""
Stash Sense Trainer - Dashboard API.

FastAPI app serving the React dashboard and build management endpoints.
"""
import asyncio
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import yaml

from build_runner import BuildRunner, BuildProgress
from run_history import Run

# Configuration
DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))
CONFIG_PATH = Path(__file__).parent / "sources.yaml"

# Initialize app
app = FastAPI(title="Stash Sense Trainer", version="1.0.0")

# Initialize build runner
runner = BuildRunner(DATA_DIR, CONFIG_PATH)


# --- Models ---

class BuildParams(BaseModel):
    sources: list[str]
    enable_faces: bool = True
    max_faces_per_source: int = 5
    max_faces_total: int = 12
    reference_mode: str = "url"


class StatusResponse(BaseModel):
    running: bool
    run_id: Optional[str] = None
    current_source: Optional[str] = None
    performers_processed: int = 0
    performers_total: int = 0
    faces_added: int = 0
    errors: int = 0
    by_source: dict = {}


class ConfigResponse(BaseModel):
    sources: dict


class VersionInfo(BaseModel):
    filename: str
    size_mb: float
    created: str


# --- API Endpoints ---

@app.get("/api/status", response_model=StatusResponse)
async def get_status():
    """Get current build status."""
    progress = runner.get_progress()
    return StatusResponse(
        running=progress.running,
        run_id=progress.run_id,
        current_source=progress.current_source,
        performers_processed=progress.performers_processed,
        performers_total=progress.performers_total,
        faces_added=progress.faces_added,
        errors=progress.errors,
        by_source=progress.by_source,
    )


@app.get("/api/config")
async def get_config():
    """Get current sources configuration."""
    if CONFIG_PATH.exists():
        config = yaml.safe_load(CONFIG_PATH.read_text())
        return {"sources": config.get("sources", {})}
    return {"sources": {}}


@app.put("/api/config")
async def update_config(config: ConfigResponse):
    """Update sources configuration."""
    current = {}
    if CONFIG_PATH.exists():
        current = yaml.safe_load(CONFIG_PATH.read_text()) or {}

    current["sources"] = config.sources
    CONFIG_PATH.write_text(yaml.dump(current, default_flow_style=False))
    return {"status": "ok"}


@app.post("/api/build/start")
async def start_build(params: BuildParams):
    """Start a new build."""
    if runner.is_running:
        raise HTTPException(status_code=409, detail="Build already in progress")

    run = await runner.start_build(params.model_dump())
    return {"status": "started", "run_id": run.id}


@app.post("/api/build/stop")
async def stop_build():
    """Request graceful shutdown of current build."""
    if not runner.is_running:
        raise HTTPException(status_code=400, detail="No build in progress")

    await runner.stop_build()
    return {"status": "stopping"}


@app.get("/api/runs")
async def get_runs(limit: int = 50):
    """Get run history."""
    runs = runner.history.get_runs(limit)
    return {"runs": [r.to_dict() for r in runs]}


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str):
    """Get details of a specific run."""
    run = runner.history.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    # Calculate diff from previous if completed
    diff = None
    if run.status == "completed":
        runs = runner.history.get_runs(100)
        for i, r in enumerate(runs):
            if r.id == run_id and i + 1 < len(runs):
                prev = runs[i + 1]
                if prev.status == "completed":
                    diff = {
                        "performers_delta": run.stats.performers_processed - prev.stats.performers_processed,
                        "faces_delta": run.stats.faces_added - prev.stats.faces_added,
                    }
                break

    result = run.to_dict()
    result["diff"] = diff
    return result


@app.get("/api/versions")
async def get_versions():
    """List available database versions."""
    versions = []
    for f in sorted(DATA_DIR.glob("performers_*.db"), reverse=True):
        stat = f.stat()
        versions.append({
            "filename": f.name,
            "size_mb": round(stat.st_size / 1024 / 1024, 2),
            "created": stat.st_mtime,
        })
    return {"versions": versions}


@app.get("/api/logs")
async def stream_logs():
    """Stream live logs via SSE."""
    async def event_generator():
        queue: asyncio.Queue = asyncio.Queue()

        def log_callback(msg: str):
            try:
                queue.put_nowait(msg)
            except asyncio.QueueFull:
                pass

        runner.add_log_callback(log_callback)
        try:
            while True:
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=30)
                    yield f"data: {msg}\n\n"
                except asyncio.TimeoutError:
                    yield f"data: heartbeat\n\n"
        finally:
            runner.remove_log_callback(log_callback)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


# --- Static Files (React app) ---

# Mount static files for production
static_dir = Path(__file__).parent.parent / "web" / "dist"
if static_dir.exists():
    app.mount("/assets", StaticFiles(directory=static_dir / "assets"), name="assets")

    @app.get("/")
    async def serve_spa():
        return FileResponse(static_dir / "index.html")

    @app.get("/{path:path}")
    async def serve_spa_routes(path: str):
        # Serve index.html for client-side routing
        file_path = static_dir / path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(static_dir / "index.html")
```

**Step 4: Commit**

```bash
git add api/main.py api/build_runner.py api/run_history.py
git commit -m "feat: add FastAPI dashboard backend"
```

---

## Task 7: Create React Dashboard - Project Setup

**Files:**
- Create: `web/` directory with Vite + React + Tailwind

**Step 1: Create React app with Vite**

```bash
cd ~/code/stash-sense-trainer
npm create vite@latest web -- --template react
cd web
npm install
```

**Step 2: Install dependencies**

```bash
npm install -D tailwindcss postcss autoprefixer
npm install @tanstack/react-query react-router-dom
npx tailwindcss init -p
```

**Step 3: Configure tailwind.config.js**

Replace `web/tailwind.config.js`:

```javascript
/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
```

**Step 4: Update web/src/index.css**

Replace contents:

```css
@tailwind base;
@tailwind components;
@tailwind utilities;

body {
  @apply bg-gray-900 text-gray-100;
}
```

**Step 5: Commit**

```bash
cd ~/code/stash-sense-trainer
git add web/
git commit -m "feat: initialize React app with Vite and Tailwind"
```

---

## Task 8: Create React Dashboard - API Client

**Files:**
- Create: `web/src/api.js`

**Step 1: Create API client**

Create `web/src/api.js`:

```javascript
const API_BASE = '/api';

export async function fetchStatus() {
  const res = await fetch(`${API_BASE}/status`);
  if (!res.ok) throw new Error('Failed to fetch status');
  return res.json();
}

export async function fetchConfig() {
  const res = await fetch(`${API_BASE}/config`);
  if (!res.ok) throw new Error('Failed to fetch config');
  return res.json();
}

export async function updateConfig(config) {
  const res = await fetch(`${API_BASE}/config`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  });
  if (!res.ok) throw new Error('Failed to update config');
  return res.json();
}

export async function startBuild(params) {
  const res = await fetch(`${API_BASE}/build/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || 'Failed to start build');
  }
  return res.json();
}

export async function stopBuild() {
  const res = await fetch(`${API_BASE}/build/stop`, {
    method: 'POST',
  });
  if (!res.ok) throw new Error('Failed to stop build');
  return res.json();
}

export async function fetchRuns(limit = 50) {
  const res = await fetch(`${API_BASE}/runs?limit=${limit}`);
  if (!res.ok) throw new Error('Failed to fetch runs');
  return res.json();
}

export async function fetchRun(runId) {
  const res = await fetch(`${API_BASE}/runs/${runId}`);
  if (!res.ok) throw new Error('Failed to fetch run');
  return res.json();
}

export async function fetchVersions() {
  const res = await fetch(`${API_BASE}/versions`);
  if (!res.ok) throw new Error('Failed to fetch versions');
  return res.json();
}

export function subscribeToLogs(onMessage) {
  const eventSource = new EventSource(`${API_BASE}/logs`);
  eventSource.onmessage = (event) => {
    if (event.data !== 'heartbeat') {
      onMessage(event.data);
    }
  };
  return () => eventSource.close();
}
```

**Step 2: Commit**

```bash
git add web/src/api.js
git commit -m "feat: add React API client"
```

---

## Task 9: Create React Dashboard - Components

**Files:**
- Create: `web/src/components/StatusCard.jsx`
- Create: `web/src/components/BuildForm.jsx`
- Create: `web/src/components/RunsTable.jsx`

**Step 1: Create StatusCard.jsx**

Create `web/src/components/StatusCard.jsx`:

```jsx
export default function StatusCard({ status }) {
  if (!status) return null;

  const isRunning = status.running;

  return (
    <div className={`rounded-lg p-6 ${isRunning ? 'bg-blue-900' : 'bg-gray-800'}`}>
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold">Build Status</h2>
        <span className={`px-3 py-1 rounded-full text-sm ${
          isRunning ? 'bg-blue-500' : 'bg-gray-600'
        }`}>
          {isRunning ? 'Running' : 'Idle'}
        </span>
      </div>

      {isRunning && (
        <div className="space-y-3">
          <div>
            <div className="text-sm text-gray-400">Current Source</div>
            <div className="text-lg">{status.current_source || 'Starting...'}</div>
          </div>

          <div className="grid grid-cols-3 gap-4">
            <div>
              <div className="text-sm text-gray-400">Performers</div>
              <div className="text-2xl font-bold">
                {status.performers_processed.toLocaleString()}
              </div>
            </div>
            <div>
              <div className="text-sm text-gray-400">Faces Added</div>
              <div className="text-2xl font-bold text-green-400">
                {status.faces_added.toLocaleString()}
              </div>
            </div>
            <div>
              <div className="text-sm text-gray-400">Errors</div>
              <div className="text-2xl font-bold text-red-400">
                {status.errors}
              </div>
            </div>
          </div>

          {Object.keys(status.by_source).length > 0 && (
            <div className="mt-4">
              <div className="text-sm text-gray-400 mb-2">By Source</div>
              <div className="flex flex-wrap gap-2">
                {Object.entries(status.by_source).map(([source, count]) => (
                  <span key={source} className="px-2 py-1 bg-gray-700 rounded text-sm">
                    {source}: {count.toLocaleString()}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
```

**Step 2: Create BuildForm.jsx**

Create `web/src/components/BuildForm.jsx`:

```jsx
import { useState } from 'react';

const ALL_SOURCES = [
  { id: 'stashdb', name: 'StashDB', type: 'stash-box' },
  { id: 'theporndb', name: 'ThePornDB', type: 'stash-box' },
  { id: 'pmvstash', name: 'PMVStash', type: 'stash-box' },
  { id: 'javstash', name: 'JAVStash', type: 'stash-box' },
  { id: 'fansdb', name: 'FansDB', type: 'stash-box' },
  { id: 'babepedia', name: 'Babepedia', type: 'reference' },
  { id: 'iafd', name: 'IAFD', type: 'reference' },
  { id: 'freeones', name: 'FreeOnes', type: 'reference' },
  { id: 'boobpedia', name: 'Boobpedia', type: 'reference' },
  { id: 'thenude', name: 'TheNude', type: 'reference' },
  { id: 'afdb', name: 'AFDB', type: 'reference' },
  { id: 'pornpics', name: 'PornPics', type: 'reference' },
  { id: 'elitebabes', name: 'EliteBabes', type: 'reference' },
  { id: 'javdatabase', name: 'JavDatabase', type: 'reference' },
];

export default function BuildForm({ onSubmit, onCancel, isSubmitting }) {
  const [sources, setSources] = useState(
    ALL_SOURCES.filter(s => s.type === 'stash-box').map(s => s.id)
  );
  const [enableFaces, setEnableFaces] = useState(true);
  const [maxFacesPerSource, setMaxFacesPerSource] = useState(5);
  const [maxFacesTotal, setMaxFacesTotal] = useState(12);
  const [referenceMode, setReferenceMode] = useState('url');

  const toggleSource = (id) => {
    setSources(prev =>
      prev.includes(id) ? prev.filter(s => s !== id) : [...prev, id]
    );
  };

  const selectAll = () => setSources(ALL_SOURCES.map(s => s.id));
  const selectNone = () => setSources([]);

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit({
      sources,
      enable_faces: enableFaces,
      max_faces_per_source: maxFacesPerSource,
      max_faces_total: maxFacesTotal,
      reference_mode: referenceMode,
    });
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <div>
        <div className="flex justify-between items-center mb-3">
          <label className="block text-sm font-medium">Sources</label>
          <div className="space-x-2">
            <button type="button" onClick={selectAll} className="text-xs text-blue-400 hover:underline">
              Select All
            </button>
            <button type="button" onClick={selectNone} className="text-xs text-blue-400 hover:underline">
              Select None
            </button>
          </div>
        </div>

        <div className="space-y-3">
          <div>
            <div className="text-xs text-gray-400 mb-2">Stash-Boxes</div>
            <div className="flex flex-wrap gap-2">
              {ALL_SOURCES.filter(s => s.type === 'stash-box').map(source => (
                <label key={source.id} className="flex items-center space-x-2 bg-gray-700 px-3 py-2 rounded cursor-pointer hover:bg-gray-600">
                  <input
                    type="checkbox"
                    checked={sources.includes(source.id)}
                    onChange={() => toggleSource(source.id)}
                    className="rounded"
                  />
                  <span>{source.name}</span>
                </label>
              ))}
            </div>
          </div>

          <div>
            <div className="text-xs text-gray-400 mb-2">Reference Sites</div>
            <div className="flex flex-wrap gap-2">
              {ALL_SOURCES.filter(s => s.type === 'reference').map(source => (
                <label key={source.id} className="flex items-center space-x-2 bg-gray-700 px-3 py-2 rounded cursor-pointer hover:bg-gray-600">
                  <input
                    type="checkbox"
                    checked={sources.includes(source.id)}
                    onChange={() => toggleSource(source.id)}
                    className="rounded"
                  />
                  <span>{source.name}</span>
                </label>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="flex items-center space-x-3">
        <input
          type="checkbox"
          id="enableFaces"
          checked={enableFaces}
          onChange={(e) => setEnableFaces(e.target.checked)}
          className="rounded"
        />
        <label htmlFor="enableFaces" className="text-sm">Enable face detection (requires GPU)</label>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium mb-1">Max Faces Per Source</label>
          <input
            type="number"
            value={maxFacesPerSource}
            onChange={(e) => setMaxFacesPerSource(parseInt(e.target.value))}
            min={1}
            max={20}
            className="w-full bg-gray-700 rounded px-3 py-2"
          />
        </div>
        <div>
          <label className="block text-sm font-medium mb-1">Max Faces Total</label>
          <input
            type="number"
            value={maxFacesTotal}
            onChange={(e) => setMaxFacesTotal(parseInt(e.target.value))}
            min={1}
            max={50}
            className="w-full bg-gray-700 rounded px-3 py-2"
          />
        </div>
      </div>

      <div>
        <label className="block text-sm font-medium mb-2">Reference Site Mode</label>
        <div className="flex space-x-4">
          <label className="flex items-center space-x-2">
            <input
              type="radio"
              name="referenceMode"
              value="url"
              checked={referenceMode === 'url'}
              onChange={(e) => setReferenceMode(e.target.value)}
            />
            <span>URL Lookup (safer)</span>
          </label>
          <label className="flex items-center space-x-2">
            <input
              type="radio"
              name="referenceMode"
              value="name"
              checked={referenceMode === 'name'}
              onChange={(e) => setReferenceMode(e.target.value)}
            />
            <span>Name Lookup (broader)</span>
          </label>
        </div>
      </div>

      <div className="flex justify-end space-x-3 pt-4 border-t border-gray-700">
        <button
          type="button"
          onClick={onCancel}
          className="px-4 py-2 bg-gray-600 rounded hover:bg-gray-500"
        >
          Cancel
        </button>
        <button
          type="submit"
          disabled={isSubmitting || sources.length === 0}
          className="px-4 py-2 bg-blue-600 rounded hover:bg-blue-500 disabled:opacity-50"
        >
          {isSubmitting ? 'Starting...' : 'Start Build'}
        </button>
      </div>
    </form>
  );
}
```

**Step 3: Create RunsTable.jsx**

Create `web/src/components/RunsTable.jsx`:

```jsx
import { Link } from 'react-router-dom';

export default function RunsTable({ runs }) {
  if (!runs || runs.length === 0) {
    return <div className="text-gray-400">No runs yet</div>;
  }

  const formatDate = (iso) => {
    return new Date(iso).toLocaleString();
  };

  const formatDuration = (start, end) => {
    if (!end) return 'In progress';
    const ms = new Date(end) - new Date(start);
    const mins = Math.floor(ms / 60000);
    const hours = Math.floor(mins / 60);
    if (hours > 0) return `${hours}h ${mins % 60}m`;
    return `${mins}m`;
  };

  return (
    <div className="overflow-x-auto">
      <table className="w-full">
        <thead>
          <tr className="text-left text-gray-400 text-sm">
            <th className="pb-3">Date</th>
            <th className="pb-3">Status</th>
            <th className="pb-3">Duration</th>
            <th className="pb-3">Performers</th>
            <th className="pb-3">Faces</th>
            <th className="pb-3">Sources</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-700">
          {runs.map((run) => (
            <tr key={run.id} className="hover:bg-gray-800">
              <td className="py-3">
                <Link to={`/runs/${run.id}`} className="text-blue-400 hover:underline">
                  {formatDate(run.started_at)}
                </Link>
              </td>
              <td className="py-3">
                <span className={`px-2 py-1 rounded text-xs ${
                  run.status === 'completed' ? 'bg-green-900 text-green-300' :
                  run.status === 'running' ? 'bg-blue-900 text-blue-300' :
                  run.status === 'failed' ? 'bg-red-900 text-red-300' :
                  'bg-gray-700'
                }`}>
                  {run.status}
                </span>
              </td>
              <td className="py-3 text-gray-400">
                {formatDuration(run.started_at, run.completed_at)}
              </td>
              <td className="py-3">
                {run.stats?.performers_processed?.toLocaleString() || '-'}
              </td>
              <td className="py-3 text-green-400">
                {run.stats?.faces_added?.toLocaleString() || '-'}
              </td>
              <td className="py-3 text-sm text-gray-400">
                {run.params?.sources?.join(', ') || '-'}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
```

**Step 4: Commit**

```bash
git add web/src/components/
git commit -m "feat: add React dashboard components"
```

---

## Task 10: Create React Dashboard - Pages

**Files:**
- Create: `web/src/pages/Dashboard.jsx`
- Create: `web/src/pages/Runs.jsx`
- Create: `web/src/pages/RunDetail.jsx`
- Create: `web/src/pages/Config.jsx`
- Create: `web/src/pages/Versions.jsx`

**Step 1: Create Dashboard.jsx**

Create `web/src/pages/Dashboard.jsx`:

```jsx
import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { fetchStatus, fetchRuns, startBuild, stopBuild } from '../api';
import StatusCard from '../components/StatusCard';
import BuildForm from '../components/BuildForm';
import RunsTable from '../components/RunsTable';

export default function Dashboard() {
  const [showForm, setShowForm] = useState(false);
  const queryClient = useQueryClient();

  const { data: status } = useQuery({
    queryKey: ['status'],
    queryFn: fetchStatus,
    refetchInterval: 2000,
  });

  const { data: runsData } = useQuery({
    queryKey: ['runs'],
    queryFn: () => fetchRuns(5),
  });

  const startMutation = useMutation({
    mutationFn: startBuild,
    onSuccess: () => {
      setShowForm(false);
      queryClient.invalidateQueries(['status']);
      queryClient.invalidateQueries(['runs']);
    },
  });

  const stopMutation = useMutation({
    mutationFn: stopBuild,
    onSuccess: () => {
      queryClient.invalidateQueries(['status']);
    },
  });

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold">Dashboard</h1>
        {status?.running ? (
          <button
            onClick={() => stopMutation.mutate()}
            disabled={stopMutation.isPending}
            className="px-4 py-2 bg-red-600 rounded hover:bg-red-500 disabled:opacity-50"
          >
            Stop Build
          </button>
        ) : (
          <button
            onClick={() => setShowForm(true)}
            className="px-4 py-2 bg-blue-600 rounded hover:bg-blue-500"
          >
            Start Build
          </button>
        )}
      </div>

      <StatusCard status={status} />

      {showForm && !status?.running && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center p-4">
          <div className="bg-gray-800 rounded-lg p-6 w-full max-w-2xl max-h-[90vh] overflow-y-auto">
            <h2 className="text-xl font-semibold mb-4">Start New Build</h2>
            <BuildForm
              onSubmit={(params) => startMutation.mutate(params)}
              onCancel={() => setShowForm(false)}
              isSubmitting={startMutation.isPending}
            />
            {startMutation.isError && (
              <div className="mt-4 p-3 bg-red-900 text-red-200 rounded">
                {startMutation.error.message}
              </div>
            )}
          </div>
        </div>
      )}

      <div className="bg-gray-800 rounded-lg p-6">
        <h2 className="text-xl font-semibold mb-4">Recent Runs</h2>
        <RunsTable runs={runsData?.runs} />
      </div>
    </div>
  );
}
```

**Step 2: Create Runs.jsx**

Create `web/src/pages/Runs.jsx`:

```jsx
import { useQuery } from '@tanstack/react-query';
import { fetchRuns } from '../api';
import RunsTable from '../components/RunsTable';

export default function Runs() {
  const { data, isLoading } = useQuery({
    queryKey: ['runs', 'all'],
    queryFn: () => fetchRuns(100),
  });

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Run History</h1>

      <div className="bg-gray-800 rounded-lg p-6">
        {isLoading ? (
          <div className="text-gray-400">Loading...</div>
        ) : (
          <RunsTable runs={data?.runs} />
        )}
      </div>
    </div>
  );
}
```

**Step 3: Create RunDetail.jsx**

Create `web/src/pages/RunDetail.jsx`:

```jsx
import { useParams, Link } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { fetchRun } from '../api';

export default function RunDetail() {
  const { runId } = useParams();

  const { data: run, isLoading } = useQuery({
    queryKey: ['run', runId],
    queryFn: () => fetchRun(runId),
  });

  if (isLoading) return <div className="text-gray-400">Loading...</div>;
  if (!run) return <div className="text-red-400">Run not found</div>;

  const formatDate = (iso) => new Date(iso).toLocaleString();

  return (
    <div className="space-y-6">
      <div className="flex items-center space-x-4">
        <Link to="/runs" className="text-blue-400 hover:underline">&larr; Back</Link>
        <h1 className="text-2xl font-bold">Run: {run.id}</h1>
        <span className={`px-2 py-1 rounded text-sm ${
          run.status === 'completed' ? 'bg-green-900 text-green-300' :
          run.status === 'running' ? 'bg-blue-900 text-blue-300' :
          run.status === 'failed' ? 'bg-red-900 text-red-300' :
          'bg-gray-700'
        }`}>
          {run.status}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-6">
        <div className="bg-gray-800 rounded-lg p-6">
          <h2 className="text-lg font-semibold mb-4">Summary</h2>
          <dl className="space-y-3">
            <div>
              <dt className="text-sm text-gray-400">Started</dt>
              <dd>{formatDate(run.started_at)}</dd>
            </div>
            {run.completed_at && (
              <div>
                <dt className="text-sm text-gray-400">Completed</dt>
                <dd>{formatDate(run.completed_at)}</dd>
              </div>
            )}
            {run.output_db && (
              <div>
                <dt className="text-sm text-gray-400">Output Database</dt>
                <dd className="font-mono text-sm">{run.output_db}</dd>
              </div>
            )}
          </dl>
        </div>

        <div className="bg-gray-800 rounded-lg p-6">
          <h2 className="text-lg font-semibold mb-4">Parameters</h2>
          <dl className="space-y-3">
            <div>
              <dt className="text-sm text-gray-400">Sources</dt>
              <dd className="flex flex-wrap gap-1">
                {run.params?.sources?.map(s => (
                  <span key={s} className="px-2 py-1 bg-gray-700 rounded text-sm">{s}</span>
                ))}
              </dd>
            </div>
            <div>
              <dt className="text-sm text-gray-400">Enable Faces</dt>
              <dd>{run.params?.enable_faces ? 'Yes' : 'No'}</dd>
            </div>
            <div>
              <dt className="text-sm text-gray-400">Max Faces</dt>
              <dd>{run.params?.max_faces_per_source} per source, {run.params?.max_faces_total} total</dd>
            </div>
          </dl>
        </div>
      </div>

      <div className="bg-gray-800 rounded-lg p-6">
        <h2 className="text-lg font-semibold mb-4">Statistics</h2>
        <div className="grid grid-cols-4 gap-4">
          <div>
            <div className="text-sm text-gray-400">Performers</div>
            <div className="text-3xl font-bold">{run.stats?.performers_processed?.toLocaleString() || 0}</div>
          </div>
          <div>
            <div className="text-sm text-gray-400">Images</div>
            <div className="text-3xl font-bold">{run.stats?.images_processed?.toLocaleString() || 0}</div>
          </div>
          <div>
            <div className="text-sm text-gray-400">Faces Added</div>
            <div className="text-3xl font-bold text-green-400">{run.stats?.faces_added?.toLocaleString() || 0}</div>
          </div>
          <div>
            <div className="text-sm text-gray-400">Errors</div>
            <div className="text-3xl font-bold text-red-400">{run.stats?.errors || 0}</div>
          </div>
        </div>

        {run.stats?.by_source && Object.keys(run.stats.by_source).length > 0 && (
          <div className="mt-6">
            <h3 className="text-md font-semibold mb-3">By Source</h3>
            <div className="grid grid-cols-3 gap-3">
              {Object.entries(run.stats.by_source).map(([source, count]) => (
                <div key={source} className="bg-gray-700 rounded p-3">
                  <div className="text-sm text-gray-400">{source}</div>
                  <div className="text-xl font-bold">{count.toLocaleString()}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {run.diff && (
          <div className="mt-6 p-4 bg-blue-900/30 rounded">
            <h3 className="text-md font-semibold mb-2">Change from Previous Run</h3>
            <div className="flex space-x-6">
              <div>
                <span className="text-gray-400">Performers:</span>{' '}
                <span className={run.diff.performers_delta >= 0 ? 'text-green-400' : 'text-red-400'}>
                  {run.diff.performers_delta >= 0 ? '+' : ''}{run.diff.performers_delta.toLocaleString()}
                </span>
              </div>
              <div>
                <span className="text-gray-400">Faces:</span>{' '}
                <span className={run.diff.faces_delta >= 0 ? 'text-green-400' : 'text-red-400'}>
                  {run.diff.faces_delta >= 0 ? '+' : ''}{run.diff.faces_delta.toLocaleString()}
                </span>
              </div>
            </div>
          </div>
        )}
      </div>

      {run.error_message && (
        <div className="bg-red-900/30 border border-red-700 rounded-lg p-6">
          <h2 className="text-lg font-semibold mb-2 text-red-300">Error</h2>
          <pre className="text-sm text-red-200 whitespace-pre-wrap">{run.error_message}</pre>
        </div>
      )}
    </div>
  );
}
```

**Step 4: Create Config.jsx**

Create `web/src/pages/Config.jsx`:

```jsx
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { fetchConfig, updateConfig } from '../api';
import { useState, useEffect } from 'react';

export default function Config() {
  const queryClient = useQueryClient();
  const [localConfig, setLocalConfig] = useState(null);

  const { data, isLoading } = useQuery({
    queryKey: ['config'],
    queryFn: fetchConfig,
  });

  useEffect(() => {
    if (data) setLocalConfig(data);
  }, [data]);

  const mutation = useMutation({
    mutationFn: updateConfig,
    onSuccess: () => {
      queryClient.invalidateQueries(['config']);
    },
  });

  if (isLoading || !localConfig) return <div className="text-gray-400">Loading...</div>;

  const updateSource = (sourceName, field, value) => {
    setLocalConfig(prev => ({
      ...prev,
      sources: {
        ...prev.sources,
        [sourceName]: {
          ...prev.sources[sourceName],
          [field]: value,
        },
      },
    }));
  };

  const handleSave = () => {
    mutation.mutate(localConfig);
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold">Configuration</h1>
        <button
          onClick={handleSave}
          disabled={mutation.isPending}
          className="px-4 py-2 bg-blue-600 rounded hover:bg-blue-500 disabled:opacity-50"
        >
          {mutation.isPending ? 'Saving...' : 'Save Changes'}
        </button>
      </div>

      {mutation.isSuccess && (
        <div className="p-3 bg-green-900 text-green-200 rounded">
          Configuration saved!
        </div>
      )}

      <div className="grid grid-cols-2 gap-4">
        {Object.entries(localConfig.sources || {}).map(([name, config]) => (
          <div key={name} className="bg-gray-800 rounded-lg p-4">
            <div className="flex justify-between items-center mb-3">
              <h3 className="font-semibold">{name}</h3>
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={config.enabled !== false}
                  onChange={(e) => updateSource(name, 'enabled', e.target.checked)}
                  className="rounded"
                />
                <span className="text-sm">Enabled</span>
              </label>
            </div>

            <div className="space-y-3">
              <div>
                <label className="block text-sm text-gray-400 mb-1">Rate Limit (req/min)</label>
                <input
                  type="number"
                  value={config.rate_limit || 60}
                  onChange={(e) => updateSource(name, 'rate_limit', parseInt(e.target.value))}
                  className="w-full bg-gray-700 rounded px-3 py-2"
                />
              </div>

              <div>
                <label className="block text-sm text-gray-400 mb-1">Trust Level</label>
                <select
                  value={config.trust_level || 'medium'}
                  onChange={(e) => updateSource(name, 'trust_level', e.target.value)}
                  className="w-full bg-gray-700 rounded px-3 py-2"
                >
                  <option value="high">High</option>
                  <option value="medium">Medium</option>
                  <option value="low">Low</option>
                </select>
              </div>

              <div>
                <label className="block text-sm text-gray-400 mb-1">Max Faces</label>
                <input
                  type="number"
                  value={config.max_faces || 5}
                  onChange={(e) => updateSource(name, 'max_faces', parseInt(e.target.value))}
                  className="w-full bg-gray-700 rounded px-3 py-2"
                />
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
```

**Step 5: Create Versions.jsx**

Create `web/src/pages/Versions.jsx`:

```jsx
import { useQuery } from '@tanstack/react-query';
import { fetchVersions } from '../api';

export default function Versions() {
  const { data, isLoading } = useQuery({
    queryKey: ['versions'],
    queryFn: fetchVersions,
  });

  if (isLoading) return <div className="text-gray-400">Loading...</div>;

  const formatDate = (timestamp) => {
    return new Date(timestamp * 1000).toLocaleString();
  };

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Database Versions</h1>

      <div className="bg-gray-800 rounded-lg p-6">
        {data?.versions?.length === 0 ? (
          <div className="text-gray-400">No database versions found</div>
        ) : (
          <table className="w-full">
            <thead>
              <tr className="text-left text-gray-400 text-sm">
                <th className="pb-3">Filename</th>
                <th className="pb-3">Created</th>
                <th className="pb-3">Size</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-700">
              {data?.versions?.map((version, idx) => (
                <tr key={version.filename} className="hover:bg-gray-700">
                  <td className="py-3 font-mono text-sm">
                    {version.filename}
                    {idx === 0 && (
                      <span className="ml-2 px-2 py-1 bg-green-900 text-green-300 text-xs rounded">
                        Latest
                      </span>
                    )}
                  </td>
                  <td className="py-3 text-gray-400">{formatDate(version.created)}</td>
                  <td className="py-3">{version.size_mb} MB</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
```

**Step 6: Commit**

```bash
git add web/src/pages/
git commit -m "feat: add React dashboard pages"
```

---

## Task 11: Create React Dashboard - App Shell

**Files:**
- Modify: `web/src/App.jsx`
- Modify: `web/src/main.jsx`

**Step 1: Update main.jsx**

Replace `web/src/main.jsx`:

```jsx
import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import App from './App.jsx'
import './index.css'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5000,
      retry: 1,
    },
  },
})

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </QueryClientProvider>
  </React.StrictMode>,
)
```

**Step 2: Update App.jsx**

Replace `web/src/App.jsx`:

```jsx
import { Routes, Route, NavLink } from 'react-router-dom';
import Dashboard from './pages/Dashboard';
import Runs from './pages/Runs';
import RunDetail from './pages/RunDetail';
import Config from './pages/Config';
import Versions from './pages/Versions';

function NavItem({ to, children }) {
  return (
    <NavLink
      to={to}
      className={({ isActive }) =>
        `px-4 py-2 rounded ${isActive ? 'bg-gray-700' : 'hover:bg-gray-800'}`
      }
    >
      {children}
    </NavLink>
  );
}

export default function App() {
  return (
    <div className="min-h-screen">
      <nav className="bg-gray-800 border-b border-gray-700">
        <div className="max-w-7xl mx-auto px-4">
          <div className="flex items-center h-16">
            <div className="text-xl font-bold mr-8">Stash Sense Trainer</div>
            <div className="flex space-x-2">
              <NavItem to="/">Dashboard</NavItem>
              <NavItem to="/runs">Runs</NavItem>
              <NavItem to="/config">Config</NavItem>
              <NavItem to="/versions">Versions</NavItem>
            </div>
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto px-4 py-8">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/runs" element={<Runs />} />
          <Route path="/runs/:runId" element={<RunDetail />} />
          <Route path="/config" element={<Config />} />
          <Route path="/versions" element={<Versions />} />
        </Routes>
      </main>
    </div>
  );
}
```

**Step 3: Commit**

```bash
git add web/src/App.jsx web/src/main.jsx
git commit -m "feat: add React app shell with routing"
```

---

## Task 12: Create Dockerfile

**Files:**
- Create: `Dockerfile`

**Step 1: Create Dockerfile**

```dockerfile
# Layer 1: CUDA base + system deps
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04 AS base
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3-pip \
    nodejs npm \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    curl && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Layer 2: ML dependencies (rarely changes)
FROM base AS ml-deps
WORKDIR /app
RUN python -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu124
RUN pip install --no-cache-dir \
    deepface onnxruntime-gpu voyager tensorflow numpy Pillow opencv-python

# Layer 3: App dependencies (changes occasionally)
FROM ml-deps AS app-deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY web/package.json web/package-lock.json ./web/
RUN cd web && npm ci

# Layer 4: Build React app
FROM app-deps AS web-build
COPY web/ ./web/
RUN cd web && npm run build

# Layer 5: Runtime (changes frequently)
FROM ml-deps AS runtime
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --from=web-build /app/web/dist ./web/dist
COPY api/ ./api/

ENV DATA_DIR=/data
ENV PYTHONUNBUFFERED=1
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/api/status || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

**Step 2: Create .dockerignore**

```
.git
.venv
venv
__pycache__
*.pyc
node_modules
web/node_modules
*.db
*.voy
data/
.env
```

**Step 3: Commit**

```bash
git add Dockerfile .dockerignore
git commit -m "feat: add layered Dockerfile"
```

---

## Task 13: Create docker-compose.yml

**Files:**
- Create: `docker-compose.yml`

**Step 1: Create docker-compose.yml**

```yaml
services:
  trainer:
    build: .
    container_name: stash-sense-trainer
    ports:
      - "8080:8080"
    volumes:
      - ./data:/data
    environment:
      - STASHDB_API_KEY=${STASHDB_API_KEY}
      - THEPORNDB_API_KEY=${THEPORNDB_API_KEY:-}
      - PMVSTASH_API_KEY=${PMVSTASH_API_KEY:-}
      - JAVSTASH_API_KEY=${JAVSTASH_API_KEY:-}
      - FANSDB_API_KEY=${FANSDB_API_KEY:-}
      - FLARESOLVERR_URL=${FLARESOLVERR_URL:-http://flaresolverr:8191}
      - CHROME_CDP_URL=${CHROME_CDP_URL:-}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    depends_on:
      - flaresolverr

  flaresolverr:
    image: ghcr.io/flaresolverr/flaresolverr:latest
    container_name: flaresolverr
    environment:
      - LOG_LEVEL=info
    ports:
      - "8191:8191"
```

**Step 2: Create .env.example**

```
# Required
STASHDB_API_KEY=your_stashdb_api_key

# Optional stash-boxes
THEPORNDB_API_KEY=
PMVSTASH_API_KEY=
JAVSTASH_API_KEY=
FANSDB_API_KEY=

# Optional external services
FLARESOLVERR_URL=http://flaresolverr:8191
CHROME_CDP_URL=
```

**Step 3: Commit**

```bash
git add docker-compose.yml .env.example
git commit -m "feat: add docker-compose with FlareSolverr"
```

---

## Task 14: Create unRAID Template

**Files:**
- Create: `unraid-template.xml`

**Step 1: Create unraid-template.xml**

```xml
<?xml version="1.0"?>
<Container version="2">
  <Name>stash-sense-trainer</Name>
  <Repository>ghcr.io/yourusername/stash-sense-trainer</Repository>
  <Registry>https://ghcr.io</Registry>
  <Network>bridge</Network>
  <Privileged>false</Privileged>
  <Support/>
  <Project>https://github.com/yourusername/stash-sense-trainer</Project>
  <Overview>Database builder for Stash Sense face recognition. Scrapes performer images from StashDB and reference sites, generates face embeddings. Includes web dashboard for monitoring and control.</Overview>
  <Category>Tools: MediaApp:Video</Category>
  <WebUI>http://[IP]:[PORT:8080]/</WebUI>
  <Icon>https://raw.githubusercontent.com/yourusername/stash-sense-trainer/main/icon.png</Icon>
  <ExtraParams>--gpus all</ExtraParams>
  <DateInstalled/>
  <Config Name="Web UI Port" Target="8080" Default="8080" Mode="tcp" Description="Dashboard web interface port" Type="Port" Display="always" Required="true">8080</Config>
  <Config Name="Data Directory" Target="/data" Default="/mnt/user/appdata/stash-sense-trainer" Mode="rw" Description="Database output, run history, and configuration" Type="Path" Display="always" Required="true">/mnt/user/appdata/stash-sense-trainer</Config>
  <Config Name="StashDB API Key" Target="STASHDB_API_KEY" Default="" Description="Required. Get from stashdb.org account settings." Type="Variable" Display="always" Required="true"/>
  <Config Name="ThePornDB API Key" Target="THEPORNDB_API_KEY" Default="" Description="Optional. For ThePornDB enrichment." Type="Variable" Display="always" Required="false"/>
  <Config Name="PMVStash API Key" Target="PMVSTASH_API_KEY" Default="" Description="Optional. For PMVStash enrichment." Type="Variable" Display="always" Required="false"/>
  <Config Name="JAVStash API Key" Target="JAVSTASH_API_KEY" Default="" Description="Optional. For JAVStash enrichment." Type="Variable" Display="always" Required="false"/>
  <Config Name="FansDB API Key" Target="FANSDB_API_KEY" Default="" Description="Optional. For FansDB enrichment." Type="Variable" Display="always" Required="false"/>
  <Config Name="FlareSolverr URL" Target="FLARESOLVERR_URL" Default="http://localhost:8191" Description="Optional. Required for Babepedia, IAFD, FreeOnes scrapers. Point to your FlareSolverr container." Type="Variable" Display="always" Required="false">http://localhost:8191</Config>
  <Config Name="Chrome CDP URL" Target="CHROME_CDP_URL" Default="" Description="Optional. Required for Indexxx scraper (currently blocked)." Type="Variable" Display="always" Required="false"/>
</Container>
```

**Step 2: Commit**

```bash
git add unraid-template.xml
git commit -m "feat: add unRAID template"
```

---

## Task 15: Clean Up Source Repo - Remove Trainer Files

**Files:**
- Delete: Trainer files from `stash-face-recognition/api/`
- Create: `api/database_reader.py` (slimmed down version)

**Step 1: Create database_reader.py**

Create `~/code/stash-face-recognition/api/database_reader.py` with read-only database operations (extract from database.py - approximately 200-300 lines of query methods only, no write operations or schema migrations).

This step should extract these methods from `database.py`:
- `__init__` (simplified, no migrations)
- `get_performer`
- `get_performer_by_stashdb_id`
- `get_performers_with_faces`
- `search_performers`
- `get_stats`
- `get_face_ids_for_performer`
- `get_all_performers_with_stashdb_ids`

**Step 2: Remove trainer files from source repo**

```bash
cd ~/code/stash-face-recognition/api

# Remove scrapers
rm -f stashdb_client.py theporndb_client.py stashbox_clients.py
rm -f babepedia_client.py boobpedia_client.py iafd_client.py
rm -f freeones_client.py indexxx_client.py thenude_client.py
rm -f afdb_client.py pornpics_client.py elitebabes_client.py
rm -f javdatabase_client.py

# Remove enrichment infrastructure
rm -f enrichment_builder.py enrichment_coordinator.py enrichment_config.py
rm -f base_scraper.py flaresolverr_client.py url_normalizer.py url_fetcher.py
rm -f face_processor.py face_validator.py quality_filters.py
rm -f index_manager.py write_queue.py sources.yaml

# Remove builders
rm -f database_builder.py build_theporndb.py metadata_refresh.py
rm -f migrate_to_sqlite.py
```

**Step 3: Update imports in main.py if needed**

Verify `main.py` doesn't import any removed files. Update any imports to use `database_reader.py` instead of `database.py`.

**Step 4: Commit**

```bash
cd ~/code/stash-face-recognition
git add -A
git commit -m "refactor: remove trainer files (moved to private repo)"
```

---

## Task 16: Clean Up Source Repo - Organize Documentation

**Files:**
- Delete: Trainer docs from `stash-face-recognition/docs/plans/`
- Update: `SESSION-CONTEXT.md` for sidecar-only focus

**Step 1: Remove trainer docs**

```bash
cd ~/code/stash-face-recognition/docs/plans

# Remove trainer-specific docs
rm -f 2026-01-26-database-builder-improvements.md
rm -f 2026-01-26-concurrent-scraping-architecture.md
rm -f 2026-01-27-performer-identity-graph.md
rm -f 2026-01-26-reference-site-enrichment.md
rm -f 2026-01-29-multi-source-enrichment-design.md
rm -f 2026-01-29-multi-source-enrichment-implementation.md
rm -f 2026-01-29-scraper-orchestration-implementation.md
rm -f 2026-01-29-face-enrichment-integration.md
rm -f 2026-01-30-reference-site-scraper-integration.md
rm -f 2026-01-30-reference-site-scraper-integration-plan.md
rm -f 2026-01-27-data-sources-catalog.md
rm -f 2026-01-30-stash-sense-trainer-design.md
rm -f 2026-01-30-stash-sense-trainer-implementation.md

# Remove obsolete docs
rm -f 2026-01-26-scene-tagging-strategy.md

# Remove from docs root
rm -f ../stash-box-endpoints.md
rm -f ../url-domains-analysis.md
```

**Step 2: Update SESSION-CONTEXT.md**

Update to reflect sidecar-only focus, remove trainer commands and references.

**Step 3: Commit**

```bash
cd ~/code/stash-face-recognition
git add -A
git commit -m "docs: clean up - remove trainer docs (moved to private repo)"
```

---

## Task 17: Test Trainer Locally

**Files:**
- None (testing)

**Step 1: Copy .env file**

```bash
cd ~/code/stash-sense-trainer
cp ~/code/stash-face-recognition/api/.env api/.env
```

**Step 2: Create Python venv and install deps**

```bash
cd ~/code/stash-sense-trainer
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Step 3: Test CLI still works**

```bash
cd ~/code/stash-sense-trainer/api
python enrichment_builder.py --status
```

Expected: Should show enrichment progress (or "No scraping progress recorded yet" if fresh DB)

**Step 4: Test dashboard backend**

```bash
cd ~/code/stash-sense-trainer/api
uvicorn main:app --reload --port 8080
```

Visit http://localhost:8080/api/status - should return JSON with `{"running": false, ...}`

**Step 5: Build and test React app**

```bash
cd ~/code/stash-sense-trainer/web
npm install
npm run dev
```

Visit http://localhost:5173 - should see dashboard UI (API calls will fail until backend is running)

---

## Task 18: Test Docker Build

**Files:**
- None (testing)

**Step 1: Build Docker image**

```bash
cd ~/code/stash-sense-trainer
docker build -t stash-sense-trainer:local .
```

**Step 2: Run container**

```bash
docker run --gpus all -p 8080:8080 \
  -v $(pwd)/data:/data \
  -e STASHDB_API_KEY=${STASHDB_API_KEY} \
  stash-sense-trainer:local
```

**Step 3: Test endpoints**

```bash
curl http://localhost:8080/api/status
curl http://localhost:8080/api/versions
```

**Step 4: Visit dashboard**

Visit http://localhost:8080 - should see React dashboard

---

## Task 19: Push to GitHub

**Files:**
- None (git operations)

**Step 1: Create private repo on GitHub**

Go to https://github.com/new and create `stash-sense-trainer` as a **private** repository.

**Step 2: Add remote and push**

```bash
cd ~/code/stash-sense-trainer
git remote add origin git@github.com:yourusername/stash-sense-trainer.git
git push -u origin main
```

**Step 3: Verify**

Visit https://github.com/yourusername/stash-sense-trainer - should show private repo with all code.

---

## Summary

After completing all tasks:

1. **New private repo** `stash-sense-trainer` with:
   - All scrapers and enrichment code
   - FastAPI dashboard backend
   - React dashboard frontend
   - Docker + unRAID support
   - Organized documentation

2. **Updated public repo** `stash-face-recognition` with:
   - Sidecar API only (face recognition, recommendations)
   - Plugin code
   - Slimmed-down database reader
   - Clean documentation

3. **Working Docker image** that:
   - Runs on GPU with NVIDIA runtime
   - Serves React dashboard on port 8080
   - Can trigger builds via web UI
   - Produces versioned database files

---

*Plan complete. Ready for execution.*
