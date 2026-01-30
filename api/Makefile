# Stash Face Recognition - Development Commands
#
# Usage: make <command>
# All commands automatically activate the virtual environment.

SHELL := /bin/bash
VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
UVICORN := $(VENV)/bin/uvicorn

# Colors for output
CYAN := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RESET := \033[0m

.PHONY: help venv install test sidecar enrich enrich-all status clean

# Default target
help:
	@echo ""
	@echo "$(CYAN)Stash Face Recognition - Available Commands$(RESET)"
	@echo ""
	@echo "$(GREEN)Setup:$(RESET)"
	@echo "  make venv        Create virtual environment"
	@echo "  make install     Install dependencies"
	@echo ""
	@echo "$(GREEN)Services:$(RESET)"
	@echo "  make sidecar     Start the FastAPI sidecar (face recognition API)"
	@echo ""
	@echo "$(GREEN)Enrichment:$(RESET)"
	@echo "  make enrich      Run face enrichment (all sources, with faces)"
	@echo "  make enrich-quick  Run enrichment without indexxx (faster)"
	@echo "  make enrich-meta   Run enrichment without face processing (metadata only)"
	@echo ""
	@echo "$(GREEN)Database:$(RESET)"
	@echo "  make status      Show database and index status"
	@echo "  make progress    Show scraper progress by source"
	@echo ""
	@echo "$(GREEN)Development:$(RESET)"
	@echo "  make test        Run all tests"
	@echo "  make test-fast   Run tests (skip slow ones)"
	@echo "  make lint        Run linter"
	@echo ""
	@echo "$(GREEN)Utilities:$(RESET)"
	@echo "  make clean       Remove cache files"
	@echo "  make shell       Open Python shell with venv"
	@echo ""

# ============================================================================
# Setup
# ============================================================================

venv:
	@echo "$(CYAN)Creating virtual environment...$(RESET)"
	python3 -m venv $(VENV)
	@echo "$(GREEN)Done. Run 'make install' to install dependencies.$(RESET)"

install: $(VENV)
	@echo "$(CYAN)Installing dependencies...$(RESET)"
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)Done.$(RESET)"

$(VENV):
	@echo "$(YELLOW)Virtual environment not found. Creating...$(RESET)"
	python3 -m venv $(VENV)

# ============================================================================
# Services
# ============================================================================

sidecar: $(VENV)
	@echo "$(CYAN)Starting FastAPI sidecar...$(RESET)"
	@echo "$(YELLOW)API will be available at http://localhost:8000$(RESET)"
	@echo "$(YELLOW)Docs at http://localhost:8000/docs$(RESET)"
	@echo ""
	$(UVICORN) main:app --host 0.0.0.0 --port 8000 --reload

sidecar-prod: $(VENV)
	@echo "$(CYAN)Starting FastAPI sidecar (production mode)...$(RESET)"
	$(UVICORN) main:app --host 0.0.0.0 --port 8000 --workers 1

# ============================================================================
# Enrichment
# ============================================================================

enrich: $(VENV)
	@echo "$(CYAN)Running face enrichment (all sources)...$(RESET)"
	@echo "$(YELLOW)Press Ctrl+C for graceful shutdown$(RESET)"
	@echo ""
	$(PYTHON) enrichment_builder.py \
		--sources stashdb,theporndb,babepedia,iafd,indexxx,boobpedia \
		--enable-faces \
		--max-faces-total 12

enrich-quick: $(VENV)
	@echo "$(CYAN)Running face enrichment (without indexxx)...$(RESET)"
	@echo "$(YELLOW)Press Ctrl+C for graceful shutdown$(RESET)"
	@echo ""
	$(PYTHON) enrichment_builder.py \
		--sources stashdb,theporndb,babepedia,iafd,boobpedia \
		--enable-faces \
		--max-faces-total 12

enrich-meta: $(VENV)
	@echo "$(CYAN)Running metadata enrichment (no face processing)...$(RESET)"
	$(PYTHON) enrichment_builder.py \
		--sources stashdb,theporndb,babepedia,iafd,boobpedia

# ============================================================================
# Database & Status
# ============================================================================

status: $(VENV)
	@echo "$(CYAN)Database Status$(RESET)"
	@echo "==============="
	@$(PYTHON) -c "import sqlite3; c = sqlite3.connect('data/performers.db').cursor(); c.execute('SELECT COUNT(*) FROM performers'); p = c.fetchone()[0]; c.execute('SELECT COUNT(*), MAX(arcface_index) FROM faces'); f, m = c.fetchone(); print(f'Performers: {p:,}\nFaces: {f:,}\nMax index: {m}')"
	@echo ""
	@echo "$(CYAN)Voyager Index$(RESET)"
	@echo "=============="
	@$(PYTHON) -c "from index_manager import IndexManager; im = IndexManager('data'); print(f'Embeddings: {im.current_index:,}')" 2>/dev/null || echo "Index not loaded"

progress: $(VENV)
	@echo "$(CYAN)Scraper Progress$(RESET)"
	@echo "================="
	@$(PYTHON) -c "import sqlite3; conn = sqlite3.connect('data/performers.db'); cur = conn.cursor(); cur.execute('SELECT source, performers_processed, faces_added FROM scrape_progress ORDER BY performers_processed DESC'); print('\n'.join([f\"{r[0]}: {r[1]} performers, {r[2]} faces\" for r in cur.fetchall()]))"

# ============================================================================
# Development
# ============================================================================

test: $(VENV)
	@echo "$(CYAN)Running tests...$(RESET)"
	$(PYTHON) -m pytest tests/ -v

test-fast: $(VENV)
	@echo "$(CYAN)Running fast tests...$(RESET)"
	$(PYTHON) -m pytest tests/ -v -x --tb=short

lint: $(VENV)
	@echo "$(CYAN)Running linter...$(RESET)"
	$(PYTHON) -m ruff check .

shell: $(VENV)
	@echo "$(CYAN)Opening Python shell...$(RESET)"
	$(PYTHON)

# ============================================================================
# Utilities
# ============================================================================

clean:
	@echo "$(CYAN)Cleaning cache files...$(RESET)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)Done.$(RESET)"
