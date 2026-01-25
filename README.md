# Stash Face Recognition

Face recognition system for identifying performers in Stash scenes.

## Project Structure

```
stash-face-recognition/
├── upstream-stashface/     # Cloned reference from HuggingFace
├── api/                    # Our FastAPI backend (to build)
├── plugin/                 # Stash plugin (to build)
├── data/                   # Our own face database
└── scripts/                # Utility scripts
```

## Quick Start

### 1. Install git-lfs (needed for database files)

```bash
sudo apt-get install git-lfs
cd upstream-stashface && git lfs pull
```

### 2. Set up Python environment

```bash
cd upstream-stashface
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Enable GPU (edit app.py)

The upstream disables CUDA by default. To enable:
```python
# Comment out or remove this line in app.py:
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
```

### 4. Run stashface

```bash
# Without VISAGE_KEY - will work but performer DB will be empty
python app.py

# With VISAGE_KEY - if you have it
VISAGE_KEY=your_key_here python app.py
```

Visit http://localhost:7860

## Important Notes

### The Encrypted Database Problem

The upstream stashface uses an **encrypted** performer database (`persons.zip`).
Without the `VISAGE_KEY` environment variable, you can't access the pre-built
performer data.

**Options:**

1. **Contact the developer** - Ask cc1234 for the key or public access
2. **Build your own database** - Create embeddings from your own performer images
3. **Use StashDB directly** - Build a scraper to download performer images and create our own index

### What Still Works Without VISAGE_KEY

- Face detection (YOLOv8)
- Face embedding generation (FaceNet512 + ArcFace)
- The Voyager index files (if pulled via git-lfs)

You just won't get performer names/metadata until we build our own database.

## Development Roadmap

- [ ] Get stashface running locally
- [ ] Verify GPU acceleration works
- [ ] Build our own performer database from Stash
- [ ] Create FastAPI backend
- [ ] Build Stash plugin
- [ ] Dockerize for Unraid deployment

## Hardware

- GPU: NVIDIA RTX 5060 Ti (16GB VRAM)
- Plenty of headroom for face recognition + other tasks (Tdarr, etc.)
