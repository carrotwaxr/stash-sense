# Configuration

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `STASH_URL` | Yes | - | URL to your Stash instance |
| `STASH_API_KEY` | Yes | - | Stash API key for fetching sprites |
| `DATA_DIR` | No | `/data` | Path to database files |
| `LOG_LEVEL` | No | `info` | Logging verbosity: debug, info, warning, error |

## Volume Mounts

| Container Path | Purpose | Mode |
|----------------|---------|------|
| `/data` | Database files | Read-only |
| `/root/.insightface` | Model weights cache | Read-write |

### Database Files

The `/data` volume should contain:

```
/data/
├── face_facenet.voy     # FaceNet embedding index
├── face_arcface.voy     # ArcFace embedding index
├── performers.json      # Performer metadata
└── manifest.json        # Database version info
```

### Model Cache

The `/root/.insightface` volume caches downloaded model weights. Without this mount, models are re-downloaded on every container start (~500MB).

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check, returns database status |
| `/database/info` | GET | Database version and stats |
| `/identify` | POST | Identify faces in a single image |
| `/identify/scene` | POST | Identify performers in a scene (uses sprite sheet) |

### Example: Health Check

```bash
curl http://localhost:5000/health
```

```json
{
  "status": "healthy",
  "database_loaded": true,
  "performer_count": 50000,
  "face_count": 150000
}
```

### Example: Scene Identification

```bash
curl -X POST http://localhost:5000/identify/scene \
  -H "Content-Type: application/json" \
  -d '{"scene_id": "123"}'
```

## Performance Tuning

### GPU Memory

The default configuration uses ~2-3GB VRAM. If you have memory constraints:

- Face detection (RetinaFace): ~1GB
- Embeddings run on CPU, don't use VRAM

### Concurrent Requests

The API handles one request at a time by default. For higher throughput, increase uvicorn workers (requires more VRAM):

```bash
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000", "--workers", "2"]
```

## Confidence Thresholds

Scores are cosine distances (lower = better match):

| Score | Interpretation |
|-------|----------------|
| < 0.4 | High confidence match |
| 0.4 - 0.6 | Likely match, verify visually |
| 0.6 - 0.8 | Possible match, needs review |
| > 0.8 | Unlikely match |
