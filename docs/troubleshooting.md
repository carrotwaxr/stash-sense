# Troubleshooting

## Container Issues

### Container won't start

**Check logs:**
```bash
docker logs stash-sense
```

**Common causes:**

- Missing database files - ensure `/data` contains the `.voy` files
- GPU driver issues - try without GPU first to isolate
- Port conflict - change from 5000 to another port

### "Database not loaded"

The container started but can't find database files.

```bash
# Check what's in the data volume
docker exec stash-sense ls -la /data
```

Should show:
```
face_facenet.voy
face_arcface.voy
performers.json
manifest.json
```

If empty, the volume mount is misconfigured or database wasn't extracted.

### Health check failing

```bash
# Test manually
curl http://localhost:5000/health

# Check container status
docker ps -a | grep stash-sense
```

If container is restarting, check logs for the error.

---

## GPU Issues

### "CUDA not available"

The container fell back to CPU mode.

**Verify GPU is accessible:**
```bash
docker exec stash-sense nvidia-smi
```

If this fails:

1. Ensure nvidia-container-toolkit is installed
2. Container needs `--runtime=nvidia --gpus all`
3. Restart Docker daemon after toolkit install

### "CUDA out of memory"

GPU memory exhausted. Check what's using it:

```bash
nvidia-smi
```

Solutions:
- Stop other GPU-using containers temporarily
- Use CPU mode (slower but works)

### Slow performance on GPU

If GPU is detected but still slow:

1. Check GPU utilization during requests: `watch nvidia-smi`
2. Ensure models aren't being re-downloaded (mount model cache volume)
3. First request is slow (model loading) - subsequent requests should be faster

---

## Connection Issues

### Plugin can't reach sidecar

**From Stash container, test connectivity:**
```bash
docker exec stash curl http://stash-sense:5000/health
```

**Common fixes:**

- Use container name if on same Docker network
- Use host IP if on different networks
- Check firewall isn't blocking port 5000

### Sidecar can't reach Stash

**Test from sidecar:**
```bash
docker exec stash-sense curl http://stash:9999/graphql
```

**Common fixes:**

- Verify STASH_URL is correct
- Verify STASH_API_KEY is valid
- Check Stash is running and accessible

---

## Recognition Issues

### No faces detected

- **Low quality sprites**: Regenerate with higher quality settings
- **No clear face shots**: Some scenes don't have good face visibility
- **Very small faces**: Detection has minimum size threshold

### Wrong matches

- **Low confidence score**: Scores > 0.6 are less reliable
- **Similar looking performers**: Some faces are genuinely similar
- **Database coverage**: Performer may not be in StashDB or have poor reference images

### Performer matched but not in library

This is expected behavior - the face matched someone in StashDB who you haven't added to your Stash yet. Options:

1. Click "View on StashDB" to verify the match
2. Add the performer to your Stash manually
3. Use Stash's "Identify" feature to import from StashDB

---

## Database Issues

### Corrupt database files

If you see index errors or crashes on load:

```bash
# Remove and re-download
rm -rf /path/to/data/*
# Re-extract from release
tar -xzf stash-sense-db.tar.gz -C /path/to/data/
```

### Outdated database

Check current version:
```bash
curl http://localhost:5000/database/info
```

Compare with latest release on GitHub. New databases include more performers and improved embeddings.

---

## Getting Help

If you're stuck:

1. Check container logs: `docker logs stash-sense`
2. Test health endpoint: `curl http://localhost:5000/health`
3. Verify database files exist and have correct permissions
4. Open an issue on GitHub with:
   - Container logs
   - Health endpoint response
   - Docker/Unraid version
   - GPU model (if applicable)
