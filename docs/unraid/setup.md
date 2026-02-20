# Unraid Setup

## Prerequisites

- Unraid 6.12+
- [Nvidia Driver plugin](gpu-passthrough.md) installed (for GPU acceleration)
- Stash running on Unraid or accessible from Unraid

## Option 1: XML Template (Recommended)

### 1. Download the template

Copy [stash-sense.xml](stash-sense.xml) to your Unraid flash drive:

```
/boot/config/plugins/dockerMan/templates-user/stash-sense.xml
```

### 2. Add container

Go to **Docker → Add Container → Template → User Templates → stash-sense**

### 3. Configure

| Setting | Value |
|---------|-------|
| Stash URL | `http://stash:9999` or your Stash IP/hostname |
| Stash API Key | From Stash: Settings → Security → API Key |
| Data Directory | Path to extracted database files |

### 4. Apply

Click **Apply**. The container will download and start.

---

## Option 2: Docker Compose

If you prefer docker-compose on Unraid:

### 1. Create directory

```bash
mkdir -p /mnt/user/appdata/stash-sense
```

### 2. Create docker-compose.yml

```yaml
version: "3.8"

services:
  stash-sense:
    image: carrotwaxr/stash-sense:latest
    container_name: stash-sense
    runtime: nvidia
    restart: unless-stopped
    ports:
      - "6960:5000"
    environment:
      - STASH_URL=http://stash:9999
      - STASH_API_KEY=your-api-key
      - NVIDIA_VISIBLE_DEVICES=all
    volumes:
      - /mnt/user/appdata/stash-sense/data:/data:ro
      - /mnt/user/appdata/stash-sense/models:/root/.insightface
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### 3. Start

```bash
cd /mnt/user/appdata/stash-sense
docker compose up -d
```

---

## Database Setup

Download the database release and extract:

```bash
cd /mnt/user/appdata/stash-sense/data
wget https://github.com/carrotwaxr/stash-sense/releases/download/db-latest/stash-sense-db.tar.gz
tar -xzf stash-sense-db.tar.gz
rm stash-sense-db.tar.gz
```

You should have:
```
data/
├── face_facenet.voy
├── face_arcface.voy
├── performers.json
└── manifest.json
```

---

## Networking

If Stash runs on the same Unraid server, use Docker's internal networking:

- **Stash URL:** `http://stash:9999` (if Stash container is named "stash")
- Or use Unraid's IP: `http://192.168.1.x:9999`

---

## Next Steps

- [GPU Passthrough](gpu-passthrough.md) - ensure GPU is working
- [Plugin Setup](../plugin.md) - install the Stash plugin
- [Troubleshooting](../troubleshooting.md) - if something's not working
