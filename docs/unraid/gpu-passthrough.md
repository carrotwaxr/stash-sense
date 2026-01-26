# GPU Passthrough for Unraid

## Prerequisites

- NVIDIA GPU (GTX 10-series or newer recommended)
- Unraid 6.12+

## Step 1: Install Nvidia Driver Plugin

1. Go to **Apps** → Search "Nvidia Driver"
2. Install **Nvidia-Driver** by ich777
3. Reboot Unraid

## Step 2: Verify GPU Detection

Open Unraid terminal and run:

```bash
nvidia-smi
```

You should see your GPU listed:

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 550.xx       Driver Version: 550.xx       CUDA Version: 12.x     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:XX:00.0 Off |                  N/A |
+-------------------------------+----------------------+----------------------+
```

## Step 3: Container Configuration

The Stash Sense template automatically includes GPU settings.

If configuring manually, ensure these are set:

| Setting | Value |
|---------|-------|
| Extra Parameters | `--runtime=nvidia --gpus all` |
| NVIDIA_VISIBLE_DEVICES | `all` (or specific GPU UUID) |

## Troubleshooting

### "nvidia-smi: command not found"

Driver plugin not installed or Unraid needs reboot.

```bash
# Check if plugin is installed
ls /boot/config/plugins/nvidia-driver.plg
# If missing, reinstall from Apps
```

### "No GPU detected in container"

Check container has GPU access:

```bash
docker exec stash-sense nvidia-smi
```

If this fails, verify:

1. Extra Parameters includes `--runtime=nvidia`
2. Container was recreated after driver install (not just restarted)

### "CUDA out of memory"

Another process is using the GPU. Check what's using it:

```bash
nvidia-smi
```

Look at "Processes" section. Common culprits:
- Plex hardware transcoding
- Tdarr
- Frigate
- Other ML containers

### "Failed to initialize NVML"

The Nvidia container toolkit isn't configured. Re-install the driver plugin and reboot.

### Selecting a Specific GPU (Multi-GPU Systems)

If you have multiple GPUs, specify which one to use:

```bash
# List GPU UUIDs
nvidia-smi -L

# Use specific GPU
NVIDIA_VISIBLE_DEVICES=GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
```

Or by index:
```bash
NVIDIA_VISIBLE_DEVICES=0  # First GPU
NVIDIA_VISIBLE_DEVICES=1  # Second GPU
```
