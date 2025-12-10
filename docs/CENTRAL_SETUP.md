# MIRA Central Storage Setup Guide

This guide explains how to set up MIRA with centralized storage (Qdrant + PostgreSQL) on a new development machine.

## Architecture Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Windows PC     │     │  MacBook        │     │  Codespace      │
│  + Tailscale    │     │  + Tailscale    │     │  + Tailscale    │
│  + MIRA         │     │  + MIRA         │     │  + MIRA         │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │ Tailscale VPN (encrypted)
                    ┌────────────▼────────────┐
                    │   GCP VM (mira-server)  │
                    │   100.107.224.88        │
                    │  ┌──────────────────┐   │
                    │  │ Qdrant :6333     │   │
                    │  │ Postgres :5432   │   │
                    │  └──────────────────┘   │
                    └─────────────────────────┘
```

## Prerequisites

- MIRA server already running on GCP (see Server Setup section below if not)
- Access to Tailscale account used for the server

## Step 1: Install Tailscale

### Windows

1. Download from https://tailscale.com/download/windows
2. Install and sign in with your account
3. Verify connection: `tailscale status`

### macOS

```bash
# Via Homebrew
brew install tailscale

# Or download from App Store
```

Sign in and verify: `tailscale status`

### GitHub Codespaces

```bash
# Install Tailscale
curl -fsSL https://tailscale.com/install.sh | sh

# Authenticate (use a pre-generated auth key for automation)
# Generate key at: https://login.tailscale.com/admin/settings/keys
sudo tailscale up --authkey=tskey-auth-XXXXX

# Verify
tailscale status
```

### Verify Connectivity

From any Tailscale-connected device:

```bash
# Test Qdrant
curl http://100.107.224.88:6333/collections

# Test Postgres
psql -h 100.107.224.88 -U mira -d mira
# Password: (see server.json or ask admin)
```

## Step 2: Install MIRA

### Via npm (recommended)

```bash
npm install -g mira3
```

### From source

```bash
git clone https://github.com/yourusername/MIRA3.git
cd MIRA3
npm install
npm run build
npm link
```

## Step 3: Configure Central Storage

### Create the config file

Create `~/.mira/server.json` with the following content:

```json
{
  "version": 1,
  "central": {
    "enabled": true,
    "qdrant": {
      "host": "100.107.224.88",
      "port": 6333,
      "collection": "mira",
      "timeout_seconds": 30
    },
    "postgres": {
      "host": "100.107.224.88",
      "port": 5432,
      "database": "mira",
      "user": "mira",
      "password": "YOUR_PASSWORD_HERE",
      "pool_size": 3,
      "timeout_seconds": 30
    }
  },
  "fallback": {
    "enabled": true,
    "warn_on_fallback": true
  },
  "cache": {
    "custodian_ttl_seconds": 300,
    "project_id_ttl_seconds": 3600
  }
}
```

### Set secure permissions (Unix/macOS/Linux)

```bash
chmod 600 ~/.mira/server.json
```

### Environment variable override (optional)

You can override the password via environment variable instead of storing in file:

```bash
export MIRA_POSTGRES_PASSWORD="your_password"
```

## Step 4: Verify Setup

### Check MIRA status

```bash
# MIRA will show central storage status
mira status
```

Expected output should show:
- Central storage: connected
- Qdrant: healthy
- Postgres: healthy

### Test search

```bash
# Should search across all projects
mira search "authentication"
```

## Troubleshooting

### Cannot connect to central storage

1. **Check Tailscale is running**: `tailscale status`
2. **Verify server is reachable**: `ping 100.107.224.88`
3. **Check services are running** (on server):
   ```bash
   gcloud compute ssh mira-server --zone=us-central1-a --command="sudo docker compose ps"
   ```

### "Central storage dependencies not installed"

MIRA installs central dependencies on first run if `server.json` exists. Force reinstall:

```bash
rm ~/.mira/config.json
# Next MIRA run will reinstall dependencies
```

### Permission denied on server.json

```bash
# Check permissions
ls -la ~/.mira/server.json

# Fix permissions
chmod 600 ~/.mira/server.json
```

### Fallback to local storage

If central storage is unavailable, MIRA automatically falls back to local ChromaDB + SQLite. Check logs for:
```
WARNING: Central storage unavailable, falling back to local
```

## Security Notes

1. **Network security**: All traffic goes through Tailscale (WireGuard encrypted)
2. **No public exposure**: Services bind only to Tailscale IP, not public internet
3. **File permissions**: `server.json` should be 600 (owner read/write only)
4. **Password handling**: Can use environment variable instead of file

## Server Administration

### SSH to server

```bash
gcloud compute ssh mira-server --zone=us-central1-a
```

### View logs

```bash
gcloud compute ssh mira-server --zone=us-central1-a --command="sudo docker compose logs -f"
```

### Restart services

```bash
gcloud compute ssh mira-server --zone=us-central1-a --command="cd /opt/mira && sudo docker compose restart"
```

### Manual backup

```bash
gcloud compute ssh mira-server --zone=us-central1-a --command="sudo /opt/mira/backup.sh"
```

### Check backup status

```bash
gcloud compute ssh mira-server --zone=us-central1-a --command="ls -lh /opt/mira/backups/"
```

---

## Server Setup (Reference)

If you need to set up a new server, here's the quick version:

### 1. Create GCP VM

```bash
gcloud compute instances create mira-server \
  --zone=us-central1-a \
  --machine-type=e2-small \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=30GB
```

### 2. Install Docker & Tailscale

```bash
gcloud compute ssh mira-server --zone=us-central1-a --command="
  curl -fsSL https://get.docker.com | sh
  curl -fsSL https://tailscale.com/install.sh | sh
  sudo tailscale up
"
```

### 3. Deploy services

Copy docker-compose.yml and .env to /opt/mira/, then:

```bash
gcloud compute ssh mira-server --zone=us-central1-a --command="
  cd /opt/mira
  sudo docker compose up -d
"
```

### 4. Initialize database

Run the schema SQL against Postgres to create tables.

---

## Connection Details

| Service | Host | Port | Notes |
|---------|------|------|-------|
| Qdrant | 100.107.224.88 | 6333 | HTTP API |
| Qdrant gRPC | 100.107.224.88 | 6334 | gRPC API |
| PostgreSQL | 100.107.224.88 | 5432 | Database: mira |

Tailscale IP may change if server is recreated - update server.json accordingly.
