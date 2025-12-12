# MIRA Server Setup

This guide walks you through deploying MIRA's central storage services on your own server.

## Why Run a Server?

Without a server, MIRA uses local SQLite with keyword search. Adding a server enables:

- **Semantic search** - Find conversations by meaning, not just keywords
- **Cross-project search** - Search across all your projects at once
- **Cross-machine sync** - Access your history from any machine

## What Gets Deployed

| Service | Port | Purpose |
|---------|------|---------|
| PostgreSQL | 5432 | Stores all session data |
| Qdrant | 6333 | Vector database for semantic search |
| Embedding Service | 8200 | Computes embeddings automatically |

## Requirements

- Linux server (or any machine that can run Docker)
- Docker and Docker Compose
- 2GB RAM minimum
- Network access from your dev machines (LAN, VPN, or Tailscale)

## Quick Start

### 1. Get the server files

```bash
mkdir -p /opt/mira && cd /opt/mira

# Download the required files
curl -O https://raw.githubusercontent.com/Interstitch/MIRA3/master/server/docker-compose.yml
curl -O https://raw.githubusercontent.com/Interstitch/MIRA3/master/server/init.sql
curl -O https://raw.githubusercontent.com/Interstitch/MIRA3/master/server/.env.example
```

That's it - just 3 files. The embedding service image is pulled automatically from Docker Hub.

### 2. Configure

```bash
cp .env.example .env
nano .env  # or vim, whatever you prefer
```

Set these two values:

```bash
POSTGRES_PASSWORD=pick_a_strong_password
TAILSCALE_IP=192.168.1.100  # Your server's IP address
```

### 3. Start

```bash
docker compose up -d
```

First run downloads images and builds the embedding service (~2-5 min).

### 4. Verify

```bash
# All containers running?
docker compose ps

# Services responding?
curl http://localhost:6333/collections        # Qdrant
curl http://localhost:8200/health             # Embedding service
```

### 5. Configure your MIRA clients

On each machine where you use MIRA, create `~/.mira/server.json`:

```json
{
  "version": 1,
  "central": {
    "enabled": true,
    "qdrant": {
      "host": "YOUR_SERVER_IP",
      "port": 6333,
      "collection": "mira"
    },
    "postgres": {
      "host": "YOUR_SERVER_IP",
      "port": 5432,
      "database": "mira",
      "user": "mira",
      "password": "your_password_here"
    }
  }
}
```

```bash
chmod 600 ~/.mira/server.json
```

## Troubleshooting

**Can't connect from client?**
- Check firewall allows ports 5432, 6333, 8200
- Verify the IP in `.env` matches your server's actual IP
- Test: `curl http://YOUR_SERVER_IP:8200/health`

**Services won't start?**
- Check logs: `docker compose logs`
- Check RAM: `free -h` (need 2GB+)

**Need to start fresh?**
```bash
docker compose down -v
docker compose up -d
```

## More Details

See [server/README.md](server/README.md) for:
- Detailed API documentation
- Backup and restore procedures
- Resource usage breakdown
- Advanced configuration options
