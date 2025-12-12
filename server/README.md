# MIRA Server Setup

Deploy MIRA's central storage services on a Linux server using Docker.

## What You Get

| Service | Port | Purpose |
|---------|------|---------|
| PostgreSQL | 5432 | Stores all session data |
| Qdrant | 6333 | Vector search for semantic queries |
| Embedding Service | 8200 | Computes embeddings automatically |

## Prerequisites

- Linux server with Docker and Docker Compose installed
- At least 2GB RAM (embedding model needs ~500MB)
- Network access from your development machines

## Quick Install

```bash
curl -sL https://raw.githubusercontent.com/Interstitch/MIRA3/master/server/install.sh | bash
```

The script prompts for your server IP and PostgreSQL password, then starts everything.

## Manual Install

### Step 1: Download Files

```bash
mkdir -p /opt/mira && cd /opt/mira
curl -O https://raw.githubusercontent.com/Interstitch/MIRA3/master/server/docker-compose.yml
curl -O https://raw.githubusercontent.com/Interstitch/MIRA3/master/server/.env.example
```

### Step 2: Configure

```bash
cp .env.example .env
nano .env
```

Set these values:

```bash
POSTGRES_PASSWORD=your_secure_password_here
BIND_IP=192.168.1.100  # Your server's IP address
```

### Step 3: Start

```bash
docker compose up -d
```

First run pulls images from Docker Hub (~1GB). Takes 1-2 minutes.

## Verify

Check containers are running:

```bash
docker compose ps
```

Expected output:
```
NAME                STATUS
embedding-service   Up (healthy)
postgres            Up (healthy)
qdrant              Up
```

Test each service:

```bash
# Qdrant - should return {"result":{"collections":[]}}
curl http://localhost:6333/collections

# Embedding service - should return {"status":"healthy",...}
curl http://localhost:8200/health

# Postgres - check it accepts connections
docker compose exec postgres psql -U mira -c "SELECT 1"
```

## Connect MIRA Clients

On each development machine, create `~/.mira/server.json`:

```json
{
  "version": 1,
  "central": {
    "enabled": true,
    "qdrant": {
      "host": "YOUR_SERVER_IP",
      "port": 6333,
      "collection": "mira",
      "timeout_seconds": 30
    },
    "postgres": {
      "host": "YOUR_SERVER_IP",
      "port": 5432,
      "database": "mira",
      "user": "mira",
      "password": "your_secure_password_here",
      "pool_size": 3,
      "timeout_seconds": 30
    }
  },
  "fallback": {
    "enabled": true,
    "warn_on_fallback": true
  }
}
```

Set secure permissions:

```bash
chmod 600 ~/.mira/server.json
```

---

## Common Commands

```bash
# View logs (all services)
docker compose logs -f

# View logs (specific service)
docker compose logs -f embedding-service

# Restart all services
docker compose restart

# Stop all services
docker compose down

# Stop and remove volumes (DELETES ALL DATA)
docker compose down -v
```

## Backup & Restore

### Backup PostgreSQL

```bash
docker compose exec postgres pg_dump -U mira mira > backup_$(date +%Y%m%d).sql
```

### Restore PostgreSQL

```bash
cat backup.sql | docker compose exec -T postgres psql -U mira mira
```

### Backup Qdrant

```bash
curl -X POST "http://localhost:6333/collections/mira_sessions/snapshots"
# Snapshots saved to /opt/mira/qdrant/snapshots/
```

## Troubleshooting

### Services won't start

```bash
# Check what's wrong
docker compose logs

# Common issue: port already in use
sudo lsof -i :5432  # Check what's using the port
```

### Embedding service unhealthy

```bash
# Check logs
docker compose logs embedding-service

# Common issue: out of memory (need 2GB+ RAM)
free -h
```

### Can't connect from client

1. Check firewall allows ports 5432, 6333, 8200
2. Verify `BIND_IP` in `.env` matches your server's IP
3. Test connectivity: `curl http://YOUR_SERVER_IP:8200/health`

### Reset everything

```bash
docker compose down -v
rm -rf /opt/mira/postgres/data /opt/mira/qdrant/storage
docker compose up -d
```

## Resource Usage

| Service | RAM | Disk |
|---------|-----|------|
| PostgreSQL | 128-300MB | Grows with data |
| Qdrant | 50-100MB | Grows with vectors |
| Embedding Service | ~500MB | ~200MB (model) |

**Total:** ~1GB RAM minimum, 2GB recommended
