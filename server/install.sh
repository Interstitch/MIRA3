#!/bin/bash
#
# MIRA Server Install Script
# Usage: curl -sL https://raw.githubusercontent.com/Interstitch/MIRA3/master/server/install.sh | bash
#

set -e

INSTALL_DIR="${MIRA_DIR:-/opt/mira}"
REPO_URL="https://raw.githubusercontent.com/Interstitch/MIRA3/master/server"

echo "==================================="
echo "  MIRA Server Installer"
echo "==================================="
echo ""

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed."
    echo "Install it first: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check for Docker Compose
if ! docker compose version &> /dev/null; then
    echo "Error: Docker Compose is not installed."
    echo "Install it first: https://docs.docker.com/compose/install/"
    exit 1
fi

# Create install directory
echo "Installing to: $INSTALL_DIR"
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# Download files
echo "Downloading docker-compose.yml..."
curl -sO "$REPO_URL/docker-compose.yml"

echo "Downloading .env.example..."
curl -sO "$REPO_URL/.env.example"

# Configure
if [ -f .env ]; then
    echo ""
    echo "Found existing .env - keeping current configuration"
else
    cp .env.example .env

    echo ""
    echo "Configuration required:"
    echo ""

    # Get server IP
    DEFAULT_IP=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "127.0.0.1")
    read -p "Server IP address [$DEFAULT_IP]: " SERVER_IP
    SERVER_IP="${SERVER_IP:-$DEFAULT_IP}"

    # Get password
    read -s -p "PostgreSQL password: " PG_PASSWORD
    echo ""

    if [ -z "$PG_PASSWORD" ]; then
        echo "Error: Password cannot be empty"
        exit 1
    fi

    # Update .env
    sed -i "s/SERVER_IP=.*/SERVER_IP=$SERVER_IP/" .env
    sed -i "s/POSTGRES_PASSWORD=.*/POSTGRES_PASSWORD=$PG_PASSWORD/" .env

    echo ""
    echo "Configuration saved to .env"
fi

# Start services
echo ""
echo "Starting services..."
docker compose up -d

echo ""
echo "==================================="
echo "  Installation Complete!"
echo "==================================="
echo ""
echo "Services starting at:"
echo "  - PostgreSQL: $SERVER_IP:5432"
echo "  - Qdrant:     $SERVER_IP:6333"
echo "  - Embedding:  $SERVER_IP:8200"
echo ""
echo "Check status:  docker compose ps"
echo "View logs:     docker compose logs -f"
echo ""
echo "Next: Configure your MIRA clients with ~/.mira/server.json"
echo "See: https://github.com/Interstitch/MIRA3/blob/master/SERVER_SETUP.md"
