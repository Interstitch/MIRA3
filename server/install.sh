#!/bin/bash
#
# MIRA Server Install Script
# Usage: curl -sL https://raw.githubusercontent.com/Interstitch/MIRA3/master/server/install.sh | bash
#

INSTALL_DIR="${MIRA_DIR:-/opt/mira}"
REPO_URL="https://raw.githubusercontent.com/Interstitch/MIRA3/master/server"
LOG_FILE="/tmp/mira-install.log"
MIN_RAM_MB=1500
MIN_DISK_MB=2000

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Logging
exec > >(tee -a "$LOG_FILE") 2>&1
echo "=== MIRA Install Log: $(date) ===" >> "$LOG_FILE"

print_header() {
    echo -e "${BLUE}${BOLD}"
    echo "==================================="
    echo "  [◉▽◉] MIRA Server Installer"
    echo "==================================="
    echo -e "${NC}"
}

print_section() {
    echo ""
    echo -e "${BOLD}-----------------------------------${NC}"
    echo -e "${BOLD}  $1${NC}"
    echo -e "${BOLD}-----------------------------------${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

fail() {
    print_error "$1"
    echo ""
    echo "Install log saved to: $LOG_FILE"
    exit 1
}

# Check for required commands
check_requirements() {
    print_section "CHECKING REQUIREMENTS"

    local missing=()

    # Check bash version (need 4+ for mapfile)
    if [ "${BASH_VERSINFO[0]}" -lt 4 ]; then
        fail "Bash 4.0+ required (found ${BASH_VERSION})"
    fi
    print_success "Bash ${BASH_VERSION}"

    # Check curl
    if ! command -v curl &> /dev/null; then
        missing+=("curl")
    else
        print_success "curl found"
    fi

    # Check openssl
    if ! command -v openssl &> /dev/null; then
        missing+=("openssl")
    else
        print_success "openssl found"
    fi

    # Check Docker
    if ! command -v docker &> /dev/null; then
        missing+=("docker")
    else
        print_success "Docker found"
    fi

    # Check Docker Compose
    if ! docker compose version &> /dev/null 2>&1; then
        missing+=("docker-compose")
    else
        print_success "Docker Compose found"
    fi

    if [ ${#missing[@]} -gt 0 ]; then
        echo ""
        print_error "Missing required tools: ${missing[*]}"
        echo ""
        echo "Install instructions:"
        [[ " ${missing[*]} " =~ " docker " ]] && echo "  Docker: https://docs.docker.com/get-docker/"
        [[ " ${missing[*]} " =~ " docker-compose " ]] && echo "  Docker Compose: https://docs.docker.com/compose/install/"
        [[ " ${missing[*]} " =~ " curl " ]] && echo "  curl: apt install curl / yum install curl"
        [[ " ${missing[*]} " =~ " openssl " ]] && echo "  openssl: apt install openssl / yum install openssl"
        fail "Please install missing tools and re-run"
    fi
}

# Check system resources
check_resources() {
    print_section "CHECKING RESOURCES"

    # Check RAM
    local ram_kb=$(grep MemTotal /proc/meminfo 2>/dev/null | awk '{print $2}')
    if [ -n "$ram_kb" ]; then
        local ram_mb=$((ram_kb / 1024))
        if [ "$ram_mb" -lt "$MIN_RAM_MB" ]; then
            print_warning "Low RAM: ${ram_mb}MB (recommended: 2GB+)"
            echo "  Services may run slowly or fail to start"
        else
            print_success "RAM: ${ram_mb}MB"
        fi
    fi

    # Check disk space
    local disk_mb=$(df -m "$INSTALL_DIR" 2>/dev/null | awk 'NR==2 {print $4}' || df -m / | awk 'NR==2 {print $4}')
    if [ -n "$disk_mb" ]; then
        if [ "$disk_mb" -lt "$MIN_DISK_MB" ]; then
            print_warning "Low disk space: ${disk_mb}MB free (recommended: 2GB+)"
            echo "  Embedding model alone is ~500MB"
        else
            print_success "Disk space: ${disk_mb}MB available"
        fi
    fi
}

# Generate secure random string
generate_secret() {
    openssl rand -base64 32 | tr -d '/+=' | head -c 32
}

# Get various IPs
get_lan_ip() {
    hostname -I 2>/dev/null | awk '{print $1}' | grep -E '^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$' || echo ""
}

get_public_ip() {
    curl -s --max-time 5 ifconfig.me 2>/dev/null || curl -s --max-time 5 icanhazip.com 2>/dev/null || echo ""
}

get_tailscale_ip() {
    if command -v tailscale &> /dev/null; then
        tailscale ip -4 2>/dev/null || echo ""
    else
        echo ""
    fi
}

# Install Tailscale
install_tailscale() {
    echo ""
    echo "Installing Tailscale..."
    if curl -fsSL https://tailscale.com/install.sh | sh; then
        echo ""
        echo "Tailscale installed. Starting authentication..."
        echo "(A browser window may open for login)"
        echo ""
        sudo tailscale up
        sleep 3
        TAILSCALE_IP=$(get_tailscale_ip)
        if [ -n "$TAILSCALE_IP" ]; then
            print_success "Tailscale connected: $TAILSCALE_IP"
            return 0
        else
            print_warning "Tailscale installed but not connected"
            echo "  Run 'sudo tailscale up' to authenticate later"
            return 1
        fi
    else
        print_error "Tailscale installation failed"
        return 1
    fi
}

# Check/offer Tailscale
setup_tailscale() {
    if command -v tailscale &> /dev/null; then
        TAILSCALE_IP=$(get_tailscale_ip)
        if [ -n "$TAILSCALE_IP" ]; then
            print_success "Tailscale connected: $TAILSCALE_IP"
            return 0
        else
            print_warning "Tailscale installed but not connected"
            read -p "  Connect now? [Y/n]: " connect
            if [[ ! "$connect" =~ ^[Nn] ]]; then
                sudo tailscale up
                sleep 3
                TAILSCALE_IP=$(get_tailscale_ip)
                [ -n "$TAILSCALE_IP" ] && return 0
            fi
        fi
        return 1
    fi

    # Tailscale not installed - explain and offer
    print_section "RECOMMENDED: Tailscale VPN"

    echo "Tailscale is a free mesh VPN that makes connecting to this server"
    echo "easy and secure - no firewall configuration needed."
    echo ""
    echo -e "${BOLD}Without Tailscale:${NC}"
    echo "  • Open firewall ports 5432, 6333, 8200"
    echo "  • Configure port forwarding on your router"
    echo "  • Services potentially exposed to internet"
    echo ""
    echo -e "${BOLD}With Tailscale:${NC}"
    echo "  • No firewall or router changes"
    echo "  • Encrypted connections between your devices"
    echo "  • Works through NAT automatically"
    echo "  • Free for personal use (100 devices)"
    echo ""
    echo -e "${BOLD}How it works:${NC}"
    echo "  Install Tailscale on this server AND your dev machines."
    echo "  Each gets a private IP (100.x.y.z) only your devices can reach."
    echo ""
    echo "  Learn more: https://tailscale.com"
    echo ""
    echo "Options:"
    echo "  1) Install Tailscale now (recommended)"
    echo "  2) Skip - I'll configure firewall manually"
    echo ""
    read -p "Select [1]: " choice
    choice="${choice:-1}"

    if [ "$choice" = "1" ]; then
        install_tailscale
        return $?
    fi
    return 1
}

# Select server IP
select_server_ip() {
    print_section "SELECT SERVER IP"

    echo "Which IP should clients use to connect?"
    echo ""

    # Build options array
    local options=()
    local labels=()

    if [ -n "$TAILSCALE_IP" ]; then
        options+=("$TAILSCALE_IP")
        labels+=("$TAILSCALE_IP (Tailscale - recommended, no firewall needed)")
    fi

    LAN_IP=$(get_lan_ip)
    if [ -n "$LAN_IP" ]; then
        options+=("$LAN_IP")
        labels+=("$LAN_IP (LAN - same network only)")
    fi

    PUBLIC_IP=$(get_public_ip)
    if [ -n "$PUBLIC_IP" ]; then
        options+=("$PUBLIC_IP")
        labels+=("$PUBLIC_IP (Public - requires firewall ports open)")
    fi

    if [ ${#options[@]} -eq 0 ]; then
        print_warning "Could not detect any IP addresses"
        read -p "Enter IP for clients to connect: " SERVER_IP
        return
    fi

    for i in "${!labels[@]}"; do
        echo "  $((i+1))) ${labels[$i]}"
    done
    echo "  $((${#options[@]}+1))) Enter a different IP"
    echo ""
    read -p "Select [1]: " choice
    choice="${choice:-1}"

    if [ "$choice" -gt 0 ] && [ "$choice" -le "${#options[@]}" ] 2>/dev/null; then
        SERVER_IP="${options[$((choice-1))]}"
    else
        read -p "Enter IP for clients to connect: " SERVER_IP
    fi

    echo ""
    print_success "Using: $SERVER_IP"

    # Firewall guidance for public IP
    if [ "$SERVER_IP" = "$PUBLIC_IP" ] && [ -n "$PUBLIC_IP" ]; then
        echo ""
        print_warning "You selected the public IP. Make sure these ports are open:"
        echo ""
        echo "  Port 5432  - PostgreSQL"
        echo "  Port 6333  - Qdrant"
        echo "  Port 8200  - Embedding Service"
        echo ""
        if command -v ufw &> /dev/null; then
            echo "  UFW commands:"
            echo "    sudo ufw allow 5432/tcp"
            echo "    sudo ufw allow 6333/tcp"
            echo "    sudo ufw allow 8200/tcp"
        elif command -v firewall-cmd &> /dev/null; then
            echo "  Firewalld commands:"
            echo "    sudo firewall-cmd --add-port={5432,6333,8200}/tcp --permanent"
            echo "    sudo firewall-cmd --reload"
        fi
        echo ""
        read -p "Press Enter to continue..."
    fi
}

# Wait for service with retries
wait_for_service() {
    local url=$1
    local name=$2
    local max_attempts=${3:-30}
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if curl -s --max-time 2 "$url" > /dev/null 2>&1; then
            return 0
        fi
        sleep 2
        ((attempt++))
    done
    return 1
}

# Check all services are healthy
verify_services() {
    print_section "VERIFYING SERVICES"

    echo "Waiting for services to start (this may take a minute)..."
    echo ""

    local all_healthy=true

    # Check Qdrant
    echo -n "  Qdrant:     "
    if wait_for_service "http://localhost:6333/collections" "Qdrant" 15; then
        echo -e "${GREEN}ready${NC}"
    else
        echo -e "${YELLOW}starting...${NC}"
        all_healthy=false
    fi

    # Check Embedding service (takes longer - downloads model on first run)
    echo -n "  Embedding:  "
    if wait_for_service "http://localhost:8200/health" "Embedding" 30; then
        echo -e "${GREEN}ready${NC}"
    else
        echo -e "${YELLOW}starting (model download may take a few minutes)...${NC}"
        all_healthy=false
    fi

    # Check Postgres
    echo -n "  PostgreSQL: "
    if docker compose exec -T postgres pg_isready -U mira > /dev/null 2>&1; then
        echo -e "${GREEN}ready${NC}"
    else
        echo -e "${YELLOW}starting...${NC}"
        all_healthy=false
    fi

    echo ""
    if [ "$all_healthy" = false ]; then
        print_warning "Some services still starting. Check: docker compose logs -f"
    else
        print_success "All services healthy"
    fi
}

# Main installation
main() {
    print_header

    check_requirements
    check_resources

    # Create install directory
    print_section "INSTALLING"

    echo "Install directory: $INSTALL_DIR"
    if ! mkdir -p "$INSTALL_DIR" 2>/dev/null; then
        echo "Need sudo to create $INSTALL_DIR"
        sudo mkdir -p "$INSTALL_DIR"
        sudo chown "$(id -u):$(id -g)" "$INSTALL_DIR"
    fi
    cd "$INSTALL_DIR" || fail "Cannot access $INSTALL_DIR"

    # Download files
    echo "Downloading configuration files..."
    curl -sfO "$REPO_URL/docker-compose.yml" || fail "Failed to download docker-compose.yml"
    curl -sfO "$REPO_URL/.env.example" || fail "Failed to download .env.example"
    print_success "Files downloaded"

    # Configure
    if [ -f .env ]; then
        echo ""
        print_warning "Existing configuration found (.env)"
        echo "  Using existing settings. Delete .env to reconfigure."
        source .env
    else
        cp .env.example .env

        # Network setup
        print_section "NETWORK SETUP"
        TAILSCALE_IP=""
        setup_tailscale && TAILSCALE_IP=$(get_tailscale_ip)

        select_server_ip

        # Generate credentials
        echo ""
        echo "Generating secure credentials..."
        PG_PASSWORD=$(generate_secret)
        QDRANT_KEY=$(generate_secret)

        # Update .env
        sed -i "s|SERVER_IP=.*|SERVER_IP=$SERVER_IP|" .env
        sed -i "s|POSTGRES_PASSWORD=.*|POSTGRES_PASSWORD=$PG_PASSWORD|" .env
        sed -i "s|QDRANT_API_KEY=.*|QDRANT_API_KEY=$QDRANT_KEY|" .env

        print_success "Configuration saved to .env"
    fi

    # Start services
    print_section "STARTING SERVICES"
    echo "Pulling Docker images (first run may take 2-3 minutes)..."
    docker compose pull
    docker compose up -d || fail "Failed to start services"
    print_success "Containers started"

    # Verify
    verify_services

    # Read credentials for display
    source .env

    # Final output
    print_section "INSTALLATION COMPLETE"

    echo -e "${GREEN}${BOLD}Server: $SERVER_IP${NC}"
    echo ""
    echo "Services running:"
    echo "  • PostgreSQL  $SERVER_IP:5432"
    echo "  • Qdrant      $SERVER_IP:6333"
    echo "  • Embedding   $SERVER_IP:8200"

    print_section "CLIENT CONFIGURATION"

    echo "Create this file on each machine using MIRA:"
    echo ""
    echo -e "${BOLD}~/.mira/server.json${NC}"
    echo ""
    cat << EOF
{
  "version": 1,
  "central": {
    "enabled": true,
    "qdrant": { "host": "$SERVER_IP", "port": 6333, "api_key": "$QDRANT_KEY" },
    "postgres": { "host": "$SERVER_IP", "password": "$PG_PASSWORD" }
  }
}
EOF
    echo ""
    echo "Then run: chmod 600 ~/.mira/server.json"

    # Tailscale reminder
    if [[ "$SERVER_IP" == 100.* ]]; then
        print_section "TAILSCALE ON CLIENTS"
        echo "You're using Tailscale. Install it on each client machine:"
        echo ""
        echo "  curl -fsSL https://tailscale.com/install.sh | sh"
        echo "  sudo tailscale up"
        echo ""
        echo "Use the same Tailscale account to join your network."
    fi

    print_section "USEFUL COMMANDS"

    echo "  Status:     cd $INSTALL_DIR && docker compose ps"
    echo "  Logs:       cd $INSTALL_DIR && docker compose logs -f"
    echo "  Stop:       cd $INSTALL_DIR && docker compose down"
    echo "  Restart:    cd $INSTALL_DIR && docker compose restart"
    echo ""
    echo -e "  ${BOLD}View credentials again:${NC}"
    echo "    cat $INSTALL_DIR/.env"
    echo ""
    echo "Install log: $LOG_FILE"
    echo ""
}

# Run main
main
