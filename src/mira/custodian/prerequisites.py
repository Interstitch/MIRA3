"""
MIRA3 Prerequisites Module

Handles detection and checking of environment-specific prerequisites
that users have taught MIRA about (e.g., "In Codespaces, start tailscaled first").
"""

import os
import socket
import platform
import subprocess

from mira.core import log
from mira.core.database import get_db_manager
from mira.core.constants import DB_CUSTODIAN

CUSTODIAN_DB = DB_CUSTODIAN

# =============================================================================
# PREREQUISITE LEARNING PATTERNS
# =============================================================================

# Pattern matching for prerequisite statements in conversations
PREREQ_STATEMENT_PATTERNS = [
    # "In X, I need to Y"
    r"(?:in|on|when (?:on|using|in))\s+([a-zA-Z][a-zA-Z0-9_\s-]{1,30}?),?\s+(?:i |we )?(?:need|have) to\s+(.+?)(?:\.|,|$)",
    # "On X, run Y first"
    r"(?:in|on)\s+([a-zA-Z][a-zA-Z0-9_\s-]{1,30}?),?\s+(?:run|start|execute)\s+(.+?)(?:\s+first)?(?:\.|,|$)",
    # "When using X, Y must be running"
    r"when (?:using|on|in)\s+([a-zA-Z][a-zA-Z0-9_\s-]{1,30}?),?\s+(.+?)\s+(?:must|should|needs to) be (?:running|started|active)",
    # "X requires Y" / "For X, we need Y"
    r"(?:for\s+)?([a-zA-Z][a-zA-Z0-9_\s-]{1,30}?)\s+(?:requires?|needs?)\s+(.+?)(?:\s+(?:to be )?running)?(?:\.|,|$)",
    # "Before using MIRA on X, do Y"
    r"before (?:using )?(?:mira|this|anything)(?:\s+(?:on|in)\s+([a-zA-Z][a-zA-Z0-9_\s-]{1,30}?))?,?\s+(.+?)(?:\.|,|$)",
    # "Y doesn't auto-start in X" (reversed capture groups)
    r"([a-zA-Z][a-zA-Z0-9_-]+)\s+(?:doesn't|does not|won't)\s+(?:auto-?start|start automatically)\s+(?:in|on)\s+([a-zA-Z][a-zA-Z0-9_\s-]{1,30})",
    # "Remember to Y on X"
    r"(?:remember|don't forget)\s+to\s+(.+?)\s+(?:on|in|when using)\s+([a-zA-Z][a-zA-Z0-9_\s-]{1,30})",
    # "Always Y when on X"
    r"always\s+(.+?)\s+when\s+(?:on|in|using)\s+([a-zA-Z][a-zA-Z0-9_\s-]{1,30})",
    # "First thing on X is to Y"
    r"first (?:thing|step)\s+(?:on|in)\s+([a-zA-Z][a-zA-Z0-9_\s-]{1,30}?)\s+is\s+(?:to\s+)?(.+?)(?:\.|,|$)",
    # "X won't work until Y" / "X fails without Y"
    r"([a-zA-Z][a-zA-Z0-9_\s-]{1,30}?)\s+(?:won't work|fails?|can't connect|doesn't work)\s+(?:until|without|unless)\s+(.+?)(?:\.|,|$)",
]

# Command extraction patterns
PREREQ_COMMAND_PATTERNS = [
    # Backtick inline code
    r"`([^`]+)`",
    # Code blocks (bash/sh/shell/zsh)
    r"```(?:bash|sh|shell|zsh)?\n?(.*?)```",
    # "Run: X" / "Execute: X"
    r"(?:run|execute|type|use|try):\s*(.+?)(?:\n|$)",
    # "the command is X"
    r"the command is\s+(.+?)(?:\.|,|\n|$)",
    # "I usually run X"
    r"i (?:usually |always )?run\s+(.+?)(?:\.|,|\n|$)",
    # Common command prefixes
    r"\b((?:sudo\s+)?(?:docker(?:-compose)?|systemctl|tailscale[d]?|kubectl|minikube|brew|apt(?:-get)?|yum|dnf|pacman|snap|npm|yarn|pnpm|pip|cargo|make|service|launchctl)\s+[^\s]+(?:\s+[^\s]+)*)",
]

# Reason extraction patterns
PREREQ_REASON_PATTERNS = [
    r"\bfor\s+(?:the\s+)?(.{5,50}?)(?:\.|,|$)",
    r"\bbecause\s+(.{5,50}?)(?:\.|,|$)",
    r"\bso\s+(?:that\s+)?(.{5,50}?)(?:\.|,|$)",
    r"\botherwise\s+(.{5,50}?)(?:\.|,|$)",
    r"\b(?:required|needed|necessary)\s+(?:for|by)\s+(.{5,50}?)(?:\.|,|$)",
    r"\bto\s+(?:connect to|access|reach|use)\s+(.{5,50}?)(?:\.|,|$)",
]

# Check command templates for known services
PREREQ_CHECK_TEMPLATES = {
    # Network/VPN
    'tailscale': 'tailscale status 2>/dev/null | grep -q "offers\\|online"',
    'tailscaled': 'pgrep -x tailscaled >/dev/null',
    'wireguard': 'wg show >/dev/null 2>&1',
    'openvpn': 'pgrep -x openvpn >/dev/null',
    'vpn': 'ip route 2>/dev/null | grep -qE "tun|tap|wg"',
    # Containers
    'docker': 'docker info >/dev/null 2>&1',
    'docker-compose': 'docker-compose ps 2>/dev/null | grep -q Up',
    'podman': 'podman info >/dev/null 2>&1',
    'colima': 'colima status 2>/dev/null | grep -q Running',
    'orbstack': 'orb status 2>/dev/null | grep -qi running',
    'kubernetes': 'kubectl cluster-info >/dev/null 2>&1',
    'minikube': 'minikube status 2>/dev/null | grep -q Running',
    'kind': 'kind get clusters 2>/dev/null | grep -q .',
    # Databases
    'postgres': 'pg_isready -q 2>/dev/null',
    'postgresql': 'pg_isready -q 2>/dev/null',
    'mysql': 'mysqladmin ping -s 2>/dev/null',
    'mariadb': 'mysqladmin ping -s 2>/dev/null',
    'redis': 'redis-cli ping 2>/dev/null | grep -q PONG',
    'mongodb': 'mongosh --quiet --eval "db.runCommand({ping:1})" >/dev/null 2>&1',
    'memcached': 'echo stats | nc -q1 localhost 11211 >/dev/null 2>&1',
    # Web servers
    'nginx': 'pgrep -x nginx >/dev/null || systemctl is-active --quiet nginx 2>/dev/null',
    'apache': 'pgrep -x apache2 >/dev/null || pgrep -x httpd >/dev/null',
    # Auth
    'ssh-agent': 'ssh-add -l >/dev/null 2>&1',
    # Message queues
    'rabbitmq': 'rabbitmqctl status >/dev/null 2>&1',
    # Search/Vector
    'elasticsearch': 'curl -sf localhost:9200 >/dev/null',
    'qdrant': 'curl -sf localhost:6333 >/dev/null',
    'chromadb': 'curl -sf localhost:8000/api/v1/heartbeat >/dev/null',
    'weaviate': 'curl -sf localhost:8080/v1/.well-known/ready >/dev/null',
}

# Keywords that suggest a prerequisite statement
PREREQ_KEYWORDS = [
    'need to', 'have to', 'must', 'should', 'first', 'before',
    'start', 'run', 'launch', 'connect', "doesn't auto", "won't start",
    'remember to', "don't forget", 'always', 'requires', 'required'
]


def detect_environment() -> list:
    """
    Detect current environment identifiers.

    Returns list of strings that might match user-defined environment names.
    More specific identifiers come first.
    """
    envs = []

    # === User-defined (highest priority) ===
    user_env = os.environ.get('MIRA_ENVIRONMENT')
    if user_env:
        envs.extend([e.strip().lower() for e in user_env.split(',')])

    # === Cloud/Container Environments ===
    env_indicators = {
        'CODESPACES': ['codespaces', 'github codespaces'],
        'GITPOD_WORKSPACE_ID': ['gitpod'],
        'CLOUD_SHELL': ['cloud shell', 'cloudshell'],
        'REPLIT_DB_URL': ['replit'],
        'RAILWAY_ENVIRONMENT': ['railway'],
        'FLY_APP_NAME': ['fly', 'fly.io'],
        'RENDER': ['render'],
        'VERCEL': ['vercel'],
    }

    for var, identifiers in env_indicators.items():
        if os.environ.get(var):
            envs.extend(identifiers)

    # AWS - check for execution env
    aws_env = os.environ.get('AWS_EXECUTION_ENV')
    if aws_env:
        envs.extend(['aws', aws_env.lower()])

    # GCP
    if os.environ.get('GOOGLE_CLOUD_PROJECT'):
        envs.extend(['gcp', 'google cloud'])

    # Azure
    if os.environ.get('AZURE_CLIENT_ID'):
        envs.append('azure')

    # === WSL Detection ===
    wsl_distro = os.environ.get('WSL_DISTRO_NAME')
    if wsl_distro:
        envs.extend(['wsl', f'wsl-{wsl_distro.lower()}', wsl_distro.lower()])

    # === SSH/Remote Detection ===
    if os.environ.get('SSH_CONNECTION') or os.environ.get('SSH_CLIENT'):
        envs.extend(['ssh', 'remote', 'remote-ssh'])

    # === Container Detection ===
    if os.path.exists('/.dockerenv'):
        envs.extend(['docker', 'container'])
    if os.path.exists('/run/.containerenv'):
        envs.extend(['podman', 'container'])

    # === VS Code Remote ===
    if os.environ.get('VSCODE_IPC_HOOK_CLI'):
        envs.append('vscode-remote')

    # === Hostname ===
    try:
        hostname = socket.gethostname().lower()
        envs.append(hostname)
        # Also add short hostname (before first dot)
        short = hostname.split('.')[0]
        if short != hostname:
            envs.append(short)
    except Exception:
        pass

    # === OS Detection ===
    os_type = platform.system().lower()
    envs.append(os_type)

    os_aliases = {
        'darwin': ['mac', 'macos', 'osx'],
        'linux': ['linux'],
        'windows': ['windows', 'win'],
        'sunos': ['solaris', 'sunos'],
        'freebsd': ['freebsd', 'bsd'],
        'openbsd': ['openbsd', 'bsd'],
    }
    envs.extend(os_aliases.get(os_type, []))

    # === Linux Distro Detection ===
    if os_type == 'linux':
        distro_files = {
            '/etc/arch-release': 'arch',
            '/etc/debian_version': 'debian',
            '/etc/fedora-release': 'fedora',
            '/etc/redhat-release': 'rhel',
            '/etc/gentoo-release': 'gentoo',
            '/etc/alpine-release': 'alpine',
            '/etc/nixos': 'nixos',
        }
        for path, name in distro_files.items():
            if os.path.exists(path):
                envs.append(name)
                break

    # === Terminal Detection ===
    term = os.environ.get('TERM_PROGRAM', '').lower()
    if term:
        envs.append(term)

    # === Local vs Remote ===
    remote_indicators = {'ssh', 'remote', 'codespaces', 'gitpod', 'container'}
    if not any(e in remote_indicators for e in envs):
        envs.append('local')

    # Dedupe while preserving order
    seen = set()
    result = []
    for e in envs:
        if e and e not in seen:
            seen.add(e)
            result.append(e)

    return result


def get_applicable_prerequisites(detected_envs: list = None) -> list:
    """
    Get prerequisites that apply to the current environment.

    Args:
        detected_envs: List of environment identifiers. If None, auto-detect.

    Returns:
        List of prerequisite dicts sorted by confidence.
    """
    if detected_envs is None:
        detected_envs = detect_environment()

    db = get_db_manager()

    # Get all prerequisites with reasonable confidence
    try:
        rows = db.execute_read(CUSTODIAN_DB, """
            SELECT environment, action, command, check_command, reason,
                   confidence, frequency, suppressed
            FROM prerequisites
            WHERE confidence >= 0.5 AND suppressed = 0
            ORDER BY confidence DESC, frequency DESC
        """)
    except Exception as e:
        log(f"Error fetching prerequisites: {e}")
        return []

    applicable = []
    detected_set = set(detected_envs)

    for row in rows:
        prereq_env = row['environment'].lower()

        # Check for matches
        matched = False

        # Exact match
        if prereq_env in detected_set:
            matched = True

        # Partial/fuzzy match (e.g., "codespaces" matches "github codespaces")
        if not matched:
            for detected in detected_envs:
                if prereq_env in detected or detected in prereq_env:
                    matched = True
                    break

        # "all" or "always" environment matches everything
        if prereq_env in ('all', 'always', 'any', 'everywhere'):
            matched = True

        if matched:
            applicable.append({
                'environment': row['environment'],
                'action': row['action'],
                'command': row['command'],
                'check_command': row['check_command'],
                'reason': row['reason'],
                'confidence': row['confidence'],
            })

    return applicable


def check_prerequisites_and_alert() -> list:
    """
    Check all applicable prerequisites and generate alerts for unmet ones.

    Returns list of alert dicts for inclusion in mira_init response.
    """
    applicable = get_applicable_prerequisites()
    alerts = []

    # Cache check results to avoid running same command multiple times
    check_cache = {}

    for prereq in applicable:
        check_cmd = prereq.get('check_command')

        # If we have a check command, verify the prerequisite
        if check_cmd:
            # Use cached result if we've already run this command
            if check_cmd in check_cache:
                if check_cache[check_cmd]:
                    continue  # Prerequisite met
            else:
                try:
                    result = subprocess.run(
                        check_cmd,
                        shell=True,
                        capture_output=True,
                        timeout=1  # Quick timeout - don't block startup
                    )
                    check_cache[check_cmd] = (result.returncode == 0)
                    if result.returncode == 0:
                        # Prerequisite met, no alert needed
                        continue
                except subprocess.TimeoutExpired:
                    # Check timed out - treat as unmet
                    check_cache[check_cmd] = False
                except Exception as e:
                    log(f"Error checking prerequisite '{prereq['action']}': {e}")
                    check_cache[check_cmd] = False
                    continue

        # Prerequisite not met (or no check command) - generate alert
        priority = 'high' if prereq.get('confidence', 0) > 0.75 else 'medium'

        alert = {
            'type': 'prerequisite',
            'priority': priority,
            'message': prereq['action'],
            'environment': prereq['environment'],
        }

        if prereq.get('command'):
            alert['suggestion'] = f"Run: {prereq['command']}"

        if prereq.get('reason'):
            alert['context'] = prereq['reason']

        alerts.append(alert)

    return alerts
