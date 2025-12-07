"""
MIRA3 Codebase Concepts Module

Extracts and tracks key concepts about the codebase from conversation analysis:
- Architectural components (Python backend, Node.js frontend, etc.)
- Module purposes (what each file does)
- Technology roles (ChromaDB for vector search, etc.)
- Integration patterns (JSON-RPC over stdio, etc.)
- Design patterns and conventions
- User-provided facts and rules about the codebase

Uses centralized db_manager for thread-safe writes during parallel ingestion.
"""

import json
import re
import hashlib
from datetime import datetime
from typing import Optional, List, Dict, Tuple
from collections import Counter

from .utils import log
from .db_manager import get_db_manager


# Database name for concepts
CONCEPTS_DB = "concepts.db"

# Concept types
CONCEPT_COMPONENT = "component"       # Major architectural pieces
CONCEPT_MODULE = "module"             # Specific files and their purposes
CONCEPT_TECHNOLOGY = "technology"     # Tech stack with context/roles
CONCEPT_INTEGRATION = "integration"   # How things connect
CONCEPT_PATTERN = "pattern"           # Design patterns and conventions
CONCEPT_FACT = "fact"                 # User-provided facts about the codebase
CONCEPT_RULE = "rule"                 # User-provided rules/conventions

# Schema for concepts database
CONCEPTS_SCHEMA = """
-- Main concepts table
CREATE TABLE IF NOT EXISTS codebase_concepts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_path TEXT NOT NULL,
    concept_type TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    related_files TEXT,
    related_concepts TEXT,
    metadata TEXT,
    frequency INTEGER DEFAULT 1,
    confidence REAL DEFAULT 0.5,
    source_sessions TEXT,
    first_seen TEXT,
    last_updated TEXT,
    UNIQUE(project_path, concept_type, name)
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_concepts_project ON codebase_concepts(project_path);
CREATE INDEX IF NOT EXISTS idx_concepts_type ON codebase_concepts(concept_type);
CREATE INDEX IF NOT EXISTS idx_concepts_confidence ON codebase_concepts(confidence DESC);
CREATE INDEX IF NOT EXISTS idx_concepts_name ON codebase_concepts(name);

-- FTS for concept search
CREATE VIRTUAL TABLE IF NOT EXISTS concepts_fts USING fts5(
    name,
    description,
    content='codebase_concepts',
    content_rowid='id'
);
"""


def init_concepts_db():
    """Initialize the codebase concepts database."""
    db = get_db_manager()
    db.init_schema(CONCEPTS_DB, CONCEPTS_SCHEMA)
    log("Concepts database initialized")


class ConceptExtractor:
    """
    Extract codebase concepts from conversation content.

    Looks for:
    - Definitional statements ("X handles Y", "X is responsible for Y")
    - File purpose descriptions ("handlers.py - RPC request handlers")
    - Technology roles ("use ChromaDB for vector search")
    - Integration patterns ("communicates via JSON-RPC")
    - User-provided facts and rules
    """

    # Patterns for definitional statements (component detection)
    DEFINITIONAL_PATTERNS = [
        (r"(?:The\s+)?(\w+\s+(?:backend|frontend|server|daemon|service|layer))\s+(?:handles?|manages?|provides?|is responsible for)\s+([^.!?\n]{15,120})", 0.75),
        (r"(?:The\s+)?(\w+\s+(?:component|module|system))\s+(?:is the|serves as|acts as)\s+([^.!?\n]{15,100})", 0.7),
    ]

    # File purpose patterns
    FILE_PURPOSE_PATTERNS = [
        (r"(\w+\.(?:py|tsx|ts|jsonl|json|jsx|js|go|rs|java|rb))\s*[-:–]\s*([^.!?\n]{10,120})", 0.75),
        (r"(\w+\.(?:py|tsx|ts|jsonl|json|jsx|js|go|rs|java|rb))\s+(?:contains?|implements?|defines?|handles?|provides?)\s+([^.!?\n]{10,100})", 0.7),
        (r"(?:The\s+)?(\w+\.(?:py|tsx|ts|jsx|js))\s+file\s+(?:does|handles?|contains?)\s+([^.!?\n]{10,100})", 0.7),
    ]

    # Technology role patterns
    TECHNOLOGY_PATTERNS = [
        (r"(?:use|using|uses)\s+(ChromaDB|Redis|PostgreSQL|MongoDB|SQLite|Elasticsearch|sentence-transformers?|watchdog|FastAPI|Express|React|Vue|Angular)\s+(?:for|to)\s+([^.!?\n]{10,80})", 0.85),
        (r"([\w\-]+)\s+(?:database|library|framework|package)\s+(?:provides?|enables?|handles?)\s+([^.!?\n]{15,80})", 0.7),
    ]

    # Integration/communication patterns
    INTEGRATION_PATTERNS = [
        (r"(\w+(?:\s+(?:backend|frontend|server|client|layer))?)\s+communicates?\s+with\s+(\w+(?:\s+(?:backend|frontend|server|client|layer))?)\s+(?:via|using|over)\s+([^.!?\n]{5,60})", 0.8),
        (r"(\w+(?:\s+server)?)\s+(?:spawns|calls|invokes)\s+(\w+(?:\s+(?:backend|process|daemon))?)", 0.75),
    ]

    # Design pattern patterns
    PATTERN_PATTERNS = [
        (r"(?:we use|uses?|follows?|implements?)\s+(?:the\s+)?([\w\-]+(?:\s+[\w\-]+)?)\s+pattern", 0.75),
        (r"(two-layer|microservice|monolithic|event-driven|serverless|layered)\s+architecture", 0.8),
        (r"([\w\-]+)-based\s+(?:design|approach)", 0.7),
    ]

    # User-provided fact patterns
    FACT_PATTERNS = [
        (r"(?:This|The)\s+(?:codebase|project)\s+uses?\s+([^.!?\n]{10,100})", 0.8),
        (r"(?:Important|Note):\s*([^.!?\n]{20,150})", 0.85),
        (r"(?:Key\s+point|Key\s+thing|Remember):\s*([^.!?\n]{20,150})", 0.85),
    ]

    # User-provided rule patterns
    RULE_PATTERNS = [
        (r"^Always\s+([^.!?\n]{15,100})", 0.85),
        (r"^Never\s+([^.!?\n]{15,100})", 0.85),
        (r"We\s+(?:always|never)\s+([^.!?\n]{15,100})", 0.8),
        (r"(?:Convention|Rule|Standard):\s*([^.!?\n]{15,150})", 0.9),
    ]

    # Known technology names
    KNOWN_TECHNOLOGIES = {
        'chromadb', 'chroma', 'faiss', 'sqlite', 'postgresql', 'postgres', 'mysql', 'mongodb',
        'redis', 'elasticsearch', 'pinecone', 'weaviate', 'milvus', 'qdrant',
        'react', 'vue', 'angular', 'svelte', 'next', 'nuxt', 'remix', 'astro',
        'express', 'fastapi', 'django', 'flask', 'fastify', 'koa', 'hono',
        'typescript', 'python', 'javascript', 'golang', 'rust', 'java',
        'node', 'nodejs', 'deno', 'bun',
        'docker', 'kubernetes', 'k8s',
        'sentence-transformers', 'transformers', 'openai', 'anthropic',
        'watchdog', 'celery', 'rabbitmq', 'kafka',
        'json-rpc', 'grpc', 'rest', 'graphql', 'websocket', 'websockets',
        'mcp', 'stdio', 'ipc',
    }

    def __init__(self, project_path: str = ""):
        self.project_path = project_path
        self._file_mentions = Counter()

    def extract_from_message(self, content: str, role: str, session_id: str) -> List[Dict]:
        """Extract concept candidates from a single message."""
        candidates = []

        if not content or len(content) < 20:
            return candidates

        self._track_file_mentions(content)

        if role == 'user':
            candidates.extend(self._extract_facts(content, session_id))
            candidates.extend(self._extract_rules(content, session_id))

        confidence_boost = 0.1 if role == 'assistant' else 0.0

        candidates.extend(self._extract_components(content, session_id, confidence_boost))
        candidates.extend(self._extract_file_purposes(content, session_id, confidence_boost))
        candidates.extend(self._extract_technologies(content, session_id, confidence_boost))
        candidates.extend(self._extract_integrations(content, session_id, confidence_boost))
        candidates.extend(self._extract_patterns(content, session_id, confidence_boost))

        return candidates

    def _track_file_mentions(self, content: str):
        """Track which files are mentioned in conversation."""
        file_pattern = r'(\w+\.(?:py|tsx|ts|jsonl|json|jsx|js|go|rs|java|rb|yaml|yml|md|sql))'
        for match in re.finditer(file_pattern, content):
            filename = match.group(1)
            self._file_mentions[filename] += 1

    def _extract_components(self, content: str, session_id: str, confidence_boost: float) -> List[Dict]:
        """Extract architectural component concepts."""
        candidates = []

        for pattern, base_confidence in self.DEFINITIONAL_PATTERNS:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                name = match.group(1).strip()
                description = match.group(2).strip()

                if not self._is_valid_component_name(name):
                    continue
                if not self._is_valid_description(description):
                    continue

                candidates.append({
                    'concept_type': CONCEPT_COMPONENT,
                    'name': self._normalize_name(name),
                    'description': description,
                    'confidence': min(0.95, base_confidence + confidence_boost),
                    'session_id': session_id,
                })

        return candidates

    def _extract_file_purposes(self, content: str, session_id: str, confidence_boost: float) -> List[Dict]:
        """Extract file/module purpose concepts."""
        candidates = []
        skip_filenames = {'node.js', 'vue.js', 'next.js', 'nuxt.js', 'express.js', 'react.js', 'angular.js'}

        for pattern, base_confidence in self.FILE_PURPOSE_PATTERNS:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                filename = match.group(1).strip()
                purpose = match.group(2).strip()

                if filename.lower() in skip_filenames:
                    continue

                if not self._is_valid_description(purpose):
                    continue

                candidates.append({
                    'concept_type': CONCEPT_MODULE,
                    'name': filename,
                    'description': purpose,
                    'confidence': min(0.95, base_confidence + confidence_boost),
                    'session_id': session_id,
                    'metadata': {'file': filename}
                })

        return candidates

    def _extract_technologies(self, content: str, session_id: str, confidence_boost: float) -> List[Dict]:
        """Extract technology role concepts."""
        candidates = []
        skip_tech_names = {
            'daemon', 'server', 'client', 'backend', 'frontend', 'layer',
            'component', 'module', 'service', 'handler', 'the', 'a', 'an',
            'it', 'this', 'that', 'to', 'and', 'or', 'but', 'for', 'with',
            'blocks', 'case', 'timestamps', 'will', 'i', 'archives', '-',
            'be', 'is', 'was', 'are', 'were', 'been', 'being', 'have', 'has',
        }

        for pattern, base_confidence in self.TECHNOLOGY_PATTERNS:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                tech_name = match.group(1).strip()
                role = match.group(2).strip()

                if tech_name.lower() in skip_tech_names:
                    continue

                if len(tech_name) < 3:
                    continue

                if not tech_name[0].isalpha():
                    continue

                confidence = base_confidence + confidence_boost
                if tech_name.lower() in self.KNOWN_TECHNOLOGIES:
                    confidence = min(0.95, confidence + 0.15)
                else:
                    if len(role) < 15:
                        continue
                    confidence = min(confidence, 0.6)

                if not self._is_valid_description(role):
                    continue

                candidates.append({
                    'concept_type': CONCEPT_TECHNOLOGY,
                    'name': tech_name,
                    'description': role,
                    'confidence': confidence,
                    'session_id': session_id,
                })

        return candidates

    def _extract_integrations(self, content: str, session_id: str, confidence_boost: float) -> List[Dict]:
        """Extract integration/communication pattern concepts."""
        candidates = []
        skip_words = {
            'the', 'a', 'an', 'and', 'or', 'to', 'in', 'on', 'at', 'for', 'with',
            'it', 'this', 'that', 'any', 'all', 'needed', 'work', 'tool', 'api',
            'never', 'always', 'extraction', 'rpc', 'js', 'py',
        }

        for pattern, base_confidence in self.INTEGRATION_PATTERNS:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                groups = match.groups()

                if len(groups) >= 3:
                    from_component = groups[0].strip()
                    to_component = groups[1].strip()
                    mechanism = groups[2].strip()

                    name = f"{from_component} -> {to_component}"
                    description = f"Communicates via {mechanism}"
                elif len(groups) == 2:
                    from_component = groups[0].strip()
                    to_component = groups[1].strip()

                    name = f"{from_component} -> {to_component}"
                    description = "Connected"
                else:
                    continue

                if from_component.lower() in skip_words or to_component.lower() in skip_words:
                    continue

                if len(from_component) < 3 or len(to_component) < 3:
                    continue

                candidates.append({
                    'concept_type': CONCEPT_INTEGRATION,
                    'name': name,
                    'description': description,
                    'confidence': min(0.95, base_confidence + confidence_boost),
                    'session_id': session_id,
                    'metadata': {
                        'from': from_component,
                        'to': to_component,
                        'mechanism': mechanism if len(groups) >= 3 else None
                    }
                })

        return candidates

    def _extract_patterns(self, content: str, session_id: str, confidence_boost: float) -> List[Dict]:
        """Extract design pattern concepts."""
        candidates = []
        skip_prefixes = ['- ', '* ', '# ', '## ', '### ']

        for pattern, base_confidence in self.PATTERN_PATTERNS:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                pattern_name = match.group(1).strip()
                full_match = match.group(0).strip()

                if any(full_match.startswith(p) for p in skip_prefixes):
                    continue

                if len(pattern_name) < 5 or '{' in pattern_name or '(' in pattern_name:
                    continue

                skip_pattern_names = {
                    'the', 'a', 'an', 'this', 'that', 'current', 'new', 'old',
                    'error', 'implement', 'basic', 'simple', 'complex', 'general',
                }
                if pattern_name.lower() in skip_pattern_names:
                    continue

                candidates.append({
                    'concept_type': CONCEPT_PATTERN,
                    'name': pattern_name.title(),
                    'description': full_match[:200],
                    'confidence': min(0.95, base_confidence + confidence_boost),
                    'session_id': session_id,
                })

        return candidates

    def _extract_facts(self, content: str, session_id: str) -> List[Dict]:
        """Extract user-provided facts about the codebase."""
        candidates = []

        for pattern, base_confidence in self.FACT_PATTERNS:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                groups = match.groups()

                if len(groups) >= 2:
                    subject = groups[0].strip()
                    predicate = groups[1].strip()
                    fact_text = f"{subject}: {predicate}"
                else:
                    fact_text = groups[0].strip()

                if not self._is_valid_description(fact_text):
                    continue

                fact_lower = fact_text.lower()
                if any(garbage in fact_lower for garbage in [
                    'no response', '/mcp', 'command', '<', '>', 'error:',
                    'warning:', 'loading', 'running'
                ]):
                    continue

                if not fact_text[0].isupper():
                    continue

                name = self._generate_fact_name(fact_text)

                candidates.append({
                    'concept_type': CONCEPT_FACT,
                    'name': name,
                    'description': fact_text,
                    'confidence': base_confidence,
                    'session_id': session_id,
                })

        return candidates

    def _extract_rules(self, content: str, session_id: str) -> List[Dict]:
        """Extract user-provided rules and conventions."""
        candidates = []

        for pattern, base_confidence in self.RULE_PATTERNS:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                groups = match.groups()
                full_match = match.group(0)

                if len(groups) >= 2:
                    rule_text = f"{groups[0].strip()} - {groups[1].strip()}"
                else:
                    rule_text = groups[0].strip()

                if not self._is_valid_description(rule_text):
                    continue

                rule_type = 'guideline'
                match_lower = full_match.lower()
                if match_lower.startswith('always') or 'always' in match_lower[:20]:
                    rule_type = 'always'
                elif match_lower.startswith('never') or 'never' in match_lower[:20]:
                    rule_type = 'never'
                elif match_lower.startswith("don't") or match_lower.startswith('do not'):
                    rule_type = 'never'

                name = self._generate_rule_name(rule_text, rule_type)

                candidates.append({
                    'concept_type': CONCEPT_RULE,
                    'name': name,
                    'description': rule_text,
                    'confidence': base_confidence,
                    'session_id': session_id,
                    'metadata': {'rule_type': rule_type}
                })

        return candidates

    def _is_valid_component_name(self, name: str) -> bool:
        """Validate a component name."""
        if not name or len(name) < 5 or len(name) > 50:
            return False

        name_lower = name.lower()

        words = name.split()
        if not any(len(w) >= 4 for w in words):
            return False

        skip_words = {
            'the', 'this', 'that', 'which', 'what', 'when', 'where', 'how', 'why',
            'it', 'they', 'we', 'you', 'i', 'he', 'she', 'and', 'or', 'but',
            'code', 'file', 'data', 'text', 'string', 'line', 'output',
            'function', 'method', 'class', 'variable', 'let me', 'did i',
            'code to', 'still uniquely', 'showing venv', 'output now', 'then',
            'backend and', 'frontend and', 'node.js', 'nodejs', 'which will',
        }
        if name_lower in skip_words:
            return False

        skip_starts = ['let ', 'did ', 'and ', 'but ', 'the ', 'a ', 'an ', 'to ', 'i ']
        for prefix in skip_starts:
            if name_lower.startswith(prefix):
                return False

        skip_ends = [' and', ' or', ' to', ' the', ' a']
        for suffix in skip_ends:
            if name_lower.endswith(suffix):
                return False

        if not any(c.isalpha() for c in name):
            return False

        return True

    def _is_valid_description(self, desc: str) -> bool:
        """Validate a description string."""
        if not desc or len(desc) < 10 or len(desc) > 200:
            return False

        alpha_ratio = sum(1 for c in desc if c.isalpha() or c.isspace()) / len(desc)
        if alpha_ratio < 0.7:
            return False

        if '{' in desc or '(' in desc and ')' in desc:
            return False
        if '`' in desc or '```' in desc:
            return False

        return True

    def _normalize_name(self, name: str) -> str:
        """Normalize a concept name for consistency."""
        return ' '.join(name.split()).title()

    def _generate_fact_name(self, fact_text: str) -> str:
        """Generate a short name from a fact description."""
        words = fact_text.split()[:5]
        name = ' '.join(words)
        if len(name) > 40:
            name = name[:40] + '...'
        return name

    def _generate_rule_name(self, rule_text: str, rule_type: str) -> str:
        """Generate a short name from a rule description."""
        prefix = rule_type.upper() + ': ' if rule_type != 'guideline' else ''
        words = rule_text.split()[:4]
        name = prefix + ' '.join(words)
        if len(name) > 50:
            name = name[:50] + '...'
        return name

    def get_hot_files(self) -> List[Tuple[str, int]]:
        """Get the most frequently mentioned files."""
        return self._file_mentions.most_common(10)


class ConceptStore:
    """
    Persistent storage for codebase concepts.

    Uses centralized db_manager for thread-safe writes.
    """

    def __init__(self, project_path: str = ""):
        self.project_path = project_path

    def upsert(self, concept: Dict) -> bool:
        """
        Insert or update a concept.

        If concept exists, updates confidence based on frequency.
        Returns True if new concept, False if updated existing.
        """
        db = get_db_manager()
        now = datetime.now().isoformat()

        concept_type = concept.get('concept_type')
        name = concept.get('name', '')
        description = concept.get('description', '')
        confidence = concept.get('confidence', 0.5)
        session_id = concept.get('session_id', '')
        metadata = json.dumps(concept.get('metadata', {}))
        project_path = self.project_path

        def upsert_concept(cursor):
            cursor.execute("""
                SELECT id, frequency, confidence, source_sessions, description
                FROM codebase_concepts
                WHERE project_path = ? AND concept_type = ? AND name = ?
            """, (project_path, concept_type, name))

            row = cursor.fetchone()
            if row:
                concept_id, freq, old_confidence, sources, old_desc = row
                sources_list = json.loads(sources) if sources else []
                if session_id and session_id not in sources_list:
                    sources_list.append(session_id)

                new_confidence = min(0.95, old_confidence + (confidence * 0.1))

                if len(description) > len(old_desc or ''):
                    new_desc = description
                else:
                    new_desc = old_desc

                cursor.execute("""
                    UPDATE codebase_concepts
                    SET frequency = ?, confidence = ?, source_sessions = ?,
                        description = ?, last_updated = ?, metadata = ?
                    WHERE id = ?
                """, (freq + 1, new_confidence, json.dumps(sources_list[-20:]),
                      new_desc, now, metadata, concept_id))
                return False
            else:
                cursor.execute("""
                    INSERT INTO codebase_concepts
                    (project_path, concept_type, name, description, metadata,
                     confidence, source_sessions, first_seen, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (project_path, concept_type, name, description, metadata,
                      confidence, json.dumps([session_id] if session_id else []),
                      now, now))

                cursor.execute("""
                    INSERT INTO concepts_fts(rowid, name, description)
                    VALUES (last_insert_rowid(), ?, ?)
                """, (name, description))

                return True

        try:
            return db.execute_write_func(CONCEPTS_DB, upsert_concept)
        except Exception as e:
            log(f"Error upserting concept: {e}")
            return False

    def _is_valid_purpose(self, purpose: str) -> bool:
        """Validate that a purpose string is meaningful."""
        if not purpose or len(purpose.strip()) < 10:
            return False

        purpose_lower = purpose.lower()

        garbage_indicators = [
            'return type', 'redundant code', 'import ', 'def ', 'class ',
            '():', '()', '{}', '[]', '===', '!==', '=>',
            ' and ', ' or ', ' the ', ' that ', ' which ',
            'sanitized', 'exception', 'error:', 'warning:',
            'todo:', 'fixme:', 'hack:',
            'let me', "i'll", 'now we', 'checking', 'looking at',
            'has been', 'was changed', 'is now', 'updated to',
        ]

        for indicator in garbage_indicators:
            if indicator in purpose_lower:
                return False

        valid_endings = '.!?)'
        if not purpose.rstrip()[-1] in valid_endings:
            words = purpose.split()
            if len(words) < 2 or len(words) > 15:
                return False

        if purpose.rstrip().endswith(':'):
            return False

        return True

    def get_concepts_for_init(self, min_confidence: float = 0.6) -> Dict:
        """Get codebase concepts for mira_init response."""
        db = get_db_manager()

        result = {
            'architecture_summary': '',
            'components': [],
            'integrations': [],
            'technologies': [],
            'key_modules': [],
            'patterns': [],
            'facts': [],
            'rules': [],
            'hot_files': [],
        }

        try:
            # Get components
            rows = db.execute_read(CONCEPTS_DB, """
                SELECT name, description, related_files, confidence, frequency
                FROM codebase_concepts
                WHERE project_path = ? AND concept_type = ? AND confidence >= ?
                ORDER BY confidence DESC, frequency DESC
                LIMIT 10
            """, (self.project_path, CONCEPT_COMPONENT, min_confidence))

            for row in rows:
                result['components'].append({
                    'name': row['name'],
                    'purpose': row['description'],
                    'files': json.loads(row['related_files']) if row['related_files'] else [],
                    'confidence': f"{int(row['confidence'] * 100)}%",
                    'mentions': row['frequency']
                })

            if result['components']:
                top_components = [c['name'] for c in result['components'][:3]]
                result['architecture_summary'] = f"Key components: {', '.join(top_components)}"

            # Get integrations
            rows = db.execute_read(CONCEPTS_DB, """
                SELECT name, description, metadata, confidence
                FROM codebase_concepts
                WHERE project_path = ? AND concept_type = ? AND confidence >= ?
                ORDER BY confidence DESC
                LIMIT 10
            """, (self.project_path, CONCEPT_INTEGRATION, min_confidence))

            for row in rows:
                meta = json.loads(row['metadata']) if row['metadata'] else {}
                mechanism = meta.get('mechanism') or row['description'] or 'Connected'
                result['integrations'].append({
                    'flow': row['name'],
                    'mechanism': mechanism,
                    'confidence': f"{int(row['confidence'] * 100)}%"
                })

            # Get technologies
            rows = db.execute_read(CONCEPTS_DB, """
                SELECT name, description, confidence, frequency
                FROM codebase_concepts
                WHERE project_path = ? AND concept_type = ? AND confidence >= ?
                ORDER BY confidence DESC, frequency DESC
                LIMIT 10
            """, (self.project_path, CONCEPT_TECHNOLOGY, min_confidence))

            for row in rows:
                result['technologies'].append({
                    'name': row['name'],
                    'role': row['description'],
                    'confidence': f"{int(row['confidence'] * 100)}%"
                })

            # Get key modules
            rows = db.execute_read(CONCEPTS_DB, """
                SELECT name, description, confidence
                FROM codebase_concepts
                WHERE project_path = ? AND concept_type = ? AND confidence >= ?
                ORDER BY confidence DESC, frequency DESC
                LIMIT 25
            """, (self.project_path, CONCEPT_MODULE, min_confidence))

            for row in rows:
                if self._is_valid_purpose(row['description']):
                    result['key_modules'].append({
                        'file': row['name'],
                        'purpose': row['description'],
                        'confidence': f"{int(row['confidence'] * 100)}%"
                    })
                elif row['confidence'] >= 0.9:
                    result['key_modules'].append({
                        'file': row['name'],
                        'purpose': f"Frequently discussed file ({row['name']})",
                        'confidence': f"{int(row['confidence'] * 100)}%"
                    })

            # Get patterns
            rows = db.execute_read(CONCEPTS_DB, """
                SELECT name, description, confidence
                FROM codebase_concepts
                WHERE project_path = ? AND concept_type = ? AND confidence >= ?
                ORDER BY confidence DESC
                LIMIT 8
            """, (self.project_path, CONCEPT_PATTERN, min_confidence))

            for row in rows:
                result['patterns'].append({
                    'name': row['name'],
                    'description': row['description'],
                    'confidence': f"{int(row['confidence'] * 100)}%"
                })

            # Get user-provided facts
            rows = db.execute_read(CONCEPTS_DB, """
                SELECT name, description, confidence
                FROM codebase_concepts
                WHERE project_path = ? AND concept_type = ? AND confidence >= ?
                ORDER BY confidence DESC, last_updated DESC
                LIMIT 10
            """, (self.project_path, CONCEPT_FACT, min_confidence - 0.1))

            for row in rows:
                result['facts'].append({
                    'fact': row['description'],
                    'confidence': f"{int(row['confidence'] * 100)}%"
                })

            # Get user-provided rules
            rows = db.execute_read(CONCEPTS_DB, """
                SELECT name, description, metadata, confidence
                FROM codebase_concepts
                WHERE project_path = ? AND concept_type = ? AND confidence >= ?
                ORDER BY confidence DESC
                LIMIT 10
            """, (self.project_path, CONCEPT_RULE, min_confidence - 0.1))

            for row in rows:
                meta = json.loads(row['metadata']) if row['metadata'] else {}
                result['rules'].append({
                    'rule': row['description'],
                    'type': meta.get('rule_type', 'guideline'),
                    'confidence': f"{int(row['confidence'] * 100)}%"
                })

        except Exception as e:
            log(f"Error getting concepts for init: {e}")
            return self._empty_response()

        return result

    def _empty_response(self) -> Dict:
        """Return empty concept structure."""
        return {
            'architecture_summary': '',
            'components': [],
            'integrations': [],
            'technologies': [],
            'key_modules': [],
            'patterns': [],
            'facts': [],
            'rules': [],
            'hot_files': [],
        }

    def get_stats(self) -> Dict:
        """Get statistics about stored concepts."""
        db = get_db_manager()

        try:
            rows = db.execute_read(CONCEPTS_DB, """
                SELECT concept_type, COUNT(*) as cnt
                FROM codebase_concepts
                WHERE project_path = ?
                GROUP BY concept_type
            """, (self.project_path,))

            by_type = {row['concept_type']: row['cnt'] for row in rows}
            total = sum(by_type.values())

            return {'total': total, 'by_type': by_type}
        except Exception as e:
            log(f"Error getting concept stats: {e}")
            return {'total': 0, 'by_type': {}}


def extract_concepts_from_conversation(conversation: dict, session_id: str, project_path: str = ""):
    """
    Extract codebase concepts from a conversation.

    Called during ingestion to learn about the codebase.
    """
    messages = conversation.get('messages', [])
    if not messages:
        return {'concepts_found': 0}

    extractor = ConceptExtractor(project_path)
    store = ConceptStore(project_path)

    concepts_found = 0

    for msg in messages:
        role = msg.get('role', '')
        content = msg.get('content', '')

        if isinstance(content, list):
            content = ' '.join(
                item.get('text', '') for item in content
                if isinstance(item, dict) and item.get('type') == 'text'
            )

        if not content:
            continue

        candidates = extractor.extract_from_message(content, role, session_id)

        for candidate in candidates:
            if store.upsert(candidate):
                concepts_found += 1

    skip_hot_files = {'node.js', 'vue.js', 'next.js', 'react.js', 'express.js'}
    for filename, count in extractor.get_hot_files():
        if count >= 3 and filename.lower() not in skip_hot_files:
            store.upsert({
                'concept_type': CONCEPT_MODULE,
                'name': filename,
                'description': f'Frequently discussed file ({count} mentions)',
                'confidence': min(0.9, 0.5 + count * 0.05),
                'session_id': session_id,
            })

    return {'concepts_found': concepts_found}


def get_codebase_knowledge(project_path: str = "") -> Dict:
    """
    Get codebase knowledge for mira_init.

    Main entry point for handlers.py to get codebase concepts.
    """
    store = ConceptStore(project_path)
    return store.get_concepts_for_init()
