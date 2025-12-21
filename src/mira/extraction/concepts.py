"""
MIRA Codebase Concepts Module

Extracts and tracks key concepts about the codebase from conversations.
"""

import hashlib
import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import Counter

from mira.core import log, DB_CONCEPTS
from mira.core.database import get_db_manager


# Concept types
CONCEPT_COMPONENT = "component"
CONCEPT_MODULE = "module"
CONCEPT_TECHNOLOGY = "technology"
CONCEPT_INTEGRATION = "integration"
CONCEPT_PATTERN = "pattern"
CONCEPT_FACT = "fact"
CONCEPT_RULE = "rule"

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

-- Indexes
CREATE INDEX IF NOT EXISTS idx_concepts_project ON codebase_concepts(project_path);
CREATE INDEX IF NOT EXISTS idx_concepts_type ON codebase_concepts(concept_type);
CREATE INDEX IF NOT EXISTS idx_concepts_confidence ON codebase_concepts(confidence DESC);

-- FTS for concept search
CREATE VIRTUAL TABLE IF NOT EXISTS concepts_fts USING fts5(
    name,
    description,
    content='codebase_concepts',
    content_rowid='id'
);
"""


def init_concepts_db():
    """Initialize the concepts database."""
    db = get_db_manager()
    db.init_schema(DB_CONCEPTS, CONCEPTS_SCHEMA)


class ConceptExtractor:
    """Extract codebase concepts from conversation content."""

    KNOWN_TECHNOLOGIES = {
        'chromadb', 'sqlite', 'postgresql', 'postgres', 'mysql', 'mongodb',
        'redis', 'elasticsearch', 'qdrant', 'pinecone',
        'react', 'vue', 'angular', 'svelte', 'next', 'nuxt',
        'express', 'fastapi', 'django', 'flask',
        'typescript', 'python', 'javascript', 'golang', 'rust',
        'docker', 'kubernetes', 'aws', 'gcp', 'azure',
        'mcp', 'json-rpc', 'grpc', 'graphql', 'rest',
    }

    def extract(self, text: str, project_path: str) -> List[Dict]:
        """Extract concepts from text."""
        concepts = []

        # Technology detection
        for tech in self.KNOWN_TECHNOLOGIES:
            if tech in text.lower():
                # Find context around the technology mention
                pattern = rf'\b{re.escape(tech)}\b[^.]*(?:for|to|handles?|provides?|enables?)[^.]*\.'
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches[:1]:  # Take first match
                    concepts.append({
                        "type": CONCEPT_TECHNOLOGY,
                        "name": tech,
                        "description": match.strip(),
                        "confidence": 0.8,
                    })

        # Component detection
        component_patterns = [
            r"(?:The\s+)?(\w+\s+(?:backend|frontend|server|service))\s+(?:handles?|provides?)\s+([^.]+)\.",
        ]
        for pattern in component_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                concepts.append({
                    "type": CONCEPT_COMPONENT,
                    "name": match.group(1).strip(),
                    "description": match.group(2).strip(),
                    "confidence": 0.75,
                })

        return concepts


class ConceptStore:
    """Store and retrieve codebase concepts."""

    def __init__(self, project_path: str):
        self.project_path = project_path
        init_concepts_db()

    def store(self, concept_type: str, name: str, description: str,
              confidence: float = 0.5, session_id: Optional[str] = None):
        """Store a concept."""
        db = get_db_manager()

        # Check if exists
        row = db.execute_read_one(
            DB_CONCEPTS,
            """SELECT id, frequency, source_sessions FROM codebase_concepts
               WHERE project_path = ? AND concept_type = ? AND name = ?""",
            (self.project_path, concept_type, name)
        )

        now = datetime.now().isoformat()

        if row:
            # Update
            sessions = json.loads(row['source_sessions'] or '[]')
            if session_id and session_id not in sessions:
                sessions.append(session_id)

            db.execute_write(
                DB_CONCEPTS,
                """UPDATE codebase_concepts SET
                   frequency = frequency + 1,
                   confidence = MAX(?, confidence),
                   source_sessions = ?,
                   last_updated = ?
                WHERE id = ?""",
                (confidence, json.dumps(sessions), now, row['id'])
            )
        else:
            # Insert
            sessions = [session_id] if session_id else []
            db.execute_write(
                DB_CONCEPTS,
                """INSERT INTO codebase_concepts
                   (project_path, concept_type, name, description, confidence,
                    source_sessions, first_seen, last_updated)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (self.project_path, concept_type, name, description,
                 confidence, json.dumps(sessions), now, now)
            )

    def get_all(self, min_confidence: float = 0.3) -> List[Dict]:
        """Get all concepts for this project."""
        db = get_db_manager()

        rows = db.execute_read(
            DB_CONCEPTS,
            """SELECT * FROM codebase_concepts
               WHERE project_path = ? AND confidence >= ?
               ORDER BY confidence DESC, frequency DESC""",
            (self.project_path, min_confidence)
        )

        results = []
        for row in rows:
            result = dict(row)
            if result.get('source_sessions'):
                try:
                    result['source_sessions'] = json.loads(result['source_sessions'])
                except json.JSONDecodeError:
                    result['source_sessions'] = []
            results.append(result)
        return results


def get_codebase_knowledge(project_path: str) -> Dict:
    """Get organized codebase knowledge for a project."""
    store = ConceptStore(project_path)
    concepts = store.get_all()

    knowledge = {
        "components": [],
        "technologies": [],
        "integrations": [],
        "patterns": [],
        "facts": [],
        "rules": [],
    }

    for concept in concepts:
        ctype = concept.get('concept_type', '')
        entry = {
            "name": concept.get('name', ''),
            "description": concept.get('description', ''),
            "confidence": concept.get('confidence', 0.5),
        }

        if ctype == CONCEPT_COMPONENT:
            knowledge["components"].append(entry)
        elif ctype == CONCEPT_TECHNOLOGY:
            knowledge["technologies"].append(entry)
        elif ctype == CONCEPT_INTEGRATION:
            knowledge["integrations"].append(entry)
        elif ctype == CONCEPT_PATTERN:
            knowledge["patterns"].append(entry)
        elif ctype == CONCEPT_FACT:
            knowledge["facts"].append(entry)
        elif ctype == CONCEPT_RULE:
            knowledge["rules"].append(entry)

    return knowledge


def get_concepts_stats() -> Dict:
    """Get statistics about recorded concepts."""
    init_concepts_db()
    db = get_db_manager()

    try:
        total = db.execute_read_one(DB_CONCEPTS, "SELECT COUNT(*) as cnt FROM codebase_concepts", ())
        by_type = db.execute_read(
            DB_CONCEPTS,
            "SELECT concept_type, COUNT(*) as cnt FROM codebase_concepts GROUP BY concept_type",
            ()
        )

        return {
            "total": total['cnt'] if total else 0,
            "by_type": {row['concept_type']: row['cnt'] for row in by_type}
        }
    except Exception as e:
        log(f"Concepts stats failed: {e}")
        return {"total": 0, "by_type": {}}


def extract_concepts_from_conversation(
    conversation: dict,
    session_id: str,
    project_path: str = "",
    postgres_session_id: Optional[int] = None,
    storage=None
) -> Dict:
    """
    Extract codebase concepts from a conversation.

    Called during ingestion to learn about the codebase.
    """
    import time

    messages = conversation.get('messages', [])
    if not messages:
        return {'concepts_found': 0}

    short_id = session_id[:12]
    t_start = time.time()

    extractor = ConceptExtractor()
    store = ConceptStore(project_path)

    concepts_found = 0

    for msg in messages:
        content = msg.get('content', '')

        if isinstance(content, list):
            content = ' '.join(
                item.get('text', '') for item in content
                if isinstance(item, dict) and item.get('type') == 'text'
            )

        if not content or len(content) < 20:
            continue

        candidates = extractor.extract(content, project_path)

        for candidate in candidates:
            store.store(
                concept_type=candidate.get('type', CONCEPT_FACT),
                name=candidate.get('name', ''),
                description=candidate.get('description', ''),
                confidence=candidate.get('confidence', 0.5),
                session_id=session_id,
            )
            concepts_found += 1

    t_total = (time.time() - t_start) * 1000
    log(f"[{short_id}] Concepts: {len(messages)} msgs, {concepts_found} found ({t_total:.0f}ms)")

    return {'concepts_found': concepts_found}
