-- MIRA Central Storage Schema
-- PostgreSQL initialization script

-- ===========================================
-- Schema Version Tracking
-- ===========================================

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMPTZ DEFAULT NOW()
);

-- ===========================================
-- Projects
-- ===========================================

CREATE TABLE IF NOT EXISTS projects (
    id SERIAL PRIMARY KEY,
    project_path TEXT UNIQUE NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_projects_path ON projects(project_path);

-- ===========================================
-- Sessions
-- ===========================================

CREATE TABLE IF NOT EXISTS sessions (
    id SERIAL PRIMARY KEY,
    session_id TEXT UNIQUE NOT NULL,
    project_id INTEGER REFERENCES projects(id),
    slug TEXT,
    summary TEXT,
    task_description TEXT,
    git_branch TEXT,
    keywords TEXT[],
    message_count INTEGER DEFAULT 0,
    file_hash TEXT,
    indexed_at TIMESTAMPTZ DEFAULT NOW(),
    llm_processed_at TIMESTAMPTZ  -- When LLM extraction was run
);

CREATE INDEX IF NOT EXISTS idx_sessions_session_id ON sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project_id);
CREATE INDEX IF NOT EXISTS idx_sessions_indexed ON sessions(indexed_at DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_llm_processed ON sessions(llm_processed_at);

-- ===========================================
-- Archives (conversation content)
-- ===========================================

CREATE TABLE IF NOT EXISTS archives (
    id SERIAL PRIMARY KEY,
    session_id TEXT UNIQUE NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    content TEXT,
    archived_at TIMESTAMPTZ DEFAULT NOW()
);

-- ===========================================
-- Artifacts
-- ===========================================

CREATE TABLE IF NOT EXISTS artifacts (
    id SERIAL PRIMARY KEY,
    project_id INTEGER REFERENCES projects(id),
    session_id TEXT REFERENCES sessions(session_id) ON DELETE CASCADE,
    artifact_type TEXT NOT NULL,
    content TEXT NOT NULL,
    language TEXT,
    context TEXT,
    line_count INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(session_id, artifact_type, content)
);

CREATE INDEX IF NOT EXISTS idx_artifacts_project ON artifacts(project_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_session ON artifacts(session_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_type ON artifacts(artifact_type);

-- ===========================================
-- Decisions
-- ===========================================

CREATE TABLE IF NOT EXISTS decisions (
    id SERIAL PRIMARY KEY,
    project_id INTEGER REFERENCES projects(id),
    decision TEXT NOT NULL,
    category TEXT DEFAULT 'general',
    reasoning TEXT,
    alternatives TEXT[],
    confidence REAL DEFAULT 0.5,
    source TEXT DEFAULT 'regex',  -- 'regex' or 'llm'
    session_id TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(project_id, decision)
);

CREATE INDEX IF NOT EXISTS idx_decisions_project ON decisions(project_id);
CREATE INDEX IF NOT EXISTS idx_decisions_category ON decisions(category);
CREATE INDEX IF NOT EXISTS idx_decisions_source ON decisions(source);

-- ===========================================
-- Error Patterns
-- ===========================================

CREATE TABLE IF NOT EXISTS error_patterns (
    id SERIAL PRIMARY KEY,
    project_id INTEGER REFERENCES projects(id),
    signature TEXT NOT NULL,
    error_type TEXT,
    error_text TEXT NOT NULL,
    solution TEXT,
    file_path TEXT,
    occurrences INTEGER DEFAULT 1,
    first_seen TIMESTAMPTZ DEFAULT NOW(),
    last_seen TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(project_id, signature)
);

CREATE INDEX IF NOT EXISTS idx_errors_project ON error_patterns(project_id);
CREATE INDEX IF NOT EXISTS idx_errors_signature ON error_patterns(signature);
CREATE INDEX IF NOT EXISTS idx_errors_type ON error_patterns(error_type);

-- ===========================================
-- Custodian (user preferences/identity)
-- ===========================================

CREATE TABLE IF NOT EXISTS custodian (
    id SERIAL PRIMARY KEY,
    key TEXT UNIQUE NOT NULL,
    value TEXT NOT NULL,
    category TEXT,
    confidence REAL DEFAULT 0.5,
    frequency INTEGER DEFAULT 1,
    source_sessions TEXT[],
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_custodian_key ON custodian(key);
CREATE INDEX IF NOT EXISTS idx_custodian_category ON custodian(category);

-- ===========================================
-- Name Candidates
-- ===========================================

CREATE TABLE IF NOT EXISTS name_candidates (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    confidence REAL DEFAULT 0.5,
    pattern_type TEXT,
    source_session TEXT,
    context TEXT,
    extracted_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(name, source_session)
);

CREATE INDEX IF NOT EXISTS idx_name_candidates_name ON name_candidates(name);
CREATE INDEX IF NOT EXISTS idx_name_candidates_confidence ON name_candidates(confidence DESC);

-- ===========================================
-- Lifecycle Patterns
-- ===========================================

CREATE TABLE IF NOT EXISTS lifecycle_patterns (
    id SERIAL PRIMARY KEY,
    pattern TEXT UNIQUE NOT NULL,
    confidence REAL DEFAULT 0.5,
    occurrences INTEGER DEFAULT 1,
    source_sessions TEXT[],
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_lifecycle_confidence ON lifecycle_patterns(confidence DESC);

-- ===========================================
-- Concepts
-- ===========================================

CREATE TABLE IF NOT EXISTS concepts (
    id SERIAL PRIMARY KEY,
    project_id INTEGER REFERENCES projects(id),
    concept_type TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    confidence REAL DEFAULT 0.5,
    frequency INTEGER DEFAULT 1,
    source_sessions TEXT[],
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(project_id, concept_type, name)
);

CREATE INDEX IF NOT EXISTS idx_concepts_project ON concepts(project_id);
CREATE INDEX IF NOT EXISTS idx_concepts_type ON concepts(concept_type);

-- ===========================================
-- File Operations
-- ===========================================

CREATE TABLE IF NOT EXISTS file_operations (
    id SERIAL PRIMARY KEY,
    project_id INTEGER REFERENCES projects(id),
    session_id TEXT,
    file_path TEXT NOT NULL,
    operation TEXT NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_file_ops_project ON file_operations(project_id);
CREATE INDEX IF NOT EXISTS idx_file_ops_file ON file_operations(file_path);

-- ===========================================
-- Insert initial schema version
-- ===========================================

INSERT INTO schema_version (version) VALUES (5)
ON CONFLICT (version) DO NOTHING;
