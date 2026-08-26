"""SQLite storage layer for Imprint. Zero-infra, notebook-friendly."""

import sqlite3
import time
from pathlib import Path

DEFAULT_DB = "data/imprint.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS pairs (
  id TEXT PRIMARY KEY,
  ts INTEGER,
  signature_id TEXT,
  prompt TEXT,
  response TEXT,
  model TEXT,
  provider TEXT,
  cost_usd REAL,
  latency_ms INTEGER,
  cache_hit INTEGER DEFAULT 0,
  accepted INTEGER DEFAULT NULL
);
CREATE INDEX IF NOT EXISTS idx_pairs_sig ON pairs(signature_id, ts);

CREATE TABLE IF NOT EXISTS signatures (
  id TEXT PRIMARY KEY,
  centroid_json TEXT,
  sample_ids TEXT,            -- JSON array of pair ids
  volume_7d INTEGER,
  avg_cost_usd REAL,
  priority_score REAL,
  status TEXT DEFAULT 'candidate',   -- candidate|active|plateaued|retired
  created_at INTEGER,
  updated_at INTEGER
);

CREATE TABLE IF NOT EXISTS skills (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  signature_id TEXT,
  version INTEGER,
  prompt_hash TEXT UNIQUE,
  template TEXT,
  few_shot_json TEXT,
  stage TEXT DEFAULT 'shadow',       -- shadow|canary|preferred|pinned
  samples INTEGER DEFAULT 0,
  regressions INTEGER DEFAULT 0,
  cost_saved_usd REAL DEFAULT 0,
  baseline_cost_usd REAL DEFAULT 0,
  created_at INTEGER,
  updated_at INTEGER
);
"""


def connect(db_path: str = DEFAULT_DB) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def now_ms() -> int:
    return int(time.time() * 1000)
