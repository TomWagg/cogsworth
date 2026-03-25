"""
SQLite-backed job queue for cogsworth web interface.
Shared by app.py (submits jobs, polls status) and worker.py (executes jobs).
"""

import json
import os
import sqlite3
import uuid
from datetime import datetime

import numpy as np
from astropy import units as u

DB_PATH = os.path.join(os.path.dirname(__file__), "cogsworth_queue.db")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


# ──────────────────────────────────────────────
# Param serialisation (handles astropy Quantities)
# ──────────────────────────────────────────────

def serialize_params(v):
    """Recursively convert a params dict to a JSON-safe form."""
    if isinstance(v, u.Quantity):
        val = v.value
        if isinstance(val, np.ndarray):
            val = val.tolist()
        elif isinstance(val, (np.floating, np.integer)):
            val = val.item()
        return {"__qty__": True, "value": val, "unit": str(v.unit)}
    elif isinstance(v, dict):
        return {k: serialize_params(vv) for k, vv in v.items()}
    elif isinstance(v, (list, tuple)):
        return [serialize_params(x) for x in v]
    elif isinstance(v, np.integer):
        return int(v)
    elif isinstance(v, np.floating):
        return float(v)
    return v


def deserialize_params(v):
    """Inverse of serialize_params."""
    if isinstance(v, dict):
        if v.get("__qty__"):
            return v["value"] * u.Unit(v["unit"])
        return {k: deserialize_params(vv) for k, vv in v.items()}
    elif isinstance(v, list):
        return [deserialize_params(x) for x in v]
    return v


# ──────────────────────────────────────────────
# DB helpers
# ──────────────────────────────────────────────

def _conn():
    c = sqlite3.connect(DB_PATH, timeout=30)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    return c


def init_db():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id           TEXT PRIMARY KEY,
                job_type     TEXT NOT NULL,
                status       TEXT NOT NULL DEFAULT 'pending',
                params_json  TEXT NOT NULL,
                result_path  TEXT,
                error_msg    TEXT,
                created_at   REAL NOT NULL,
                started_at   REAL,
                finished_at  REAL
            )
        """)
        c.commit()


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def submit_job(job_type: str, params: dict) -> str:
    """Serialise params and insert a new pending job. Returns the job UUID."""
    job_id = str(uuid.uuid4())
    with _conn() as c:
        c.execute(
            "INSERT INTO jobs (id, job_type, status, params_json, created_at) VALUES (?, ?, 'pending', ?, ?)",
            (job_id, job_type, json.dumps(serialize_params(params)), datetime.now().timestamp()),
        )
        c.commit()
    return job_id


def get_job(job_id: str) -> dict | None:
    with _conn() as c:
        row = c.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return dict(row) if row else None


def get_queue_position(job_id: str) -> int:
    """
    Number of pending or running jobs created *before* this one.
    0 means this job is next up (or already running).
    """
    job = get_job(job_id)
    if job is None or job["status"] not in ("pending",):
        return 0
    with _conn() as c:
        count = c.execute(
            "SELECT COUNT(*) FROM jobs WHERE status IN ('pending', 'running') AND created_at < ?",
            (job["created_at"],),
        ).fetchone()[0]
    return count


def get_next_pending() -> dict | None:
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM jobs WHERE status = 'pending' ORDER BY created_at ASC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None


def set_running(job_id: str):
    with _conn() as c:
        c.execute(
            "UPDATE jobs SET status = 'running', started_at = ? WHERE id = ?",
            (datetime.now().timestamp(), job_id),
        )
        c.commit()


def set_done(job_id: str, result_path: str):
    with _conn() as c:
        c.execute(
            "UPDATE jobs SET status = 'done', result_path = ?, finished_at = ? WHERE id = ?",
            (result_path, datetime.now().timestamp(), job_id),
        )
        c.commit()


def set_failed(job_id: str, error_msg: str):
    with _conn() as c:
        c.execute(
            "UPDATE jobs SET status = 'failed', error_msg = ?, finished_at = ? WHERE id = ?",
            (error_msg, datetime.now().timestamp(), job_id),
        )
        c.commit()


def reset_stale_running():
    """Reset any 'running' jobs to 'pending' — used on worker startup after a crash."""
    with _conn() as c:
        c.execute("UPDATE jobs SET status = 'pending', started_at = NULL WHERE status = 'running'")
        c.commit()


def cleanup_old_results(max_age_hours: float = 2.0):
    """Delete finished (done/failed) jobs older than max_age_hours and remove their result files."""
    cutoff = datetime.now().timestamp() - max_age_hours * 3600
    with _conn() as c:
        old = c.execute(
            "SELECT id, result_path FROM jobs WHERE status IN ('done', 'failed') AND finished_at < ?",
            (cutoff,),
        ).fetchall()
        if not old:
            return
        for row in old:
            rp = row["result_path"]
            if rp and os.path.exists(rp):
                os.remove(rp)
        ids = [r["id"] for r in old]
        c.execute(f"DELETE FROM jobs WHERE id IN ({','.join('?' * len(ids))})", ids)
        c.commit()


# Initialise on import
init_db()
