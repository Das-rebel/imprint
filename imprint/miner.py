"""Signature Miner: cluster pairs into recurring task patterns, ranked by economics.

Phase 0: deterministic prefix heuristic (no ML deps required).
Phase 1+: local embeddings (BAAI/bge-small-en) + cosine-threshold clustering.
Ranking is economics-first: priority = volume_7d x avg_cost x cache_miss_ratio.
"""
import hashlib
import json

from .store import connect, now_ms

WEEK_MS = 7 * 24 * 3600 * 1000
MIN_VOLUME_PER_WEEK = 100  # training refusal threshold


def _prefix_key(prompt: str, words: int = 8) -> str:
    """Cheap deterministic clustering key: first N content words.
    Pure numbers are dropped so '...for region 3' and '...for region 7' cluster."""
    tokens = [t for t in prompt.lower().split()
             if not t.lstrip("-+").isdigit()]
    return " ".join(tokens[:words])


def _sig_id(key: str) -> str:
    return "sig_" + hashlib.sha256(key.encode()).hexdigest()[:10]


def mine(db_path: str = "data/imprint.db", min_volume: int = 5,
         use_embeddings: bool = False) -> list[dict]:
    """Cluster unassigned pairs into signatures; persist and return ranked rows."""
    conn = connect(db_path)
    week_ago = now_ms() - WEEK_MS
    rows = conn.execute(
        "SELECT id, prompt, cost_usd, cache_hit FROM pairs WHERE ts >= ?",
        (week_ago,),
    ).fetchall()

    groups: dict[str, list] = {}
    for r in rows:
        key = _prefix_key(r["prompt"])
        groups.setdefault(key, []).append(r)

    ranked = []
    for key, items in groups.items():
        volume_week = len(items)
        avg_cost = sum(i["cost_usd"] or 0 for i in items) / volume_week
        cache_miss = sum(1 for i in items if not i["cache_hit"]) / volume_week
        priority = volume_week * avg_cost * cache_miss
        sig_id = _sig_id(key)
        status = "active" if volume_week >= MIN_VOLUME_PER_WEEK else "candidate"
        conn.execute(
            "INSERT INTO signatures (id, centroid_json, sample_ids, volume_7d,"
            " avg_cost_usd, priority_score, status, created_at, updated_at)"
            " VALUES (?,?,?,?,?,?,?,?,?)"
            " ON CONFLICT(id) DO UPDATE SET volume_7d=?, avg_cost_usd=?,"
            " priority_score=?, status=?, updated_at=?",
            (
                sig_id, json.dumps({"prefix": key}),
                json.dumps([i["id"] for i in items[:50]]),
                volume_week, avg_cost, priority, status, now_ms(), now_ms(),
                volume_week, avg_cost, priority, status, now_ms(),
            ),
        )
        ranked.append({
            "signature_id": sig_id, "prefix": key, "volume_week": volume_week,
            "avg_cost_usd": round(avg_cost, 6), "priority": round(priority, 4),
            "status": status,
        })

    # assign signature_id back onto pairs
    for key, items in groups.items():
        sig_id = _sig_id(key)
        conn.execute(
            "UPDATE pairs SET signature_id=? WHERE id IN (%s)"
            % ",".join("?" * len(items)),
            [sig_id] + [i["id"] for i in items],
        )
    conn.commit()
    return sorted(ranked, key=lambda r: -r["priority"])


def report(db_path: str = "data/imprint.db", top: int = 10) -> list[dict]:
    ranked = mine(db_path)
    for r in ranked[:top]:
        print(
            f"{r['volume_week']:>6}/wk  ${r['avg_cost_usd']:.5f}/req  "
            f"prio={r['priority']:.3f}  [{r['status']:9s}] {r['signature_id']}  {r['prefix'][:50]}"
        )
    return ranked


if __name__ == "__main__":
    report()
