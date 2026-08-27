"""Imprint CLI — python -m imprint <command>

Commands:
  collect <jsonl>   ingest model-agnostic request logs
  mine              cluster pairs into signatures ranked by economics
  status            show signatures + skill ladder state
  serve [port]      run imprint-local endpoint (model-agnostic)
"""

import sys

from .store import connect


def cmd_collect(args: list[str]) -> int:
    from .collector import run

    if not args:
        print("usage: python -m imprint collect <jsonl>")
        return 1
    run(args[0])
    return 0


def cmd_mine(args: list[str]) -> int:
    from .miner import report

    report()
    return 0


def cmd_status(args: list[str]) -> int:
    conn = connect()
    sigs = conn.execute(
        "SELECT id, volume_7d, avg_cost_usd, priority_score, status"
        " FROM signatures ORDER BY priority_score DESC LIMIT 10"
    ).fetchall()
    print("== signatures ==")
    for s in sigs:
        print(
            f"  {s['id']}  {s['volume_7d']}/wk  ${s['avg_cost_usd']:.5f}  "
            f"prio={s['priority_score']:.3f}  [{s['status']}]"
        )
    skills = conn.execute(
        "SELECT signature_id, version, stage, samples FROM skills"
        " ORDER BY updated_at DESC LIMIT 10"
    ).fetchall()
    print("\n== skills ==")
    for k in skills:
        print(
            f"  {k['signature_id']}  v{k['version']}  [{k['stage']}]  {k['samples']} samples"
        )
    if not sigs and not skills:
        print("  (empty — run collect + mine first)")
    return 0


def cmd_bases(args: list[str]) -> int:
    from .basemodel import detect_hardware, select_base

    hw = detect_hardware()
    print(f"hardware: {hw.kind} ({hw.vram_gb}GB) {hw.gpu_name}")
    sel = select_base([], {})
    if not sel.distiller_enabled:
        print(f"distiller: DISABLED — {sel.block_reason}")
        print("prompt-evolution (v1) remains fully functional.")
        return 0
    print(f"selected base : {sel.candidate.name} [{sel.candidate.tier}]")
    print(f"  hf id       : {sel.candidate.hf_id}")
    print(f"  backend     : {sel.candidate.backend}")
    print(f"  score       : {sel.score:.1f}")
    if sel.runner_up:
        print(f"  runner-up   : {sel.runner_up.name}")
    return 0


def cmd_serve(args: list[str]) -> int:
    from .server import serve

    port = int(args[0]) if args else None
    serve(port=port) if port else serve()
    return 0


COMMANDS = {
    "collect": cmd_collect,
    "mine": cmd_mine,
    "status": cmd_status,
    "serve": cmd_serve,
    "bases": cmd_bases,
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        return 1
    return COMMANDS[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    sys.exit(main())
