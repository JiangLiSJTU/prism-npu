#!/usr/bin/env python3
"""Issue #9 Phase 1d — Ingest high-batch msprof + report v8 current-model prediction error.

Workflow:
  1. Scans ``msprof_data/msprof_issue9_*_PipeUtilization/`` dirs (rsync'd back from NPU).
  2. For each, derives the config_name (e.g.
     ``msprof_issue9_qwen3_06b_prefill_S512_b16_sdpa_PipeUtilization`` →
     ``Qwen3-prefill-S512-b16-sdpa``) and calls ``parse_pipeutil_to_baseline.py``
     to merge into ``data/calibration/pipe_baseline_per_model.json``.
  3. Computes v8 wall-clock prediction for each new config against measured;
     reports a table grouped by expected bucket vs actual classifier output.
     This is the "before-fix" baseline that Issue #9 Phase 2 will improve on.

Usage:
  python3 scripts/ingest_issue9_msprof.py
  python3 scripts/ingest_issue9_msprof.py --dry-run    # just preview the table, no merge
  python3 scripts/ingest_issue9_msprof.py --loop 5     # override default ais_bench loop count
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO = Path(__file__).resolve().parent.parent
MSPROF_ROOT = REPO / "msprof_data"
BASELINE_PATH = REPO / "data" / "calibration" / "pipe_baseline_per_model.json"
PARSE_TOOL = REPO / "scripts" / "parse_pipeutil_to_baseline.py"

# Map msprof dir basename stem (after `msprof_issue9_`) → canonical config_name.
# Pattern: <model>_S<S>_b<B>_sdpa  →  <Model-fancy>-prefill-S<S>-b<B>-sdpa
DIR_NAME_PATTERN = re.compile(
    r"^msprof_issue9_(?P<model>[a-z0-9_]+?)_prefill_S(?P<S>\d+)_b(?P<B>\d+)_sdpa_PipeUtilization$"
)
MODEL_DISPLAY = {
    "qwen3_06b":      "Qwen3-prefill",
    "llama_3_2_1b":   "Llama-3.2-1B-prefill",
    "qwen2_5_05b":    "Qwen2.5-0.5B-prefill",
    "smollm2_360m":   "SmolLM2-360M-prefill",
    "phi3_mini":      "Phi-3-mini-prefill",
    "modernbert_base": "ModernBERT-base-prefill",
}


def derive_config_name(dirname: str) -> "str|None":
    m = DIR_NAME_PATTERN.match(dirname)
    if not m:
        return None
    model = MODEL_DISPLAY.get(m.group("model"))
    if model is None:
        return None
    return f"{model}-S{m.group('S')}-b{m.group('B')}-sdpa"


def find_issue9_dirs() -> List[Path]:
    if not MSPROF_ROOT.is_dir():
        return []
    return sorted(d for d in MSPROF_ROOT.iterdir()
                  if d.is_dir() and d.name.startswith("msprof_issue9_"))


def parse_and_merge(msprof_dir: Path, config_name: str, loop: int, dry_run: bool) -> bool:
    cmd = [
        sys.executable, str(PARSE_TOOL),
        "--msprof-dir", str(msprof_dir),
        "--config-name", config_name,
        "--loop", str(loop),
        "--merge-into", str(BASELINE_PATH),
    ]
    if dry_run:
        cmd.append("--dry-run")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"  ✗ parse failed: {res.stderr.strip()[:400]}")
        return False
    return True


def v8_predict_for_config(config_name: str) -> "Dict|None":
    """Run predict_pipe v8 path for a synthesized config — picks up calibrated
    params + bucket classifier output."""
    # The simpler path: load merged baseline, call predict_wallclock_v3 with
    # baseline arch (full code path includes v8 gm_frac injection).
    sys.path.insert(0, str(REPO / "src"))
    from prism.sweep.runner import (  # noqa: E402
        BASELINE_910B4,
        DEFAULT_AIV_MTE3_GM_FRAC,
        DEFAULT_FIXPIPE_GM_FRAC,
        predict_wallclock_v3,
    )

    with BASELINE_PATH.open(encoding="utf-8") as f:
        baseline = json.load(f)
    pipe = baseline["configs"].get(config_name)
    if pipe is None:
        return None

    # Inject gm_frac (Issue #7 path) the way runner.main() does
    dest_bw_path = REPO / "data" / "calibration" / "pipe_dest_bw.json"
    if dest_bw_path.is_file():
        with dest_bw_path.open(encoding="utf-8") as f:
            dest_bw = json.load(f)
        gm_fp = dest_bw.get("aic_fixpipe", {}).get(config_name, {}).get("gm_frac")
        gm_m3 = dest_bw.get("aiv_mte3", {}).get(config_name, {}).get("gm_frac")
        pipe = dict(pipe)
        pipe["_aic_fixpipe_gm_frac"] = (
            gm_fp if isinstance(gm_fp, (int, float)) else DEFAULT_FIXPIPE_GM_FRAC
        )
        pipe["_aiv_mte3_gm_frac"] = (
            gm_m3 if isinstance(gm_m3, (int, float)) else DEFAULT_AIV_MTE3_GM_FRAC
        )

    pred = predict_wallclock_v3(pipe, BASELINE_910B4, BASELINE_910B4)
    meas = pipe.get("wall_clock_us", 0)
    err_pct = 100 * (pred["wall_clock_us"] - meas) / meas if meas > 0 else 0
    return {
        "wall_pred_us": pred["wall_clock_us"],
        "wall_meas_us": meas,
        "wall_err_pct": err_pct,
        "aic_pred_us": pred["aic_time_us"],
        "aiv_pred_us": pred["aiv_time_us"],
        "aiv_aic_ratio": pred["aiv_time_us"] / max(pred["aic_time_us"], 1e-9),
    }


def classify_with_v7(config_name: str) -> "str|None":
    """Run the v7 classifier on a config_name — same logic predict_pipe uses."""
    sys.path.insert(0, str(REPO / "src"))
    # Build a ModelSpec from the config_name and yaml; fall back to None if yaml missing.
    # For now we infer from name alone: any (S>1) high-batch → check current rules.
    # This avoids depending on yaml plumbing.
    from prism.predict_pipe.physics_v7 import classify_bottleneck_v7  # noqa: E402

    class _Spec:
        pass

    # Best-effort parse from config_name
    name_lower = config_name.lower()
    spec = _Spec()
    spec.S = 1 if "decode" in name_lower else int(
        re.search(r"-S(\d+)-", config_name).group(1)
    )
    spec.layers = 28 if "qwen3" in name_lower else (
        16 if "llama-3.2-1b" in name_lower else 22
    )
    spec.d_model = 1024 if "qwen3" in name_lower else (
        2048 if "llama-3.2-1b" in name_lower else 768
    )
    m_batch = re.search(r"-b(\d+)", config_name)
    batch = int(m_batch.group(1)) if m_batch else 1
    return classify_bottleneck_v7(spec, batch)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--loop", type=int, default=5,
                   help="ais_bench loop count used during NPU capture (default 5; "
                        "must match what run_issue9_high_batch.sh used)")
    p.add_argument("--dry-run", action="store_true",
                   help="Don't merge into pipe_baseline_per_model.json, just preview")
    args = p.parse_args()

    dirs = find_issue9_dirs()
    if not dirs:
        print(f"⚠ no msprof_issue9_* dirs found under {MSPROF_ROOT}")
        print(f"  expected: msprof_data/msprof_issue9_<model>_prefill_S<S>_b<B>_sdpa_PipeUtilization/")
        print(f"  did you rsync from the NPU host yet?")
        return 1

    print(f"Found {len(dirs)} issue9 msprof dirs under {MSPROF_ROOT}")
    print()

    parsed: List[Tuple[str, Path]] = []
    for d in dirs:
        cfg = derive_config_name(d.name)
        if cfg is None:
            print(f"  ⚠ skip (unknown dir pattern): {d.name}")
            continue
        print(f"→ {d.name} → {cfg}")
        ok = parse_and_merge(d, cfg, args.loop, args.dry_run)
        if ok:
            parsed.append((cfg, d))

    if not parsed:
        print("\n✗ no configs parsed successfully")
        return 1

    # Phase 1d core: report v8 current prediction error per new high-batch config
    print()
    print("═════════════════════════════════════════════════════════════════════")
    print("v8 CURRENT prediction error on newly-ingested high-batch configs")
    print("(= 'before-fix' baseline; Issue #9 Phase 2 should reduce these)")
    print("═════════════════════════════════════════════════════════════════════")
    print(f"{'config':<48} {'v7 bucket':<12} {'aiv/aic':>8} {'wall err':>10}")
    print("-" * 80)
    for cfg, _ in parsed:
        bucket = classify_with_v7(cfg) or "(unknown)"
        pred = v8_predict_for_config(cfg) if not args.dry_run else None
        if pred is None:
            print(f"{cfg:<48} {bucket:<12} {'—':>8} {'—':>10}  (no merged data; dry-run?)")
        else:
            ratio = pred["aiv_aic_ratio"]
            err = pred["wall_err_pct"]
            print(f"{cfg:<48} {bucket:<12} {ratio:>8.2f} {err:>+9.1f}%  "
                  f"(pred={pred['wall_pred_us']:.0f} meas={pred['wall_meas_us']:.0f})")

    print()
    print("→ Interpretation:")
    print("  - 'AIV_BOUND' classification + high aiv/aic ratio + large positive wall_err")
    print("    confirms the Issue #9 hypothesis (classifier over-amplifies AIV at high batch).")
    print("  - 'aiv/aic < 1.0' on a config still bucketed AIV_BOUND is the smoking gun.")
    print("  - Next: scripts/calib_issue9_aic_compute.py to fit the new AIC_COMPUTE bucket.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
