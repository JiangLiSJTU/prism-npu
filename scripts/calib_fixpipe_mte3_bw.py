#!/usr/bin/env python3
"""Calibrate aic_fixpipe / aiv_mte3 effective bandwidth by destination (Issue #7).

Both pipes write a Cube/Vector result *out*, and the bottleneck bandwidth depends
on the destination:

    aic_fixpipe : L0C → {L1/UB on-chip  | GM direct-write}   on-chip ref = fixpipe_bw
    aiv_mte3    : UB  → {L1 on-chip      | GM}                on-chip ref = ub_l1_bw
                  GM-bound for both → hbm_bw (~5-10x slower)

msprof reports a single aggregate `*_time`. We recover the GM byte fraction
`gm_frac` via **prior-based 2-cluster classification**: for each op compute its
implied bandwidth (bytes/time) and classify against the threshold
`sqrt(hbm_bw * onchip_bw)` (the physical midpoint of the two regimes). Ops below
threshold → GM cluster; above → on-chip cluster. `gm_frac` is the byte fraction
in the GM cluster.

This is universal — bimodal configs split cleanly into two clusters, unimodal
configs collapse to '1cluster' (one bucket empty). It avoids both pitfalls of
the naive approaches:
- ``Σbytes/Σtime``: contaminated by per-op fixed overhead (e.g. two 3 MB
  GatherV2 ops with 11× different mte3_time → aggregate bw meaningless).
- Single pooled OLS slope: gives a leverage-weighted blended slope for bimodal
  data; the back-solved gm_frac is biased.

Each cluster's OLS slope is reported as a sanity check — GM cluster's fitted
bw should land near hbm_bw, on-chip cluster's near the on-chip reference.

`gm_frac` is consumed by prism.sweep.runner.scale_aic_pipes / scale_aiv_pipes.

Output: data/calibration/pipe_dest_bw.json

Usage:
    python3 scripts/calib_fixpipe_mte3_bw.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Reference bandwidths — must match arch/ascend_910b4_for_sweep_v2.yaml baseline
# (and prism.sweep.runner.BASELINE_910B4).
HBM_BW_GBS = 392.0
FIXPIPE_BW_GBS = 4096.0   # on-chip reference for aic_fixpipe (L0C→L1/UB)
UB_L1_BW_GBS = 2048.0     # on-chip reference for aiv_mte3 (UB→L1)

# r2 below this → the bandwidth model does not explain pipe_time (overhead- or
# noise-dominated); gm_frac is unreliable → confidence low.
R2_TRUST = 0.50
MIN_POINTS = 5

_DT = {"FLOAT16": 2, "FLOAT": 4, "FLOAT32": 4, "BF16": 2, "BFLOAT16": 2,
       "INT8": 1, "UINT8": 1, "BOOL": 1, "INT32": 4, "UINT32": 4,
       "INT64": 8, "UINT64": 8, "INT16": 2, "UINT16": 2, "DOUBLE": 8}

# pipe → (msprof time column, on-chip reference bandwidth)
PIPES = {
    "aic_fixpipe": ("aic_fixpipe_time(us)", FIXPIPE_BW_GBS),
    "aiv_mte3":    ("aiv_mte3_time(us)",    UB_L1_BW_GBS),
}

# config key → msprof PipeUtilization dir basename (without _PipeUtilization).
MEASURE = {
    "Net-Transformer-S256-L1-b1":  "msprof_net_transformer_b1",
    "Net-Transformer-S256-L1-b4":  "msprof_net_transformer_b4",
    "Net-Transformer-S256-L1-b8":  "msprof_net_transformer_b8",
    "Net-Transformer-S256-L1-b16": "msprof_net_transformer_b16",
    "BERT-base-S128-b4":  "msprof_bert_base_b4",
    "BERT-base-S128-b8":  "msprof_bert_base_b8",
    "BERT-base-S128-b16": "msprof_bert_base_b16",
    "GPT-2-S512-b4":  "msprof_gpt2_small_b4",
    "GPT-2-S512-b8":  "msprof_gpt2_small_b8",
    "GPT-2-S512-b16": "msprof_gpt2_small_b16",
    "HF-BERT-S128-b1":  "msprof_hf_bert_b1",
    "HF-BERT-S128-b4":  "msprof_hf_bert_b4",
    "HF-BERT-S128-b8":  "msprof_hf_bert_b8",
    "HF-BERT-S128-b16": "msprof_hf_bert_b16",
    "ModernBERT-base-S4096-b1":      "msprof_modernbert_S4096_b1",
    "ModernBERT-base-S4096-b1-sdpa": "msprof_modernbert_base_prefill_S4096_b1_sdpa",
    "Llama-3.2-1B-prefill-S2048-b1":      "msprof_llama_3_2_1b_prefill_S2048_b1",
    "Llama-3.2-1B-prefill-S2048-b1-sdpa": "msprof_llama_3_2_1b_prefill_S2048_b1_sdpa",
    "Qwen2.5-0.5B-prefill-S2048-b1":      "msprof_qwen2_5_05b_prefill_S2048_b1",
    "Qwen2.5-0.5B-prefill-S2048-b1-sdpa": "msprof_qwen2_5_05b_prefill_S2048_b1_sdpa",
    "SmolLM2-360M-prefill-S2048-b1":      "msprof_smollm2_360m_prefill_S2048_b1",
    "SmolLM2-360M-prefill-S2048-b1-sdpa": "msprof_smollm2_360m_prefill_S2048_b1_sdpa",
    "Phi-3-mini-prefill-S2048-b1-sdpa":   "msprof_phi3_mini_prefill_S2048_b1_sdpa",
    "Qwen3-prefill-S4096-b1-sdpa": "msprof_qwen3_06b_prefill_S4096_b1_sdpa",
    "Qwen3-prefill-S256-b1-sdpa":  "msprof_qwen3_06b_prefill_S256_b1_sdpa",
    "Qwen3-prefill-S256-b4-sdpa":  "msprof_qwen3_06b_prefill_S256_b4_sdpa",
    "Qwen3-prefill-S256-b8-sdpa":  "msprof_qwen3_06b_prefill_S256_b8_sdpa",
    "Qwen3-prefill-S512-b4-sdpa":  "msprof_qwen3_06b_prefill_S512_b4_sdpa",
    "Qwen3-prefill-S512-b8-sdpa":  "msprof_qwen3_06b_prefill_S512_b8_sdpa",
}

# config key → sibling config key (no usable local msprof; copy sibling result).
INHERIT = {
    "BERT-base-S128-b1": "BERT-base-S128-b4",
    "GPT-2-S512-b1":     "GPT-2-S512-b4",
    "Qwen3-prefill-S256-b1": "Qwen3-prefill-S256-b1-sdpa",
    "Qwen3-prefill-S256-b4": "Qwen3-prefill-S256-b4-sdpa",
    "Qwen3-prefill-S256-b8": "Qwen3-prefill-S256-b8-sdpa",
    "Qwen3-prefill-S512-b4": "Qwen3-prefill-S512-b4-sdpa",
    "Qwen3-prefill-S512-b8": "Qwen3-prefill-S512-b8-sdpa",
    "Qwen3-prefill-S4096-b1":   "Qwen3-prefill-S4096-b1-sdpa",
    "Qwen3-Embedding-S4096-b1": "Qwen3-prefill-S4096-b1-sdpa",
}

# decode: no msprof, tiny S=1 vector/cube outputs → on-chip → gm_frac 0.
ASSUMED = {"Qwen3-decode-Min4-Skv128-b1": "decode S=1: tiny outputs, on-chip"}


def _dtype_bytes(s: str) -> int:
    return _DT.get((s or "").strip().upper().replace("DT_", ""), 2)


def _output_bytes(shapes: str, dtypes: str) -> float:
    shapes = (shapes or "").strip().strip('"')
    if not shapes:
        return 0.0
    tensors = [t for t in shapes.split(";") if t.strip()]
    dts = [d for d in (dtypes or "").strip().strip('"').split(";") if d.strip()]
    total = 0.0
    for i, t in enumerate(tensors):
        dims = [int(x) for x in t.split(",") if x.strip().lstrip("-").isdigit()]
        if not dims:
            continue
        n = 1
        for d in dims:
            n *= d
        dt = dts[i] if i < len(dts) else (dts[0] if dts else "FLOAT16")
        total += n * _dtype_bytes(dt)
    return total


def ols(xy: list) -> "dict|None":
    """OLS fit time(us) = intercept + slope*bytes. Returns eff_bw_gbs / overhead / n / r2."""
    n = len(xy)
    if n < MIN_POINTS:
        return None
    mx = sum(x for x, _ in xy) / n
    my = sum(y for _, y in xy) / n
    sxx = sum((x - mx) ** 2 for x, _ in xy)
    if sxx == 0:
        return None
    slope = sum((x - mx) * (y - my) for x, y in xy) / sxx
    intercept = my - slope * mx
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in xy)
    ss_tot = sum((y - my) ** 2 for _, y in xy)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    if slope <= 0:                       # negative slope = noise; bandwidth undefined
        return {"eff_bw_gbs": None, "overhead_us": round(intercept, 3),
                "n": n, "r2": round(r2, 3)}
    eff_bw = (1.0 / slope) / 1e3         # us/byte → GB/s
    return {"eff_bw_gbs": round(eff_bw, 1), "overhead_us": round(intercept, 4),
            "n": n, "r2": round(r2, 3)}


def gm_frac_from_eff_bw(eff_bw: "float|None", onchip_bw: float) -> "float|None":
    """Back-solve GM byte fraction: eff_bw = 1/(gm/HBM + (1-gm)/onchip_bw)."""
    if eff_bw is None or eff_bw <= 0:
        return None
    inv_hbm, inv_on = 1.0 / HBM_BW_GBS, 1.0 / onchip_bw
    if inv_hbm == inv_on:
        return 0.0
    g = (1.0 / eff_bw - inv_on) / (inv_hbm - inv_on)
    return max(0.0, min(1.0, g))


def calibrate_pipe(pts: list, hbm_bw: float, onchip_bw: float) -> dict:
    """Per-pipe calibration via prior-based 2-cluster classification.

    For each op, implied bandwidth = bytes / time. Ops with implied_bw below
    sqrt(hbm * onchip) are classified GM-bound (HBM-write), else on-chip.
    `gm_frac` is the *byte fraction* in the GM cluster — the actionable output
    consumed by `prism.sweep.runner.scale_*_pipes`.

    This is universal: bimodal configs are correctly split; unimodal configs
    end up with all ops in one bucket (method collapses to '1cluster').

    Per-cluster OLS slope is reported as a sanity check (should fall near hbm_bw
    for GM cluster, near onchip_bw for on-chip cluster).
    """
    if not pts:
        return {"gm_frac": None, "method": "none", "n": 0, "confidence": "none"}

    # 1. classify ops by per-op implied bandwidth vs geometric-mean threshold
    threshold_gbs = (hbm_bw * onchip_bw) ** 0.5
    gm_pts, oc_pts = [], []
    for b, t in pts:
        if b <= 0 or t <= 0:
            continue
        bw_op = b / (t * 1e-6) / 1e9     # GB/s
        (gm_pts if bw_op < threshold_gbs else oc_pts).append((b, t))

    gm_bytes = sum(b for b, _ in gm_pts)
    oc_bytes = sum(b for b, _ in oc_pts)
    tot_bytes = gm_bytes + oc_bytes
    if tot_bytes <= 0:
        return {"gm_frac": None, "method": "none", "n": len(pts), "confidence": "none"}
    gm_frac = gm_bytes / tot_bytes

    # 2. per-cluster OLS sanity (does each cluster's fitted bw match its regime?)
    gm_fit = ols(gm_pts) if len(gm_pts) >= MIN_POINTS else None
    oc_fit = ols(oc_pts) if len(oc_pts) >= MIN_POINTS else None
    pooled_fit = ols(pts)

    # 3. method label: '2cluster' iff both buckets non-empty
    method = "2cluster" if gm_pts and oc_pts else "1cluster"

    # 4. confidence: prefer per-cluster r² where applicable, else pooled
    if method == "2cluster":
        r2s = [f["r2"] for f in (gm_fit, oc_fit) if f and f.get("r2") is not None]
        conf = "high" if r2s and min(r2s) >= R2_TRUST else "low"
    else:
        conf = "high" if (pooled_fit and pooled_fit["r2"] >= R2_TRUST) else "low"

    # 5. blended effective bandwidth (for diagnostic display only; the runner
    #    re-computes it per variant from gm_frac).
    eff_bw = 1.0 / (gm_frac / hbm_bw + (1 - gm_frac) / onchip_bw)

    return {
        "gm_frac": round(gm_frac, 3),
        "method": method,
        "eff_bw_gbs": round(eff_bw, 1),
        "n_gm": len(gm_pts), "n_oc": len(oc_pts),
        "gm_bw_eff_gbs": (gm_fit or {}).get("eff_bw_gbs"),
        "oc_bw_eff_gbs": (oc_fit or {}).get("eff_bw_gbs"),
        "pooled_r2": (pooled_fit or {}).get("r2"),
        "confidence": conf,
    }


def calibrate_dir(msprof_dir: Path) -> dict:
    """Calibrate both pipes from one PipeUtilization run."""
    csvs = list(msprof_dir.glob("PROF_*/mindstudio_profiler_output/op_summary*.csv"))
    if not csvs:
        raise FileNotFoundError(f"no op_summary under {msprof_dir}")
    pts = {p: [] for p in PIPES}
    with open(csvs[0], encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ob = _output_bytes(r.get("Output Shapes"), r.get("Output Data Types"))
            if ob <= 0:
                continue
            for pipe, (col, _) in PIPES.items():
                try:
                    t = float(r.get(col) or 0)
                except (TypeError, ValueError):
                    t = 0.0
                if t > 0:
                    pts[pipe].append((ob, t))
    return {p: calibrate_pipe(pts[p], HBM_BW_GBS, onchip_bw)
            for p, (_, onchip_bw) in PIPES.items()}


def main() -> int:
    result = {p: {} for p in PIPES}
    measured = {}
    print(f"{'config':<40}  {'aic_fixpipe (method gm  gm_bw/oc_bw)':<36}  "
          f"{'aiv_mte3 (method gm  gm_bw/oc_bw)':<36}")
    for cfg, basename in MEASURE.items():
        d = REPO / "msprof_data" / f"{basename}_PipeUtilization"
        try:
            rec = calibrate_dir(d)
        except (FileNotFoundError, ValueError) as e:
            print(f"  ⚠ SKIP {cfg}: {e}")
            continue
        measured[cfg] = rec
        for pipe in PIPES:
            r = dict(rec[pipe])
            r["source"] = f"measured:{basename}"
            result[pipe][cfg] = r

        def _fmt(p):
            r = rec[p]
            if r.get("gm_frac") is None:
                return "  (insufficient)"
            g = r["gm_frac"]
            gb = r.get("gm_bw_eff_gbs"); ob = r.get("oc_bw_eff_gbs")
            gbs = f"{gb:.0f}" if isinstance(gb, (int, float)) else "—"
            obs = f"{ob:.0f}" if isinstance(ob, (int, float)) else "—"
            return f"{r['method']:<9} gm={g:.2f}  {gbs}/{obs}"
        print(f"  {cfg:<38}  {_fmt('aic_fixpipe'):<36}  {_fmt('aiv_mte3'):<36}")

    # inherited
    for cfg, sib in INHERIT.items():
        if sib not in measured:
            print(f"  ⚠ SKIP {cfg}: inherit source {sib} missing")
            continue
        for pipe in PIPES:
            src = measured[sib][pipe]
            result[pipe][cfg] = {"gm_frac": src.get("gm_frac"),
                                 "eff_bw_gbs": src.get("eff_bw_gbs"),
                                 "confidence": src.get("confidence", "low"),
                                 "source": f"inherited:{sib}"}
        print(f"  {cfg:<38}  (inherited ← {sib})")

    # assumed
    for cfg, note in ASSUMED.items():
        for pipe in PIPES:
            result[pipe][cfg] = {"gm_frac": 0.0, "eff_bw_gbs": None,
                                 "confidence": "low", "source": f"assumed ({note})"}
        print(f"  {cfg:<38}  (assumed: {note})")

    out = {
        "_meta": {
            "hbm_bw_gbs": HBM_BW_GBS, "fixpipe_bw_gbs": FIXPIPE_BW_GBS,
            "ub_l1_bw_gbs": UB_L1_BW_GBS,
            "method": "Prior-based 2-cluster: classify each op by implied bw "
                      "(bytes/time) against threshold sqrt(hbm*onchip); gm_frac = "
                      "byte fraction in GM cluster. Universal — bimodal configs "
                      "split cleanly, unimodal collapse to '1cluster'. Per-cluster "
                      "OLS slopes reported as sanity (gm_bw_eff_gbs / oc_bw_eff_gbs).",
            "consumer": "prism.sweep.runner.scale_aic_pipes / scale_aiv_pipes",
            "issue": "issue-7-fixpipe-mte3-destination-bw",
            "date": "2026-05-21",
        },
        "aic_fixpipe": result["aic_fixpipe"],
        "aiv_mte3": result["aiv_mte3"],
    }
    out_path = REPO / "data" / "calibration" / "pipe_dest_bw.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n{len(result['aiv_mte3'])} configs × 2 pipes → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
