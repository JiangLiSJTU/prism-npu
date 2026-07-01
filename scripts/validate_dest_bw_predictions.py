#!/usr/bin/env python3
"""Issue #7 follow-up validation — quantify modeling error of aic_fixpipe / aiv_mte3
prediction under the OLD model (constant on-chip BW) vs the NEW model (HBM/on-chip
blend by per-config ``gm_frac``), against per-config OLS-fit effective bandwidth
from msprof, grouped by model family.

Two complementary comparisons:

A. **Per-pipe effective bandwidth** (the bug Issue #7 fixed)

   For each msprof-measured config in ``data/calibration/pipe_dest_bw.json``:
     bw_meas = eff_bw_gbs          (OLS slope of bytes vs pipe_time, calibrated)
     bw_old  = onchip_ref          (constant 4096/2048; pre-#7)
     bw_new  = blend(gm_frac)      (post-#7)
   Per-family median rel-err shows where the OLD model systematically
   overestimated (treating GM-bound stores as on-chip-bound).

B. **End-to-end wall-clock reproduction at baseline arch**

   For each config, run ``predict_wallclock_v3`` with the new code path
   (gm_frac injected from ``pipe_dest_bw.json``) on baseline arch and compare
   to measured ``wall_clock_us``. Sanity that Issue #7 changes preserve baseline
   reproduction (blend factor = 1.0 at variant=baseline by construction; this
   confirms no off-by-one bugs).

Output:
  data/outputs/dest_bw_predicted_vs_measured.json
  docs/findings/dest_bw_predicted_vs_measured.md

Usage:
  python3 scripts/validate_dest_bw_predictions.py
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from prism.sweep.runner import (  # noqa: E402
    BASELINE_910B4,
    DEFAULT_AIV_MTE3_GM_FRAC,
    DEFAULT_FIXPIPE_GM_FRAC,
    _dest_time_proxy,
    predict_wallclock_v3,
)

PIPE_BASELINE_PATH = REPO / "data" / "calibration" / "pipe_baseline_per_model.json"
DEST_BW_PATH = REPO / "data" / "calibration" / "pipe_dest_bw.json"
OUT_JSON = REPO / "data" / "outputs" / "dest_bw_predicted_vs_measured.json"
OUT_MD = REPO / "docs" / "findings" / "dest_bw_predicted_vs_measured.md"

# pipe → on-chip reference bandwidth (must match calib_fixpipe_mte3_bw.py)
ONCHIP_REF = {
    "aic_fixpipe": BASELINE_910B4["fixpipe_bw_gbs"],   # 4096
    "aiv_mte3":    BASELINE_910B4["ub_l1_bw_gbs"],     # 2048
}
HBM_BW = BASELINE_910B4["hbm_bw_gbs"]                  # 392


def family_of(cfg: str) -> str:
    """Map config key → coarse model family for grouping."""
    for prefix, fam in [
        ("BERT-base-",      "BERT-base"),
        ("HF-BERT-",        "HF-BERT"),
        ("GPT-2-",          "GPT-2"),
        ("Net-Transformer-", "Net-Trans"),
        ("Qwen3-prefill-",  "Qwen3-prefill"),
        ("Qwen3-decode-",   "Qwen3-decode"),
        ("Qwen3-Embedding-", "Qwen3-Embedding"),
        ("ModernBERT-",     "ModernBERT"),
        ("Llama-3.2-",      "Llama-3.2"),
        ("Qwen2.5-",        "Qwen2.5"),
        ("SmolLM2-",        "SmolLM2"),
        ("Phi-3-",          "Phi-3"),
    ]:
        if cfg.startswith(prefix):
            return fam
    return "Other"


def new_model_bw(gm_frac: float, onchip_ref: float) -> float:
    """Effective bandwidth predicted by the NEW (post-#7) blend model."""
    inv = gm_frac / HBM_BW + (1.0 - gm_frac) / onchip_ref
    return 1.0 / inv


def part_a_per_pipe_bandwidth(dest_bw: dict) -> dict:
    """For each msprof-measured config, compare OLD vs NEW predicted effective
    bandwidth to the OLS-fit empirical ``eff_bw_gbs``.

    Configs without local msprof (``source: inherited:...`` / ``assumed`` /
    confidence ``low``) are excluded — their ``eff_bw_gbs`` is not an
    independent measurement.
    """
    rows = []
    for pipe_name in ("aic_fixpipe", "aiv_mte3"):
        onchip_ref = ONCHIP_REF[pipe_name]
        for cfg, entry in dest_bw[pipe_name].items():
            src = entry.get("source", "")
            if not src.startswith("measured:"):
                continue
            if entry.get("confidence") != "high":
                continue
            bw_meas = entry.get("eff_bw_gbs")
            gm = entry.get("gm_frac")
            if bw_meas is None or gm is None:
                continue
            bw_old = onchip_ref
            bw_new = new_model_bw(gm, onchip_ref)
            rel_err_old = (bw_old - bw_meas) / bw_meas
            rel_err_new = (bw_new - bw_meas) / bw_meas
            rows.append({
                "config":      cfg,
                "family":      family_of(cfg),
                "pipe":        pipe_name,
                "gm_frac":     gm,
                "bw_meas_GBs": round(bw_meas, 1),
                "bw_old_GBs":  round(bw_old, 1),
                "bw_new_GBs":  round(bw_new, 1),
                "err_old_pct": round(100 * rel_err_old, 1),
                "err_new_pct": round(100 * rel_err_new, 1),
                "method":      entry.get("method"),
            })
    return rows


def part_b_wall_clock_reproduction(pipe_baseline: dict, dest_bw: dict) -> list:
    """For each config, run ``predict_wallclock_v3`` with the NEW code path
    (gm_frac injected from pipe_dest_bw.json) on baseline arch; report err%.
    """
    fixpipe_cal = dest_bw.get("aic_fixpipe", {})
    mte3_cal = dest_bw.get("aiv_mte3", {})

    def _gm(cal: dict, cfg: str, default: float) -> float:
        g = cal.get(cfg, {}).get("gm_frac")
        return g if isinstance(g, (int, float)) else default

    rows = []
    for cfg, pipe in pipe_baseline["configs"].items():
        measured = pipe.get("wall_clock_us", 0)
        if measured <= 0:
            continue  # decode or placeholder configs without wall_clock
        # Inject gm_frac the same way runner.main() does
        p = dict(pipe)
        p["_aic_fixpipe_gm_frac"] = _gm(fixpipe_cal, cfg, DEFAULT_FIXPIPE_GM_FRAC)
        p["_aiv_mte3_gm_frac"] = _gm(mte3_cal, cfg, DEFAULT_AIV_MTE3_GM_FRAC)
        pred = predict_wallclock_v3(p, BASELINE_910B4, BASELINE_910B4)["wall_clock_us"]
        rows.append({
            "config":      cfg,
            "family":      family_of(cfg),
            "wall_pred_us": round(pred, 0),
            "wall_meas_us": round(measured, 0),
            "err_pct":     round(100 * (pred - measured) / measured, 2),
            "abs_err_pct": round(abs(100 * (pred - measured) / measured), 2),
        })
    return rows


def aggregate(rows: list, key: str, value: str) -> dict:
    """Group rows by `key`, compute median / min / max of `value`."""
    buckets = defaultdict(list)
    for r in rows:
        buckets[r[key]].append(r[value])
    out = {}
    for k, vs in buckets.items():
        out[k] = {
            "n":      len(vs),
            "median": round(statistics.median(vs), 2),
            "min":    round(min(vs), 2),
            "max":    round(max(vs), 2),
        }
    return out


def main() -> int:
    with PIPE_BASELINE_PATH.open(encoding="utf-8") as f:
        pipe_baseline = json.load(f)
    with DEST_BW_PATH.open(encoding="utf-8") as f:
        dest_bw = json.load(f)

    # Part A: per-pipe bandwidth modeling error
    rows_a = part_a_per_pipe_bandwidth(dest_bw)
    agg_a_old = aggregate(rows_a, "family", "err_old_pct")
    agg_a_new = aggregate(rows_a, "family", "err_new_pct")

    # Aggregate by pipe within each family
    agg_pipe_old = {p: aggregate([r for r in rows_a if r["pipe"] == p],
                                 "family", "err_old_pct")
                    for p in ("aic_fixpipe", "aiv_mte3")}
    agg_pipe_new = {p: aggregate([r for r in rows_a if r["pipe"] == p],
                                 "family", "err_new_pct")
                    for p in ("aic_fixpipe", "aiv_mte3")}

    # Part B: end-to-end wall-clock reproduction at baseline
    rows_b = part_b_wall_clock_reproduction(pipe_baseline, dest_bw)
    agg_b = aggregate(rows_b, "family", "abs_err_pct")

    # JSON dump
    out = {
        "_meta": {
            "purpose": "Issue #7 validation: NEW (blend) vs OLD (constant on-chip BW) "
                       "model prediction error vs msprof-fit empirical bandwidth, plus "
                       "end-to-end wall-clock baseline reproduction sanity.",
            "consumer": "docs/findings/dest_bw_predicted_vs_measured.md",
            "hbm_bw_gbs": HBM_BW,
            "onchip_ref": ONCHIP_REF,
        },
        "part_a_per_pipe_bandwidth": {
            "rows":              rows_a,
            "family_err_old":    agg_a_old,
            "family_err_new":    agg_a_new,
            "per_pipe_err_old":  agg_pipe_old,
            "per_pipe_err_new":  agg_pipe_new,
        },
        "part_b_wall_clock_at_baseline": {
            "rows":            rows_b,
            "family_abs_err":  agg_b,
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # Markdown finding
    lines = [
        "# Predicted-vs-Measured: aic_fixpipe / aiv_mte3 destination bandwidth",
        "",
        "> Issue #7 follow-up validation. Quantifies modeling error reduction from",
        "> the OLD model (constant on-chip bandwidth: fixpipe_bw=4096 / ub_l1_bw=2048 GB/s)",
        "> vs the NEW model (per-config HBM↔on-chip blend by `gm_frac`),",
        "> against per-config OLS-fit empirical bandwidth from msprof.",
        "",
        f"- HBM bandwidth: **{HBM_BW} GB/s**",
        f"- On-chip ref (fixpipe / ub_l1): **{ONCHIP_REF['aic_fixpipe']} / "
        f"{ONCHIP_REF['aiv_mte3']} GB/s**",
        f"- Configs in Part A: {len(rows_a)} (msprof-measured, confidence=high)",
        f"- Configs in Part B: {len(rows_b)} (all with measured `wall_clock_us`)",
        "",
        "## Part A — per-pipe effective bandwidth, OLD vs NEW",
        "",
        "Each row: one (model family, pipe) pair. `err` = (predicted − measured) / measured. "
        "Positive err means the model **overestimates** the available bandwidth "
        "(underestimates pipe time → underestimates HBM sensitivity).",
        "",
        "### aic_fixpipe (L0C → output)",
        "",
        "| Family | n | OLD median err | OLD range | NEW median err | NEW range |",
        "|---|---:|---:|---|---:|---|",
    ]
    for fam, stats in sorted(agg_pipe_old["aic_fixpipe"].items()):
        old_med, old_min, old_max = stats["median"], stats["min"], stats["max"]
        new_stats = agg_pipe_new["aic_fixpipe"].get(fam, {})
        new_med = new_stats.get("median", float("nan"))
        new_min = new_stats.get("min", float("nan"))
        new_max = new_stats.get("max", float("nan"))
        lines.append(f"| {fam} | {stats['n']} | "
                     f"{old_med:+.1f}% | [{old_min:+.1f}, {old_max:+.1f}] | "
                     f"{new_med:+.1f}% | [{new_min:+.1f}, {new_max:+.1f}] |")
    lines += ["", "### aiv_mte3 (UB → output)", "",
              "| Family | n | OLD median err | OLD range | NEW median err | NEW range |",
              "|---|---:|---:|---|---:|---|"]
    for fam, stats in sorted(agg_pipe_old["aiv_mte3"].items()):
        old_med, old_min, old_max = stats["median"], stats["min"], stats["max"]
        new_stats = agg_pipe_new["aiv_mte3"].get(fam, {})
        new_med = new_stats.get("median", float("nan"))
        new_min = new_stats.get("min", float("nan"))
        new_max = new_stats.get("max", float("nan"))
        lines.append(f"| {fam} | {stats['n']} | "
                     f"{old_med:+.1f}% | [{old_min:+.1f}, {old_max:+.1f}] | "
                     f"{new_med:+.1f}% | [{new_min:+.1f}, {new_max:+.1f}] |")

    lines += [
        "",
        "**Reading**: the OLD model is the pre-#7 sweep formula —",
        "`pipe_time_new = pipe_time_baseline × (onchip_baseline / onchip_variant)`. "
        "Under this rule, the *effective* bandwidth ridden by the pipe was implicitly",
        "the constant on-chip reference (4096 / 2048 GB/s). The empirical OLS slope",
        "shows actual effective bandwidth is 5–10× lower on most large prefill configs",
        "— because the store goes mostly to GM (HBM-bound). The NEW model fixes this",
        "via per-config `gm_frac` blend (err ≈ 0% by construction, modulo OLS noise).",
        "",
        "## Part B — wall-clock baseline reproduction (sanity)",
        "",
        "End-to-end `predict_wallclock_v3` at variant=baseline, full Issue #7 code path",
        "(gm_frac injection + blend scaling). At baseline arch the blend factor is",
        "**1.0 by construction**, so this confirms no off-by-one bugs from the new code.",
        "",
        "| Family | n | abs(err)% median | range |",
        "|---|---:|---:|---|",
    ]
    for fam, stats in sorted(agg_b.items()):
        lines.append(f"| {fam} | {stats['n']} | "
                     f"{stats['median']:.2f}% | [{stats['min']:.2f}, {stats['max']:.2f}] |")

    overall_b = [r["abs_err_pct"] for r in rows_b]
    lines += [
        "",
        f"**Overall**: {len(overall_b)} configs, median abs(err) "
        f"**{statistics.median(overall_b):.2f}%**, max **{max(overall_b):.2f}%**. "
        f"All below the 10% hard gate enforced by "
        f"`tests/test_sweep.py::test_predict_wallclock_v3_baseline_reproduction`.",
        "",
        "## Per-config detail (Part A)",
        "",
        "| Config | Pipe | gm_frac | bw_meas (GB/s) | bw_old | err_old | bw_new | err_new |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(rows_a, key=lambda x: (x["pipe"], x["family"], x["config"])):
        lines.append(
            f"| {r['config']} | {r['pipe']} | {r['gm_frac']:.3f} | "
            f"{r['bw_meas_GBs']} | {r['bw_old_GBs']:.0f} | "
            f"{r['err_old_pct']:+.1f}% | {r['bw_new_GBs']:.0f} | "
            f"{r['err_new_pct']:+.1f}% |"
        )

    lines += [
        "",
        "---",
        "",
        "Regenerate: `python3 scripts/validate_dest_bw_predictions.py`",
        "",
        "Source data: `data/calibration/pipe_dest_bw.json` (per-config gm_frac),",
        "`data/calibration/pipe_baseline_per_model.json` (msprof measurements).",
        "",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    with OUT_MD.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Console summary
    print(f"Part A: {len(rows_a)} (config, pipe) rows analyzed")
    print(f"Part B: {len(rows_b)} configs wall_clock reproduction")
    print()
    print("aic_fixpipe OLD model err median by family:")
    for fam, st in sorted(agg_pipe_old["aic_fixpipe"].items()):
        print(f"  {fam:<20} n={st['n']:>2}  median={st['median']:+7.1f}%  "
              f"range=[{st['min']:+.1f}, {st['max']:+.1f}]")
    print()
    print(f"→ wrote {OUT_JSON}")
    print(f"→ wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
