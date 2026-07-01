#!/usr/bin/env python3
"""
Phase G+ G1 — Roofline 决策门工具（regime gate）

输入：arch yaml（chip + calib）+ model roofline yaml（layers, ops_b1, bytes_total, ...）
输出：每对 (model, arch, batch) 的 regime + timeloop_needed flag

依据：predict_910b4_v2() in scripts/perf_estimation/roofline_910b4_calibrated_v2.py
决策：见 docs/timeloop_decision_gate.md

用法：
  # 单点：
  python regime_gate.py --arch ../../arch/regime/ascend_910b4_with_calib.yaml \
                        --model ../../models/regime/qwen3_0.6b.yaml --batch 1

  # 全量 sweep：
  python regime_gate.py --sweep --output ../../data/regime_matrix.json
"""

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import yaml

# ── 导入 predict_910b4_v2 ──────────────────────────────────────────
HERE = Path(__file__).resolve().parent
from prism.roofline import predict as r  # noqa: E402


# Names of `predict` module globals we override per arch_yaml. Keep this list
# in sync with what predict_910b4_v2() reads at runtime.
_PATCHED_GLOBALS = ("FP16_TFLOPS", "HBM_BW_GBS", "L2_MB", "ETA_COMPUTE", "CALIB_V2")


def _build_overrides(arch_yaml: dict) -> dict:
    """Compute the per-arch override dict — pure function, no side effects."""
    chip = arch_yaml["chip"]
    calib = arch_yaml["calib"]
    return {
        "FP16_TFLOPS": float(chip["fp16_tflops"]),
        "HBM_BW_GBS":  float(chip["hbm_bw_gbs"]),
        "L2_MB":       float(chip["l2_mb"]),
        "ETA_COMPUTE": float(calib.get("eta_compute", 0.70)),
        "CALIB_V2": {
            "beta_device_us":               float(calib["beta_device_us"]),
            "beta_layer_enc_us":            float(calib["beta_layer_enc_us"]),
            "beta_layer_enc_d768_us":       float(calib.get("beta_layer_enc_d768_us", calib["beta_layer_enc_us"])),
            "beta_layer_enc_d256_us":       float(calib.get("beta_layer_enc_d256_us", calib["beta_layer_enc_us"])),
            "beta_layer_dec_S256_d384_us":  float(calib.get("beta_layer_dec_S256_d384_us", 80.4)),
            "beta_layer_dec_S512_d768_us":  float(calib.get("beta_layer_dec_S512_d768_us", 355.6)),
            "beta_layer_dec_S512_d1024_us": float(calib.get("beta_layer_dec_S512_d1024_us", 560.4)),
            "eta_compute":                  float(calib.get("eta_compute", 0.70)),
            "l2_mb":                        float(chip["l2_mb"]),
        },
    }


@contextmanager
def arch_context(arch_yaml: dict):
    """Temporarily override `predict` module globals from arch_yaml.

    Why a context manager: `predict_910b4_v2()` reads hardware constants and
    CALIB_V2 from module globals (legacy design). Mutating them in a loop is
    fragile — exceptions or future parallelism would leak state. This wrapper
    snapshots the originals on entry and restores them on exit (even on error).

    The pattern still requires single-threaded callers within the `with` block.
    """
    overrides = _build_overrides(arch_yaml)
    saved = {name: getattr(r, name) for name in _PATCHED_GLOBALS}
    try:
        for name, value in overrides.items():
            setattr(r, name, value)
        yield
    finally:
        for name, value in saved.items():
            setattr(r, name, value)


def predict(arch_yaml: dict, model_yaml: dict, batch: int) -> dict:
    """对一对 (arch, model, batch) 调用 predict_910b4_v2，返回扩展 dict。"""
    with arch_context(arch_yaml):
        out = r.predict_910b4_v2(
            layers=int(model_yaml["layers"]),
            ops_b1=float(model_yaml["ops_b1"]),
            bytes_total=float(model_yaml["bytes_total"]),
            batch=int(batch),
            arch=str(model_yaml["arch"]),
            beta_layer_override=model_yaml.get("beta_layer_override"),
            alpha_per_batch=model_yaml.get("alpha_per_batch"),
        )

    # 决策：仅"调度受限（β 主导）"才 timeloop_needed = False
    out["timeloop_needed"] = (out["regime"] != "调度受限（β 主导）")

    # 加 dominant 字段
    dominant_us = max(out["T_compute_us"], out["T_memory_us"], out["T_overhead_us"])
    if dominant_us == out["T_overhead_us"]:
        out["dominant"] = "T_overhead_us"
    elif dominant_us == out["T_compute_us"]:
        out["dominant"] = "T_compute_us"
    else:
        out["dominant"] = "T_memory_us"

    return out


def load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_single(args) -> None:
    arch = load_yaml(args.arch)
    model = load_yaml(args.model)
    out = predict(arch, model, args.batch)
    out["model"] = model["name"]
    out["arch"] = arch["name"]
    print(json.dumps(out, ensure_ascii=False, indent=2))


def run_sweep(args) -> int:
    from prism import data_root_or_fallback
    repo = data_root_or_fallback()
    arch_dir = repo / "arch" / "regime"
    model_dir = repo / "models" / "regime"

    arches = sorted(arch_dir.glob("*.yaml"))
    models = sorted(model_dir.glob("*.yaml"))
    batches = [1, 4, 8, 16, 32, 64, 128]   # H 阶段扩展：加 32/64/128 暴露 batch 放大下的 compute-bound

    results = []
    failures: list[tuple[str, str, int, str]] = []
    for arch_path in arches:
        arch = load_yaml(arch_path)
        for model_path in models:
            model = load_yaml(model_path)
            for B in batches:
                try:
                    out = predict(arch, model, B)
                    out["model"] = model["name"]
                    out["arch"] = arch["name"]
                    results.append(out)
                except (KeyError, ValueError, TypeError, ZeroDivisionError) as e:
                    triple = (arch["name"], model["name"], B)
                    failures.append((*triple, f"{type(e).__name__}: {e}"))
                    print(
                        f"  WARN  {triple[0]} × {triple[1]} × B={triple[2]}: {type(e).__name__}: {e}",
                        file=sys.stderr,
                    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    total = len(results) + len(failures)
    print(f"OK  {len(results)}/{total} entries → {out_path}")
    if results:
        n_need = sum(1 for r_ in results if r_["timeloop_needed"])
        print(f"    timeloop_needed=True: {n_need} / {len(results)} ({100 * n_need / len(results):.1f}%)")

    # 统计：每个 (model, arch) 在哪些 batch 切到 timeloop_needed=True
    by_pair = {}
    for entry in results:
        if entry["timeloop_needed"]:
            key = (entry["model"], entry["arch"])
            by_pair.setdefault(key, []).append(entry["batch"])
    if by_pair:
        print(f"    Triples switching to non-overhead regime:")
        for (m, a), batches_ in sorted(by_pair.items()):
            print(f"      {m:20s} × {a:30s}  B={batches_}")

    if failures:
        print(
            f"\n  FAIL  {len(failures)} sweep triple(s) failed; results JSON is incomplete.",
            file=sys.stderr,
        )
        return 1
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", help="arch yaml path (single mode)")
    p.add_argument("--model", help="model yaml path (single mode)")
    p.add_argument("--batch", type=int, default=1, help="batch size (single mode)")
    p.add_argument("--sweep", action="store_true", help="full matrix sweep")
    p.add_argument("--output", default="data/regime_matrix.json", help="output JSON for sweep")
    args = p.parse_args()

    if args.sweep:
        return run_sweep(args)
    if not (args.arch and args.model):
        p.error("--arch and --model required in single mode")
    run_single(args)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
