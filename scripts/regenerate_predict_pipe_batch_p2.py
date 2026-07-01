#!/usr/bin/env python3
"""Regenerate data/calibration/predict_pipe_batch_p2.json on 10 model YAMLs.

Used to refresh the batch after any AIV-model change (e.g. v3 -> v4 Method B).
Output matches pipe_baseline_per_model.json schema so prism-sweep can consume it
directly.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from prism.predict_pipe import ModelSpec, predict_pipe_baseline   # noqa: E402
from prism.predict_pipe.predict import _arch_dict_from_yaml        # noqa: E402

# (yaml_filename, batch) — preserve the 10 configs from the v3 batch.
CONFIGS = [
    ("bert_base.yaml", 1),
    ("bge_small_en_v15_S512.yaml", 1),
    ("deberta_base_S512.yaml", 1),
    ("flan_t5_base_encoder_S512.yaml", 1),
    ("llama_3_2_1b_prefill_S2048.yaml", 1),
    ("modernbert_base_prefill_S4096.yaml", 1),
    ("qwen2_5_0_5b_prefill_S2048.yaml", 1),
    ("smollm2_135m_decode.yaml", 1),
    ("smollm2_360m_prefill_S2048.yaml", 1),
    ("t5_small_encoder_S512.yaml", 1),
]

ARCH_YAML = _REPO / "arch" / "ascend_910b4_for_sweep_v2.yaml"
PARAMS_JSON = _REPO / "data" / "calibration" / "predict_pipe_params.json"
OUT_JSON = _REPO / "data" / "calibration" / "predict_pipe_batch_p2.json"


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--params", default=str(PARAMS_JSON),
                   help="fitted params JSON (e.g. predict_pipe_params_v6.json)")
    p.add_argument("--output", default=str(OUT_JSON))
    args = p.parse_args()

    arch = _arch_dict_from_yaml(ARCH_YAML)
    params = json.load(open(args.params, encoding="utf-8"))
    print(f"  Using params: {args.params}  (v_model={params.get('v_model','v4')})")

    out = {
        "baseline_arch_name": "ascend_910b4_for_sweep_v2",
        "configs": {},
    }

    for yaml_name, batch in CONFIGS:
        yaml_path = _REPO / "models" / "regime" / yaml_name
        spec = ModelSpec.from_yaml(yaml_path)
        entry = predict_pipe_baseline(spec, arch, params, batch=batch)
        # Use entry name from spec (matches the v3 batch's keying convention)
        out["configs"][spec.name] = entry
        print(f"  {spec.name:42s} wall={entry['wall_clock_us']:>9.0f}  "
              f"aic={entry['aic_time_us']:>8.0f}  aiv={entry['aiv_time_us']:>9.0f}  "
              f"conf={entry['confidence'][:30]}")

    out_path = Path(args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {len(out['configs'])} configs -> {out_path}")


if __name__ == "__main__":
    main()
