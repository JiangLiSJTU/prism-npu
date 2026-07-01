# Agent Context for PRISM

> Read this before doing any work in this repo. Auto-loaded by opencode / Claude Code / Cursor.

## What this tool is

**PRISM** = Pipeline-aware Roofline & Inference Sweep Model. A wall-clock + TCO prediction toolkit for **NPU architecture exploration**. Given a target inference workload + candidate NPU configurations, it answers: *"Should we increase Cube units? More HBM bandwidth? Add UB+L1 fusion? What's the optimization ceiling?"*

The empirical anchor is **Ascend 910B4** (publicly documented NPU), calibrated via **msprof PipeUtilization** on real silicon. The methodology generalizes to any NPU with comparable pipe-level performance counters.

## Architecture (5-second mental model)

```
msprof CSV → prism-extract → cube_util_extracted.json
                              ↓
                     prism-fit → eta_physics_fit.json (5 params, BERT MAE < 15 pp gate)
                              ↓
                     pipe_baseline_per_model.json  ←──── prism-predict-pipe
                              ↓                         (new: predict baseline
              ┌─────────────────────┬──────────────────────┐  for new models
              ↓                     ↓                      ↓   without msprof)
          prism-sweep        prism-ceiling         prism-regime
        (12-dim arch)      (5 optimization       (host/compute/
                            scenarios)           memory-bound)
              ↓                     ↓
                       prism-render → docs/findings/
```

## Critical invariants — DO NOT BREAK

1. **`prism-render --check` exits 0** — sanity gate for templates ↔ rendered findings
2. **BERT validation MAE < 15 pp** — η_real fit hard gate (`tests/test_eta_real.py`)
3. **Baseline reproduction within 10%** — sweep MODELS dict integrity (`tests/test_sweep.py`)
4. **`pytest tests/` all pass** (25 unit + 5 e2e-skip)
5. **No identity references**: never reintroduce the original company / business-unit / chip-name terms — see `## Sanitization rules` below for the canonical "do not reintroduce" set

## Locked files — DO NOT MODIFY

- `arch/ascend_910b4*.yaml` — calibrated against real hardware, modifying invalidates all measurements
- `arch/ascend_310p*.yaml` — same
- `models/*.yaml` — calibration target specs
- `data/calibration/eta_physics_fit.json` — fitted parameters with documented MAE
- `data/calibration/pipe_baseline_per_model.json` — empirical msprof measurements

To change these: must re-run Tier 3 calibration (Ascend NPU + msprof) and document new MAE in commit.

## Tier-aware capability map

| Tier | Capability | Hardware needed |
|------|-----------|-----------------|
| 1 — Predict | `prism-sweep`, `prism-ceiling`, `prism-render`, `prism-regime`, `prism-fit`, `prism-extract`, `prism-predict-pipe` | Any laptop (incl. Windows) |
| 2 — Mapping verify | `prism-mapping` | Any laptop + Docker (`accelergy/timeloop:latest`) |
| 3 — Recalibrate | `benchmark/*` + Tier 1 | **Ascend 910B/910B4 NPU** + CANN 8.5 |

> `prism-predict-pipe` (Issue #2) closes the Tier 1 capability gap: it analytically
> predicts the per-pipe baseline for a new model (without msprof) so `prism-sweep`
> and `prism-ceiling` become usable on models that haven't been profiled yet.
> See `docs/methodology/08_predict_pipe.md`.

If you're an agent reviewing on Windows without NPU/Docker:
- Tier 1 is fully testable
- Tier 2 testable with Docker Desktop (slower than Linux but works)
- Tier 3 cannot be exercised — review the scripts statically

## Sanitization rules (project-wide)

This repo was sanitized for public release. Future commits **must not** reintroduce identifying terms from the original internal context:

- The original company brand name (drop entirely)
- The original business-unit name → use "固定网络" (CN) / "fixed network" (EN), the industry-neutral term
- The original "BU + AI 芯片" framing for the project → use generic "NPU"
- "Self-developed X-chip" framing → just "NPU"

**Keep as factual references** (publicly documented hardware/toolkit names): Ascend, CANN, DaVinci, msprof, ais_bench. These describe the empirical calibration target, not project identity.

**Exception**: academic citations referencing a paper's institutional byline are factual and OK.

If you're unsure whether a term is acceptable, treat the four bullets above as the canonical "do not reintroduce" set; when still in doubt, prefer the industry-neutral term.

## Windows-specific gotchas

- **Chinese filenames in repo**: `docs/findings/主报告.md`, `reports/templates/主报告_v2.md.j2` etc. PowerShell handles UTF-8 if `[Console]::OutputEncoding = [System.Text.Encoding]::UTF8` is set in profile.
- **Line endings**: `.gitattributes` should handle, but if you see CRLF/LF diff churn, that's the issue.
- **pytest in venv**: `python -m venv .venv && .venv\Scripts\activate && pip install -e .[dev]` then `pytest tests/`.
- **Docker for Tier 2**: install Docker Desktop, then `prism-mapping` works the same as on Linux. Docker on Windows is slower (~2-3× the Linux runtime).
- **`scripts/prism_*.py` shebang**: works regardless on Windows since you'd invoke as `python scripts/prism_*.py`.

## Modeling discipline — lessons baked in

These are hard-won from prior sessions. Violating them produces plausible-looking but wrong work.

1. **Persist findings before they can be compacted.** A literature survey / analysis delivered only in chat gets mangled or lost when the session summary is compacted. A real example: a 5-paper survey delivered chat-only later resurfaced with a fabricated author attribution ("Liang & Gong" instead of the correct "Wang et al., ASPLOS 2025"). If a finding matters, write it to a file (doc, handoff, or `docs/methodology/`) in the same turn you produce it.

2. **Never dress an empirical fit as physics.** If a term / exponent / constant was chosen because it made the fit converge, the code comment and doc must say "empirically fit" — not "physically motivated". A real example: `(w_proxy/1000)²` in `predict_aiv_v2` was a dynamic-range hack (3.4× → 11.4×), not a derived law. Honest labeling tells the next person it is a hyperparameter to re-fit, not a constant to trust.

3. **Always label a MAE as in-sample or cross-validated.** `predict_pipe` AIV v4 reports 4.9% — but that is in-sample (n=6); LOO CV is not yet run. A bare "4.9% MAE" without the in-sample qualifier overstates confidence. Only OOS / LOO numbers justify extrapolation claims.

4. **Verify the call graph, not just function existence.** `physics.py` defines `aiv_per_kernel_overhead`, `eta_repeat`, `eta_ub_bandwidth` — none are called in the v4 hot path (dead code from v3). Before reasoning about a function's effect, `grep` for its call sites; a definition is not a usage.

## When in doubt

- Methodology questions → `docs/methodology/01_overview.md` is the entry point for the 7-doc series
- API questions → `docs/reference/api.md`
- Adding a model → `docs/tutorials/04_add_new_model.md`
- Recalibrating → `docs/tutorials/03_recalibrate_with_new_msprof.md`
- Architecture decisions → `docs/findings/主报告.md` is the synthesis report
