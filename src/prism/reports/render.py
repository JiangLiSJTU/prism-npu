#!/usr/bin/env python3
"""
render_reports.py — Jinja2-based markdown report rendering for the
Phase G+ macro-variable / templating system.

USAGE:
    # Render templates to output directory
    python scripts/render_reports.py \
        --vars data/experiment_variables.json \
        --templates-dir reports/templates \
        --output-dir reports/current

    # Verify rendered output matches the existing files (CI mode);
    # exit 1 if any drift is detected.
    python scripts/render_reports.py --check

    # Show what would change without writing files
    python scripts/render_reports.py --dry-run

PURPOSE:
    Single source of truth for ~65 experiment-driven values that appear in
    multiple Markdown reports under the repo root. Editing
    experiment_variables.json and re-running this script keeps every
    consuming report in lock-step, eliminating drift between values cited
    in narrative text and the underlying experimental data.

DESIGN:
    * Uses jinja2.StrictUndefined so any missing key surfaces as a hard
      error instead of silently rendering an empty string.
    * Vars are loaded as a nested dictionary: templates may reference
      `{{ msprof_910b4.wall_clock_bert_ms }}`. For convenience, every
      leaf value is also exposed at the top level (e.g. `{{ wall_clock_bert_ms }}`).
      When a leaf name appears under multiple groups, the first occurrence
      wins. Templates should prefer top-level shortnames for readability.
    * Templates live in `reports/templates/<name>.md.j2` and render to
      `reports/current/<name>.md` (or other output paths matching the
      original locations of the reports — see the OUTPUT_PATH_MAP table).

OUTPUT PATH MAP:
    Some templates render to files outside reports/current. The
    OUTPUT_PATH_MAP dictionary records each template's intended output
    location relative to the repo root.
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path
from typing import Any, Dict

import jinja2


from prism import data_root_or_fallback

SCRIPT_DIR = Path(__file__).resolve().parent
SIM_ROOT = data_root_or_fallback()  # repo root containing data/, arch/, reports/
PROJECT_ROOT = SIM_ROOT.parent  # one level above repo root

DEFAULT_VARS = SIM_ROOT / "data" / "experiment_variables.json"
DEFAULT_TEMPLATES = SIM_ROOT / "reports" / "templates"
DEFAULT_OUTPUT = SIM_ROOT / "docs" / "findings"


# Template-name -> absolute output path. 输入（reports/templates/）与输出（docs/findings/）分离。
# 主报告.md 是面向决策者的核心产物，单独取直接命名（不含 v2/v3 版本号印记）。
OUTPUT_PATH_MAP: Dict[str, Path] = {
    "主报告_v2.md.j2":                    SIM_ROOT / "docs" / "findings" / "主报告.md",
    "910B4_roofline_校准报告_v3.md.j2":   SIM_ROOT / "docs" / "findings" / "roofline校准报告.md",
    "微架构探索_报告_v3.md.j2":           SIM_ROOT / "docs" / "findings" / "微架构探索报告.md",
    "msprof_breakdown_summary.md.j2":     SIM_ROOT / "docs" / "findings" / "msprof分解报告.md",
}


def load_vars(path: Path) -> Dict[str, Any]:
    """Load JSON vars file. Flatten leaves to top level for shortname access."""
    with path.open("r", encoding="utf-8") as f:
        nested = json.load(f)

    flat: Dict[str, Any] = {}
    for group_name, group_val in nested.items():
        if isinstance(group_val, dict):
            for leaf_name, leaf_val in group_val.items():
                # Only add to top level if not already present (first-wins).
                if leaf_name not in flat:
                    flat[leaf_name] = leaf_val

    # Merge: keep nested groups accessible as `{{ msprof_910b4.foo }}`,
    # plus expose every leaf at the top level.
    merged = dict(nested)
    for k, v in flat.items():
        merged.setdefault(k, v)
    return merged


def render_template(env: jinja2.Environment, template_name: str, vars_dict: Dict[str, Any]) -> str:
    """Render one template and return its content as a string."""
    template = env.get_template(template_name)
    return template.render(**vars_dict)


def output_path_for(template_name: str, output_dir: Path) -> Path:
    """Resolve the absolute output path for a given template name."""
    if template_name in OUTPUT_PATH_MAP:
        return OUTPUT_PATH_MAP[template_name]
    # Default: drop the .j2 suffix and put under output_dir.
    if template_name.endswith(".j2"):
        return output_dir / template_name[: -len(".j2")]
    return output_dir / template_name


def diff_text(a: str, b: str, label_a: str, label_b: str) -> str:
    """Return a unified diff between two strings."""
    return "".join(
        difflib.unified_diff(
            a.splitlines(keepends=True),
            b.splitlines(keepends=True),
            fromfile=label_a,
            tofile=label_b,
            n=3,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render Markdown reports from Jinja2 templates and a JSON vars file.",
    )
    parser.add_argument(
        "--vars",
        type=Path,
        default=DEFAULT_VARS,
        help=f"Path to experiment_variables.json (default: {DEFAULT_VARS})",
    )
    parser.add_argument(
        "--templates-dir",
        type=Path,
        default=DEFAULT_TEMPLATES,
        help=f"Directory containing *.md.j2 templates (default: {DEFAULT_TEMPLATES})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Default directory for rendered output (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify rendered output matches existing files; exit 1 on drift.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print diffs of what would change but do not write any files.",
    )
    args = parser.parse_args()

    vars_path = args.vars.resolve()
    templates_dir = args.templates_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not vars_path.is_file():
        print(f"ERROR: vars file not found: {vars_path}", file=sys.stderr)
        return 2
    if not templates_dir.is_dir():
        print(f"ERROR: templates dir not found: {templates_dir}", file=sys.stderr)
        return 2

    vars_dict = load_vars(vars_path)

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(templates_dir)),
        undefined=jinja2.StrictUndefined,
        keep_trailing_newline=True,
        autoescape=False,
    )

    templates = sorted(p.name for p in templates_dir.glob("*.md.j2"))
    if not templates:
        print(f"WARNING: no *.md.j2 templates found in {templates_dir}", file=sys.stderr)
        return 0

    drift_detected = False
    summary_lines = []

    for tname in templates:
        try:
            rendered = render_template(env, tname, vars_dict)
        except jinja2.exceptions.UndefinedError as e:
            print(f"ERROR: undefined variable in {tname}: {e}", file=sys.stderr)
            return 2
        except jinja2.exceptions.TemplateError as e:
            print(f"ERROR: template error in {tname}: {e}", file=sys.stderr)
            return 2

        out_path = output_path_for(tname, output_dir)
        existing = out_path.read_text(encoding="utf-8") if out_path.is_file() else None

        if existing is None:
            summary_lines.append(f"NEW: {tname} -> {out_path} ({len(rendered)} bytes)")
            if args.check:
                drift_detected = True
                print(f"[CHECK] missing output file: {out_path}", file=sys.stderr)
            elif args.dry_run:
                print(f"[DRY-RUN] would create {out_path} ({len(rendered)} bytes)")
            else:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(rendered, encoding="utf-8")
                print(f"WROTE: {out_path}")
        elif existing == rendered:
            summary_lines.append(f"OK:  {tname} (identical)")
            if not args.check and not args.dry_run:
                # No-op: contents match, do not rewrite to preserve mtime.
                pass
        else:
            drift_detected = True
            d = diff_text(existing, rendered, str(out_path), f"<rendered:{tname}>")
            summary_lines.append(f"DIFF: {tname} -> {out_path} ({len(d.splitlines())} diff lines)")
            if args.check or args.dry_run:
                print(f"\n=== DIFF for {tname} ({out_path}) ===")
                print(d)
            else:
                out_path.write_text(rendered, encoding="utf-8")
                print(f"UPDATED: {out_path}")

    print("\n--- Summary ---")
    for line in summary_lines:
        print(line)

    if args.check and drift_detected:
        print("\n[CHECK] DRIFT DETECTED — see diffs above.", file=sys.stderr)
        return 1
    if args.dry_run:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
