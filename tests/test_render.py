"""Test prism.reports.render module.

Verifies:
- 4 templates render without errors
- --check mode detects drift correctly
- OUTPUT_PATH_MAP points to docs/findings/
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


def test_render_module_imports():
    from prism.reports import render
    assert hasattr(render, "OUTPUT_PATH_MAP")
    assert hasattr(render, "main")


def test_output_path_map_points_to_findings(sim_root):
    """OUTPUT_PATH_MAP 4 个模板应都渲染到 docs/findings/。"""
    from prism.reports.render import OUTPUT_PATH_MAP

    expected_findings = sim_root / "docs" / "findings"
    for tpl_name, out_path in OUTPUT_PATH_MAP.items():
        assert str(out_path).startswith(str(expected_findings)), (
            f"{tpl_name} renders to {out_path}, expected docs/findings/*"
        )


def test_render_check_mode(sim_root, tmp_path, monkeypatch):
    """prism-render --check 应当 exit 0 (4 OK identical)。"""
    from prism.reports.render import main

    # 模拟 CLI: --check
    monkeypatch.setattr(sys, "argv", [
        "prism-render", "--check",
        "--vars",          str(sim_root / "data" / "experiment_variables.json"),
        "--templates-dir", str(sim_root / "reports" / "templates"),
        "--output-dir",    str(sim_root / "docs" / "findings"),
    ])

    if not (sim_root / "data" / "experiment_variables.json").is_file():
        pytest.skip("experiment_variables.json missing — bootstrap data first")

    exit_code = main()
    assert exit_code == 0, f"prism-render --check exit {exit_code} (drift detected)"


def test_findings_dir_has_4_reports(sim_root):
    """渲染输出 docs/findings/ 应至少有 4 份报告。"""
    findings_dir = sim_root / "docs" / "findings"
    assert findings_dir.is_dir()
    md_files = sorted(findings_dir.glob("*.md"))
    expected = {"主报告.md", "roofline校准报告.md", "微架构探索报告.md", "msprof分解报告.md"}
    actual = {p.name for p in md_files}
    missing = expected - actual
    assert not missing, f"docs/findings/ 缺 {missing}"
