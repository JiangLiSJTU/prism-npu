"""Compatibility shim for legacy pip (<23) that doesn't trigger PEP 660 editable installs from pyproject.toml alone.

All metadata lives in pyproject.toml. This file just makes `pip install -e .` work on older pip versions by providing the legacy entry point setuptools requires.
"""
from setuptools import setup

setup()
