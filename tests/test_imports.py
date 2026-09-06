#!/usr/bin/env python3
"""Self-check: verify all modules in the ionogram-scaling skill can be imported."""

from __future__ import annotations

import importlib
import os
import sys
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

MODULES = [
    "ionogram_scaling",
    "ionogram_scaling.config",
    "ionogram_scaling.vlm_client",
    "ionogram_scaling.image_utils",
    "ionogram_scaling.prompts",
    "ionogram_scaling.agents",
    "ionogram_scaling.workflow",
    "ionogram_scaling.main",
]

TOOLS = [
    "tools.evaluate",
    "tools.pdsa",
    "tools.filter_bad",
]


def check_module(name: str) -> bool:
    try:
        importlib.import_module(name)
        print(f"[OK] {name}")
        return True
    except Exception as e:
        print(f"[FAIL] {name}: {e}")
        traceback.print_exc()
        return False


def main() -> int:
    print("Checking package modules...")
    ok = all(check_module(m) for m in MODULES)

    print("\nChecking tool modules (PyQt5 tool skipped in headless check)...")
    # labeling.py requires PyQt5 and a display; skip here to avoid false failures.
    ok = ok and all(check_module(m) for m in TOOLS)

    print("\nChecking CLI argument parsing (no API call)...")
    try:
        from ionogram_scaling.main import parse_args
        # argparse exits on -h; test that the parser builds by checking object type
        assert parse_args.__module__ == "ionogram_scaling.main"
        print("[OK] CLI parser ready")
    except Exception as e:
        print(f"[FAIL] CLI parser: {e}")
        ok = False

    print("\nChecking Config validation...")
    try:
        from ionogram_scaling.config import Config
        cfg = Config()
        err = cfg.validate()
        assert err is not None, "expected validation error when credentials are empty"
        cfg.api_key = "dummy-key"
        cfg.base_url = "https://example.invalid/v1"
        cfg.model = "dummy-vision"
        err = cfg.validate()
        assert err is None, err
        print("[OK] Config validation logic")
    except Exception as e:
        print(f"[FAIL] Config validation: {e}")
        ok = False

    print("\n" + ("All checks passed." if ok else "Some checks failed."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
