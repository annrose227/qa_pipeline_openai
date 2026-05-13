#!/usr/bin/env python3
"""
Standalone validation script.
Run: python validate.py
"""

import json
import sys
from pathlib import Path

# Re-use validation logic from pipeline
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from pipeline_openai import (
    _run_validation,
    QA_FRAMEWORK_FILE,
    QA_SCORES_FILE,
    COMPLIANCE_FLAGS_FILE,
    COACHING_NOTES_FILE,
    DASHBOARD_FILE,
    LLM_LOG_FILE,
    CALIBRATION_FILE,
    TEAM_TREND_FILE,
    ESCALATION_FILE,
    REBUTTAL_FILE,
)

print("=" * 60)
print("  QA PIPELINE VALIDATOR")
print("=" * 60)

print("\nChecking required artifacts...")
optional = [CALIBRATION_FILE, TEAM_TREND_FILE, ESCALATION_FILE, REBUTTAL_FILE]
for f in optional:
    if f.exists():
        print(f"  [optional] {f.name} ✓")
    else:
        print(f"  [optional] {f.name} — not present")

errors, warnings = _run_validation(state=None, standalone=True)

if not errors:
    print("\n✓ VALIDATION PASSED")
    sys.exit(0)
else:
    print(f"\n✗ VALIDATION FAILED — {len(errors)} error(s)")
    sys.exit(1)
