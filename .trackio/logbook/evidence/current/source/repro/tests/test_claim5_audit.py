"""Tests for the Claim 5 blocker and its negative control."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from claim5_audit import blocking_reasons


def test_exact_timing_audit_detects_missing_prerequisites():
    audit = {
        "python_contract": {"compatible": False},
        "module_audit": {"missing": ["torch", "botorch"]},
        "required_external_data": {"all_present": False},
    }
    assert len(blocking_reasons(audit)) == 3


def test_negative_control_removes_blocker():
    audit = {
        "python_contract": {"compatible": True},
        "module_audit": {"missing": []},
        "required_external_data": {"all_present": True},
    }
    assert blocking_reasons(audit) == []
