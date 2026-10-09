from src.flowfix.patching import safe_candidate, unified_diff

def test_diff_contains_change():
    diff = unified_diff("x = 1\n","x = 2\n")
    assert "-x = 1" in diff and "+x = 2" in diff

def test_syntax_is_checked():
    ok, _ = safe_candidate("x=1","def broken(:")
    assert not ok

def test_block_test_skip():
    ok, reason = safe_candidate("x=1","import pytest\npytest.skip('skip')")
    assert not ok and "blocked" in reason
