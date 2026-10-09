"""Validation of controlled demo code. A temp directory is not a security sandbox."""
import subprocess
import sys
from pathlib import Path

def run_tests(workspace: Path, timeout_seconds: int = 4):
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "test_solution.py"],
            cwd=str(workspace), capture_output=True, text=True,
            timeout=timeout_seconds
        )
        output = (proc.stdout + "\n" + proc.stderr).strip()[-6000:]
        return proc.returncode == 0, output or f"pytest exited {proc.returncode}"
    except subprocess.TimeoutExpired:
        return False, f"Test execution timed out after {timeout_seconds}s"
    except Exception as exc:
        return False, f"Could not execute tests: {type(exc).__name__}: {exc}"
