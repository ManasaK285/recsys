import subprocess
import sys
import tempfile
import json
import time
from pathlib import Path

def evaluate_in_subprocess(code: str, test_expression: str, timeout_s: float = 3.0):
    """Execute generated code in a short-lived subprocess.

    This is a development sandbox only. Use a hardened container/isolation layer
    for genuinely untrusted code.
    """
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "candidate.py"
        harness = f"""
import json, time, traceback
code = {code!r}
try:
    ns = {{}}
    exec(code, ns)
    result = ({test_expression})
    print(json.dumps({{"ok": True, "result": result}}))
except Exception as e:
    print(json.dumps({{"ok": False, "error": str(e), "type": type(e).__name__}}))
"""
        p.write_text(harness, encoding="utf-8")
        start = time.perf_counter()
        try:
            proc = subprocess.run(
                [sys.executable, str(p)],
                capture_output=True, text=True, timeout=timeout_s
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "timeout": True, "runtime_ms": timeout_s * 1000}
        runtime_ms = (time.perf_counter() - start) * 1000
        out = proc.stdout.strip().splitlines()
        if not out:
            return {"ok": False, "error": proc.stderr[-500:], "runtime_ms": runtime_ms}
        try:
            data = json.loads(out[-1])
        except Exception:
            data = {"ok": False, "error": out[-1]}
        data["runtime_ms"] = runtime_ms
        return data
