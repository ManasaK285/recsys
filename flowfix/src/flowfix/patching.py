import difflib
import re

def unified_diff(old: str, new: str, filename="solution.py") -> str:
    return "".join(difflib.unified_diff(
        old.splitlines(keepends=True), new.splitlines(keepends=True),
        fromfile=f"a/{filename}", tofile=f"b/{filename}"
    ))

def safe_candidate(old: str, new: str) -> tuple[bool, str]:
    if not new.strip():
        return False, "Empty candidate source"
    if len(new) > max(20000, len(old) * 5):
        return False, "Candidate exceeds source-size limit"
    forbidden = [
        r"\bpytest\.skip\b", r"\bskipif\s*\(", r"\bsubprocess\.(?:run|Popen|call)\b",
        r"\bos\.system\s*\(", r"\bshutil\.rmtree\s*\(", r"\bopen\([^\n]*['\"]w"
    ]
    for pattern in forbidden:
        if re.search(pattern, new):
            return False, f"Candidate contains a blocked pattern: {pattern}"
    try:
        compile(new, "<candidate>", "exec")
    except SyntaxError as exc:
        return False, f"Candidate has syntax error: {exc}"
    return True, "Candidate passed structural checks"
