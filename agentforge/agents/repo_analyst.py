from __future__ import annotations
from pathlib import Path
from tools.filesystem import list_files, read_file
from tools.code_search import search_code

class RepoAnalyst:
    def analyze(self, root: str) -> dict:
        files = list_files(root)
        py_files = [f for f in files if f.endswith('.py')][:80]
        tests = [f for f in files if 'test' in Path(f).name.lower()]
        return {"files": files, "python_files": py_files, "tests": tests, "test_count": len(tests), "snippets": {f: read_file(f, root)[:3000] for f in py_files[:10]}, "search": search_code(root, "def ")[:30]}
