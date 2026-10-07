from __future__ import annotations
from tools.commands import run_command

class Verifier:
    def run(self, workspace: str) -> dict:
        tests=run_command(['python','-m','pytest','-q'],workspace,120)
        lint=run_command(['ruff','check','.'],workspace,120)
        types=run_command(['mypy','.','--ignore-missing-imports'],workspace,120)
        return {'tests_passed': self._passed_count(tests['stdout']), 'tests_failed': 0 if tests['success'] else 1, 'test_run':tests, 'lint':lint['status'],'lint_run':lint,'typecheck':types['status'],'typecheck_run':types}
    @staticmethod
    def _passed_count(output: str) -> int:
        import re
        m=re.search(r'(\d+) passed',output)
        return int(m.group(1)) if m else 0
