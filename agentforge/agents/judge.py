from __future__ import annotations

class Judge:
    def evaluate(self, spec: dict, verification: dict, requirement_result: dict) -> dict:
        checks_ok = (
            verification.get('tests_failed', 1) == 0
            and verification.get('lint') in {'PASS', 'SKIPPED'}
            and verification.get('typecheck') in {'PASS', 'SKIPPED'}
            and requirement_result.get('passed', False)
        )
        return {'status': 'PASS' if checks_ok else 'FAIL', 'evidence': verification, 'requirements': requirement_result}
