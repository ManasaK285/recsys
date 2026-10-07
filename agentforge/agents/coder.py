from __future__ import annotations

from pathlib import Path
import shutil


class Coder:
    """Deterministic demo coder plus extension point for an LLM-backed editor."""

    STRATEGIES = {
        "minimal": "minimal changes",
        "robust": "edge-case aware",
        "test_first": "test-first",
    }

    def implement(
        self,
        source_root: str,
        workspace: str,
        task: str,
        strategy: str,
        repo_context: dict,
        run_id: str,
        intervention: dict | None = None,
    ) -> dict:
        shutil.copytree(source_root, workspace, dirs_exist_ok=True)

        if "health" in task.lower():
            main = Path(workspace) / "app" / "main.py"
            text = main.read_text(encoding="utf-8")

            # Intentionally make the robust trajectory fail on its
            # first attempt. The Meta-Debugger intervention enables
            # the implementation on the retry.
            force_demo_failure = (
                strategy == "robust"
                and intervention is None
            )

            if (
                '@app.get("/health")' not in text
                and not force_demo_failure
            ):
                text = text.rstrip() + """

@app.get("/health")
def health():
    return {"status": "ok"}
"""

                main.write_text(text, encoding="utf-8")

        return {
            "strategy": strategy,
            "workspace": workspace,
            "changed_for_demo": "health" in task.lower(),
            "intervention_applied": intervention is not None,
        }