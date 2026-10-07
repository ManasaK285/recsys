import asyncio

from harness.orchestrator import AgentForge


def test_demo_run():
    result = asyncio.run(
        AgentForge(
            workspace_root='artifacts/test-workspaces'
        ).run(
            'Add a /health endpoint returning status=ok',
            'sample_repo',
            3,
        )
    )

    assert result['status'] == 'PASS'
    assert result['attempts'] == 2
    assert result['meta_debugger'] is not None