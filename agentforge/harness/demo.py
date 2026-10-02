import asyncio
from harness.orchestrator import AgentForge

if __name__ == '__main__':
    result=asyncio.run(AgentForge().run('Add a /health endpoint returning status=ok; demonstrate workflow recovery', 'sample_repo', 3))
    print('\nAgentForge demo')
    print('Run:', result['run_id'])
    print('Status:', result['status'])
    print('Attempts:', result['attempts'])
    for t in result['trajectories']:
        print(f"  {t['strategy']}: {t['judgment']['status']}")
    if result['meta_debugger']:
        print('Meta-debugger:', result['meta_debugger'])
