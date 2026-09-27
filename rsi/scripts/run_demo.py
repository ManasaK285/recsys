from src.db import init_db
from src.orchestrator import run_discovery
from src.dreaming import dream
from src.recursive import recursive_improvement
from src.policy.base import GreedyPolicy

def main():
    init_db()
    print("=== Live discovery ===")
    run_id, tree = run_discovery("topk", GreedyPolicy(), width=2, cycles=3)
    print("run:", run_id)
    print("nodes:", len(tree.nodes))
    print("best:", max(n.evaluation.score for n in tree.nodes.values()))

    print("\n=== Offline dreaming ===")
    results = dream(list(tree.nodes.values()))
    for policy, result in results:
        print(policy.name, "reward=", result.reward, "work=", result.work)

    print("\n=== Recursive improvement ===")
    history = recursive_improvement("topk", cycles=3)
    for item in history:
        print(item)

if __name__ == "__main__":
    main()
