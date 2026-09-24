import uuid
from datetime import datetime, timezone
from src.db import init_db, insert_task, insert_run, insert_node
from src.models import DiscoveryNode
from src.agent.coding_agent import CodingAgent
from src.evaluator.evaluator import evaluate_candidate
from src.policy.base import GreedyPolicy
from src.discovery.tree import DiscoveryTree
from src.tasks.registry import make_registry

def now():
    return datetime.now(timezone.utc).isoformat()

def run_discovery(task_id="topk", policy=None, width=3, cycles=3):
    init_db()
    task = make_registry()[task_id]
    insert_task(task.task_id, task.name, task.description)
    policy = policy or GreedyPolicy()

    run_id = str(uuid.uuid4())
    insert_run(run_id, task.task_id, policy.name, 0, now())

    tree = DiscoveryTree()
    agent = CodingAgent()

    # Root is conceptual; candidate approaches become first-level nodes.
    generated = []
    for approach in task.candidate_factories:
        code = agent.generate(task, approach)
        ev = evaluate_candidate(task, code)
        node = DiscoveryNode(
            node_id=str(uuid.uuid4()),
            run_id=run_id,
            task_id=task.task_id,
            parent_id=None,
            depth=0,
            approach=approach,
            code=code,
            evaluation=ev,
            created_at=now(),
            policy_version=policy.name,
        )
        tree.add(node)
        insert_node(node)
        generated.append(node)

    # Additional rounds refine/retry the selected branches.
    leaves = tree.leaves()
    for depth in range(1, cycles):
        selected = policy.select(leaves, width)
        next_nodes = []
        for parent in selected:
            # For the prototype, "refinement" reuses the same approach factory.
            code = agent.generate(task, parent.approach, parent.code)
            ev = evaluate_candidate(task, code)
            node = DiscoveryNode(
                node_id=str(uuid.uuid4()),
                run_id=run_id,
                task_id=task.task_id,
                parent_id=parent.node_id,
                depth=depth,
                approach=parent.approach,
                code=code,
                evaluation=ev,
                created_at=now(),
                policy_version=policy.name,
            )
            tree.add(node)
            insert_node(node)
            next_nodes.append(node)
        leaves = tree.leaves()

    return run_id, tree
