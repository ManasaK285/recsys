class ReplayWorld:
    """A frozen discovery tree used as an offline environment."""

    def __init__(self, nodes):
        self.nodes = list(nodes)
        self.by_id = {n.node_id: n for n in self.nodes}

    def reset(self):
        return {"revealed": set(), "reward": 0.0, "work": 0}

    def replay(self, policy, width=2, steps=5):
        state = self.reset()
        available = self.nodes[:]
        for _ in range(steps):
            if not available:
                break
            chosen = policy.select(available, width)
            if not chosen:
                break
            for n in chosen:
                state["revealed"].add(n.node_id)
                state["reward"] = max(state["reward"], n.evaluation.score)
                state["work"] += 1
            # In a real environment, this would reveal child nodes conditionally.
            available = [n for n in available if n.node_id not in state["revealed"]]
        return state
