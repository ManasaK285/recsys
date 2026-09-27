from .base import ExplorationPolicy

class AdaptivePolicy(ExplorationPolicy):
    name = "adaptive"

    def select(self, leaves, width):
        def key(n):
            e = n.evaluation
            novelty = 1.0 if e.failure_class or e.score < 1.0 else 0.0
            depth_penalty = 0.02 * n.depth
            return e.score + 0.25 * novelty - depth_penalty

        return sorted(leaves, key=key, reverse=True)[:width]
