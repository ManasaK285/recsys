from abc import ABC, abstractmethod

class ExplorationPolicy(ABC):
    name = "base"
    @abstractmethod
    def select(self, leaves, width):
        raise NotImplementedError

class GreedyPolicy(ExplorationPolicy):
    name = "greedy"
    def select(self, leaves, width):
        return sorted(
            leaves,
            key=lambda n: n.evaluation.score,
            reverse=True
        )[:width]

class DiversePolicy(ExplorationPolicy):
    name = "diverse"
    def select(self, leaves, width):
        # Prefer distinct approaches and then score.
        chosen, seen = [], set()
        for n in sorted(leaves, key=lambda x: x.evaluation.score, reverse=True):
            if n.approach not in seen:
                chosen.append(n)
                seen.add(n.approach)
            if len(chosen) >= width:
                return chosen
        for n in sorted(leaves, key=lambda x: x.evaluation.score, reverse=True):
            if n not in chosen:
                chosen.append(n)
            if len(chosen) >= width:
                break
        return chosen
