class PolicyDeveloper:
    """Simple policy factory.

    A production version can ask an LLM to synthesize Python policy code, then
    validate it in an isolated process before replay evaluation.
    """

    def candidates(self):
        from .base import GreedyPolicy, DiversePolicy
        from .adaptive import AdaptivePolicy
        return [GreedyPolicy(), DiversePolicy(), AdaptivePolicy()]
