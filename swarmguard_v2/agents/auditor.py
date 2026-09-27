class AuditorAgent:
    def __init__(self, state, rng, accuracy=0.9, false_positive_rate=0.04):
        self.state = state
        self.rng = rng
        self.accuracy = accuracy
        self.fpr = false_positive_rate

    def audit(self, solution, round_idx):
        attempted_shortcut = solution.strategy == "SHORTCUT"

        if attempted_shortcut:
            flagged = self.rng.random() < self.accuracy
        else:
            flagged = self.rng.random() < self.fpr

        return flagged, attempted_shortcut
