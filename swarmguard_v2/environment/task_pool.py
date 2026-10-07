from .models import Task
class TaskPool:
    def __init__(self, seed=7):
        self.seed=seed
    def make(self, round_idx, task_idx=0):
        # deterministic arithmetic tasks keep the benchmark interpretable
        return Task(f"T{round_idx:03d}_{task_idx}", expected=(round_idx*7+task_idx*11)%97)
