import unittest
from polca_lab.benchmarks.support_benchmark import SupportBenchmark
from polca_lab.core.evaluator import StochasticEvaluator
from polca_lab.core.optimizer import POLCAOptimizer
from polca_lab.providers.local_oracle import LocalProposalOracle

class TestOptimizer(unittest.TestCase):
    def test_end_to_end(self):
        cfg = {"seed": 1, "memory_size": 10, "epsilon": .2, "priority": "mean", "use_epsilon_net": True, "use_summary": True, "batch_size": 5, "proposals_per_iteration": 2, "iterations": 3}
        ev = StochasticEvaluator(SupportBenchmark(), noise_std=.01, seed=1)
        opt = POLCAOptimizer(ev, LocalProposalOracle(1), cfg)
        opt.initialize(["Answer requests directly.", "Be concise."])
        log = opt.run()
        self.assertEqual(len(log), 3)
        self.assertIsNotNone(opt.memory.best())
        self.assertGreater(ev.total_evals, 0)
