import unittest
from polca_lab.core.candidate import Candidate
from polca_lab.core.memory import CandidateMemory

class TestMemory(unittest.TestCase):
    def test_mean_and_variance(self):
        c = Candidate("x", "policy")
        c.record(.2, "bad"); c.record(.6, "good")
        self.assertAlmostEqual(c.mean, .4)
        self.assertGreater(c.variance, 0)

    def test_capacity(self):
        m = CandidateMemory(2)
        for i, score in enumerate([.1, .9, .5]):
            c = Candidate(str(i), str(i)); c.record(score, "f"); m.add(c)
        self.assertEqual(len(m.items), 2)
        self.assertEqual(m.best().cid, "1")
