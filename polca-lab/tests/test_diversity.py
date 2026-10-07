import unittest
from polca_lab.core.diversity import EpsilonNet, cosine_distance

class TestDiversity(unittest.TestCase):
    def test_identical_text_is_rejected(self):
        net = EpsilonNet(.1)
        self.assertFalse(net.accept("be concise", ["be concise"]))

    def test_different_text_can_be_admitted(self):
        net = EpsilonNet(.2)
        self.assertTrue(net.accept("escalate to a human when uncertain", ["answer directly and briefly"]))

    def test_distance_range(self):
        d = cosine_distance([1,0], [0,1])
        self.assertAlmostEqual(d, 1.0)
