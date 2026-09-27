import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import unittest
from environment.evaluator import VulnerableEvaluator, StrictEvaluator
from environment.models import Task,Solution
from environment.swarm import Swarm
from config import Config
class TestCore(unittest.TestCase):
 def test_vulnerable_accepts_shortcut(self):
  t=Task("T",42); s=Solution("A","T",0,"SHORTCUT"); self.assertEqual(VulnerableEvaluator().evaluate(s,t),(True,"evaluator_shortcut"))
 def test_strict_rejects_shortcut(self):
  t=Task("T",42); s=Solution("A","T",0,"SHORTCUT"); self.assertFalse(StrictEvaluator().evaluate(s,t)[0])
 def test_swarm_metrics(self):
  c=Config(agents=6,rounds=5,seeds=2); m=Swarm(c,7,"baseline").run(); self.assertIn("exploit_adoption_rate",m); self.assertEqual(m["governance"],"baseline")
if __name__=="__main__": unittest.main()
