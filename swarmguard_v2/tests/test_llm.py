import sys, pathlib, unittest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from llm.client import LLMClient
class TestLLM(unittest.TestCase):
 def test_mock_json(self):
  out=LLMClient("mock").decide({"exploit_known":True})
  self.assertIn(out["action"],["EXPLOIT","LEGITIMATE"])
if __name__=="__main__": unittest.main()
