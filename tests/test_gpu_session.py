import unittest
from tools.gpu_session import violation,UUID
class GPUBudgetTests(unittest.TestCase):
 def test_process_budget(self):
  self.assertIsNone(violation(f'{UUID}, 42, 1024',UUID,42))
  self.assertIn('budget',violation(f'{UUID}, 42, 4097',UUID,42))
 def test_other_owner(self):
  self.assertIn('Another',violation(f'{UUID}, 43, 16',UUID,42))
 def test_other_gpu_ignored(self):
  self.assertIsNone(violation('GPU-other, 43, 10000',UUID,42))


class Descendants(unittest.TestCase):
 def test_children_are_allowed(self):
  import subprocess,os
  from tools.gpu_session import descendants
  child=subprocess.Popen(['sleep','5'])
  try:self.assertIn(child.pid,descendants(os.getpid()))
  finally:child.kill();child.wait()
  self.assertIsNone(violation(f'{UUID}, 42, 100\n{UUID}, 43, 200',UUID,{42,43}))
  self.assertIn('Another',violation(f'{UUID}, 44, 10',UUID,{42,43}))
