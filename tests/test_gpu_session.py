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
