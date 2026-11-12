import unittest

from evals.run_eval import main


class EvaluationTests(unittest.TestCase):
    def test_offline_evaluation_passes(self):
        self.assertEqual(main([]), 0)


if __name__ == "__main__":
    unittest.main()
