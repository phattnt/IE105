import unittest

import analyze_transfer as transfer


class TransferAnalysisTest(unittest.TestCase):
    def test_average_ranks_handles_ties(self):
        self.assertEqual(transfer.average_ranks([10, 20, 20, 40]), [0.0, 1.5, 1.5, 3.0])

    def test_committed_results_build_complete_report(self):
        report = transfer.build_report(transfer.DEFAULT_RESULTS)
        self.assertEqual(report["n_runs"], 9)
        baseline = next(r for r in report["runs"] if r["run"] == transfer.BASELINE_RUN)
        self.assertEqual(baseline["delta_vs_plain"]["secvuleval_top10"], 0.0)
        self.assertEqual(baseline["delta_vs_plain"]["primevul_top10"], 0.0)
        self.assertEqual(report["cross_run_correlations"]["comparable_converged"]["top10"]["n"], 6)

    def test_markdown_contains_research_caveat(self):
        text = transfer.report_markdown(transfer.build_report(transfer.DEFAULT_RESULTS))
        self.assertIn("không phải kiểm định độc lập", text)
        self.assertIn("không tự động chuyển", text)


if __name__ == "__main__":
    unittest.main()
