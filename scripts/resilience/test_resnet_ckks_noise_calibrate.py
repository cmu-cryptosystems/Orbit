import unittest

from .resnet_ckks_noise_calibrate import select_candidate


def candidate(candidate_id: str, latency_gain: float, stability: float, passing: bool = True) -> dict:
    baseline_latency = 1000.0
    return {
        "candidate_id": candidate_id,
        "pass": passing,
        "candidate_stats": {"tdag_latency_sec": baseline_latency * (1.0 - latency_gain)},
        "drift_metrics": {"q05_margin_retention": stability},
    }


class ParetoMiddleSelectionTest(unittest.TestCase):
    def test_knee_point_can_differ_from_fastest_passing_candidate(self) -> None:
        reports = [
            candidate("conservative", 0.05, 0.95),
            candidate("knee", 0.25, 0.93),
            candidate("middle_fast", 0.45, 0.70),
            candidate("fastest", 0.65, 0.50),
            candidate("rejected_fast", 0.80, 0.20, passing=False),
        ]

        chosen, selection = select_candidate(
            reports,
            baseline_latency=1000.0,
            selection_policy="pareto-middle",
            drift_metric="margin-retention",
        )

        self.assertIsNotNone(chosen)
        self.assertEqual(chosen["candidate_id"], "knee")
        self.assertEqual(selection["chosen_candidate_id"], "knee")
        self.assertEqual(selection["fastest_passing_candidate_id"], "fastest")
        self.assertEqual(selection["reason"], "pareto_knee")
        self.assertEqual([entry["candidate_id"] for entry in selection["frontier"]], [
            "conservative",
            "knee",
            "middle_fast",
            "fastest",
        ])

    def test_two_candidate_rule_prefers_stability_without_large_latency_step(self) -> None:
        reports = [
            candidate("stable", 0.10, 0.90),
            candidate("slightly_faster", 0.18, 0.60),
        ]

        chosen, selection = select_candidate(
            reports,
            baseline_latency=1000.0,
            selection_policy="pareto-middle",
            drift_metric="margin-retention",
        )

        self.assertIsNotNone(chosen)
        self.assertEqual(chosen["candidate_id"], "stable")
        self.assertEqual(selection["reason"], "two_candidate_stability_preferred")


if __name__ == "__main__":
    unittest.main()
