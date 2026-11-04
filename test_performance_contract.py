"""Behavior preserved while removing redundant inference work; no timing tests."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from model import DecisionModel
from network import save_weights, weights_digest
from weight_patches import stack


ROOT = Path(__file__).resolve().parent
REQUEST = {
    "question": "Which answer style?",
    "contexts": ["communication"],
    "options": [
        {"id": "concise", "attributes": {"concise": 1}},
        {"id": "verified", "attributes": {"verified": 1}},
    ],
}


class PerformanceContractTests(unittest.TestCase):
    def test_in_place_weight_edit_changes_next_decision_and_fingerprint(self):
        model = DecisionModel()
        before = model.decide(**REQUEST)
        self.assertEqual(before["decision"], "verified")
        model.weights[2][0, 0] = 10
        after = model.decide(**REQUEST)
        self.assertEqual(after["decision"], "concise")
        self.assertNotEqual(before["weights_sha256"], after["weights_sha256"])
        self.assertEqual(after["weights_sha256"], weights_digest(model.weights))

    def test_in_place_weight_edit_invalidates_verified_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "base.npz"
            save_weights(path, DecisionModel().weights)
            _, manifest = stack(path, [])
            model = DecisionModel(path, manifest)
            before = model.decide(**REQUEST)
            model.weights[2][0, 0] = 10
            with self.assertRaisesRegex(ValueError, "Weights changed after manifest verification"):
                model.decide(**REQUEST)
            model.weights[2][0, 0] = 3
            self.assertEqual(model.decide(**REQUEST), before)

    def test_nonfinite_direct_edit_is_rejected_by_decision_and_public_digest(self):
        model = DecisionModel()
        model.weights[2][0, 0] = np.nan
        with self.assertRaises(ValueError):
            model.decide(**REQUEST)
        with self.assertRaises(ValueError):
            weights_digest(model.weights)

    def test_returned_contributors_cannot_mutate_verified_bundle_provenance(self):
        from brain_bundle import create_blank, save_bundle

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "blank.brain"
            save_bundle(path, create_blank())
            model = DecisionModel.from_bundle(path)
            request = {
                "question": "Which option?", "contexts": ["general"],
                "options": [{"id": "a"}, {"id": "b"}],
            }
            first = model.decide(**request)
            self.assertEqual(first["contributing_harnesses"], [])
            first["contributing_harnesses"].append("not in verified history")
            self.assertEqual(model.decide(**request)["contributing_harnesses"], [])

    def test_all_committed_probe_outputs_remain_exactly_equal(self):
        reference = json.loads((ROOT / "evaluation_100.json").read_text())
        self.assertEqual(len(reference["results"]), 100)
        model = DecisionModel()
        for row in reference["results"]:
            with self.subTest(probe=row["id"]):
                actual = model.decide(row["question"], row["contexts"], row["options"])
                self.assertEqual(actual, row["output"])

    def test_scalar_tie_check_preserves_prior_numpy_tolerance(self):
        pairs = [
            (0, 0), (0, 1e-7), (0, 2e-7), (1, 1.000001),
            (1, 1.000002), (-10, -10.000005), (-10, -10.00002),
            (-1e-8, 1e-8), (1e30, 1e30 + 1e24),
        ]
        for a, b in pairs:
            with self.subTest(a=a, b=b):
                model = DecisionModel()
                model.weights[2][0, 0] = a
                model.weights[2][1, 0] = b
                scores = [float(np.float32(a)), float(np.float32(b))]
                highest, second = sorted(scores, reverse=True)
                prior_tie = bool(np.isclose(highest, second, rtol=1e-6, atol=1e-7))
                result = model.decide(**REQUEST)
                self.assertEqual(result["status"], "tie" if prior_tie else "decision")
                expected = None if prior_tie else REQUEST["options"][scores.index(highest)]["id"]
                self.assertEqual(result["decision"], expected)


if __name__ == "__main__":
    unittest.main()
