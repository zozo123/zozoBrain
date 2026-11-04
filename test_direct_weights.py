"""Behavioral checks for direct tensor authoring and artifact loading."""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
import zipfile

import numpy as np

from model import DecisionModel
from network import load_weights, save_weights, validate_weights, weights_digest
from weights import authored_weights


ROOT = Path(__file__).resolve().parent
TENSOR_NAMES = ("w1", "b1", "w2", "b2")


class DirectWeightTests(unittest.TestCase):
    def command(self, *args, input_text=None, check=True):
        return subprocess.run(
            [sys.executable, *map(str, args)], input=input_text,
            text=True, capture_output=True, check=check, cwd=ROOT,
        )

    def assert_invalid_artifact(self, path):
        with self.assertRaises(ValueError) as caught:
            load_weights(path)
        self.assertTrue(str(caught.exception).strip(), "A rejected artifact needs an explanation")

    def test_roundtrip_preserves_actual_tensors_and_parameter_budget(self):
        original = authored_weights()
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "weights.npz"
            save_weights(artifact, original)
            restored = load_weights(artifact)
        self.assertEqual([a.shape for a in restored], [(48, 48), (48,), (48, 3), (3,)])
        self.assertEqual(sum(a.size for a in restored), 2499)
        self.assertEqual(sum(int(np.count_nonzero(a)) for a in restored), 150)
        self.assertEqual(weights_digest(original), weights_digest(restored))
        for expected, actual in zip(original, restored):
            self.assertEqual(actual.dtype, np.dtype("float32"))
            np.testing.assert_array_equal(expected, actual)

    def test_shape_and_nonfinite_tensors_are_rejected(self):
        invalid_shape = list(authored_weights())
        invalid_shape[0] = np.zeros((47, 48), dtype=np.float32)
        with self.assertRaises(ValueError):
            validate_weights(invalid_shape)
        with self.assertRaises(ValueError):
            validate_weights(authored_weights()[:3])
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                invalid = [a.copy() for a in authored_weights()]
                invalid[2][0, 0] = value
                with self.assertRaises(ValueError):
                    validate_weights(invalid)
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "nonfinite.npz"
                    np.savez(path, **dict(zip(TENSOR_NAMES, invalid)))
                    self.assert_invalid_artifact(path)

    def test_corrupt_npy_object_and_wrong_schema_archives_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corrupt = root / "corrupt.npz"
            corrupt.write_bytes(b"this is not a NumPy archive")
            self.assert_invalid_artifact(corrupt)

            truncated_zip = root / "truncated_zip.npz"
            truncated_zip.write_bytes(b"PK\x03\x04")
            self.assert_invalid_artifact(truncated_zip)

            compressed_corrupt = root / "compressed_corrupt.npz"
            np.savez_compressed(compressed_corrupt, **dict(zip(TENSOR_NAMES, authored_weights())))
            with zipfile.ZipFile(compressed_corrupt) as archive:
                member = archive.infolist()[0]
            damaged = bytearray(compressed_corrupt.read_bytes())
            name_size, extra_size = struct.unpack_from("<HH", damaged, member.header_offset + 26)
            data_offset = member.header_offset + 30 + name_size + extra_size
            damaged[data_offset:data_offset + 5] = b"\0" * 5
            compressed_corrupt.write_bytes(damaged)
            self.assert_invalid_artifact(compressed_corrupt)

            plain_npy = root / "plain_npy.npz"
            with plain_npy.open("wb") as sink:
                np.save(sink, np.zeros((48, 48), dtype=np.float32))
            self.assert_invalid_artifact(plain_npy)

            object_archive = root / "objects.npz"
            object_tensors = [a.astype(object) for a in authored_weights()]
            np.savez(object_archive, **dict(zip(TENSOR_NAMES, object_tensors)))
            self.assert_invalid_artifact(object_archive)

            wrong_shape = root / "wrong_shape.npz"
            arrays = dict(zip(TENSOR_NAMES, authored_weights()))
            arrays["b1"] = np.zeros(47, dtype=np.float32)
            np.savez(wrong_shape, **arrays)
            self.assert_invalid_artifact(wrong_shape)

            missing_tensor = root / "missing.npz"
            np.savez(missing_tensor, w1=arrays["w1"])
            self.assert_invalid_artifact(missing_tensor)

            unexpected_tensor = root / "unexpected.npz"
            np.savez(unexpected_tensor, **dict(zip(TENSOR_NAMES, authored_weights())), surprise=np.zeros(1))
            self.assert_invalid_artifact(unexpected_tensor)

    def test_authored_export_check_does_not_replace_changed_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "authored.npz"
            exported = json.loads(self.command(ROOT / "weights.py", "--output", path).stdout)
            self.assertEqual(exported["total_parameters"], 2499)
            self.assertEqual(exported["nonzero_parameters"], 150)
            self.command(ROOT / "weights.py", "--output", path, "--check")
            changed = [a.copy() for a in load_weights(path)]
            changed[2][0, 0] = 10
            save_weights(path, changed)
            before = path.read_bytes()
            checked = self.command(ROOT / "weights.py", "--output", path, "--check", check=False)
            self.assertNotEqual(checked.returncode, 0)
            self.assertIn("differs", checked.stderr.lower())
            self.assertEqual(path.read_bytes(), before)

    def test_loading_edited_tensors_changes_decision_without_changing_source_default(self):
        request = {
            "question": "Which answer style?", "contexts": ["communication"],
            "options": [
                {"id": "concise", "attributes": {"concise": 1}},
                {"id": "verified", "attributes": {"verified": 1}},
            ],
        }
        original_digest = weights_digest(authored_weights())
        self.assertEqual(DecisionModel().decide(**request)["decision"], "verified")
        with tempfile.TemporaryDirectory() as tmp:
            edited = Path(tmp) / "edited.npz"
            tensors = [a.copy() for a in authored_weights()]
            tensors[2][0, 0] = 10
            save_weights(edited, tensors)
            model = DecisionModel(edited)
            self.assertEqual(model.parameter_count, 2499)
            self.assertEqual(model.decide(**request)["decision"], "concise")
            self.assertNotEqual(weights_digest(model.weights), original_digest)
        self.assertEqual(weights_digest(authored_weights()), original_digest)
        self.assertEqual(DecisionModel().decide(**request)["decision"], "verified")

    def test_forcing_loaded_network_cannot_select_an_all_vetoed_option(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "all_vetoed.npz"
            tensors = [a.copy() for a in authored_weights()]
            tensors[3][1] = 1  # Directly activate the veto output for every option.
            save_weights(path, tensors)
            result = DecisionModel(path).decide(
                "Which?", ["general"], [{"id": "a"}, {"id": "b"}], force=True,
            )
            self.assertIsNone(result["decision"])
            self.assertEqual(result["status"], "no_feasible_option")
            self.assertEqual(len(result["vetoed"]), 2)

    def test_tensor_editor_changes_api_cli_and_harness_consistently(self):
        request = {
            "id": "edited-style", "question": "Which answer style?",
            "contexts": ["communication"],
            "options": [
                {"id": "concise", "attributes": {"concise": 1}},
                {"id": "verified", "attributes": {"verified": 1}},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.npz"
            edited = Path(tmp) / "edited.npz"
            save_weights(source, authored_weights())
            original_bytes = source.read_bytes()
            changed = json.loads(self.command(
                ROOT / "edit_weights.py", "set", source,
                "--tensor", "w2", "--index", "0", "0", "--value", "10",
                "--output", edited,
            ).stdout)
            self.assertEqual(changed["previous"], 3)
            self.assertEqual(changed["value"], 10)
            self.assertEqual(source.read_bytes(), original_bytes)

            # A second explicit edit changes the sparsity, so inspect must count
            # the loaded tensor entries rather than report the source defaults.
            self.command(
                ROOT / "edit_weights.py", "set", edited,
                "--tensor", "b2", "--index", "0", "--value", "0.25",
                "--output", edited,
            )
            inspected = json.loads(self.command(ROOT / "edit_weights.py", "inspect", edited).stdout)
            tensors = load_weights(edited)
            digest = weights_digest(tensors)
            self.assertEqual(inspected["total_parameters"], sum(a.size for a in tensors))
            self.assertEqual(inspected["total_parameters"], 2499)
            self.assertEqual(inspected["nonzero_parameters"], 151)
            self.assertEqual(inspected["tensors"], {
                "w1": [48, 48], "b1": [48], "w2": [48, 3], "b2": [3],
            })
            self.assertEqual(inspected["weights_sha256"], digest)

            api_request = {key: value for key, value in request.items() if key != "id"}
            api = DecisionModel(edited).decide(**api_request)
            decision_file = Path(tmp) / "decision.json"
            decision_file.write_text(json.dumps(request))
            cli = json.loads(self.command(ROOT / "model.py", decision_file, "--weights", edited).stdout)
            responses = self.command(
                ROOT / "harness.py", "--weights", edited,
                input_text=json.dumps(request) + "\n" + json.dumps(request) + "\n",
            ).stdout.splitlines()
            self.assertEqual(len(responses), 2)
            streamed = [json.loads(line) for line in responses]
            for response in streamed:
                self.assertTrue(response["ok"])
                self.assertEqual(response["id"], "edited-style")
            for result in [api, cli, *[row["result"] for row in streamed]]:
                self.assertEqual(result["decision"], "concise")
                self.assertEqual(result["weights_sha256"], digest)
            self.assertEqual(DecisionModel().decide(**api_request)["decision"], "verified")

    def test_editor_rejects_invalid_values_and_indices_without_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.npz"
            output = Path(tmp) / "must_not_exist.npz"
            save_weights(source, authored_weights())
            for index, value in [(["-1", "0"], "1"), (["48", "0"], "1"),
                                 (["0"], "1"), (["0", "0"], "nan"),
                                 (["0", "0"], "inf"), (["0", "0"], "1e100")]:
                with self.subTest(index=index, value=value):
                    result = self.command(
                        ROOT / "edit_weights.py", "set", source,
                        "--tensor", "w2", "--index", *index,
                        "--value", value, "--output", output, check=False,
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("error", result.stderr.lower())
                    self.assertNotIn("traceback", result.stderr.lower())
                    self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
