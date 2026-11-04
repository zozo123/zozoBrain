"""Run a synthetic Codex-then-Claude weight-editing walkthrough in a temp folder."""
import json
from pathlib import Path
import tempfile

from model import DecisionModel
from network import save_weights
from weight_patches import diff, stack
from weights import authored_weights


def run():
    options = [{'id': 'concise', 'attributes': {'concise': 1}},
               {'id': 'verified', 'attributes': {'verified': 1}}]

    def decision(weights, manifest=None):
        return DecisionModel(weights, manifest).decide('Which style?', ['communication'], options)

    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        base = folder / 'base.npz'
        source = authored_weights()
        save_weights(base, source)

        # Synthetic instructions demonstrate mechanics, not new personal claims.
        codex = tuple(tensor.copy() for tensor in source)
        codex[2][0, 0] = 6  # Directly assign the actual concise utility weight.
        codex_snapshot = folder / 'codex.npz'
        save_weights(codex_snapshot, codex)
        codex_patch = diff(base, codex_snapshot, harness='codex', patch_id='demo-codex-001',
                           evidence='Synthetic demo instruction: set concise utility to 6.')
        after_codex, first_manifest = stack(base, [codex_patch])
        accumulated = folder / 'after-codex.npz'
        manifest_path = folder / 'after-codex.manifest.json'
        save_weights(accumulated, after_codex)
        manifest_path.write_text(json.dumps(first_manifest), encoding='utf-8')

        claude = tuple(tensor.copy() for tensor in after_codex)
        claude[2][1, 0] = 8  # A second direct edit, preserving the Codex entry.
        claude_snapshot = folder / 'claude.npz'
        save_weights(claude_snapshot, claude)
        claude_patch = diff(accumulated, claude_snapshot, harness='claude', patch_id='demo-claude-001',
                            evidence='Synthetic demo instruction: set verified utility to 8.')
        combined, manifest = stack(accumulated, [claude_patch], previous_manifest=manifest_path)
        result = decision(combined, manifest)
        print('Synthetic weight-editing demo; these edits are not new Yossi preferences.')
        print('Base: ' + decision(source)['decision'])
        print('After Codex: ' + decision(after_codex, first_manifest)['decision'])
        print('Codex + Claude: ' + result['decision'])
        print('Contributors: ' + ', '.join(result['contributing_harnesses']))
        print('Direct weights: concise=6, verified=8 | total parameters: 2499')
        print('weights_sha256: ' + result['weights_sha256'])
        return result


if __name__ == '__main__':
    run()
