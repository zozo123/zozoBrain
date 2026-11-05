"""Directly authored neural weights: edit tensor assignments, never fit or train.

Every dense tensor entry counts, including zero entries. The numerical values
below are authored policy hypotheses, not estimated personal utilities.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from network import (
    forward as network_forward,
    load_weights,
    save_weights,
    validate_weights,
    weights_digest,
)


CONTEXTS = [
    "communication", "travel", "career", "security", "research", "design",
    "cv", "website", "teaching", "analytics", "food", "music", "politics",
    "countries", "people", "general",
]
ATTRIBUTES = [
    "concise", "verified", "speculative", "autonomous_completion",
    "redundant_approval", "finished_deliverable", "shabbat_flight",
    "avoids_shabbat", "misses_one_or_two_days", "scientific_ai",
    "secure_sandbox", "reproducible", "measured_performance", "open_source",
    "relevant_expertise", "pastel", "coarse_beard", "printed_name",
    "single_page", "multi_page", "netlify_domain", "old_custom_domain",
    "merged_pr", "unmerged_pr", "required_services", "frontend_only",
    "genuine_visitors", "ci_as_visitors", "explicitly_preferred",
    "explicitly_disliked", "explicitly_disallowed", "explicitly_allowed",
]
BINARY_ATTRIBUTES = ["shabbat_flight", "explicitly_disallowed"]

# This list wires the feature neurons to their contexts. It contains no utility
# values or veto rules. None means the feature is active in every context.
ATTRIBUTE_CONTEXTS = [
    "communication", "communication", "communication", "communication",
    "communication", "communication", "travel", "travel", "travel",
    "career", "security", "research", "research", "career", "people",
    "design", "design", "design", "cv", "cv", "website", "website",
    "teaching", "teaching", "teaching", "teaching", "analytics",
    "analytics", None, None, None, None,
]


def authored_weights():
    """Return the four tensors authored here, with no optimization or fitting."""
    w1 = np.zeros((48, 48), dtype=np.float32)
    b1 = np.zeros(48, dtype=np.float32)
    w2 = np.zeros((48, 3), dtype=np.float32)
    b2 = np.zeros(3, dtype=np.float32)

    # Context-gated feature neurons: ReLU(attribute + context - 1).
    # Universal features pass directly through ReLU(attribute).
    for i, context in enumerate(ATTRIBUTE_CONTEXTS):
        w1[i, i] = 1
        if context is not None:
            w1[32 + CONTEXTS.index(context), i] = 1
            b1[i] = -1

    # Direct output tensor assignments: [utility, veto, modeled evidence].
    # Edit these actual weights to change the policy. Evidence is explicit:
    # a zero-utility permissibility feature supplies no ordering preference.
    w2[0] = [3, 0, 1]       # concise
    w2[1] = [4, 0, 1]       # verified
    w2[2] = [-5, 0, 1]      # speculative
    w2[3] = [3, 0, 1]       # autonomous_completion
    w2[4] = [-2, 0, 1]      # redundant_approval
    w2[5] = [3, 0, 1]       # finished_deliverable
    w2[6] = [-20, 1, 1]     # shabbat_flight
    w2[7] = [3, 0, 1]       # avoids_shabbat
    w2[8] = [0, 0, 0]       # misses_one_or_two_days
    w2[9] = [4, 0, 1]       # scientific_ai
    w2[10] = [4, 0, 1]      # secure_sandbox
    w2[11] = [3, 0, 1]      # reproducible
    w2[12] = [3, 0, 1]      # measured_performance
    w2[13] = [2, 0, 1]      # open_source
    w2[14] = [3, 0, 1]      # relevant_expertise
    w2[15] = [2, 0, 1]      # pastel
    w2[16] = [2, 0, 1]      # coarse_beard
    w2[17] = [-3, 0, 1]     # printed_name
    w2[18] = [3, 0, 1]      # single_page
    w2[19] = [-3, 0, 1]     # multi_page
    w2[20] = [4, 0, 1]      # netlify_domain
    w2[21] = [-4, 0, 1]     # old_custom_domain
    w2[22] = [3, 0, 1]      # merged_pr
    w2[23] = [-3, 0, 1]     # unmerged_pr
    w2[24] = [3, 0, 1]      # required_services
    w2[25] = [-3, 0, 1]     # frontend_only
    w2[26] = [3, 0, 1]      # genuine_visitors
    w2[27] = [-3, 0, 1]     # ci_as_visitors
    w2[28] = [5, 0, 1]      # explicitly_preferred
    w2[29] = [-5, 0, 1]     # explicitly_disliked
    w2[30] = [-20, 1, 1]    # explicitly_disallowed
    w2[31] = [0, 0, 0]      # explicitly_allowed

    # Hidden units 32..47 remain zero and count toward the 2,499 parameters.
    # b2 is explicitly all zero; no preference exists without active evidence.
    return validate_weights((w1, b1, w2, b2))


def forward(x, weights=None):
    """Compatibility entry point; network.py owns the shared forward pass."""
    return network_forward(x, authored_weights() if weights is None else weights)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_name("weights.npz"),
        help="Weight artifact to export, or verify with --check",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Verify that the existing artifact exactly matches authored tensors; do not write",
    )
    args = parser.parse_args(argv)
    try:
        tensors = authored_weights()
        if args.check:
            existing = load_weights(args.output)
            if weights_digest(existing) != weights_digest(tensors):
                raise ValueError("Artifact differs from directly authored tensors")
            tensors = existing  # report the artifact's own counts and fingerprint
        else:
            save_weights(args.output, tensors)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(json.dumps({
        "path": str(args.output),
        "action": "checked" if args.check else "exported",
        "total_parameters": sum(a.size for a in tensors),
        "nonzero_parameters": sum(int(np.count_nonzero(a)) for a in tensors),
        "weights_sha256": weights_digest(tensors),
    }))


if __name__ == "__main__":
    main()
