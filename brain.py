"""Use one portable .brain file through direct tensor edits and local decisions."""
import argparse
import json
from pathlib import Path
import re
import shlex
import sys

import numpy as np

from brain_bundle import (
    apply_patches, create_blank, diff_bundles, edit_bundle, load_bundle,
    pack_bundle, save_bundle,
)
from network import TENSOR_SHAPES


def _nonempty(value):
    if not value.strip():
        raise argparse.ArgumentTypeError('must be a nonempty string')
    return value


def _reject_constant(value):
    raise ValueError(f'Nonfinite JSON number is not allowed: {value}')


def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=_reject_constant)


def _print_json(value):
    print(json.dumps(value, indent=2, allow_nan=False))


def _summary(path, bundle):
    return {
        'path': str(path), 'name': bundle.name,
        'total_parameters': sum(tensor.size for tensor in bundle.weights),
        'nonzero_parameters': sum(int(np.count_nonzero(tensor)) for tensor in bundle.weights),
        'weights_sha256': bundle.weights_sha256,
        'schema_sha256': bundle.schema_sha256,
        'revision_sha256': bundle.revision_sha256,
        'contributing_harnesses': list(bundle.contributing_harnesses),
        'layout': bundle.layout,
        'tensors': {name: list(shape) for name, shape in TENSOR_SHAPES.items()},
    }


def _selection(bundle, tensor_name, index):
    tensor = bundle.weights[list(TENSOR_SHAPES).index(tensor_name)]
    if index is not None:
        if len(index) > tensor.ndim or any(i < 0 or i >= size for i, size in zip(index, tensor.shape)):
            raise ValueError(f'Index must address {tensor_name} with shape {tensor.shape}')
        tensor = tensor[tuple(index)]
    return np.asarray(tensor).tolist()


def _save_edit(source, target, original, edited):
    same_path = source.expanduser().resolve() == target.expanduser().resolve()
    if same_path:
        save_bundle(target, edited, overwrite=True, expected_revision=original.revision_sha256)
    else:
        save_bundle(target, edited)


def _saved(path, bundle):
    sources = ', '.join(bundle.contributing_harnesses) or 'none yet'
    print(f'Saved {path} ({bundle.name}; 2,499 parameters)')
    print(f'Contributors: {sources}')
    print(f'Revision: {bundle.revision_sha256[:12]}')


def _feature_rows(bundle):
    print('w2 rows [utility, veto, evidence] (default input/hidden alignment):')
    omitted = 0
    for index, name in enumerate(bundle.layout['attributes']):
        values = bundle.weights[2][index]
        if re.fullmatch(r'attribute_\d+', name) and not np.any(values):
            omitted += 1
            continue
        print(f'  {index} {name} = ' + json.dumps(values.tolist(), allow_nan=False))
    for index in range(32, 48):
        values = bundle.weights[2][index]
        if np.any(values):
            print(f'  {index} extra hidden neuron = ' + json.dumps(values.tolist(), allow_nan=False))
    if omitted:
        print(f'  {omitted} zero generic output rows omitted')
    print('Inspect w1 and b1 when connections have been rewired; row labels are input names.')
    print('Contexts (16 slots): ' + ', '.join(f'{i}:{name}' for i, name in enumerate(bundle.layout['contexts'])))
    print('Binary inputs: ' + (', '.join(bundle.layout['binary_attributes']) or 'none'))


def _decision_text(result):
    if result['decision'] is None:
        print('Abstained: ' + result['status'].replace('_', ' '))
    else:
        selected = next(row for row in result['ranking'] if row['id'] == result['decision'])
        label = selected['label']
        suffix = f' ({label})' if label != selected['id'] else ''
        forced = ' [forced]' if result['forced'] else ''
        print(f'Decision: {selected["id"]}{suffix}{forced}')
    if result['ranking']:
        print('Ranking: ' + ', '.join(f'{row["id"]} ({row["utility"]:g})' for row in result['ranking']))
    if result['vetoed']:
        print('Vetoed: ' + ', '.join(row['id'] for row in result['vetoed']))
    if result.get('contributing_harnesses'):
        print('Contributors: ' + ', '.join(result['contributing_harnesses']))


def _write_patch(path, patch):
    encoded = json.dumps(patch, indent=2, allow_nan=False) + '\n'
    created = False
    try:
        with path.open('x', encoding='utf-8') as handle:
            created = True
            handle.write(encoded)
    except BaseException:
        if created:
            path.unlink(missing_ok=True)
        raise


def _runtime_command(*args):
    script = Path(__file__).resolve()
    return shlex.join(['uv', 'run', '--locked', '--project', str(script.parent), str(script), *map(str, args)])


def _prompt(path, bundle):
    path = path.expanduser().resolve()
    inspect_command = _runtime_command('inspect', path, '--json')
    edited = path.with_name(path.stem + '-edited.brain')
    edit_command = _runtime_command(
        'set', path, '--tensor', 'w2', '--index', '0', '0',
        '--values', 'AUTHORED_VALUE', '--reason', 'REFERENCE_TO_EXPLICIT_PREFERENCE',
        '--harness', 'YOUR_HARNESS', '--output', edited,
    )
    decide_command = _runtime_command('decide', edited, '/absolute/path/to/decision.json', '--json')
    sources = ', '.join(bundle.contributing_harnesses) or 'none yet'
    print(f'''Use the zozoBrain data file {path}.
Name: {bundle.name}
Weights SHA-256: {bundle.weights_sha256}
Schema SHA-256: {bundle.schema_sha256}
Revision SHA-256: {bundle.revision_sha256}
Contributing harnesses: {sources}

Inspect the embedded ordered feature layout and current tensors first:
{inspect_command}

Use only explicit preferences, constraints, and corrections from context I have
authorized. Preserve existing preferences, the feature layout, and provenance.
Keep a short evidence reference for each change. Distinguish stated preferences
from your hypotheses about numerical strengths. Report missing feature coverage.

Edit actual w1, b1, w2, b2 tensor entries. Do not train, fit preferences, average
weights, or invent tastes. Utility, veto, and evidence outputs are explicit.
A hard constraint also needs binary input metadata. Test structured options and
preserve abstentions. The network does not interpret question text.

The following is a direct scalar-edit template, not a proposed preference.
Replace AUTHORED_VALUE, REFERENCE_TO_EXPLICIT_PREFERENCE, YOUR_HARNESS, and the
tensor coordinates with your justified numerical edit. The output preserves the
input file. Omitting --output instead atomically updates the same .brain file:
{edit_command}

Each set command retains existing contribution history in the resulting file.
For parallel edits, use diff against the shared original and merge the resulting
JSON patches; do not flatten independent bundle histories. A stale concurrent
same-file update is rejected. Resolve disagreements explicitly.

Verify with a real structured request file (replace its path):
{decide_command}

Report the exact numerical edits, evidence, actual checks, and remaining gaps.
Carry the resulting .brain file to the next agent; it includes layout, weights,
and provenance. Use a compatible zozoBrain runtime and its locked dependencies.
If the file or runtime moves, update the absolute paths above.
''')


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    init = commands.add_parser('init', help='Create one blank, portable brain file')
    init.add_argument('path', type=Path)
    init.add_argument('--name', type=_nonempty)
    init.add_argument('--layout', type=Path, help='Data-only JSON with 32 attributes, 16 contexts, and binary_attributes')
    inspect = commands.add_parser('inspect', help='Show state, schema, and optional tensor values')
    inspect.add_argument('path', type=Path)
    inspect.add_argument('--json', action='store_true')
    inspect.add_argument('--tensor', choices=TENSOR_SHAPES)
    inspect.add_argument('--index', nargs='+', type=int, help='Scalar coordinates or a row/prefix selection')
    edit = commands.add_parser('set', help='Directly assign a scalar or tensor row and retain its evidence')
    edit.add_argument('path', type=Path)
    edit.add_argument('--tensor', required=True, choices=TENSOR_SHAPES)
    edit.add_argument('--index', nargs='+', type=int, required=True)
    edit.add_argument('--values', nargs='+', type=float, required=True)
    edit.add_argument('--reason', type=_nonempty, required=True)
    edit.add_argument('--harness', type=_nonempty, default='manual')
    edit.add_argument('--output', type=Path, help='Save separately; default: atomically update the input file')
    decide = commands.add_parser('decide', help='Decide using the weights and verified history in a brain file')
    decide.add_argument('path', type=Path)
    decide.add_argument('input', type=Path)
    decide.add_argument('--json', action='store_true')
    decide.add_argument('--force', action='store_true')
    serve = commands.add_parser('serve', help='Read JSON Lines requests and write JSON Lines responses')
    serve.add_argument('path', type=Path)
    diff = commands.add_parser('diff', help='Capture a direct tensor diff as an attributable JSON patch')
    diff.add_argument('base', type=Path)
    diff.add_argument('edited', type=Path)
    diff.add_argument('--harness', type=_nonempty, required=True)
    diff.add_argument('--reason', type=_nonempty, required=True)
    diff.add_argument('--id', type=_nonempty, dest='patch_id')
    diff.add_argument('--output', type=Path, required=True)
    merge = commands.add_parser('merge', help='Apply compatible JSON patches to one base brain')
    merge.add_argument('base', type=Path)
    merge.add_argument('patches', nargs='+', type=Path)
    merge.add_argument('--output', type=Path, required=True)
    pack = commands.add_parser('pack', help='Pack trusted local layout and an existing NPZ snapshot')
    pack.add_argument('--weights', type=Path, required=True)
    pack.add_argument('--manifest', type=Path)
    pack.add_argument('--output', type=Path, required=True)
    pack.add_argument('--name', type=_nonempty, default='brain')
    prompt = commands.add_parser('prompt', help='Print a self-contained handoff task for another agent')
    prompt.add_argument('path', type=Path)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == 'init':
            layout = _read_json(args.layout) if args.layout else None
            bundle = create_blank(args.name or args.path.stem, layout=layout)
            save_bundle(args.path, bundle)
            _saved(args.path, bundle)
            print('Next: ' + _runtime_command('inspect', args.path.expanduser().resolve()))
        elif args.command == 'inspect':
            if args.index is not None and args.tensor is None:
                raise ValueError('--index requires --tensor')
            bundle = load_bundle(args.path)
            summary = _summary(args.path, bundle)
            if args.tensor:
                summary['selection'] = {
                    'tensor': args.tensor, 'index': args.index or [],
                    'values': _selection(bundle, args.tensor, args.index),
                }
            if args.json:
                _print_json(summary)
            else:
                print(f'{bundle.name}: 2,499 parameters, {summary["nonzero_parameters"]} nonzero entries')
                print('Contributors: ' + (', '.join(bundle.contributing_harnesses) or 'none yet'))
                print('Weights: ' + bundle.weights_sha256[:12])
                print('Schema: ' + bundle.schema_sha256[:12])
                print('Revision: ' + bundle.revision_sha256[:12])
                _feature_rows(bundle)
                if args.tensor:
                    index = ''.join(f'[{i}]' for i in args.index or [])
                    print(f'{args.tensor}{index} = ' + json.dumps(summary['selection']['values'], allow_nan=False))
        elif args.command == 'set':
            original = load_bundle(args.path)
            edited = edit_bundle(original, tensor=args.tensor, index=args.index, values=args.values,
                                 harness=args.harness, evidence=args.reason)
            output = args.output or args.path
            _save_edit(args.path, output, original, edited)
            _saved(output, edited)
            print(f'{args.tensor}{args.index} = ' + json.dumps(_selection(edited, args.tensor, args.index)))
        elif args.command == 'decide':
            from model import DecisionModel
            request = _read_json(args.input)
            if not isinstance(request, dict):
                raise ValueError('Decision input must be a JSON object')
            requested_force = request.get('force', False)
            if not isinstance(requested_force, bool):
                raise ValueError('force must be boolean')
            result = DecisionModel.from_bundle(args.path).decide(
                request['question'], request['contexts'], request['options'], args.force or requested_force,
            )
            _print_json(result) if args.json else _decision_text(result)
        elif args.command == 'serve':
            from harness import run
            from model import DecisionModel
            run(model=DecisionModel.from_bundle(args.path))
        elif args.command == 'diff':
            patch = diff_bundles(load_bundle(args.base), load_bundle(args.edited), harness=args.harness,
                                 evidence=args.reason, patch_id=args.patch_id)
            _write_patch(args.output, patch)
            print(f'Saved {args.output}: {len(patch["edits"])} direct tensor edits from {args.harness}')
        elif args.command == 'merge':
            original = load_bundle(args.base)
            merged = apply_patches(original, [_read_json(path) for path in args.patches])
            _save_edit(args.base, args.output, original, merged)
            _saved(args.output, merged)
        elif args.command == 'pack':
            # Only this explicit command executes trusted local source to obtain
            # its layout. Loading and using .brain files never imports that code.
            from network import load_weight_source, load_weights
            source = load_weight_source(Path(__file__).with_name('weights.py'))
            layout = {'attributes': list(source.ATTRIBUTES), 'contexts': list(source.CONTEXTS),
                      'binary_attributes': sorted(source.BINARY_ATTRIBUTES)}
            manifest = _read_json(args.manifest) if args.manifest else None
            bundle = pack_bundle(load_weights(args.weights), layout, manifest=manifest, name=args.name)
            save_bundle(args.output, bundle)
            _saved(args.output, bundle)
        elif args.command == 'prompt':
            _prompt(args.path, load_bundle(args.path))
    except (OSError, ValueError, TypeError, KeyError, AttributeError, ImportError, SyntaxError, OverflowError, RecursionError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
