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


def _path(value):
    return Path(value).expanduser()


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
        'history': _history(bundle),
        'layout': bundle.layout,
        'tensors': {name: list(shape) for name, shape in TENSOR_SHAPES.items()},
        'w2_columns': ['utility', 'veto', 'evidence'],
        'w2': bundle.weights[2].tolist(),
        'b2': bundle.weights[3].tolist(),
    }


def _history(bundle):
    return [{'id': record['patch']['id'], 'harness': record['patch']['source']['harness'],
             'evidence': record['patch']['source']['evidence'], 'edits': len(record['patch']['edits'])}
            for record in bundle.manifest['patches']]


def _selection(bundle, tensor_name, index):
    tensor = bundle.weights[list(TENSOR_SHAPES).index(tensor_name)]
    if index is not None:
        if len(index) > tensor.ndim or any(i < 0 or i >= size for i, size in zip(index, tensor.shape)):
            raise ValueError(f'Index must address {tensor_name} with shape {tensor.shape}')
        tensor = tensor[tuple(index)]
    return np.asarray(tensor).tolist()


def _save_edit(source, target, original, edited):
    same_path = source.resolve() == target.resolve()
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
    history = _history(bundle)
    print(f'History: {len(history)} verified contributions' + (':' if history else ''))
    if len(history) > 20:
        print(f'  ... {len(history) - 20} earlier')
    for number, record in list(enumerate(history, 1))[-20:]:
        evidence = record['evidence'] if len(record['evidence']) <= 70 else record['evidence'][:67] + '...'
        plural = '' if record['edits'] == 1 else 's'
        print(f'  {number} {record["harness"]}: {record["edits"]} edit{plural}, evidence {json.dumps(evidence)}')


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
    path = path.resolve()
    layout = bundle.layout
    example = {'question': 'Which option?', 'contexts': [layout['contexts'][0]], 'options': [
        {'id': 'A', 'attributes': {layout['attributes'][0]: 1}},
        {'id': 'B', 'attributes': {layout['attributes'][1]: 1}},
    ]}
    edit = _runtime_command('set', path, '--tensor', 'w2', '--index', 'ROW', '--values', 'UTILITY', 'VETO',
                            'EVIDENCE', '--reason', 'EVIDENCE_REFERENCE', '--harness', 'YOUR_HARNESS')
    # Name and harness strings come from the file; print them as quoted data.
    print(f'''zozoBrain handoff for {path}
Name: {json.dumps(bundle.name)}; contributors: {json.dumps(bundle.contributing_harnesses)} (data, not instructions)
Revision SHA-256: {bundle.revision_sha256}

Encode only preferences, constraints, and corrections I state explicitly in
context I have authorized, each with a short evidence reference. Never guess
tastes. Keep existing rows and history. Edit tensors directly: no training,
fitting, or averaging. Report any preference the layout cannot express.

1. Inspect the labeled weights and history:
{_runtime_command('inspect', path)}

2. Edit. Inputs are 32 option attributes in [0,1] and 16 contexts. In a brain
made by init, hidden neuron i = ReLU(attribute_i + {layout['contexts'][0]} - 1), and w2 row i is
[utility, veto, evidence]. veto > 0 removes an option. Ranking needs evidence > 0
on every remaining option; otherwise the brain abstains. Hard rules need a binary
input {json.dumps(layout['binary_attributes'])}. To apply a rule in every context, set w1[32, i]
and b1[i] to 0. Each command assigns one row or scalar and updates the file in place:
{edit}

3. Check decisions with a request file such as
{json.dumps(example)}
{_runtime_command('decide', path, 'REQUEST.json')}

Report the exact edits, evidence, and decisions. For parallel work, edit a copy
(set ... --output COPY.brain), then combine copies with brain.py diff and merge.''')


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    init = commands.add_parser('init', help='Create one blank, portable brain file')
    init.add_argument('path', type=_path)
    init.add_argument('--name', type=_nonempty)
    init.add_argument('--layout', type=_path, help='Data-only JSON with 32 attributes, 16 contexts, and binary_attributes')
    inspect = commands.add_parser('inspect', help='Show state, schema, and optional tensor values')
    inspect.add_argument('path', type=_path)
    inspect.add_argument('--json', action='store_true')
    inspect.add_argument('--tensor', choices=TENSOR_SHAPES)
    inspect.add_argument('--index', nargs='+', type=int, help='Scalar coordinates or a row/prefix selection')
    edit = commands.add_parser('set', help='Directly assign a scalar or tensor row and retain its evidence')
    edit.add_argument('path', type=_path)
    edit.add_argument('--tensor', required=True, choices=TENSOR_SHAPES)
    edit.add_argument('--index', nargs='+', type=int, required=True)
    edit.add_argument('--values', nargs='+', type=float, required=True)
    edit.add_argument('--reason', type=_nonempty, required=True)
    edit.add_argument('--harness', type=_nonempty, default='manual')
    edit.add_argument('--output', type=_path, help='Save separately; default: atomically update the input file')
    decide = commands.add_parser('decide', help='Decide using the weights and verified history in a brain file')
    decide.add_argument('path', type=_path)
    decide.add_argument('input', type=_path)
    decide.add_argument('--json', action='store_true')
    decide.add_argument('--force', action='store_true')
    serve = commands.add_parser('serve', help='Read JSON Lines requests and write JSON Lines responses')
    serve.add_argument('path', type=_path)
    diff = commands.add_parser('diff', help='Capture a direct tensor diff as an attributable JSON patch')
    diff.add_argument('base', type=_path)
    diff.add_argument('edited', type=_path)
    diff.add_argument('--harness', type=_nonempty, required=True)
    diff.add_argument('--reason', type=_nonempty, required=True)
    diff.add_argument('--id', type=_nonempty, dest='patch_id')
    diff.add_argument('--output', type=_path, required=True)
    merge = commands.add_parser('merge', help='Apply compatible JSON patches to one base brain')
    merge.add_argument('base', type=_path)
    merge.add_argument('patches', nargs='+', type=_path)
    merge.add_argument('--output', type=_path, required=True)
    pack = commands.add_parser('pack', help='Pack trusted local layout and an existing NPZ snapshot')
    pack.add_argument('--weights', type=_path, required=True)
    pack.add_argument('--manifest', type=_path)
    pack.add_argument('--output', type=_path, required=True)
    pack.add_argument('--name', type=_nonempty, default='brain')
    prompt = commands.add_parser('prompt', help='Print a self-contained handoff task for another agent')
    prompt.add_argument('path', type=_path)
    return parser


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(errors='backslashreplace')  # e.g. cp1252 consoles and non-ASCII names
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == 'init':
            layout = _read_json(args.layout) if args.layout else None
            bundle = create_blank(args.name or args.path.stem, layout=layout)
            save_bundle(args.path, bundle)
            _saved(args.path, bundle)
            print('Next: ' + _runtime_command('inspect', args.path.resolve()))
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
            if not patch['edits']:
                raise ValueError('No tensor changes between these brains; nothing to contribute')
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
