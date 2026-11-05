"""Use one portable .brain file through direct tensor edits and local decisions."""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

import numpy as np

from brain_bundle import (
    _unique_object, apply_patches, create_blank, diff_bundles, edit_bundle, load_bundle,
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
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=_unique_object,
                      parse_constant=_reject_constant)


def _print_json(value):
    print(json.dumps(value, indent=2, allow_nan=False))


def _summary(path, bundle):
    rules, silent, _, free = _neurons(bundle)
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
        'neurons': rules,
        'silent_neurons': silent,
        'free_neurons': free,
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


def _span(numbers):
    """[1, 2, 3, 7] -> '1-3, 7'."""
    runs = []
    for number in numbers:
        if runs and number == runs[-1][1] + 1:
            runs[-1][1] = number
        else:
            runs.append([number, number])
    return ', '.join(f'{low}-{high}' if high > low else f'{low}' for low, high in runs)


def _names(names, first):
    """Number names by w1 row; shorten runs of generic names: 4:attribute_5 ... 31:attribute_32."""
    runs = []
    for row, name in enumerate(names, first):
        match = re.fullmatch(r'(.*_)([1-9][0-9]*)', name)  # plain ASCII numbers only, so the run can be rebuilt
        previous = runs[-1][-1] if runs else None
        if match and previous and previous[2] == match[1] and previous[3] + 1 == int(match[2]):
            runs[-1].append((row, name, match[1], int(match[2])))
        else:
            runs.append([(row, name, match[1] if match else None, int(match[2]) if match else None)])
    return ', '.join(f'{run[0][0]}:{run[0][1]} ... {run[-1][0]}:{run[-1][1]}' if len(run) > 2
                     else ', '.join(f'{row}:{name}' for row, name, *_ in run) for run in runs)


def _number(value):
    return f'{float(value):g}'


def _neurons(bundle):
    """Read each hidden neuron back from w1, b1, and w2 as the rule it computes."""
    w1, b1, w2, _ = bundle.weights
    names = bundle.layout['attributes'] + [f'context:{name}' for name in bundle.layout['contexts']]
    rules, silent, wired, free = [], [], [], []
    for j in range(48):
        column, bias = w1[:, j], float(b1[j])
        if not np.any(w2[j]):
            if not np.any(column) and not bias:
                free.append(j)
                continue
            if j < 32 and bias == -1 and np.count_nonzero(column) == 2 and column[j] == 1 and column[32] == 1:
                wired.append(j)  # init wiring, no output yet
                continue
        text = ''
        terms = [(column[i], names[i]) for i in np.flatnonzero(column)] + ([(bias, '')] if bias else [])
        for coefficient, name in terms:
            size = abs(float(coefficient))
            term = name if size == 1 and name else f'{size:g}*{name}' if name else f'{size:g}'
            sign = '-' if coefficient < 0 else '+'
            text += f' {sign} {term}' if text else (f'-{term}' if sign == '-' else term)
        # Inputs are at most 1: if the attributes alone cannot lift it past zero, the neuron needs a context.
        alone = float(np.maximum(column[:32], 0).sum()) + bias
        can_fire = alone + float(np.maximum(column[32:], 0).sum()) > 0
        only_with = [] if alone > 0 else [bundle.layout['contexts'][i] for i in np.flatnonzero(column[32:] > 0)]
        if not np.any(w2[j]):
            silent.append({'neuron': j, 'rule': f'ReLU({text or 0})'})
            continue
        rules.append({'neuron': j, 'rule': f'ReLU({text or 0})', 'output': w2[j].tolist(),
                      'only_with': only_with, 'can_fire': can_fire})
    return rules, silent, wired, free


def _print_brain(bundle, full_history=False):
    rules, silent, wired, free = _neurons(bundle)
    print('Neurons that drive decisions (rule -> [utility, veto, evidence]):' + ('' if rules else ' none yet'))
    for row in rules:
        scope = (' (never fires)' if not row['can_fire']
                 else f' (only with {", ".join(row["only_with"])})' if row['only_with'] else '')
        print(f'  {row["neuron"]} {row["rule"]} -> [' + ', '.join(map(_number, row['output'])) + ']' + scope)
    w1, w2 = bundle.weights[0], bundle.weights[2]
    driving = [j for j in range(48) if np.any(w2[j])]
    read = [name for i, name in enumerate(bundle.layout['attributes']) if np.any(w1[i, driving])]
    print('Attributes read by rules: ' + (', '.join(read) or 'none'))
    general = bundle.layout['contexts'][0]
    if silent:
        print('Wired, no output yet: ' + ', '.join(f'{row["neuron"]} {row["rule"]}' for row in silent))
    if wired:
        print(f'Silent (init wiring, neuron i = ReLU(attribute i + {general} - 1)): {_span(wired)}')
    if free:
        print(f'Free for new rules: {_span(free)}')
    if np.any(bundle.weights[3]):
        print('b2, added to every option: [' + ', '.join(map(_number, bundle.weights[3])) + ']')
    print('Attributes (w1 rows 0-31): ' + _names(bundle.layout['attributes'], 0))
    print('Contexts (w1 rows 32-47): ' + _names(bundle.layout['contexts'], 32))
    print('Binary inputs: ' + (', '.join(bundle.layout['binary_attributes']) or 'none'))
    history = _history(bundle)
    print(f'History: {len(history)} verified contributions' + (':' if history else ''))
    shown = list(enumerate(zip(history, bundle.manifest['patches']), 1))
    if len(history) > 20 and not full_history:
        print(f'  ... {len(history) - 20} earlier (inspect --history shows all, with every edit)')
        shown = shown[-20:]
    for number, (record, stored) in shown:
        evidence = record['evidence']
        if not full_history and len(evidence) > 70:
            evidence = evidence[:67] + '...'
        plural = '' if record['edits'] == 1 else 's'
        print(f'  {number} {record["harness"]}: {record["edits"]} edit{plural}, evidence {json.dumps(evidence)}')
        if full_history:
            actions = stored.get('edit_actions', ['applied'] * len(stored['patch']['edits']))
            for edit, action in zip(stored['patch']['edits'], actions):
                index = ', '.join(map(str, edit['index']))
                note = ' (already present)' if action == 'already_present' else ''
                print(f'      {edit["tensor"]}[{index}]: {_number(edit["before"])} -> {_number(edit["after"])}{note}')


def _decision_text(result):
    if result['decision'] is None:
        missing = [row['id'] for row in result['ranking'] if not row['modeled_evidence']]
        detail = f' (no evidence for {", ".join(missing)})' if result['status'] == 'insufficient_preference_evidence' else ''
        print('Abstained: ' + result['status'].replace('_', ' ') + detail)
    else:
        selected = next(row for row in result['ranking'] if row['id'] == result['decision'])
        label = selected['label']
        suffix = f' ({label})' if label != selected['id'] else ''
        forced = ' [forced]' if result['forced'] else ''
        print(f'Decision: {selected["id"]}{suffix}{forced}')
    if result['ranking']:
        print('Ranking: ' + ', '.join(f'{row["id"]} ({row["utility"]:g})' for row in result['ranking']))
        why = []
        for row in result['ranking']:
            terms = [f'neuron {n["neuron"]} ({n["output"][0]:+g})' for n in row['neurons'] if n['output'][0]]
            if result['b2'][0]:
                terms.append(f'b2 ({result["b2"][0]:+g})')
            why.append(f'{row["id"]} = ' + (' + '.join(terms) or 'no neuron'))
        print('Utility: ' + '; '.join(why))
    if result['vetoed']:
        causes = []
        for row in result['vetoed']:
            sources = [f'neuron {n["neuron"]}' for n in row['neurons'] if n['output'][1] > 0]
            sources += ['b2'] if result['b2'][1] > 0 else []
            causes.append(f'{row["id"]} ({", ".join(sources)})' if sources else row['id'])
        print('Vetoed: ' + ', '.join(causes))
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
    command = ['uv', 'run', '--locked', '--project', str(script.parent), str(script), *map(str, args)]
    if sys.platform == 'win32':
        return subprocess.list2cmdline(command)
    return shlex.join(command)


def _prompt(path, bundle):
    path = path.resolve()
    layout = bundle.layout
    example = {'question': 'Which option?', 'contexts': [layout['contexts'][0]], 'options': [
        {'id': 'A', 'attributes': {layout['attributes'][0]: 1}},
        {'id': 'B', 'attributes': {layout['attributes'][1]: 1}},
    ]}
    edit = _runtime_command('set', path, '--tensor', 'w2', '--index', 'ROW', '--values', 'UTILITY', 'VETO',
                            'EVIDENCE', '--reason', 'EVIDENCE_REFERENCE', '--harness', 'YOUR_HARNESS')
    wire = _runtime_command('set', path, '--tensor', 'w1', '--index', 'ATTRIBUTE_ROW', 'J', '--values', 1,
                            '--tensor', 'w1', '--index', 'CONTEXT_ROW', 'J', '--values', 1,
                            '--tensor', 'b1', '--index', 'J', '--values', -1,
                            '--tensor', 'w2', '--index', 'J', '--values', 'UTILITY', 'VETO', 'EVIDENCE',
                            '--reason', 'EVIDENCE_REFERENCE', '--harness', 'YOUR_HARNESS')
    general = layout['contexts'][0]
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

2. Edit. w1 rows 0-31 are option attributes in [0,1]; rows 32-47 are contexts,
1 when active (a request can activate several). Hidden neuron j computes
ReLU(sum of input_i * w1[i, j] + b1[j]). Its w2 row j is
[utility, veto, evidence], and inspect prints each neuron as that rule.
veto > 0 removes an option. Ranking needs evidence > 0 on every remaining
option; otherwise the brain abstains. Hard rules need a binary input {json.dumps(layout['binary_attributes'])}.
- In a brain made by init, neuron i < 32 is ReLU(attribute_i + {general} - 1):
  write w2 row i for a rule in {general}. To apply it in every context, also set
  w1[32, i] and b1[i] to 0.
- For a rule in one context only, wire a free neuron j: w1[ATTRIBUTE_ROW, j] = 1,
  w1[CONTEXT_ROW, j] = 1, b1[j] = -1, then w2 row j.
- To fix or remove a rule, assign the wrong entries their correct values (0 to unwire).
The outputs of all firing neurons add up, and decide prints each option's
utility as that sum. A rule meant to outweigh existing ones needs a utility
larger than the gap it must close.
Each command updates the file in place. Repeat --tensor, --index, --values to
make several assignments one verified edit:
{edit}
{wire}

3. Check decisions with a request file such as
{json.dumps(example)}
{_runtime_command('decide', path, 'REQUEST.json')}
List every active context: rules wired through {general} apply only when it is
listed, so check a context rule with ["{general}", "CONTEXT"].

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
    inspect.add_argument('--history', action='store_true', help='Show every verified edit in full, entry by entry')
    inspect.add_argument('--tensor', choices=TENSOR_SHAPES)
    inspect.add_argument('--index', nargs='+', type=int, help='Scalar coordinates or a row/prefix selection')
    edit = commands.add_parser('set', help='Directly assign scalars or tensor rows as one edit with its evidence')
    edit.add_argument('path', type=_path)
    edit.add_argument('--tensor', required=True, choices=TENSOR_SHAPES, action='append',
                      help='Repeat --tensor, --index, --values to make several assignments one edit')
    edit.add_argument('--index', nargs='+', type=int, required=True, action='append')
    edit.add_argument('--values', nargs='+', type=float, required=True, action='append')
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
                _print_brain(bundle, full_history=args.history)
                if args.tensor:
                    index = ''.join(f'[{i}]' for i in args.index or [])
                    print(f'{args.tensor}{index} = ' + json.dumps(summary['selection']['values'], allow_nan=False))
        elif args.command == 'set':
            if not len(args.tensor) == len(args.index) == len(args.values):
                raise ValueError('Give --tensor, --index, and --values once for each assignment')
            assignments = list(zip(args.tensor, args.index, args.values))
            original = load_bundle(args.path)
            edited = edit_bundle(original, assignments=assignments, harness=args.harness, evidence=args.reason)
            output = args.output or args.path
            _save_edit(args.path, output, original, edited)
            _saved(output, edited)
            for tensor, index, _ in assignments:
                print(f'{tensor}{index} = ' + json.dumps(_selection(edited, tensor, index)))
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
                explain=True,
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
