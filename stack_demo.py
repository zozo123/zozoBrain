"""Two agents edit one .brain file directly; decisions change and history stays verified."""
import os
from pathlib import Path
import sys
import tempfile

from brain_bundle import apply_patches, create_blank, diff_bundles, edit_bundle, load_bundle, save_bundle
from model import DecisionModel

REQUESTS = [
    ('cheap vs pricey', [('cheap', {'low_cost': 1}), ('pricey', {'high_cost': 1})]),
    ('banned_cheap vs pricey', [('banned_cheap', {'low_cost': 1, 'explicitly_disallowed': 1}),
                                ('pricey', {'high_cost': 1})]),
    ('cheap_cloud vs cheap_offline', [('cheap_cloud', {'low_cost': 1}),
                                      ('cheap_offline', {'low_cost': 1, 'offline': 1})]),
    ('X vs Y, nothing stated', [('X', {'attribute_5': 1}), ('Y', {'attribute_6': 1})]),
]
# Synthetic statements for the demo, not anyone's real preferences.
EDITS = [
    ('codex', 0, [3, 0, 1], 'I prefer lower cost'),
    ('codex', 1, [-3, 0, 1], 'I prefer lower cost'),
    ('claude', 3, [0, 1, 0], 'Never pick disallowed options'),
    ('claude', 2, [2, 0, 1], 'I prefer offline options'),
]
COLOR = sys.stdout.isatty() or bool(os.environ.get('FORCE_COLOR'))


def paint(text, code):
    return f'\033[{code}m{text}\033[0m' if COLOR else text


def decide(bundle, options):
    model = DecisionModel.from_bundle(bundle) if isinstance(bundle, Path) else DecisionModel(
        bundle.weights, bundle.manifest, layout=bundle.layout)
    result = model.decide('Which option?', ['general'],
                          [{'id': name, 'attributes': attributes} for name, attributes in options])
    vetoed = ', '.join(row['id'] for row in result['vetoed'])
    text = paint(result['decision'], '1;32') if result['decision'] else paint('abstain', '33')
    return text + (paint(f' ({vetoed} vetoed)', '31') if vetoed else ''), result


def run():
    blank = create_blank('demo')
    print(paint('zozoBrain: two agents, one 2,499-parameter brain', '1') + '\n')
    print(f'blank brain: cheap vs pricey -> {decide(blank, REQUESTS[0][1])[0]}')
    brain = blank
    for harness, row, values, evidence in EDITS:
        brain = edit_bundle(brain, tensor='w2', index=[row], values=values, harness=harness, evidence=evidence)
        name = brain.layout['attributes'][row]
        print(f'{paint(f"{harness:7}", "36" if harness == "codex" else "35")} w2[{row}] = {str(values):11} '
              f'{name:22} "{evidence}"')

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / 'me.brain'
        save_bundle(path, brain)
        size = path.stat().st_size
        print(f'\nsaved me.brain ({size:,} bytes), reloaded and verified:')
        for title, options in REQUESTS:
            print(f'  {title:28} -> {decide(path, options)[0]}')
        reloaded = load_bundle(path)
    contributors = ', '.join(reloaded.contributing_harnesses)
    print(f'history: {len(reloaded.manifest["patches"])} verified edits by {contributors}')

    codex = edit_bundle(blank, tensor='w2', index=[0, 0], values=[3], harness='codex', evidence='lower cost: 3')
    claude = edit_bundle(blank, tensor='w2', index=[0, 0], values=[5], harness='claude', evidence='lower cost: 5')
    patches = [diff_bundles(blank, edited, harness=edited.contributing_harnesses[0], evidence='parallel copy')
               for edited in (codex, claude)]
    try:
        apply_patches(blank, patches)
        conflict = None
    except ValueError as exc:
        conflict = str(exc)
        print(f'parallel copies: codex sets w2[0,0] = 3, claude sets 5 -> {paint("merge refused", "31")}')
    return {'brain': reloaded, 'decisions': [decide(reloaded, options)[1] for _, options in REQUESTS],
            'conflict': conflict, 'bytes': size}


if __name__ == '__main__':
    run()
