"""Capture and combine explicit tensor edits from independent agent harnesses.

Patches carry the author's evidence and exact before/after float32 values. They
never train, infer a preference, average edits, or silently resolve a conflict.
"""
import argparse
from collections import Counter
import copy
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

import numpy as np

from network import TENSOR_SHAPES, check_label, layout_digest, load_weight_source, load_weights, save_weights, validate_weights, weights_digest


PATCH_FORMAT='zozobrain.weight-patch.v1'
STACK_FORMAT='zozobrain.weight-stack.v1'
TENSOR_NAMES=tuple(TENSOR_SHAPES)


def schema_digest():
    """Bind a patch to feature order, binary inputs, dtype, and tensor layout."""
    source=load_weight_source(Path(__file__).with_name('weights.py'))
    return layout_digest(source.ATTRIBUTES,source.CONTEXTS,source.BINARY_ATTRIBUTES)


def _canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def _text(value,label):
    if not isinstance(value,str) or not value.strip():
        raise ValueError(f'{label} must be a nonempty string')
    return value


def _keys(value,expected,label):
    if not isinstance(value,dict) or set(value)!=set(expected):
        raise ValueError(f'{label} must contain exactly: '+', '.join(expected))


def _digest(value,label):
    if not isinstance(value,str) or not re.fullmatch('[0-9a-f]{64}',value):
        raise ValueError(f'{label} must be a lowercase SHA-256 digest')
    return value


def _expected_schema(value):
    return schema_digest() if value is None else _digest(value,'Expected schema_sha256')


def _weights(value):
    return validate_weights(value) if isinstance(value,(tuple,list)) else load_weights(value)


def _number(value,label):
    if isinstance(value,bool) or not isinstance(value,(int,float)):
        raise ValueError(f'{label} must be a finite float32 number')
    try:
        with np.errstate(over='ignore',invalid='ignore'):
            number=np.float32(value)
    except (OverflowError,ValueError,TypeError) as exc:
        raise ValueError(f'{label} must be a finite float32 number') from exc
    if not np.isfinite(number):
        raise ValueError(f'{label} must be a finite float32 number')
    return number


def validate_patch(patch,*,schema_sha256=None):
    """Validate a v1 patch completely before any tensor is changed."""
    expected_schema=_expected_schema(schema_sha256)
    _keys(patch,['format','id','source','schema_sha256','base_weights_sha256','edits'],'Patch')
    if patch['format']!=PATCH_FORMAT:
        raise ValueError(f'Unsupported patch format; expected {PATCH_FORMAT}')
    check_label(patch['id'],'Patch id')
    _keys(patch['source'],['harness','evidence'],'Patch source')
    check_label(patch['source']['harness'],'Source harness')
    _text(patch['source']['evidence'],'Source evidence')
    _digest(patch['schema_sha256'],'schema_sha256')
    if patch['schema_sha256']!=expected_schema:
        raise ValueError('Patch schema does not match the current attributes, contexts, and tensors')
    _digest(patch['base_weights_sha256'],'base_weights_sha256')
    if not isinstance(patch['edits'],list):
        raise ValueError('Patch edits must be a list')
    entries=set()
    for edit in patch['edits']:
        _keys(edit,['tensor','index','before','after','reason'],'Edit')
        tensor=edit['tensor']
        if not isinstance(tensor,str) or tensor not in TENSOR_SHAPES:
            raise ValueError('Edit tensor must be one of: '+', '.join(TENSOR_NAMES))
        shape=TENSOR_SHAPES[tensor]
        index=edit['index']
        if (not isinstance(index,list) or len(index)!=len(shape)
                or any(isinstance(i,bool) or not isinstance(i,int) or not 0<=i<size
                       for i,size in zip(index,shape))):
            raise ValueError(f'Invalid index for {tensor}: expected coordinates within {shape}')
        entry=(tensor,tuple(index))
        if entry in entries:
            raise ValueError(f'Patch repeats tensor entry {tensor}{tuple(index)}')
        entries.add(entry)
        _number(edit['before'],'Edit before')
        _number(edit['after'],'Edit after')
        _text(edit['reason'],'Edit reason')
    # Reject values that cannot be represented in JSON, including nonfinite
    # metadata. The full patch is retained unchanged in the provenance manifest.
    _canonical(patch)
    return patch


def diff(base_path,edited_path,*,harness,evidence,patch_id,schema_sha256=None):
    """Return every changed tensor entry as an explicit, attributable patch."""
    expected_schema=_expected_schema(schema_sha256)
    check_label(harness,'Source harness')
    _text(evidence,'Source evidence')
    check_label(patch_id,'Patch id')
    base=_weights(base_path)
    edited=_weights(edited_path)
    edits=[]
    for tensor,before,after in zip(TENSOR_NAMES,base,edited):
        for coordinates in np.argwhere(before.view(np.uint32)!=after.view(np.uint32)):
            index=tuple(int(i) for i in coordinates)
            edits.append({'tensor':tensor,'index':list(index),
                          'before':float(before[index]),'after':float(after[index]),
                          'reason':evidence})
    patch={'format':PATCH_FORMAT,'id':patch_id,
           'source':{'harness':harness,'evidence':evidence},
           'schema_sha256':expected_schema,'base_weights_sha256':weights_digest(base),
           'edits':edits}
    return validate_patch(patch,schema_sha256=expected_schema)


def _read_patch(value):
    if isinstance(value,dict):
        return copy.deepcopy(value)
    return json.loads(Path(value).read_text(encoding='utf-8'))


def _same(left,right):
    return np.float32(left).view(np.uint32)==np.float32(right).view(np.uint32)


def _actions(record):
    # Earlier v1 records applied every edit. New records also identify agreed
    # overlapping values, which add provenance without changing the tensor.
    actions=record.get('edit_actions',['applied']*len(record['patch']['edits']))
    if (not isinstance(actions,list) or len(actions)!=len(record['patch']['edits'])
            or any(action not in ('applied','already_present') for action in actions)):
        raise ValueError('Manifest edit_actions must describe every patch edit')
    return actions


def validate_manifest(manifest,base_weights,*,schema_sha256=None):
    """Validate saved provenance against the actual final tensors and schema.

    Reversing recorded changes checks every intermediate tensor digest. It
    validates consistency, not the truth of an author's evidence statement.
    """
    expected_schema=_expected_schema(schema_sha256)
    manifest=_read_patch(manifest)
    _keys(manifest,['format','schema_sha256','original_weights_sha256',
                    'final_weights_sha256','patches','duplicate_patch_ids_skipped'],'Manifest')
    if manifest['format']!=STACK_FORMAT:
        raise ValueError(f'Unsupported manifest format; expected {STACK_FORMAT}')
    if manifest['schema_sha256']!=expected_schema:
        raise ValueError('Manifest schema does not match the current attributes, contexts, and tensors')
    _digest(manifest['original_weights_sha256'],'original_weights_sha256')
    _digest(manifest['final_weights_sha256'],'final_weights_sha256')
    tensors=_weights(base_weights)
    if manifest['final_weights_sha256']!=weights_digest(tensors):
        raise ValueError('Manifest final_weights_sha256 does not match the supplied base weights')
    if not isinstance(manifest['patches'],list):
        raise ValueError('Manifest patches must be a list')
    seen={}
    expected=manifest['original_weights_sha256']
    for record in manifest['patches']:
        keys=['patch','input_weights_sha256','output_weights_sha256']
        if isinstance(record,dict) and 'edit_actions' in record:
            keys.append('edit_actions')
        _keys(record,keys,'Manifest patch record')
        patch=validate_patch(record['patch'],schema_sha256=expected_schema)
        if patch['id'] in seen:
            raise ValueError('Manifest contains duplicate patch records')
        seen[patch['id']]=_canonical(patch)
        _actions(record)
        _digest(record['input_weights_sha256'],'input_weights_sha256')
        _digest(record['output_weights_sha256'],'output_weights_sha256')
        if record['input_weights_sha256']!=expected:
            raise ValueError('Manifest patch digest chain is broken')
        expected=record['output_weights_sha256']
    if expected!=manifest['final_weights_sha256']:
        raise ValueError('Manifest final digest does not match its patch chain')
    skipped=manifest['duplicate_patch_ids_skipped']
    if not isinstance(skipped,list) or any(not isinstance(patch_id,str) or patch_id not in seen for patch_id in skipped):
        raise ValueError('Manifest skipped IDs must refer to retained patch records')
    reverse=tuple(array.copy() for array in tensors)
    for record in reversed(manifest['patches']):
        for edit,action in reversed(list(zip(record['patch']['edits'],_actions(record)))):
            tensor=reverse[TENSOR_NAMES.index(edit['tensor'])]
            index=tuple(edit['index'])
            if not _same(tensor[index],_number(edit['after'],'Edit after')):
                raise ValueError('Manifest after value does not match recorded tensor changes')
            if action=='applied':
                tensor[index]=_number(edit['before'],'Edit before')
    if weights_digest(reverse)!=manifest['original_weights_sha256']:
        raise ValueError('Manifest original digest does not match recorded tensor changes')
    # Replay from the reconstructed original, retaining a snapshot only while
    # later patches still refer to that historical base. A linear chain needs
    # no growing collection of dense 2,499-parameter snapshots.
    replay=reverse
    remaining=Counter(record['patch']['base_weights_sha256'] for record in manifest['patches'])
    snapshots={}
    for record in manifest['patches']:
        patch=record['patch']
        base_digest=patch['base_weights_sha256']
        input_digest=record['input_weights_sha256']
        if base_digest==input_digest:
            declared=replay
        elif base_digest in snapshots:
            declared=snapshots[base_digest]
        else:
            raise ValueError('Manifest patch declares an unrelated base state')
        actions=_actions(record)
        for edit,action in zip(patch['edits'],actions):
            value=declared[TENSOR_NAMES.index(edit['tensor'])][tuple(edit['index'])]
            if not _same(value,_number(edit['before'],'Edit before')):
                raise ValueError('Manifest patch before value does not match its declared base')
            current=replay[TENSOR_NAMES.index(edit['tensor'])][tuple(edit['index'])]
            expected_value=edit['before'] if action=='applied' else edit['after']
            if not _same(current,_number(expected_value,'Edit value')):
                raise ValueError('Manifest edit action does not match its input tensor state')
        remaining[base_digest]-=1
        if (record['input_weights_sha256']!=record['output_weights_sha256']
                and remaining[input_digest]>0 and input_digest not in snapshots):
            snapshots[input_digest]=tuple(array.copy() for array in replay)
        for edit,action in zip(patch['edits'],actions):
            if action=='applied':
                replay[TENSOR_NAMES.index(edit['tensor'])][tuple(edit['index'])]=_number(edit['after'],'Edit after')
        if weights_digest(replay)!=record['output_weights_sha256']:
            raise ValueError('Manifest output digest does not match recorded tensor changes')
        if remaining[base_digest]==0:
            snapshots.pop(base_digest,None)
    return manifest


def stack(base_path,patches,previous_manifest=None,*,schema_sha256=None):
    """Apply ordered, compatible edits in memory; return tensors and provenance.

    A new patch may name this run's base or the current state at its start. This
    supports both disjoint parallel edits and an explicit chain of revisions.
    A prior manifest preserves and verifies history across separate invocations.
    """
    expected_schema=_expected_schema(schema_sha256)
    current=_weights(base_path)
    run_base=tuple(array.copy() for array in current)
    run_base_digest=weights_digest(run_base)
    original_digest=run_base_digest
    seen={}
    records=[]
    skipped=[]
    claimed={}
    if previous_manifest is not None:
        previous=validate_manifest(previous_manifest,current,schema_sha256=expected_schema)
        original_digest=previous['original_weights_sha256']
        records=previous['patches']
        skipped=previous['duplicate_patch_ids_skipped']
        seen={record['patch']['id']:_canonical(record['patch']) for record in records}
    for value in patches:
        patch=validate_patch(_read_patch(value),schema_sha256=expected_schema)
        identity=_canonical(patch)
        patch_id=patch['id']
        if patch_id in seen:
            if seen[patch_id]!=identity:
                raise ValueError(f'Patch id {patch_id!r} is reused with different content')
            skipped.append(patch_id)
            continue
        start_digest=weights_digest(current)
        if patch['base_weights_sha256'] not in (run_base_digest,start_digest):
            raise ValueError(f'Patch {patch_id!r} was authored against an unrelated weight state')
        parallel=patch['base_weights_sha256']==run_base_digest
        declared=run_base if parallel else tuple(array.copy() for array in current)
        rules={}
        for edit in patch['edits']:
            index=tuple(edit['index'])
            before=_number(edit['before'],'Edit before')
            declared_value=declared[TENSOR_NAMES.index(edit['tensor'])][index]
            if not _same(declared_value,before):
                raise ValueError(f'Patch {patch_id!r} before value does not match its declared base: '
                                 f'{edit["tensor"]}{index}')
            neuron=_neuron(edit)
            if neuron is not None:
                rules.setdefault(neuron,set()).add((edit['tensor'],index,float(_number(edit['after'],'Edit after'))))
        # A neuron's weights are one rule. Entries from two parallel copies would combine
        # into a rule neither wrote, so a copy may change a neuron that an earlier patch
        # in this merge changed only with exactly the same edits.
        for neuron,rule in rules.items():
            owner,owned=claimed.setdefault(neuron,(patch_id,rule))
            if parallel and owner!=patch_id and owned!=rule:
                raise ValueError(f'Conflicting edit in patch {patch_id!r}: hidden neuron {neuron} '
                                 f'was already changed differently by patch {owner!r} in this merge')
        actions=[]
        for edit in patch['edits']:
            tensor=current[TENSOR_NAMES.index(edit['tensor'])]
            index=tuple(edit['index'])
            before=_number(edit['before'],'Edit before')
            after=_number(edit['after'],'Edit after')
            if _same(tensor[index],before):
                tensor[index]=after
                actions.append('applied')
            elif _same(tensor[index],after):
                actions.append('already_present')
            else:
                raise ValueError(f'Conflicting edit in patch {patch_id!r}: '
                                 f'{edit["tensor"]}{index} expects {float(before)}, '
                                 f'currently {float(tensor[index])}')
        validate_weights(current)
        records.append({'patch':patch,'input_weights_sha256':start_digest,
                        'output_weights_sha256':weights_digest(current),'edit_actions':actions})
        seen[patch_id]=identity
    manifest={'format':STACK_FORMAT,'schema_sha256':expected_schema,
              'original_weights_sha256':original_digest,
              'final_weights_sha256':weights_digest(current),
              'patches':records,'duplicate_patch_ids_skipped':skipped}
    return current,manifest


def _neuron(edit):
    """The hidden neuron an entry belongs to: w1 column, b1 index, or w2 row; b2 belongs to none."""
    if edit['tensor']=='w1':
        return edit['index'][1]
    return None if edit['tensor']=='b2' else edit['index'][0]


def _distinct_output(path,inputs):
    resolved=Path(path).resolve()
    if any(resolved==Path(value).resolve() for value in inputs):
        raise ValueError('Output must be a separate path from the input artifacts')


def _output_parent(path):
    path=Path(path)
    if not path.parent.is_dir() or not os.access(path.parent,os.W_OK|os.X_OK):
        raise ValueError(f'Output parent must exist and be writable: {path.parent}')
    if path.exists() and not path.is_file():
        raise ValueError(f'Output must be a file path: {path}')


def _save_stack(output,manifest_path,tensors,manifest_text):
    _output_parent(output)
    _output_parent(manifest_path)
    temporary=None
    backup=None
    replaced=False
    try:
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=manifest_path.parent,
                                         suffix='.json',delete=False) as handle:
            temporary=Path(handle.name)
            handle.write(manifest_text)
        if output.exists():
            with tempfile.NamedTemporaryFile(dir=output.parent,suffix='.backup',delete=False) as handle:
                backup=Path(handle.name)
            shutil.copy2(output,backup)
        # Stage the manifest before replacing the snapshot. Restore the previous
        # snapshot if the final manifest rename fails, keeping the pair coherent.
        save_weights(output,tensors)
        replaced=True
        temporary.replace(manifest_path)
    except BaseException:
        if replaced:
            if backup is None:
                output.unlink(missing_ok=True)
            else:
                backup.replace(output)
        raise
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        if backup is not None:
            backup.unlink(missing_ok=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    capture=commands.add_parser('diff',help='Capture directly edited tensor entries')
    capture.add_argument('base',type=Path)
    capture.add_argument('edited',type=Path)
    capture.add_argument('--harness',required=True)
    capture.add_argument('--evidence',required=True)
    capture.add_argument('--id',required=True,dest='patch_id')
    capture.add_argument('--output',required=True,type=Path)
    combine=commands.add_parser('stack',help='Combine compatible edits without averaging')
    combine.add_argument('base',type=Path)
    combine.add_argument('patches',nargs='+',type=Path)
    combine.add_argument('--output',required=True,type=Path)
    combine.add_argument('--manifest',required=True,type=Path)
    combine.add_argument('--resume-manifest',type=Path,help='Retain verified provenance from a previous stack')
    args=parser.parse_args(argv)
    try:
        if args.command=='diff':
            _distinct_output(args.output,[args.base,args.edited])
            patch=diff(args.base,args.edited,harness=args.harness,evidence=args.evidence,
                       patch_id=args.patch_id)
            args.output.write_text(json.dumps(patch,indent=2,allow_nan=False)+'\n',encoding='utf-8')
            print(json.dumps({'patch':str(args.output),'edits':len(patch['edits']),
                              'base_weights_sha256':patch['base_weights_sha256']}))
        else:
            inputs=[args.base,*args.patches]
            if args.resume_manifest is not None:
                inputs.append(args.resume_manifest)
            _distinct_output(args.output,inputs)
            _distinct_output(args.manifest,[*inputs,args.output])
            tensors,manifest=stack(args.base,args.patches,previous_manifest=args.resume_manifest)
            # No files are written until every patch, base digest, and before
            # value has passed validation. Conflicts preserve existing outputs.
            manifest_text=json.dumps(manifest,indent=2,allow_nan=False)+'\n'
            _save_stack(args.output,args.manifest,tensors,manifest_text)
            print(json.dumps({'weights':str(args.output),'manifest':str(args.manifest),
                              'weights_sha256':manifest['final_weights_sha256']}))
    except (ValueError,TypeError,OSError) as exc:
        print(f'Weight patch error: {exc}',file=sys.stderr)
        return 1
    return 0


if __name__=='__main__':
    sys.exit(main())
