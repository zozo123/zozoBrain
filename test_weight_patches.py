import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch as mock_patch

import numpy as np

from model import DecisionModel
from network import load_weights, save_weights, weights_digest
import weight_patches
from weight_patches import diff, main, schema_digest, stack, validate_manifest, validate_patch
from weights import authored_weights


class WeightPatchTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder=Path(temporary.name)
        self.base=self.folder/'base.npz'
        self.tensors=authored_weights()
        save_weights(self.base,self.tensors)

    def edited(self,tensor,index,value,source=None):
        tensors=tuple(array.copy() for array in (self.tensors if source is None else source))
        tensors[tensor][index]=value
        return tensors

    def patch(self,tensors,patch_id,harness='codex',base=None):
        edited=self.folder/f'{patch_id}-edited.npz'
        save_weights(edited,tensors)
        return diff(self.base if base is None else base,edited,harness=harness,
                    evidence='Explicit test instruction to author these numerical weights.',
                    patch_id=patch_id)

    def test_independent_harness_edits_combine_and_change_actual_decision(self):
        codex_edited=self.folder/'codex-edited.npz'
        claude_edited=self.folder/'claude-edited.npz'
        save_weights(codex_edited,self.edited(2,(0,0),6))
        save_weights(claude_edited,self.edited(2,(1,0),2))
        codex_patch=self.folder/'codex.json'
        claude_patch=self.folder/'claude.json'
        combined=self.folder/'combined.npz'
        manifest_path=self.folder/'combined.json'
        with contextlib.redirect_stdout(io.StringIO()):
            for edited,harness,output in [(codex_edited,'codex',codex_patch),(claude_edited,'claude',claude_patch)]:
                self.assertEqual(main(['diff',str(self.base),str(edited),'--harness',harness,
                                       '--evidence','Explicit preference instruction.',
                                       '--id',harness+'-001','--output',str(output)]),0)
            self.assertEqual(main(['stack',str(self.base),str(codex_patch),str(claude_patch),
                                   '--output',str(combined),'--manifest',str(manifest_path)]),0)
        options=[{'id':'concise','attributes':{'concise':1}},
                 {'id':'verified','attributes':{'verified':1}}]
        self.assertEqual(DecisionModel().decide('Which?',['communication'],options)['decision'],'verified')
        self.assertEqual(DecisionModel(combined).decide('Which?',['communication'],options)['decision'],'concise')
        result=load_weights(combined)
        self.assertEqual(result[2][0,0],6)
        self.assertEqual(result[2][1,0],2)
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        self.assertEqual(manifest['original_weights_sha256'],weights_digest(self.tensors))
        self.assertEqual(manifest['final_weights_sha256'],weights_digest(result))
        self.assertEqual([record['patch']['source']['harness'] for record in manifest['patches']],['codex','claude'])
        self.assertEqual(manifest['patches'][0]['patch'],json.loads(codex_patch.read_text(encoding='utf-8')))

    def test_conflict_preserves_existing_outputs_and_base(self):
        codex=self.patch(self.edited(2,(0,0),6),'codex-001')
        claude=self.patch(self.edited(2,(0,0),2),'claude-001','claude')
        paths=[]
        for patch in (codex,claude):
            path=self.folder/(patch['id']+'.json')
            path.write_text(json.dumps(patch),encoding='utf-8')
            paths.append(path)
        output=self.folder/'combined.npz'
        manifest=self.folder/'combined.json'
        output.write_bytes(b'existing weights must survive')
        manifest.write_bytes(b'existing manifest must survive')
        before={path:path.read_bytes() for path in (self.base,output,manifest)}
        errors=io.StringIO()
        with contextlib.redirect_stderr(errors):
            status=main(['stack',str(self.base),*[str(path) for path in paths],
                         '--output',str(output),'--manifest',str(manifest)])
        self.assertEqual(status,1)
        self.assertIn('Conflicting edit',errors.getvalue())
        self.assertEqual(before,{path:path.read_bytes() for path in before})

    def test_sequential_revision_uses_current_state_as_base(self):
        first_tensors=self.edited(2,(0,0),6)
        first=self.patch(first_tensors,'codex-001')
        current=self.folder/'current.npz'
        save_weights(current,first_tensors)
        second=self.patch(self.edited(2,(0,0),2,first_tensors),'claude-002','claude',base=current)
        result,manifest=stack(self.base,[first,second])
        self.assertEqual(result[2][0,0],2)
        self.assertEqual(manifest['patches'][0]['output_weights_sha256'],
                         manifest['patches'][1]['input_weights_sha256'])
        second['base_weights_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'unrelated weight state'):
            stack(self.base,[first,second])

    def test_patch_before_must_match_its_declared_base(self):
        first_tensors=self.edited(2,(0,0),6)
        first=self.patch(first_tensors,'codex-001')
        current=self.folder/'current.npz'
        save_weights(current,first_tensors)
        second=self.patch(self.edited(2,(0,0),2,first_tensors),'claude-002','claude',base=current)
        # A value from the current state cannot be attributed to the old base.
        second['base_weights_sha256']=weights_digest(self.tensors)
        with self.assertRaisesRegex(ValueError,'before value does not match its declared base'):
            stack(self.base,[first,second])

    def test_agreeing_independent_overlap_retains_both_sources(self):
        first=self.patch(self.edited(2,(0,0),6),'codex-001')
        second=self.patch(self.edited(2,(0,0),6),'claude-001','claude')
        tensors,manifest=stack(self.base,[first,second])
        self.assertEqual(tensors[2][0,0],6)
        self.assertEqual(len(manifest['patches']),2)
        self.assertEqual(manifest['patches'][1]['edit_actions'],['already_present'])
        self.assertEqual(manifest['patches'][1]['input_weights_sha256'],
                         manifest['patches'][1]['output_weights_sha256'])
        self.assertEqual(validate_manifest(manifest,tensors),manifest)

    def test_resume_across_cli_invocations_keeps_complete_provenance(self):
        first=self.patch(self.edited(2,(0,0),6),'codex-001')
        first_patch=self.folder/'codex.json'
        first_patch.write_text(json.dumps(first),encoding='utf-8')
        first_output=self.folder/'first.npz'
        first_manifest=self.folder/'first.json'
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['stack',str(self.base),str(first_patch),
                                   '--output',str(first_output),'--manifest',str(first_manifest)]),0)
        first_tensors=load_weights(first_output)
        second=self.patch(self.edited(2,(1,0),2,first_tensors),'claude-002','claude',base=first_output)
        second_patch=self.folder/'claude.json'
        second_patch.write_text(json.dumps(second),encoding='utf-8')
        output=self.folder/'combined.npz'
        manifest_path=self.folder/'combined.json'
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['stack',str(first_output),str(second_patch),
                                   '--resume-manifest',str(first_manifest),
                                   '--output',str(output),'--manifest',str(manifest_path)]),0)
        combined=load_weights(output)
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        self.assertEqual([record['patch']['source']['harness'] for record in manifest['patches']],['codex','claude'])
        self.assertEqual(manifest['original_weights_sha256'],weights_digest(self.tensors))
        self.assertEqual(manifest['final_weights_sha256'],weights_digest(combined))
        self.assertEqual(validate_manifest(manifest,combined),manifest)
        result=DecisionModel(output,manifest_path).decide('Which?',['communication'],[
            {'id':'concise','attributes':{'concise':1}},
            {'id':'verified','attributes':{'verified':1}},
        ])
        self.assertEqual(result['decision'],'concise')
        self.assertEqual(result['contributing_harnesses'],['codex','claude'])
        self.assertEqual(result['schema_sha256'],manifest['schema_sha256'])
        with self.assertRaisesRegex(ValueError,'does not match the supplied base'):
            DecisionModel(output,first_manifest)
        unchanged,retained=stack(output,[first],previous_manifest=manifest)
        self.assertEqual(weights_digest(unchanged),weights_digest(combined))
        self.assertEqual(len(retained['patches']),2)
        self.assertEqual(retained['duplicate_patch_ids_skipped'],['codex-001'])
        stale=self.patch(self.edited(2,(2,0),-6),'stale-003','claude')
        with self.assertRaisesRegex(ValueError,'unrelated weight state'):
            stack(output,[stale],previous_manifest=manifest)

    def test_mismatched_resume_manifest_preserves_output_files(self):
        first=self.patch(self.edited(2,(0,0),6),'codex-001')
        tensors,manifest=stack(self.base,[first])
        current=self.folder/'current.npz'
        save_weights(current,tensors)
        next_patch=self.patch(self.edited(2,(1,0),2,tensors),'claude-002','claude',base=current)
        next_path=self.folder/'next.json'
        next_path.write_text(json.dumps(next_patch),encoding='utf-8')
        output=self.folder/'output.npz'
        output_manifest=self.folder/'output.json'
        output.write_bytes(b'old output')
        output_manifest.write_bytes(b'old provenance')
        for field in ('final_weights_sha256','schema_sha256'):
            with self.subTest(field=field):
                invalid=copy.deepcopy(manifest)
                invalid[field]='0'*64
                previous=self.folder/'invalid.json'
                previous.write_text(json.dumps(invalid),encoding='utf-8')
                with contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(main(['stack',str(current),str(next_path),
                                           '--resume-manifest',str(previous),
                                           '--output',str(output),'--manifest',str(output_manifest)]),1)
                self.assertEqual(output.read_bytes(),b'old output')
                self.assertEqual(output_manifest.read_bytes(),b'old provenance')
        tampered=copy.deepcopy(manifest)
        tampered['patches'][0]['patch']['edits'][0]['after']=7
        with self.assertRaisesRegex(ValueError,'after value'):
            validate_manifest(tampered,tensors)

    def test_manifest_parent_failure_does_not_replace_weights(self):
        patch=self.patch(self.edited(2,(0,0),6),'codex-001')
        patch_path=self.folder/'codex.json'
        patch_path.write_text(json.dumps(patch),encoding='utf-8')
        output=self.folder/'output.npz'
        output.write_bytes(b'old output')
        manifest=self.folder/'missing'/'manifest.json'
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(['stack',str(self.base),str(patch_path),
                                   '--output',str(output),'--manifest',str(manifest)]),1)
        self.assertEqual(output.read_bytes(),b'old output')
        self.assertFalse(manifest.exists())

    def test_manifest_replace_failure_restores_outputs_or_removes_new_snapshot(self):
        for existing_output in (True,False):
            with self.subTest(existing_output=existing_output):
                output=self.folder/'combined.npz'
                manifest=self.folder/'combined.json'
                original_weights=b'previous complete weights snapshot'
                original_manifest=b'previous complete manifest'
                output.unlink(missing_ok=True)
                if existing_output:
                    output.write_bytes(original_weights)
                manifest.write_bytes(original_manifest)
                replace=Path.replace

                def fail_manifest_rename(source,target):
                    if Path(target)==manifest:
                        raise PermissionError('Manifest destination rename failed')
                    return replace(source,target)

                with mock_patch.object(Path,'replace',fail_manifest_rename):
                    with self.assertRaises(OSError):
                        weight_patches._save_stack(output,manifest,self.tensors,'new provenance')
                if existing_output:
                    self.assertEqual(output.read_bytes(),original_weights)
                else:
                    self.assertFalse(output.exists())
                self.assertEqual(manifest.read_bytes(),original_manifest)

    def test_schema_reads_rapid_same_size_source_edits(self):
        source=self.folder/'weights.py'
        source.write_text("ATTRIBUTES=['concise']; CONTEXTS=['general']; BINARY_ATTRIBUTES=[]\n",encoding='utf-8')
        stamp=source.stat()
        with mock_patch.object(weight_patches,'__file__',str(self.folder/'weight_patches.py')):
            before=schema_digest()
            source.write_text(source.read_text(encoding='utf-8').replace('concise','compact'),encoding='utf-8')
            os.utime(source,ns=(stamp.st_atime_ns,stamp.st_mtime_ns))
            self.assertEqual(source.stat().st_size,stamp.st_size)
            self.assertNotEqual(before,schema_digest())

    def test_schema_mismatch_is_rejected(self):
        patch=self.patch(self.edited(2,(0,0),6),'codex-001')
        self.assertEqual(patch['schema_sha256'],schema_digest())
        patch['schema_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'schema'):
            stack(self.base,[patch])

    def test_exact_duplicate_is_idempotent_but_reused_id_is_rejected(self):
        patch=self.patch(self.edited(2,(0,0),6),'codex-001')
        tensors,manifest=stack(self.base,[patch,copy.deepcopy(patch)])
        self.assertEqual(tensors[2][0,0],6)
        self.assertEqual(len(manifest['patches']),1)
        self.assertEqual(manifest['duplicate_patch_ids_skipped'],['codex-001'])
        other=copy.deepcopy(patch)
        other['source']['evidence']='A different claimed instruction.'
        with self.assertRaisesRegex(ValueError,'reused with different content'):
            stack(self.base,[patch,other])

    def test_invalid_edit_values_and_coordinates_are_rejected(self):
        original=self.patch(self.edited(2,(0,0),6),'codex-001')
        for changes in [{'index':[True,0]},{'index':[-1,0]},{'index':[48,0]},
                        {'index':[0]},{'before':float('nan')},{'after':float('inf')},
                        {'after':1e100},{'after':True},{'reason':' '}]:
            with self.subTest(changes=changes):
                patch=copy.deepcopy(original)
                patch['edits'][0].update(changes)
                with self.assertRaises(ValueError):
                    validate_patch(patch)
        repeated=copy.deepcopy(original)
        repeated['edits'].append(copy.deepcopy(repeated['edits'][0]))
        with self.assertRaisesRegex(ValueError,'repeats tensor entry'):
            validate_patch(repeated)

    def test_signed_zero_edit_preserves_exact_tensor_fingerprint(self):
        edited=self.edited(3,(0,),np.float32(-0.0))
        patch=self.patch(edited,'zero-001')
        self.assertEqual(len(patch['edits']),1)
        tensors,manifest=stack(self.base,[patch])
        self.assertEqual(weights_digest(tensors),weights_digest(edited))
        self.assertEqual(manifest['final_weights_sha256'],weights_digest(edited))


if __name__=='__main__':
    unittest.main()
