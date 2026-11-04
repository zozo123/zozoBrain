"""Independent data-only transport, authoring, integrity, and writer checks."""
import copy
import io
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile

import numpy as np

import brain_bundle
from brain_bundle import (
    create_blank, edit_bundle, load_bundle, pack_bundle, save_bundle,
)
from model import DecisionModel
from network import TENSOR_SHAPES, layout_digest, weights_digest


ROOT=Path(__file__).resolve().parent


class BrainBundleTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder=Path(temporary.name)
        self.blank=create_blank('Portable test brain')
        self.path=self.folder/'blank.brain'
        save_bundle(self.path,self.blank)

    def members(self):
        with zipfile.ZipFile(self.path) as archive:
            return {name:archive.read(name) for name in archive.namelist()}

    def archive(self,members,name='changed.brain'):
        path=self.folder/name
        entries=members.items() if isinstance(members,dict) else members
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',UserWarning)
            with zipfile.ZipFile(path,'w') as archive:
                for member,data in entries:
                    archive.writestr(member,data)
        return path

    def decide(self,bundle):
        return DecisionModel(bundle.weights,bundle.manifest,layout=bundle.layout).decide(
            'Which option?',['general'],[
                {'id':'lower','attributes':{'low_cost':1}},
                {'id':'higher','attributes':{'high_cost':1}},
            ])

    def author(self,bundle,tensor,index,values,harness,identifier):
        return edit_bundle(bundle,tensor=tensor,index=index,values=values,
                           harness=harness,patch_id=identifier,
                           evidence=f'Synthetic explicit instruction from {harness}: set {tensor}{index} to {values}.')

    def test_roundtrip_contains_only_standard_numeric_tensors_layout_and_history(self):
        restored=load_bundle(self.path)
        self.assertEqual(restored.layout,self.blank.layout)
        self.assertEqual(restored.manifest,self.blank.manifest)
        self.assertEqual(restored.weights_sha256,weights_digest(self.blank.weights))
        self.assertEqual(restored.schema_sha256,layout_digest(**self.blank.layout))
        self.assertEqual(restored.revision_sha256,self.blank.revision_sha256)
        self.assertEqual(sum(array.size for array in restored.weights),2499)
        with zipfile.ZipFile(self.path) as archive:
            self.assertEqual(set(archive.namelist()),{'brain.json','w1.npy','b1.npy','w2.npy','b2.npy'})
            metadata=json.loads(archive.read('brain.json'))
            self.assertEqual(metadata['schema_sha256'],restored.schema_sha256)
            self.assertEqual(metadata['weights_sha256'],restored.weights_sha256)
            for name,expected in zip(TENSOR_SHAPES,self.blank.weights):
                tensor=np.load(io.BytesIO(archive.read(name+'.npy')),allow_pickle=False)
                self.assertEqual(tensor.dtype,np.dtype('<f4'))
                self.assertTrue(tensor.flags.c_contiguous)
                self.assertEqual(tensor.tobytes(),expected.astype('<f4').tobytes())
        self.assertIsNone(self.decide(restored)['decision'])

    def test_blank_import_does_not_execute_nearby_weight_source(self):
        runtime=self.folder/'runtime'
        runtime.mkdir()
        for filename in ('model.py','network.py','weight_patches.py','brain_bundle.py'):
            shutil.copy2(ROOT/filename,runtime/filename)
        (runtime/'weights.py').write_text(
            "from pathlib import Path\nPath(__file__).with_name('SOURCE_EXECUTED').write_text('bad')\n"
            "raise RuntimeError('A data-only import executed weights.py')\n",encoding='utf-8')
        script=(
            'import json,sys; from model import DecisionModel; '
            "result=DecisionModel.from_bundle(sys.argv[1]).decide('Which?',['general'],"
            "[{'id':'A','attributes':{'low_cost':1}},{'id':'B','attributes':{'high_cost':1}}]); "
            'print(json.dumps(result))'
        )
        proc=subprocess.run([sys.executable,'-c',script,str(self.path)],cwd=runtime,
                            capture_output=True,text=True,timeout=10)
        self.assertEqual(proc.returncode,0,proc.stderr)
        self.assertIsNone(json.loads(proc.stdout)['decision'])
        self.assertFalse((runtime/'SOURCE_EXECUTED').exists())

    def test_identical_bundle_saves_have_identical_bytes(self):
        other=self.folder/'same.brain'
        save_bundle(other,self.blank)
        self.assertEqual(self.path.read_bytes(),other.read_bytes())
        original=self.path.read_bytes()
        with self.assertRaises(FileExistsError):
            save_bundle(self.path,self.blank)
        self.assertEqual(self.path.read_bytes(),original)

    def test_row_and_scalar_edits_change_actual_choices_and_keep_caller_evidence(self):
        first=self.author(self.blank,'w2',[0],[3,0,1],'codex','codex-001')
        self.assertIsNone(self.decide(first)['decision'])
        second=self.author(first,'w2',[1],[-3,0,1],'codex','codex-002')
        self.assertEqual(self.decide(second)['decision'],'lower')
        third=self.author(second,'w2',[0,0],[-6],'claude','claude-001')
        self.assertEqual(self.decide(third)['decision'],'higher')
        self.assertEqual(third.contributing_harnesses,['codex','claude'])
        self.assertEqual(self.blank.weights[2][0,0],0)
        edited_path=self.folder/'edited.brain'
        save_bundle(edited_path,third)
        restored=load_bundle(edited_path)
        result=DecisionModel.from_bundle(edited_path).decide('Which?',['general'],[
            {'id':'lower','attributes':{'low_cost':1}},
            {'id':'higher','attributes':{'high_cost':1}},
        ])
        self.assertEqual(result['decision'],'higher')
        self.assertEqual(result['contributing_harnesses'],['codex','claude'])
        self.assertEqual(restored.manifest,third.manifest)
        self.assertIn('Synthetic explicit instruction from claude',
                      restored.manifest['patches'][-1]['patch']['source']['evidence'])
        self.assertNotEqual(restored.revision_sha256,self.blank.revision_sha256)

    def test_corrupt_unknown_duplicate_and_missing_archive_members_are_rejected(self):
        corrupt=self.folder/'corrupt.brain'
        corrupt.write_bytes(b'PK\x03\x04')
        with self.assertRaises(ValueError):
            load_bundle(corrupt)
        members=self.members()
        cases={
            'unknown':{**members,'weights.py':b'raise RuntimeError()'},
            'missing':{name:data for name,data in members.items() if name!='b2.npy'},
            'duplicate':[*members.items(),('brain.json',members['brain.json'])],
        }
        for label,content in cases.items():
            with self.subTest(case=label),self.assertRaises(ValueError):
                load_bundle(self.archive(content,label+'.brain'))

    def test_invalid_metadata_schema_fingerprints_and_provenance_are_rejected(self):
        members=self.members()
        original=json.loads(members['brain.json'])
        mutations={
            'format':lambda m:m.update(format='zozobrain.brain.v999'),
            'extra':lambda m:m.update(unexpected=True),
            'schema':lambda m:m.update(schema_sha256='0'*64),
            'weights':lambda m:m.update(weights_sha256='0'*64),
            'layout':lambda m:m['layout']['attributes'].__setitem__(1,m['layout']['attributes'][0]),
            'binary':lambda m:m['layout'].update(binary_attributes=['unknown']),
            'history-final':lambda m:m['manifest'].update(final_weights_sha256='0'*64),
            'history-schema':lambda m:m['manifest'].update(schema_sha256='0'*64),
        }
        for label,mutate in mutations.items():
            with self.subTest(case=label):
                metadata=copy.deepcopy(original)
                mutate(metadata)
                bad={**members,'brain.json':json.dumps(metadata).encode('utf-8')}
                with self.assertRaises(ValueError):
                    load_bundle(self.archive(bad,label+'.brain'))
        duplicate=b'{"format":"duplicate",'+members['brain.json'][1:]
        with self.assertRaisesRegex(ValueError,'Duplicate JSON field'):
            load_bundle(self.archive({**members,'brain.json':duplicate},'duplicate-json.brain'))
        nonfinite=copy.deepcopy(original)
        nonfinite['name']=float('nan')
        with self.assertRaisesRegex(ValueError,'Nonfinite JSON number'):
            load_bundle(self.archive({**members,'brain.json':json.dumps(nonfinite).encode()},'nan-json.brain'))

    def test_changed_tensor_payload_cannot_keep_old_hash_or_history(self):
        members=self.members()
        tensor=self.blank.weights[2].copy()
        tensor[0,0]=1
        data=io.BytesIO()
        np.save(data,tensor,allow_pickle=False)
        with self.assertRaises(ValueError):
            load_bundle(self.archive({**members,'w2.npy':data.getvalue()},'changed-tensor.brain'))
        tensor[0,0]=np.nan
        data=io.BytesIO()
        np.save(data,tensor,allow_pickle=False)
        with self.assertRaisesRegex(ValueError,'finite float32'):
            load_bundle(self.archive({**members,'w2.npy':data.getvalue()},'nan-tensor.brain'))

    def test_crafted_npy_shapes_and_headers_fail_before_tensor_allocation(self):
        members=self.members()
        headers=[]
        for shape,descr,fortran in [((2**62,48),'<f4',False),((48,48),'|O',False),
                                    ((48,48),'>f4',False),((48,48),'<f4',True)]:
            stream=io.BytesIO()
            np.lib.format.write_array_header_1_0(stream,{'shape':shape,'descr':descr,'fortran_order':fortran})
            headers.append(stream.getvalue())
        huge_header=repr({'shape':(48,48),'descr':'<f4','fortran_order':False}).encode()+b' '*2048+b'\n'
        headers.append(np.lib.format.magic(1,0)+struct.pack('<H',len(huge_header))+huge_header)
        for number,header in enumerate(headers):
            path=self.archive({**members,'w1.npy':header},f'header-{number}.brain')
            with self.subTest(header=number), \
                 patch.object(brain_bundle.np,'frombuffer',side_effect=AssertionError('Header allocated a tensor')), \
                 patch.object(brain_bundle.np,'empty',side_effect=AssertionError('Header allocated a tensor')), \
                 self.assertRaises(ValueError):
                load_bundle(path)
        for raw in (members['w1.npy'][:-1],members['w1.npy']+b'\0'):
            with self.assertRaisesRegex(ValueError,'tensor byte count'):
                load_bundle(self.archive({**members,'w1.npy':raw},'bad-length.brain'))

    def test_member_size_limits_apply_before_reading_payloads(self):
        with patch.object(brain_bundle,'MAX_METADATA_BYTES',1),self.assertRaisesRegex(ValueError,'oversized'):
            load_bundle(self.path)
        with patch.object(brain_bundle,'MAX_TENSOR_BYTES',1),self.assertRaisesRegex(ValueError,'oversized'):
            load_bundle(self.path)

    def test_stale_revision_and_disappeared_destination_do_not_overwrite(self):
        first=load_bundle(self.path)
        stale=load_bundle(self.path)
        codex=self.author(first,'w2',[0],[3,0,1],'codex','codex-001')
        save_bundle(self.path,codex,overwrite=True,expected_revision=first.revision_sha256)
        current=self.path.read_bytes()
        claude=self.author(stale,'w2',[1],[-3,0,1],'claude','claude-001')
        with self.assertRaisesRegex(ValueError,'changed since it was loaded'):
            save_bundle(self.path,claude,overwrite=True,expected_revision=stale.revision_sha256)
        self.assertEqual(self.path.read_bytes(),current)
        absent=self.folder/'absent.brain'
        with self.assertRaisesRegex(ValueError,'disappeared'):
            save_bundle(absent,claude,overwrite=True,expected_revision=stale.revision_sha256)
        self.assertFalse(absent.exists())

    def test_other_process_cannot_write_during_lock_and_can_write_after_release(self):
        script=(
            'import sys; from brain_bundle import load_bundle,save_bundle; '
            'path=sys.argv[1]; bundle=load_bundle(path); '
            'save_bundle(path,bundle,overwrite=True,expected_revision=bundle.revision_sha256)'
        )
        before=self.path.read_bytes()
        with brain_bundle._write_lock(self.path):
            proc=subprocess.run([sys.executable,'-c',script,str(self.path)],cwd=ROOT,
                                capture_output=True,text=True,timeout=10)
            self.assertNotEqual(proc.returncode,0)
            self.assertIn('another process',proc.stderr)
            self.assertEqual(self.path.read_bytes(),before)
        proc=subprocess.run([sys.executable,'-c',script,str(self.path)],cwd=ROOT,
                            capture_output=True,text=True,timeout=10)
        self.assertEqual(proc.returncode,0,proc.stderr)
        self.assertEqual(self.path.read_bytes(),before)


if __name__=='__main__':
    unittest.main()
