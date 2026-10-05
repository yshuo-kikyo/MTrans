"""Synthetic CPU tests only; never load a real checkpoint or validation case."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import h5py
import numpy as np
import torch
from config import build_config
from models import build_model_from_name
from models.cmmt_reconstruction_multi_cross import CrossCMMT
from cross_modal_dependency_audit import AuditDataset,run_condition
from cross_modal_dependency_utils import VolumeAccumulator,volume_metrics,pair_catalog
from frozen_b0_pathway_audit import run_pathway,verify_forward_contract
from frozen_b0_mismatch_localization import CONDITIONS,run_four,run_localization,mapping_record
from relation_distortion_analysis import make_mask
from data import transforms as T


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(51)
        cfg=build_config("reconstruction_multi_cross").clone()
        cfg.INPUT_SIZE,cfg.MODEL.P1,cfg.MODEL.P2=16,2,4
        cfg.MODEL.HEAD_HIDDEN_DIM,cfg.MODEL.TRANSFORMER_NUM_HEADS=2,2
        self.model=build_model_from_name(cfg,"reconstruction_multi_cross").eval().requires_grad_(False)
        self.x,self.matched,self.donor=[torch.randn(1,1,16,16) for _ in range(3)]

    def clean(self):
        self.assertNotIn("forward",self.model.__dict__)
        self.assertIs(self.model.forward.__func__,CrossCMMT.forward)
        for layer in self.model.cross_transformer.layers:
            for direction in layer:
                self.assertFalse(direction.fn.attn.attn_drop._forward_pre_hooks)

    def test_three_reference_conditions_exact_and_new_combination(self):
        with torch.no_grad():
            results=run_four(self.model,self.x,self.matched,self.donor)
            reference={"normal_matched":self.model(self.x,self.matched)[0],
                "normal_mismatched":run_condition(self.model,self.x,self.matched,self.donor,"mismatched_aux"),
                "no_early_matched":run_pathway(self.model,self.x,self.matched,"no_early"),
                "no_early_mismatched":run_pathway(self.model,self.x,self.donor,"no_early")}
        for name in CONDITIONS:
            self.assertTrue(torch.equal(results[name],reference[name]))
        self.clean()

    def test_same_donor_tensor_both_mismatch_conditions_no_cross_hooks(self):
        inputs=[]
        def record(module,args):
            inputs.append(args[0])
        handle=self.model.head2.register_forward_pre_hook(record)
        try:
            with torch.no_grad(),patch("cross_modal_dependency_audit.zero_cross",side_effect=AssertionError("NoCross forbidden")):
                run_four(self.model,self.x,self.matched,self.donor)
            self.assertIs(inputs[1],self.donor)
            self.assertIs(inputs[3],self.donor)
            self.assertIs(inputs[0],self.matched)
            self.assertIs(inputs[2],self.matched)
        finally:
            handle.remove()
        self.clean()

    def test_noearly_ast_complement_and_transformer_execute(self):
        self.assertTrue(verify_forward_contract())
        calls=[]
        handles=[]
        for name in ("head2","complement_patch_embbeding","cross_transformer","tail2"):
            def record(m,args,out,name=name):
                calls.append(name)
            handles.append(getattr(self.model,name).register_forward_hook(record))
        try:
            with torch.no_grad():
                run_localization(self.model,self.x,self.matched,self.donor,"no_early_mismatched")
            self.assertEqual(calls,["head2","complement_patch_embbeding","cross_transformer","tail2"])
        finally:
            for handle in handles:
                handle.remove()

    def test_parameter_buffer_input_invariance_and_restoration(self):
        state={k:v.clone() for k,v in self.model.state_dict().items()}
        inputs=[v.clone() for v in (self.x,self.matched,self.donor)]
        with torch.no_grad():
            original=self.model(self.x,self.matched)[0]
            run_four(self.model,self.x,self.matched,self.donor)
            self.assertTrue(torch.equal(self.model(self.x,self.matched)[0],original))
        for k,v in self.model.state_dict().items():
            self.assertTrue(torch.equal(v,state[k]))
        for a,b in zip(inputs,(self.x,self.matched,self.donor)):
            self.assertTrue(torch.equal(a,b))
        self.clean()

    def test_exception_restores_noearly_for_both_identities(self):
        with torch.no_grad():
            original=self.model(self.x,self.matched)[0]
            for condition in ("no_early_matched","no_early_mismatched"):
                with patch.object(self.model.tail1,"forward",side_effect=RuntimeError("test")):
                    with self.assertRaises(RuntimeError):
                        run_localization(self.model,self.x,self.matched,self.donor,condition)
                self.clean()
                self.assertTrue(torch.equal(self.model(self.x,self.matched)[0],original))

    def test_batch_partition_numeric_equivalence(self):
        x=torch.cat([self.x,self.x*.8])
        matched=torch.cat([self.matched,self.matched*.7])
        donor=torch.cat([self.donor,self.donor*.6])
        with torch.no_grad():
            whole=run_four(self.model,x,matched,donor)
            parts=[run_four(self.model,x[i:i+1],matched[i:i+1],donor[i:i+1]) for i in range(2)]
        for condition in CONDITIONS:
            torch.testing.assert_close(whole[condition],torch.cat([r[condition] for r in parts]),atol=1e-6,rtol=1e-4)


class DataVolumeTests(unittest.TestCase):
    def test_donor_identity_slice_normalization_exact_dependency_reuse(self):
        # The new audit exports the ORIGINAL class, not an independently implemented subclass.
        from frozen_b0_mismatch_localization import AuditDataset as ReusedDataset
        self.assertIs(ReusedDataset,AuditDataset)
        rng=np.random.RandomState(6)
        k=(rng.randn(3,16,16)+1j*rng.randn(3,16,16)).astype(np.complex64)
        gt=np.abs(rng.randn(3,16,16)).astype(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            original=str(Path(directory)/"pd0.h5")
            donor=str(Path(directory)/"pd1.h5")
            with h5py.File(donor,"w") as file:
                file["kspace"],file["reconstruction_esc"]=k,gt
            examples=[(original,"target0.h5",s,{}, {},0) for s in range(2)]
            examples += [(donor,"target1.h5",s,{}, {},1) for s in range(3)]
            dataset=AuditDataset.__new__(AuditDataset)
            from types import SimpleNamespace
            dataset.raw=SimpleNamespace(examples=examples)
            dataset.pairs=pair_catalog(examples)
            dataset.pd_counts={original:2,donor:3}
            dataset.native=T.ReconstructionTransform("singlecoil",make_mask(4,.08),use_seed=True)
            mapping=dataset.mapping(1)
            sample=dataset.donor_sample(mapping)
            expected=dataset.native(k[2],None,gt[2],{},donor,2)
            record=mapping_record(dataset,1,sample[2],sample[3])
            self.assertEqual(record["mismatched_pd_slice"],2)
            self.assertEqual(record["mismatched_pd_fname"],donor)
            for key,value in mapping.items():
                self.assertEqual(record[key],value)
            for i in (1,2,3):
                self.assertTrue(torch.equal(sample[i],expected[i]))
            self.assertEqual(record["mismatched_pd_normalization_mean"],float(expected[2]))
            self.assertEqual(record["mismatched_pd_normalization_std"],float(expected[3]))

    def test_volume_aggregation_partial_and_batch_invariance(self):
        gt=np.arange(128,dtype=np.float32).reshape(2,8,8)+1
        predictions={c:gt*(.9-i*.01) for i,c in enumerate(CONDITIONS)}
        results=[]
        for batch_size in (1,2):
            accumulator=VolumeAccumulator({"v":2},{"v":3},conditions=CONDITIONS)
            for start in range(0,2,batch_size):
                for i in range(start,min(2,start+batch_size)):
                    rows=accumulator.add("v",i,gt[i],{c:v[i] for c,v in predictions.items()})
            results.append(rows)
            for row in rows:
                self.assertTrue(row["partial_volume"])
                for metric,value in volume_metrics(gt,predictions[row["condition"]]).items():
                    self.assertEqual(row[metric],value)
        self.assertEqual(results[0],results[1])


if __name__ == "__main__":
    unittest.main()
