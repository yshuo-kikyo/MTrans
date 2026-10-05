"""Dependency-free math, regression and dispatch-contract tests."""
import argparse
import ast
from collections import Counter
import math
from pathlib import Path
import unittest
from cross_modal_dependency_utils import paired_effects, descriptive

SOURCE=Path(__file__).resolve().parents[1]/"frozen_b0_mismatch_localization.py"


def functions():
    tree=ast.parse(SOURCE.read_text(encoding="utf-8"))
    names={"damage_localization","quick_regression","summarize","parse_args","run_four","run_localization","mapping_record"}
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names or
           isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ("CONDITIONS","REGRESSION_PSNR","REGRESSION_ATOL") for t in n.targets)]
    def require(ok,message):
        if not ok:
            raise ValueError(message)
    ns=dict(math=math,paired_effects=paired_effects,descriptive=descriptive,Path=Path,Counter=Counter,
            json=__import__("json"),argparse=argparse,__doc__="test",require=require)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE),"exec"),ns)
    return ns


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.f=functions()

    def test_damage_within_early_state_and_localization_direction(self):
        values={c:dict(PSNR=p,SSIM=p/100,NMSE=(40-p)/100) for c,p in
                (("normal_matched",30),("normal_mismatched",28),("no_early_matched",25),("no_early_mismatched",24))}
        result=self.f["damage_localization"](values)
        for row in result:
            self.assertGreater(row["damage_on"],0)
            self.assertGreater(row["damage_off"],0)
            self.assertAlmostEqual(row["reduction_fraction"],.5)
            self.assertAlmostEqual(row["localization"],row["damage_on"]-row["damage_off"])
        self.assertEqual(result[0]["damage_off"],1) # NOT normal_matched - no_early_mismatched (6).

    def test_equal_and_enhanced_mismatch_damage(self):
        values={c:dict(PSNR=p,SSIM=p,NMSE=-p) for c,p in
                (("normal_matched",30),("normal_mismatched",28),("no_early_matched",25),("no_early_mismatched",23))}
        for row in self.f["damage_localization"](values):
            self.assertEqual(row["localization"],0)
        values["no_early_mismatched"]=dict(PSNR=22,SSIM=22,NMSE=-22)
        for row in self.f["damage_localization"](values):
            self.assertLess(row["localization"],0)

    def test_zero_negative_nonfinite_denominator_no_epsilon(self):
        for mismatch in (30,31,float("inf")):
            values={c:dict(PSNR=30,SSIM=30,NMSE=-30) for c in self.f["CONDITIONS"]}
            values["normal_mismatched"]=dict(PSNR=mismatch,SSIM=mismatch,NMSE=-mismatch)
            for row in self.f["damage_localization"](values):
                self.assertIsNone(row["reduction_fraction"])
                self.assertTrue(row["fraction_status"].startswith("undefined_"))

    def test_identical_donor_object_and_only_normal_noearly_dispatch(self):
        calls=[]
        def dependency(model,target,matched,donor,condition):
            calls.append((target,donor if condition=="mismatched_aux" else matched,condition))
            return len(calls)
        def pathway(model,target,auxiliary,condition):
            self.assertEqual(condition,"no_early")
            calls.append((target,auxiliary,condition))
            return len(calls)
        self.f.update(run_condition=dependency,run_pathway=pathway)
        target,matched,donor=object(),object(),object()
        out=self.f["run_four"](object(),target,matched,donor)
        self.assertEqual(tuple(out),self.f["CONDITIONS"])
        self.assertTrue(all(c[0] is target for c in calls))
        self.assertIs(calls[1][1],donor)
        self.assertIs(calls[3][1],donor)
        self.assertIs(calls[0][1],matched)
        self.assertIs(calls[2][1],matched)

    def test_quick_regression_pass_fail_and_inapplicable(self):
        sample=[dict(fname="/data/file1002538.h5",slice_num=0)]
        rows=[dict(condition=c,PSNR=v) for c,v in self.f["REGRESSION_PSNR"].items()]
        self.assertEqual(self.f["quick_regression"](sample,rows)[0],"passed")
        rows[0]["PSNR"]+=.01
        self.assertEqual(self.f["quick_regression"](sample,rows)[0],"failed")
        self.assertEqual(self.f["quick_regression"](sample*2,rows)[0],"not_applicable_to_this_selection")

    def test_mapping_direct_reuse_preserves_fields_and_stats(self):
        class Dataset:
            def mapping(self,index):
                return dict(target_fname="t",target_slice=2,mismatched_pd_fname="donor",mismatched_pd_slice=4)
        row=self.f["mapping_record"](Dataset(),7,1.25,.75)
        for k,v in Dataset().mapping(7).items():
            self.assertEqual(row[k],v)
        self.assertEqual(row["mismatched_pd_normalization_mean"],1.25)
        self.assertEqual(row["mismatched_pd_normalization_std"],.75)
        self.assertEqual(row["used_by_conditions"],"normal_mismatched;no_early_mismatched")

    def test_no_training_no_alternate_checkpoint_or_sampling(self):
        source=SOURCE.read_text(encoding="utf-8")
        tree=ast.parse(source)
        self.assertIn("dataset = AuditDataset(root, cfg.INPUT_SIZE)",source)
        self.assertIn("weights_reconstruction_multi_cross_paper_random4x/best.pth",source)
        for n in ast.walk(tree):
            if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute):
                self.assertNotIn(n.func.attr,("backward","train","step","save"))
        self.assertEqual(self.f["parse_args"]([]).max_batches,1)
        self.assertEqual(self.f["parse_args"]([]).output_dir,"frozen_b0_mismatch_outputs")


if __name__ == "__main__":
    unittest.main()
