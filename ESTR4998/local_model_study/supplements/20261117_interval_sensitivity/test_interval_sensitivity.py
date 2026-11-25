"""Edge cases and source-preservation tests for the post hoc t sensitivity."""
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('local_interval_sensitivity_analysis', HERE / 'analyze.py')
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class IntervalSensitivityTests(unittest.TestCase):
    def test_known_three_task_interval_and_sign_reversal(self):
        result = analysis.paired_summary([1,2,3])
        # Student t df=2 critical value: 4.302652729911275, sample SE=1/sqrt(3).
        half_width = 4.302652729911275 / np.sqrt(3)
        self.assertAlmostEqual(result['t_ci_low'],2-half_width)
        self.assertAlmostEqual(result['t_ci_high'],2+half_width)
        reverse = analysis.paired_summary([-1,-2,-3])
        self.assertAlmostEqual(reverse['t_ci_low'],-result['t_ci_high'])
        self.assertAlmostEqual(reverse['t_ci_high'],-result['t_ci_low'])
        self.assertEqual(reverse['negative_tasks'],3)

    def test_constant_and_numerical_zero_counts(self):
        self.assertEqual(analysis.paired_summary([0,0,0])['t_ci_high'],0)
        same = analysis.paired_summary([2,2,2])
        self.assertEqual((same['t_ci_low'],same['t_ci_high']),(2,2))
        zeros = analysis.paired_summary([-1e-15,1e-15,1])
        self.assertEqual((zeros['positive_tasks'],zeros['negative_tasks'],zeros['zero_tasks']),(1,0,2))

    def test_rejects_insufficient_or_nonfinite_observations(self):
        for values in ([],[1],[1,float('nan')],[1,float('inf')],[[1],[2]]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                analysis.paired_summary(values)

    def test_joint_missing_support_and_task_pairing(self):
        tasks = {'a':{'x':{'m':'3'},'y':{'m':''},'z':{'m':'2'}},
                 'b':{'z':{'m':'5'},'y':{'m':''},'x':{'m':'1'}}}
        self.assertEqual(analysis.collect_pairs(tasks,'a','b','m'),[2,-3])
        tasks['a']['y']['m']='0'
        with self.assertRaisesRegex(ValueError,'Asymmetric missing'):
            analysis.collect_pairs(tasks,'a','b','m')
        del tasks['a']['y']
        with self.assertRaisesRegex(ValueError,'task identities'):
            analysis.collect_pairs(tasks,'a','b','m')

    def test_all_original_contrasts_have_correct_support(self):
        result = analysis.analyze()
        self.assertEqual(len(result['contrasts']),330)
        for row in result['contrasts']:
            self.assertEqual(row['n_tasks'],6 if row['metric']=='constraint_satisfaction' else 10)
            self.assertEqual(row['positive_tasks']+row['negative_tasks']+row['zero_tasks'],row['n_tasks'])
        self.assertEqual(len(result['naive_bootstrap_excludes_zero']),11)

    def test_rejects_modified_stored_effect(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            for name in ('task_metrics.csv','contrasts.csv'):
                shutil.copyfile(analysis.SOURCE/name,target/name)
            text = (target/'contrasts.csv').read_text()
            rows = text.splitlines()
            fields = rows[0].split(',')
            first = rows[1].split(',')
            first[fields.index('difference')]='1234'
            rows[1]=','.join(first)
            (target/'contrasts.csv').write_text('\n'.join(rows)+'\n')
            with self.assertRaisesRegex(ValueError,'Difference disagrees'):
                analysis.analyze(target)

    def test_outputs_do_not_overwrite_prior_or_aliased_files(self):
        result=analysis.analyze()
        for target in (analysis.SOURCE,analysis.SOURCE/'new',analysis.PRIOR/'new'):
            with self.subTest(target=target), self.assertRaisesRegex(ValueError,'original source'):
                analysis.write_outputs(result,target)
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)
            protected=target/'protected.json'
            protected.write_text('preserve')
            (target/'summary.json').symlink_to(protected)
            with self.assertRaisesRegex(ValueError,'alias'):
                analysis.write_outputs(result,target)
            self.assertEqual(protected.read_text(),'preserve')

    def test_byte_identical_reproduction_and_source_preservation(self):
        before={name:analysis.digest(analysis.SOURCE/name) for name in ('task_metrics.csv','contrasts.csv')}
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)
            analysis.write_outputs(analysis.analyze(),target)
            for name in ('summary.json','contrasts.csv','RESULTS.md','manifest.json'):
                self.assertEqual((target/name).read_bytes(),(HERE/name).read_bytes(),name)
            manifest=json.loads((target/'manifest.json').read_text())
            self.assertEqual(manifest['source_sha256'],before)
        self.assertEqual(before,{name:analysis.digest(analysis.SOURCE/name) for name in before})


if __name__=='__main__':
    unittest.main()
