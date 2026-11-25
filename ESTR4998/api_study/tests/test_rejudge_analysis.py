"""Hand-calculable offline checks; these fixtures are not real judgments."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rejudge_analysis as analysis


def cell(task='task-1', arm='default', original=None, repeat=None, pooled=None):
    original = original if original is not None else [[0, 0, 1], [0, 1, 2]]
    repeat = repeat if repeat is not None else copy.deepcopy(original)
    ids = [[f'{task}-{arm}-{rep}-{i}' for i in range(3)] for rep in (0, 1)]
    return {'task_id': task, 'arm': arm,
            'original': [{'repetition': rep, 'response_ids': ids[rep], 'labels': original[rep]} for rep in (0, 1)],
            'repeats': [{'repetition': rep, 'response_ids': ids[rep], 'labels': repeat[rep]} for rep in (0, 1)],
            'pooled': {'response_ids': ids[0] + ids[1],
                       'labels': pooled if pooled is not None else [0, 0, 1, 0, 2, 3]}}


def run(cells):
    return analysis.analyze(cells, bootstrap_samples=100, seed=13)


class RejudgeAnalysisTests(unittest.TestCase):
    def test_pair_counts_exclude_diagonal_and_ignore_label_names(self):
        metric = analysis.partition_change([0, 0, 1, 1], [5, 8, 5, 8])
        self.assertEqual(metric['pairs'], 6)
        self.assertEqual(metric['original_same_pairs'], 2)
        self.assertEqual(metric['original_different_pairs'], 4)
        self.assertEqual(metric['split_pairs'], 2)
        self.assertEqual(metric['merged_pairs'], 2)
        self.assertEqual(metric['agreement'], 2 / 6)
        self.assertEqual(metric['split_rate'], 1)
        self.assertEqual(metric['merge_rate'], .5)
        same = analysis.partition_change([0, 0, 1, 1], [8, 8, 5, 5])
        self.assertTrue(same['same_partition'])
        self.assertEqual(same['adjusted_rand_index'], 1)

    def test_degenerate_pair_denominators_are_unavailable_not_zero(self):
        singletons = analysis.partition_change([0, 1, 2], [3, 4, 5])
        allsame = analysis.partition_change([0, 0, 0], [4, 4, 4])
        self.assertIsNone(singletons['split_rate'])
        self.assertIsNone(allsame['merge_rate'])
        self.assertEqual(singletons['agreement'], 1)
        self.assertEqual(allsame['adjusted_rand_index'], 1)

    def test_identity_alignment_accepts_reordering_and_label_renaming(self):
        source = cell()
        base = run([source])
        reordered = copy.deepcopy(source)
        for row in reordered['repeats'] + [reordered['pooled']]:
            row['response_ids'] = list(reversed(row['response_ids']))
            row['labels'] = [f'cluster-{v}' for v in reversed(row['labels'])]
        self.assertEqual(base, run([reordered]))

    def test_pool_within_and_cross_rep_pairs_are_distinct(self):
        report = run([cell()])
        self.assertEqual([r['original_at_ceiling'] for r in report['repeat_sets']], [False, True])
        self.assertEqual([r['repeat_at_ceiling'] for r in report['repeat_sets']], [False, True])
        self.assertEqual([r['absolute_count_change'] for r in report['repeat_sets']], [0, 0])
        pooled = report['pooled_sets'][0]
        # pooled [0,0,1] / [0,2,3]: two of the nine cross pairs share 0.
        self.assertEqual(pooled['pooled_count'], 4)
        self.assertEqual(pooled['cross_repetition_pairs'], 9)
        self.assertEqual(pooled['cross_repetition_same_pairs'], 2)
        self.assertEqual(pooled['cross_repetition_same_rate'], 2 / 9)
        self.assertEqual([r['restricted_pooled_count'] for r in pooled['within_repetition']], [2, 3])
        self.assertTrue(all(r['versus_original']['same_partition'] for r in pooled['within_repetition']))
        # Original label 0 occurs in each repetition, but new disjoint groups
        # must yield zero cross co-membership regardless of that coincidence.
        disjoint = run([cell(pooled=[0, 0, 1, 2, 3, 4])])['pooled_sets'][0]
        self.assertEqual(disjoint['cross_repetition_same_pairs'], 0)

    def test_task_contrasts_average_reps_and_pair_original_repeat_change(self):
        cells = []
        # Task1: original contrast+1, repeatcontrast-1 => change-2.
        # Task2: originalcontrast0, repeatcontrast+.5 => change+.5.
        for task in ('task-1', 'task-2'):
            cells.append(cell(task, original=[[0, 0, 0], [0, 1, 1]],
                              repeat=[[0, 1, 1], [0, 1, 2]]))
        cells.append(cell('task-1', 'naive_creative', original=[[0, 1, 1], [0, 1, 2]],
                          repeat=[[0, 0, 0], [0, 1, 1]]))
        cells.append(cell('task-2', 'naive_creative', original=[[0, 0, 0], [0, 1, 1]],
                          repeat=[[0, 1, 2], [0, 1, 2]]))
        report = run(cells)
        contrast = report['contrasts'][0]
        self.assertEqual(contrast['mean'], {'original': .5, 'repeat': -.25, 'change': -.75})
        self.assertEqual(contrast['per_task']['repeat'], {'task-1': -1, 'task-2': .5})
        self.assertEqual(contrast['paired_task_bootstrap_95']['change'], [-2, .5])
        self.assertEqual(report, run(list(reversed(cells))))

    def test_missing_repetition_excludes_entire_task_from_all_paired_estimates(self):
        cells = [cell(task, arm) for task in ('task-1', 'task-2')
                 for arm in ('default', 'naive_creative')]
        cells[-1]['repeats'][0]['labels'] = None
        cells[-1]['pooled']['labels'] = None
        report = run(cells)
        contrast = report['contrasts'][0]
        self.assertEqual(contrast['tasks_used'], ['task-1'])
        self.assertEqual(contrast['tasks_missing'], ['task-2'])
        self.assertEqual(contrast['n_tasks'], 1)
        self.assertEqual(contrast['mean'], {'original': 0, 'repeat': 0, 'change': 0})
        self.assertEqual(contrast['paired_task_bootstrap_95'], {'original': None, 'repeat': None, 'change': None})
        summary = next(r for r in report['arm_summaries'] if r['arm'] == 'naive_creative')
        self.assertEqual(summary['repeat_sets_available'], 3)
        self.assertEqual(summary['complete_task_count'], 1)
        self.assertEqual(summary['pooled_sets_available'], 1)
        missingrepeat = next(r for r in report['repeat_sets'] if r['task_id'] == 'task-2'
                             and r['arm'] == 'naive_creative' and r['repetition'] == 0)
        self.assertIsNone(missingrepeat['repeat_at_ceiling'])
        self.assertIsNone(missingrepeat['absolute_count_change'])
        missingpool = next(r for r in report['pooled_sets'] if r['task_id'] == 'task-2' and r['arm'] == 'naive_creative')
        self.assertIsNone(missingpool['pooled_count'])
        self.assertIsNone(missingpool['cross_repetition_same_pairs'])

    def test_all_new_judgments_unavailable_produces_no_fabricated_estimates(self):
        cells = [cell(arm=arm) for arm in ('default', 'naive_creative')]
        for item in cells:
            item['pooled']['labels'] = None
            for row in item['repeats']:
                row['labels'] = None
        report = run(cells)
        self.assertEqual(report['contrasts'][0]['n_tasks'], 0)
        self.assertTrue(all(v is None for v in report['contrasts'][0]['mean'].values()))
        self.assertTrue(all(s['same_partition_sets'] == 0 and s['repeat_sets_available'] == 0
                            and s['task_mean_count_change'] is None for s in report['arm_summaries']))

    def test_actual_project_causal_minus_naive_identity_is_retained(self):
        report = run([cell(arm=arm) for arm in ('default', 'naive_creative',
                                              'manual_causal_shift_v1', 'manual_alternative_selection_v1')])
        self.assertEqual(len(report['contrasts']), 4)
        self.assertEqual((report['contrasts'][-1]['arm'], report['contrasts'][-1]['reference']),
                         ('manual_causal_shift_v1', 'naive_creative'))
        self.assertFalse(report['qualified_endpoint'])
        self.assertFalse(report['original_endpoint_changed'])

    def test_invalid_identity_roster_and_partition_shapes_fail(self):
        base = cell()
        for mutate in (
            lambda c: c['repeats'][0]['response_ids'].__setitem__(0, 'wrong-response'),
            lambda c: c['repeats'][0]['response_ids'].__setitem__(0, c['repeats'][0]['response_ids'][1]),
            lambda c: c['original'][0].__setitem__('labels', None),
            lambda c: c['original'][0].__setitem__('labels', [0]),
            lambda c: c['original'][0].__setitem__('labels', [False, 0, 1]),
            lambda c: c['repeats'][0].__setitem__('repetition', 1),
            lambda c: c['pooled'].__setitem__('response_ids', c['pooled']['response_ids'][:-1]),
        ):
            with self.subTest(mutation=mutate):
                corrupted = copy.deepcopy(base)
                mutate(corrupted)
                with self.assertRaises(ValueError):
                    run([corrupted])
        with self.assertRaises(ValueError):
            run([base, copy.deepcopy(base)])
        with self.assertRaises(ValueError):
            run([base, cell('task-1', 'naive_creative'), cell('task-2')])
        with self.assertRaises(ValueError):
            analysis.analyze([base], bootstrap_samples=1)


if __name__ == '__main__':
    unittest.main()
