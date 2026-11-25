"""Synthetic arithmetic/edge-case checks; no model, saved-run or network access."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import factorial_analysis as analysis


def fixture(tasks=('curated-0', 'curated-2'), n=3):
    arms = [{'id': f'{instruction}_t{temperature:g}', 'instruction_id': instruction,
             'temperature': temperature, 'instruction': f'fixture-{instruction}'}
            for instruction in analysis.INSTRUCTIONS for temperature in analysis.TEMPERATURES]
    plan = {'arms': arms, 'tasks': [{'id': task, 'text': 'Synthetic fixture only.'} for task in tasks],
            'check_tasks': list(tasks[:1]), 'n': n, 'repetitions': 2,
            'analysis': analysis.analysis_protocol(arms)}
    rows = []
    for task in tasks:
        for arm in arms:
            for repetition in (0, 1):
                rows.append({'cell': {'generator': 'anchor', 'task': task,
                                      'arm': arm['id'], 'repetition': repetition},
                             'condition': {k: arm[k] for k in ('instruction_id', 'temperature')},
                             'samples': n, 'generation_complete': True,
                             'generations': [{'status': 'ok', 'text': 'One. Two. Three. Four. Five.',
                                              'finish_reason': 'stop', 'reported_cost_usd': 0.0} for _ in range(n)],
                             'judges': {role: {'groupings': [list(range(n)) for _ in range(count)],
                                              'scores': [7] * n}
                                        for role, count in [('primary', 2)] +
                                        ([('check', 1)] if task in plan['check_tasks'] else [])}})
    return plan, {'rows': rows, 'accounting': {'synthetic': True}}


def row(results, task='curated-0', instruction='default', temperature=.8, rep=0):
    return next(r for r in results['rows'] if r['cell']['task'] == task and
                r['condition'] == {'instruction_id': instruction, 'temperature': temperature}
                and r['cell']['repetition'] == rep)


def contrast(report, name='naive_minus_default_at_0.8'):
    return next(c for c in report['contrasts'] if c['id'] == name)


class FactorialAnalysisTests(unittest.TestCase):
    def test_five_primary_contrasts_and_frozen_configuration(self):
        plan, results = fixture()
        protocol = plan['analysis']
        self.assertEqual(sum(c['analysis_role'] == 'primary' for c in protocol['contrasts']), 5)
        self.assertEqual(len(protocol['contrasts']), 15)
        self.assertEqual(protocol['bootstrap_seed'], 20261116)
        self.assertEqual(protocol['bootstrap_samples'], 5000)
        self.assertEqual(protocol['quality_guard']['quality']['minimum_tasks'], 10)
        self.assertEqual(protocol['quality_guard']['constraint_satisfaction']['minimum_tasks'], 6)
        self.assertTrue(all(sum(c['weights'].values()) == 0 for c in protocol['contrasts']))
        corrupt = copy.deepcopy(plan)
        corrupt['analysis']['bootstrap_seed'] += 1
        with self.assertRaisesRegex(ValueError, 'Frozen analysis'):
            analysis.analyze(corrupt, results)

    def test_mean_counts_repetitions_and_paired_task_bootstrap(self):
        plan, results = fixture()
        specifications = {
            ('curated-0', 'default', 0): [[0, 0, 0], [0, 1, 2]],
            ('curated-0', 'default', 1): [[0, 0, 0], [0, 0, 0]],
            ('curated-0', 'naive_creative', 0): [[0, 0, 1], [0, 0, 1]],
            ('curated-0', 'naive_creative', 1): [[0, 1, 2], [0, 1, 2]],
            ('curated-2', 'default', 0): [[0, 0, 0], [0, 0, 0]],
            ('curated-2', 'default', 1): [[0, 0, 1], [0, 0, 1]],
            ('curated-2', 'naive_creative', 0): [[0, 0, 0], [0, 1, 2]],
            ('curated-2', 'naive_creative', 1): [[0, 0, 1], [0, 0, 1]],
        }
        for (task, instruction, rep), labels in specifications.items():
            row(results, task, instruction, rep=rep)['judges']['primary']['groupings'] = labels
        report = analysis.analyze(plan, results)
        estimate = contrast(report)['estimates']['primary_count']
        self.assertEqual(estimate['per_task'], {'curated-0': 1, 'curated-2': .5})
        self.assertEqual(estimate['mean'], .75)
        self.assertEqual(estimate['paired_task_bootstrap_95'], [.5, 1])
        self.assertEqual(estimate['approximate_family_bootstrap_99'], [.5, 1])
        self.assertFalse(estimate['full_task_support_for_primary_interpretation'])
        self.assertFalse(report['qualified_endpoint'])
        self.assertFalse(report['original_failed_gates_changed'])
        self.assertEqual(report['accounting'], {'synthetic': True})
        reversed_results = results | {'rows': list(reversed(results['rows']))}
        self.assertEqual(report, analysis.analyze(plan, reversed_results))

    def test_missing_second_judgment_keeps_first_only_and_quality_support(self):
        plan, results = fixture()
        row(results, 'curated-2', 'naive_creative')['judges']['primary']['groupings'][1] = None
        report = analysis.analyze(plan, results)
        comparison = contrast(report)
        primary = comparison['estimates']['primary_count']
        self.assertEqual(primary['tasks_used'], ['curated-0'])
        self.assertEqual(primary['tasks_missing'], ['curated-2'])
        self.assertIsNone(primary['paired_task_bootstrap_95'])
        self.assertEqual(comparison['estimates']['first_judgment_count']['n_tasks'], 2)
        self.assertEqual(comparison['estimates']['primary_quality']['n_tasks'], 2)
        matched = comparison['single_judgment_sensitivities_on_primary_support']
        self.assertTrue(all(v['tasks_used'] == ['curated-0'] for v in matched.values()))
        self.assertEqual(comparison['automatic_safeguards']['quality']['verdict'], 'insufficient_support')

    def test_all_missing_retains_roster_and_unavailable_estimates(self):
        plan, results = fixture()
        for r in results['rows']:
            r['generation_complete'] = False
            r['generations'] = [None] * plan['n']
            for judge in r['judges'].values():
                judge['groupings'] = [None] * len(judge['groupings'])
                judge['scores'] = [None] * plan['n']
        report = analysis.analyze(plan, results)
        self.assertEqual(report['planned_cells'], 36)
        self.assertTrue(all(c['estimates']['primary_count']['n_tasks'] == 0 for c in report['contrasts']))
        self.assertTrue(all(c['estimates']['primary_count']['mean'] is None for c in report['contrasts']))
        self.assertTrue(all(s['repeat_comparable_cells'] == 0 for s in report['condition_summaries']))
        self.assertTrue(all(s['generation_diagnostics']['successful_outputs'] == 0
                            and s['generation_diagnostics']['unobserved_slots'] == 12
                            for s in report['condition_summaries']))

    def test_partition_disagreement_excludes_diagonal_and_ignores_label_names(self):
        plan, results = fixture(n=4)
        r = row(results)
        r['judges']['primary']['groupings'] = [[0, 0, 1, 1], [5, 8, 5, 8]]
        report = analysis.analyze(plan, results)
        c = next(c for c in report['cells'] if c['task'] == 'curated-0' and
                 c['condition'] == r['cell']['arm'] and c['repetition'] == 0)
        self.assertEqual(c['primary_mean_count'], 2)
        self.assertEqual(c['absolute_count_difference'], 0)
        self.assertEqual(c['repeat_partition_change']['pairs'], 6)
        self.assertEqual(c['repeat_partition_change']['split_pairs'], 2)
        self.assertEqual(c['repeat_partition_change']['merged_pairs'], 2)
        self.assertFalse(c['repeat_partition_change']['same_partition'])
        r['judges']['primary']['groupings'][1] = ['B', 'B', 'A', 'A']
        invariant = analysis.analyze(plan, results)
        changed = next(c for c in invariant['cells'] if c['task'] == 'curated-0' and
                       c['condition'] == r['cell']['arm'] and c['repetition'] == 0)
        self.assertTrue(changed['repeat_partition_change']['same_partition'])

    def test_parser_change_applies_symmetrically_and_empty_outputs_are_diagnostics(self):
        plan, results = fixture(tasks=('curated-0',))
        for r in results['rows']:
            for g in r['generations']:
                g['text'] = '**One.** Two. Three. Four. Five.'
        report = analysis.analyze(plan, results)
        for summary in report['condition_summaries']:
            self.assertEqual(summary['estimates']['constraint_satisfied']['mean'], 1)
            self.assertEqual(summary['estimates']['original_constraint_satisfied']['mean'], 0)
        # Duplicates and empty strings do not manufacture observations or alter
        # externally supplied group labels. They remain inspectable diagnostics.
        row(results)['generations'][0]['text'] = ''
        empty = analysis.analyze(plan, results)
        summary = next(s for s in empty['condition_summaries'] if s['condition'] == 'default_t0.8')
        self.assertEqual(summary['generation_diagnostics']['successful_outputs'], 6)
        self.assertEqual(summary['generation_diagnostics']['flags']['empty'], 1)
        self.assertEqual(summary['estimates']['primary_count']['mean'], 3)

    def test_quality_and_constraint_support_do_not_depend_on_grouping(self):
        plan, results = fixture()
        r = row(results)
        r['judges']['primary']['groupings'] = [None, None]
        r['judges']['primary']['scores'][0] = None
        report = analysis.analyze(plan, results)
        estimates = contrast(report)['estimates']
        self.assertEqual(estimates['primary_count']['tasks_used'], ['curated-2'])
        self.assertEqual(estimates['primary_quality']['tasks_used'], ['curated-2'])
        self.assertEqual(estimates['constraint_satisfied']['tasks_used'], ['curated-0', 'curated-2'])
        self.assertEqual(estimates['check_count']['tasks_used'], ['curated-0'])

    def test_check_judge_comparison_uses_matched_subset_and_both_primary_judgments(self):
        plan, results = fixture()
        for rep in (0, 1):
            r = row(results, rep=rep)
            r['judges']['primary']['groupings'] = [[0, 0, 0], [0, 1, 2]]
            r['judges']['check']['groupings'] = [[0, 0, 0]]
            r['judges']['check']['scores'] = [8, 8, 8]
        report = analysis.analyze(plan, results)
        summary = next(s for s in report['condition_summaries'] if s['condition'] == 'default_t0.8')
        self.assertEqual(summary['estimates']['check_minus_primary_count']['mean'], -1)
        self.assertEqual(summary['estimates']['check_minus_primary_quality']['mean'], 1)
        self.assertEqual(summary['estimates']['check_minus_primary_count']['tasks_used'], ['curated-0'])

    def test_cross_temperature_contrast_uses_actual_reference(self):
        plan, results = fixture()
        for task in ('curated-0', 'curated-2'):
            for rep in (0, 1):
                row(results, task, 'default', 1., rep)['judges']['primary']['scores'] = [10, 10, 10]
        report = analysis.analyze(plan, results)
        estimate = contrast(report, 'naive_at_0.8_minus_default_at_1')['estimates']['primary_quality']
        self.assertEqual(estimate['mean'], -3)
        self.assertEqual(contrast(report)['estimates']['primary_quality']['mean'], 0)

    def test_guard_inclusive_boundary_and_missing_task_bounds(self):
        estimate = {'tasks_missing': [], 'n_tasks': 10,
                    'per_task': {str(i): -.5 for i in range(10)},
                    'paired_task_bootstrap_95': [-.50000000000001, -.5]}
        policy = {'margin': .5, 'minimum_tasks': 10, 'maximum_missing_tasks': 0}
        self.assertEqual(analysis._guard(estimate, policy, 9)['verdict'], 'passes_automatic_safeguard')
        estimate['tasks_missing'] = ['missing']
        estimate['n_tasks'] = 9
        del estimate['per_task']['9']
        guarded = analysis._guard(estimate, policy, 9)
        self.assertEqual(guarded['verdict'], 'insufficient_support')
        self.assertEqual(guarded['full_eligible_task_point_mean_bounds'], [-1.35, .45])

    def test_missing_duplicate_misaligned_or_invalid_slots_fail(self):
        plan, results = fixture()
        mutations = [
            lambda r: r['rows'].pop(),
            lambda r: r['rows'].append(copy.deepcopy(r['rows'][0])),
            lambda r: r['rows'][0]['condition'].__setitem__('temperature', 9),
            lambda r: r['rows'][0]['judges']['primary']['groupings'].__setitem__(0, [0]),
            lambda r: r['rows'][0]['judges']['primary']['scores'].__setitem__(0, True),
            lambda r: r['rows'][0].__setitem__('generation_complete', False),
            lambda r: r['rows'][0]['judges']['primary']['groupings'].pop(),
            lambda r: r['rows'][0]['generations'].__setitem__(0, None),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                corrupted = copy.deepcopy(results)
                mutate(corrupted)
                with self.assertRaises(ValueError):
                    analysis.analyze(plan, corrupted)

    def test_embedding_helper_requires_canonical_backend_and_never_downloads(self):
        _, results = fixture()
        with self.assertRaisesRegex(ValueError, 'canonical pinned'):
            analysis.with_embedding_metrics(results, object())
        # No backend is instantiated by ordinary analysis; all unavailable
        # secondary estimates remain None, not substituted fake scores.
        plan, results = fixture()
        with patch('localstudy.metrics.Embedder.__init__', side_effect=AssertionError('No model construction')):
            report = analysis.analyze(plan, results)
        self.assertTrue(all(c['estimates']['vendi_score']['mean'] is None for c in report['contrasts']))

    def test_unknown_billing_preserves_text_diagnostic_but_not_complete_task_estimates(self):
        plan, results = fixture()
        r = row(results)
        r['generations'][0]['reported_cost_usd'] = None
        r['generation_complete'] = False
        for judge in r['judges'].values():
            judge['groupings'] = [None] * len(judge['groupings'])
            judge['scores'][0] = None
        report = analysis.analyze(plan, results)
        estimates = contrast(report)['estimates']
        self.assertEqual(estimates['primary_count']['tasks_used'], ['curated-2'])
        self.assertEqual(estimates['constraint_satisfied']['tasks_used'], ['curated-2'])
        summary = next(s for s in report['condition_summaries'] if s['condition'] == 'default_t0.8')
        self.assertEqual(summary['generation_diagnostics']['successful_texts_with_unresolved_billing'], 1)
        self.assertEqual(summary['generation_diagnostics']['successful_outputs'], 12)


if __name__ == '__main__':
    unittest.main()
