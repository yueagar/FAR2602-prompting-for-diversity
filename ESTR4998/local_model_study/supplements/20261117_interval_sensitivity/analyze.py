#!/usr/bin/env python3
"""Post hoc paired-task t sensitivity; preserve all original bootstrap results.

For task-level paired differences d_i, use mean(d) +/- t_(.975,n-1) *
SD(d, ddof=1)/sqrt(n). This interval assumes independent task differences
and a normal sampling model for their mean; it is not a validated replacement
for the original percentile interval or a new confirmatory analysis.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import t

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / '20261117_unified_local'
PRIOR = HERE.parent / '20261117_naive_v05'
ZERO_TOLERANCE = 1e-12
FIELDS = [
    'condition', 'comparator', 'metric', 'comparison_role', 'metric_role',
    'difference', 'bootstrap_ci_low', 'bootstrap_ci_high', 't_ci_low',
    't_ci_high', 'n_tasks', 'positive_tasks', 'negative_tasks', 'zero_tasks',
    'bootstrap_excludes_zero', 't_excludes_zero',
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_summary(differences: list[float]) -> dict:
    """Intervals use raw values; tolerance only classifies numerical zero counts."""
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('Need at least two finite paired task differences')
    n = len(values)
    mean = float(values.mean())
    margin = float(t.ppf(.975, n - 1) * values.std(ddof=1) / math.sqrt(n))
    positive = int(np.count_nonzero(values > ZERO_TOLERANCE))
    negative = int(np.count_nonzero(values < -ZERO_TOLERANCE))
    return {
        'difference': mean, 't_ci_low': mean - margin, 't_ci_high': mean + margin,
        'n_tasks': n, 'positive_tasks': positive, 'negative_tasks': negative,
        'zero_tasks': n - positive - negative,
    }


def read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def collect_pairs(tasks: dict, condition: str, comparator: str, metric: str) -> list[float]:
    a, b = tasks[condition], tasks[comparator]
    if set(a) != set(b):
        raise ValueError('Conditions must contain identical task identities')
    differences = []
    for task in sorted(a):
        x, y = a[task][metric], b[task][metric]
        if (x == '') != (y == ''):
            raise ValueError('Asymmetric missing task support')
        if x == '':
            continue
        x, y = float(x), float(y)
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError('Nonfinite source task mean')
        differences.append(x - y)
    return differences


def analyze(source: Path = SOURCE) -> dict:
    columns, task_rows = read_csv(source / 'task_metrics.csv')
    metrics = [name for name in columns if name not in ('condition', 'prompt_id')]
    if len(metrics) != 30 or len(task_rows) != 70:
        raise ValueError('Expected 70 task/condition rows and 30 metrics')
    tasks = {}
    for row in task_rows:
        arm = tasks.setdefault(row['condition'], {})
        if row['prompt_id'] in arm:
            raise ValueError('Duplicate task/condition identity')
        arm[row['prompt_id']] = row
    if len(tasks) != 7 or any(len(arm) != 10 for arm in tasks.values()):
        raise ValueError('Expected seven complete ten-task conditions')
    _, stored = read_csv(source / 'contrasts.csv')
    if len(stored) != 330:
        raise ValueError('Expected all 330 original contrast/metric rows')
    output, seen = [], set()
    for row in stored:
        key = row['condition'], row['comparator'], row['metric']
        if key in seen or row['metric'] not in metrics:
            raise ValueError('Duplicate contrast or unknown metric')
        seen.add(key)
        stats = paired_summary(collect_pairs(tasks, *key))
        if stats['n_tasks'] != int(row['n_tasks']):
            raise ValueError('Task support differs from original contrast')
        if not math.isclose(stats['difference'], float(row['difference']), rel_tol=0, abs_tol=1e-12):
            raise ValueError('Difference disagrees with original contrast')
        low, high = float(row['ci_low']), float(row['ci_high'])
        if not all(map(math.isfinite, (low, high))) or low > high:
            raise ValueError('Invalid original bootstrap bounds')
        output.append({
            **{k: row[k] for k in ('condition', 'comparator', 'metric', 'comparison_role', 'metric_role')},
            **stats, 'bootstrap_ci_low': low, 'bootstrap_ci_high': high,
            'bootstrap_excludes_zero': low > 0 or high < 0,
            't_excludes_zero': stats['t_ci_low'] > 0 or stats['t_ci_high'] < 0,
        })
    families = {(row['condition'], row['comparator']) for row in output}
    if len(families) != 11 or any(sum((r['condition'], r['comparator']) == pair for r in output) != 30 for pair in families):
        raise ValueError('Expected eleven complete 30-metric contrast families')
    embedding_metrics = ['semantic_diversity', 'vendi_score', 'rke_score']
    deltas = [collect_pairs(tasks, 'naive_creative', 'default', metric) for metric in embedding_metrics]
    naive = [r for r in output if r['condition'] == 'naive_creative' and r['comparator'] == 'default']
    return {
        'schema': 'local-interval-sensitivity-v1',
        'status': 'post_hoc_sensitivity_after_review_not_replacement_or_confirmation',
        'method': {'confidence': .95, 'interval': 'paired Student t', 'degrees_of_freedom': 'n_tasks - 1',
                   'task_unit': 'three-repetition mean, already saved in unified task_metrics.csv',
                   'zero_count_tolerance': ZERO_TOLERANCE, 'new_bootstrap_draws': 0,
                   'external_api_calls': 0, 'new_generator_responses': 0},
        'contrasts': output,
        'naive_embedding_correlations': {'metrics': embedding_metrics, 'pearson': np.corrcoef(deltas).tolist()},
        'naive_bootstrap_excludes_zero': [r['metric'] for r in naive if r['bootstrap_excludes_zero']],
        'naive_t_excludes_zero': [r['metric'] for r in naive if r['t_excludes_zero']],
    }


def interval(row: dict, prefix: str) -> str:
    return f"[{row[prefix + '_ci_low']:+.6f}, {row[prefix + '_ci_high']:+.6f}]"


def render_markdown(summary: dict) -> str:
    rows = summary['contrasts']
    naive = [r for r in rows if r['condition'] == 'naive_creative' and r['comparator'] == 'default']
    text = '''# Local interval-method sensitivity

This analysis was added after the unified results and their review were known.
It preserves the original percentile-bootstrap intervals and adds paired-task
Student t intervals to all 330 existing contrast/metric rows. It does not
replace the fixed analysis, add observations, or provide a new confirmatory test.

For paired task differences d, the interval is mean(d) ± t(0.975, n−1) ×
SD(d)/√n, with sample SD (ddof=1). Task means already average three repetitions.
Most metrics have ten paired tasks; constraints have six. Missing support is
excluded only when absent in both arms, never filled with zero. Differences and
support must match the original contrast table. Positive/negative/zero counts
use a 1e−12 numerical-zero tolerance; intervals use the unrounded differences.

The t interval relies on an independent-task normal sampling model; with ten
selected development tasks it is not a gold standard or evidence of transfer.
Neither method provides simultaneous coverage over the correlated diagnostics.
More bootstrap draws reduce Monte Carlo error, not the uncertainty from having
only ten tasks. No Wilcoxon tests or revised quality gates are used here.

## All thirty naive-minus-default diagnostics

| Metric | Mean difference | Original percentile 95% interval | Paired t 95% interval | Positive / negative / zero tasks |
|---|---:|---:|---:|---:|
'''
    for row in naive:
        text += (f"| {row['metric']} | {row['difference']:+.6f} | {interval(row, 'bootstrap')} | "
                 f"{interval(row, 't')} | {row['positive_tasks']} / {row['negative_tasks']} / {row['zero_tasks']} |\n")
    text += f"\nOriginal percentile intervals exclude zero for {len(summary['naive_bootstrap_excludes_zero'])} of 30 diagnostics. This is a descriptive inventory, not a multiple-testing claim.\n"
    text += '''
The classifier-count effect is inconclusive, not evidence of no change. RKE and
the MiniLM partition proxy have positive original percentile bounds but t
intervals including zero. Mean cosine distance and Vendi include zero under
both methods. Distinct-2 is positive under both, but is lexical and accompanies
longer answers. Thus within-set semantic diversification remains inconclusive.
Reference-relative novelty and response length are positive under both methods.

## Quality contrasts whose zero-exclusion depends on interval method

| Contrast | Mean | Original percentile 95% interval | Paired t 95% interval |
|---|---:|---:|---:|
'''
    for arm in ('naive_creative', 'temp_1.0'):
        row = next(r for r in rows if r['condition'] == arm and r['comparator'] == 'default' and r['metric'] == 'quality_mean')
        text += f"| {arm} − default, greedy quality | {row['difference']:+.6f} | {interval(row, 'bootstrap')} | {interval(row, 't')} |\n"
    text += '''
The naive weighted-quality point difference is positive on all ten tasks and
its t interval also excludes zero. This greater numerical consistency does
not validate the judge or control the response-length difference. All quality
endpoints remain automatic and unvalidated; no human-quality claim follows.

## Dependence between embedding measures

Pearson correlations of the ten naive-minus-default paired task differences:

| Pair | Correlation |
|---|---:|
'''
    corr = summary['naive_embedding_correlations']
    for i in range(3):
        for j in range(i+1,3):
            text += f"| {corr['metrics'][i]} / {corr['metrics'][j]} | {corr['pearson'][i][j]:.6f} |\n"
    text += '''
These measures are strongly correlated, not independent confirmations or
mathematically identical information. They summarize the same embeddings in
different ways. The unchanged original results and these sensitivities should
be read together, without selecting whichever interval is most favorable.

`contrasts.csv` and `summary.json` contain all eleven contrast families and all
thirty metrics, including the two methods and task sign counts. The manifest
binds the source CSVs and generated artifacts. No model or API calls are needed.
'''
    return text


def write_outputs(summary: dict, output: Path, source: Path = SOURCE) -> None:
    output = output.resolve()
    for protected in (SOURCE, PRIOR, source):
        if output == protected.resolve() or protected.resolve() in output.parents:
            raise ValueError('Cannot write into an original source/supplement directory')
    targets = [output / name for name in ('summary.json','contrasts.csv','RESULTS.md','manifest.json')]
    for target in targets:
        if target.is_symlink() or (target.exists() and target.stat().st_nlink != 1):
            raise ValueError('Output must not alias another artifact')
    output.mkdir(parents=True, exist_ok=True)
    (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=FIELDS, lineterminator='\n')
    writer.writeheader()
    writer.writerows(summary['contrasts'])
    (output/'contrasts.csv').write_text(buffer.getvalue())
    (output/'RESULTS.md').write_text(render_markdown(summary))
    manifest = {
        'schema': 'local-interval-sensitivity-manifest-v1',
        'status': summary['status'], 'analysis_sha256': digest(Path(__file__)),
        'source_sha256': {name:digest(source/name) for name in ('task_metrics.csv','contrasts.csv')},
        'output_sha256': {name:digest(output/name) for name in ('summary.json','contrasts.csv','RESULTS.md')},
        'environment': {'numpy': np.__version__, 'scipy': scipy.__version__},
        'original_bootstrap_bounds_preserved': True,
        'new_generator_responses': 0, 'external_api_calls': 0,
    }
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=HERE)
    args = parser.parse_args()
    summary = analyze()
    write_outputs(summary, args.output_dir)
    print(f"Wrote {len(summary['contrasts'])} interval comparisons to {args.output_dir}")


if __name__ == '__main__':
    main()
