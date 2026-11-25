# Local interval-method sensitivity

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
| ai_disclaimer_share | +0.006667 | [+0.000000, +0.020000] | [-0.008414, +0.021748] | 1 / 0 / 9 |
| completion_tokens_mean | +40.163333 | [+24.659833, +53.866750] | [+22.256815, +58.069852] | 9 / 1 / 0 |
| constraint_satisfaction | +0.038889 | [-0.022222, +0.111111] | [-0.058599, +0.136377] | 2 / 2 / 2 |
| distinct_1 | +0.011270 | [-0.001971, +0.025920] | [-0.005924, +0.028463] | 7 / 3 / 0 |
| distinct_2 | +0.022131 | [+0.007690, +0.037991] | [+0.003602, +0.040659] | 8 / 2 / 0 |
| empty_share | +0.000000 | [+0.000000, +0.000000] | [+0.000000, +0.000000] | 0 / 0 / 10 |
| exact_duplicate_rate | +0.000000 | [+0.000000, +0.000000] | [+0.000000, +0.000000] | 0 / 0 / 10 |
| expected_parsed_share | -0.003333 | [-0.010000, +0.000000] | [-0.010874, +0.004207] | 0 / 1 / 9 |
| expected_quality | +0.083062 | [+0.047360, +0.130553] | [+0.031260, +0.134864] | 10 / 0 / 0 |
| functional_classes | -0.066667 | [-0.933333, +1.100000] | [-1.307230, +1.173897] | 2 / 5 / 3 |
| functional_novel_share | +0.080000 | [-0.030000, +0.206667] | [-0.064478, +0.224478] | 5 / 3 / 2 |
| functional_window_truncated_share | +0.096667 | [-0.036667, +0.256667] | [-0.082992, +0.276325] | 6 / 2 / 2 |
| instruction_leak_share | +0.000000 | [+0.000000, +0.000000] | [+0.000000, +0.000000] | 0 / 0 / 10 |
| mixed_script_share | +0.000000 | [+0.000000, +0.000000] | [+0.000000, +0.000000] | 0 / 0 / 10 |
| novel_share_proxy | +0.233333 | [+0.110000, +0.400000] | [+0.051039, +0.415628] | 10 / 0 / 0 |
| partition_count_proxy | +0.766667 | [+0.133333, +1.633333] | [-0.167398, +1.700732] | 5 / 2 / 3 |
| quality_coherence | +0.057778 | [+0.007778, +0.122222] | [-0.013787, +0.129343] | 5 / 1 / 4 |
| quality_constraints | +0.036296 | [-0.016296, +0.095185] | [-0.031503, +0.104096] | 4 / 3 / 3 |
| quality_effectiveness | +0.070370 | [+0.010370, +0.140000] | [-0.007957, +0.148698] | 4 / 1 / 5 |
| quality_mean | +0.056019 | [+0.012685, +0.109537] | [-0.004072, +0.116109] | 6 / 2 / 2 |
| quality_parsed_share | -0.003333 | [-0.010000, +0.000000] | [-0.010874, +0.004207] | 0 / 1 / 9 |
| quality_relevance | +0.061111 | [+0.013306, +0.124444] | [-0.009312, +0.131534] | 6 / 1 / 3 |
| reference_novelty | +0.067645 | [+0.030152, +0.109310] | [+0.018973, +0.116316] | 9 / 1 / 0 |
| refusal_share | +0.000000 | [+0.000000, +0.000000] | [+0.000000, +0.000000] | 0 / 0 / 10 |
| repetitive_share | -0.030000 | [-0.076667, +0.016667] | [-0.088893, +0.028893] | 2 / 5 / 3 |
| rke_score | +0.396962 | [+0.014609, +0.820700] | [-0.098478, +0.892402] | 6 / 4 / 0 |
| semantic_diversity | +0.021759 | [-0.005452, +0.051120] | [-0.012775, +0.056293] | 6 / 4 / 0 |
| truncated_share | +0.006667 | [+0.000000, +0.016667] | [-0.003387, +0.016721] | 2 / 0 / 8 |
| vendi_score | +0.201325 | [-0.031047, +0.471019] | [-0.104791, +0.507440] | 6 / 4 / 0 |
| very_short_share | +0.000000 | [+0.000000, +0.000000] | [+0.000000, +0.000000] | 0 / 0 / 10 |

Original percentile intervals exclude zero for 11 of 30 diagnostics. This is a descriptive inventory, not a multiple-testing claim.

The classifier-count effect is inconclusive, not evidence of no change. RKE and
the MiniLM partition proxy have positive original percentile bounds but t
intervals including zero. Mean cosine distance and Vendi include zero under
both methods. Distinct-2 is positive under both, but is lexical and accompanies
longer answers. Thus within-set semantic diversification remains inconclusive.
Reference-relative novelty and response length are positive under both methods.

## Quality contrasts whose zero-exclusion depends on interval method

| Contrast | Mean | Original percentile 95% interval | Paired t 95% interval |
|---|---:|---:|---:|
| naive_creative − default, greedy quality | +0.056019 | [+0.012685, +0.109537] | [-0.004072, +0.116109] |
| temp_1.0 − default, greedy quality | -0.045833 | [-0.096667, -0.008333] | [-0.100704, +0.009037] |

The naive weighted-quality point difference is positive on all ten tasks and
its t interval also excludes zero. This greater numerical consistency does
not validate the judge or control the response-length difference. All quality
endpoints remain automatic and unvalidated; no human-quality claim follows.

## Dependence between embedding measures

Pearson correlations of the ten naive-minus-default paired task differences:

| Pair | Correlation |
|---|---:|
| semantic_diversity / vendi_score | 0.988367 |
| semantic_diversity / rke_score | 0.970288 |
| vendi_score / rke_score | 0.965319 |

These measures are strongly correlated, not independent confirmations or
mathematically identical information. They summarize the same embeddings in
different ways. The unchanged original results and these sensitivities should
be read together, without selecting whichever interval is most favorable.

`contrasts.csv` and `summary.json` contain all eleven contrast families and all
thirty metrics, including the two methods and task sign counts. The manifest
binds the source CSVs and generated artifacts. No model or API calls are needed.
