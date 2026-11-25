# Reproduce the interim-report results

Run the commands below from the repository root. Numeric inputs contain task- or set-level counts, means, and diagnostics. The study uses the ten development tasks as its sampling units.

| Report result | Public input and level of verification |
|---|---|
| Local study, Figure 1 | [Local model study](../local_model_study/README.md) provides the numeric inputs and plotting command. The published [SVG](figures/local_unified_sensitivity.svg) and [PNG](figures/local_unified_sensitivity.png) are included here. |
| Study A supplementary local metrics | [80 set records](data/main_local_metrics_posthoc_20261115.json) yield the task-balanced means for the MiniLM partition proxy, functional classes, and novelty measures. |
| Study B grouping repeatability | [Four arm summaries](data/study_b_grouping_by_arm.csv) and [ten task-level pooled counts per arm](data/study_b_pooled_task_counts.csv) transcribe the source tables. The pooled counts reproduce the four reported means and ceiling frequencies. The arm summaries supply the reported 67/80 exact ten-response repeat matches. |
| Studies C and D | The [DeepSeek](data/temperature_instruction_20261116.json) and [Nemotron](data/second_generator_20261116.json) snapshots supply the nine conditions, fifteen contrasts and analysis rules for each generator. [Task-level means](data/cross_generator_task_counts.csv) reconstruct the planned count comparisons and their published intervals. |
| Smaller-generator screen (Study F) | [Screen summary](data/small_generator_screen_20261117.json) records valid support, screening outcomes, missingness bounds and accounting. Of 200 planned outputs per model, Llama 3.1 8B returned 198, Llama 3.2 3B returned 196 and Qwen3.5 9B returned 190. This screen ended without a selected candidate. |
| Post hoc opening reuse | [Set counts and analysis](supplements/opening_reuse/README.md) reproduce opening-match rates, paired differences and intervals for Studies A, C and D. |

## Recalculate the reported group-count contrasts

```sh
python3 -BS ESTR4998/interim_report/replay_results.py
```

The script checks all 30 count contrasts against the [published estimates](data/cross_generator_contrasts.csv). For each generator it applies the declared contrast weights to the per-task condition means and uses the frozen 5,000-draw paired bootstrap. It verifies 30 nominal 95% intervals and ten approximate 99% primary-family intervals. At temperature 0.8 the naive-minus-default differences are +2.550 groups for DeepSeek (95% interval +1.000 to +4.325) and +1.175 for Nemotron (−0.325 to +2.575). Each estimate averages ten tasks. The script also prints the Study A supplementary means and observed coverage for each Study F generator.

## Rebuild the tables and figures

The following commands write into a temporary copy. They require Python 3.11 or later and Matplotlib. Verification used Python 3.13.7 and Matplotlib 3.10.6. All six regenerated factorial CSVs and the five rendered PDF, PNG or SVG files matched the published versions byte for byte in that environment. Matplotlib versions and fonts can affect rendered bytes.

```sh
work=$(mktemp -d)
cp -R ESTR4998/interim_report "$work/report"
export MPLCONFIGDIR="$work/mpl"
python3 -B "$work/report/figures/generate_factorial_summary.py"
python3 -B "$work/report/figures/generate_cross_generator_summary.py"
python3 -B "$work/report/figures/generate_task_effects.py"
diff -r ESTR4998/interim_report/data "$work/report/data"
diff -r ESTR4998/interim_report/figures "$work/report/figures"
```

The [task-level figure](figures/task_effects.pdf) compares naive and default at temperature 0.8. The [cross-generator summary](figures/cross_generator_summary.pdf) and [DeepSeek factorial summary](figures/temperature_instruction_summary.svg) show additional frozen contrasts.

## Verify the opening and screen calculations

```sh
(cd ESTR4998/interim_report/supplements/opening_reuse && python3 -BS analyze.py | diff -u RESULTS.md -)
python3 -BS -m unittest discover -s ESTR4998/interim_report/supplements/opening_reuse -q
python3 -BS -m unittest discover -s ESTR4998/interim_report -p 'test_small_generator_screen_summary.py' -q
```

The opening table matched exactly. Five opening tests and twelve screen-summary tests passed. The screen tests use synthetic inputs to exercise missing-data bounds and support counters. The source exports omit raw texts, individual ratings, per-request identifiers and private execution histories. The public snapshots preserve scientific estimates, settings and original source digests. Original per-response judgments would be needed to rerun automatic grouping from scratch.
