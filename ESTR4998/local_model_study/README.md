# Local Qwen study

Reproduce the Section 4 automatic-measure tables and Figure 1 from saved measurements on ten development tasks. The seven conditions each contribute three sets of ten responses per task. The released input has 210 rows of set-level measurements for the 2,100 generation observations. It counts the 900 observations shared by the two source runs once.

| Artifact | What to inspect |
|---|---|
| [Unified analysis](supplements/20261117_unified_local/) | The 210 numeric cells, three full-precision CSVs, task bootstrap and tests. |
| [Interval sensitivity](supplements/20261117_interval_sensitivity/) | The paired Student t calculation, source and result hashes, and Figure 1 plotting code. |
| [Model and decoding configurations](configs/) and [method implementations](src/localstudy/) | The two configurations used in the study, rendering of the added instruction, classifier, embedding metrics, automatic quality rubric and deterministic constraint checks. |
| [Human agreement](reports/HUMAN_RATING_SUMMARY_20261105.md) | Anonymous counts and published intervals for the separate human-rating exercise. |

## Reproduce the numbers and Figure 1

Run from this directory with Python 3.13.7 and the packages in [requirements.txt](requirements.txt). The replay and tests run offline on CPU. The commands compare the reconstructed CSVs, paired-interval summary and generated figure against the archived values.

```sh
work=$(mktemp -d)
u=supplements/20261117_unified_local
i=supplements/20261117_interval_sensitivity
python3 "$u/analyze.py" --output-dir "$work/unified"
cmp "$work/unified/arm_means.csv" "$u/arm_means.csv"
cmp "$work/unified/task_metrics.csv" "$u/task_metrics.csv"
cmp "$work/unified/contrasts.csv" "$u/contrasts.csv"
python3 "$i/analyze.py" --output-dir "$work/interval"
cmp "$work/interval/summary.json" "$i/summary.json"
MPLCONFIGDIR="$work/mpl" python3 "$i/plot.py" \
  --input "$work/interval/summary.json" --output-dir "$work/figures"
cmp "$work/figures/local_unified_sensitivity.png" ../interim_report/figures/local_unified_sensitivity.png
cmp "$work/figures/local_unified_sensitivity.svg" ../interim_report/figures/local_unified_sensitivity.svg
python3 -m pytest -q --import-mode=importlib -p no:cacheprovider tests \
  "$u/test_analyze.py" "$i/test_interval_sensitivity.py"
```

The unified analysis averages repetitions within a task, then gives each eligible task equal weight. It uses 20,000 paired task-bootstrap draws with seed 20261117. Each of 11 contrasts has 30 measurements. Six tasks have countable constraints and ten support the other measures. The sensitivity adds nominal paired t intervals on the same task differences. Its three figure panels were chosen retrospectively.

For the naive instruction versus default, the mean change in functional classes per ten responses is −0.067 (bootstrap 95% interval −0.93 to +1.10). Reference-relative novelty increases by 0.068 (+0.030 to +0.109). The evidence for moving away from the default reference pool is clearer than the evidence for a wider spread of central ideas. Automatic quality remains unvalidated.

## Provenance and scope

The analysis code and original CSVs come from thesis commit `674b954cc011c1c1fb30c3cc1a3681fd625ae767`. The release projects each saved development cell onto condition, task ID, repetition and numeric metrics. This changes the input hash embedded in the newly generated unified summary. All three unified CSVs reproduce byte for byte. [The sensitivity plot manifest](supplements/20261117_interval_sensitivity/plot_manifest.json) records the expected PNG and SVG hashes for the original figure.

The method code is supplied for inspection. The task-specific constraint map covers the six relevant development tasks and uses descriptive labels. The YAML files retain the recorded model revisions, generation settings and original split filename. The method package and example output paths use release-specific names. The release starts after generation and measurement. Prompt collections, raw responses, embeddings, model weights and individual human ratings are omitted. The classifier method follows NoveltyBench v1.0. Its [software notice](LICENSE-NOVELTYBENCH) applies to the upstream code.
