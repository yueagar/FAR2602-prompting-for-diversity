# FAR2602 · Guiding LLMs for Novel Text Generation

Code and numerical evidence accompanying the ESTR4998 interim report by Ka Leung Yue and Yu Hin Tang, supervised by Prof. Farzan Farnia at The Chinese University of Hong Kong. The report asks whether changing an instruction can make ten responses to the same task express a wider range of ideas. The local-model study, the API studies and the two-task search trial use development tasks.

## Verify the interim submission

From the repository root, run the following with Python 3.13. The default check uses only the standard library. It verifies the release file hashes, rebuilds the DeepSeek four-instruction comparison and its automatic safeguards, verifies 30 temperature and cross-generator contrasts from their task means, and reproduces the opening-reuse table.

```sh
python3 -B ESTR4998/verify.py
```

For the full table and figure replay, install the versions recorded for the local study. The replay runs on CPU and writes generated files to a temporary directory. It compares three local-study CSVs, four sensitivity artifacts, six API-study CSVs and seven figure files with their archived counterparts. Byte-identical figures were checked on Linux with Python 3.13.7 and the pinned plotting stack. The replay uses no model credentials.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r ESTR4998/local_model_study/requirements.txt
.venv/bin/python -B ESTR4998/verify.py --full
```

## Find the evidence

| Report component | Submission files | What can be checked |
|---|---|---|
| Small-model study | [Local Qwen study](ESTR4998/local_model_study/) | Reanalyze 210 measured response sets across seven conditions, compare paired intervals and rebuild Figure 1. The human-agreement file gives anonymous aggregate counts. |
| Four DeepSeek instructions (Study A) | [Numeric response-set summaries](ESTR4998/evidence/study_a/) | Recompute group counts, differences versus default, quality and constraint guards, ceiling counts and the sum of the recorded request costs. |
| Saved-response re-evaluation, temperature comparison and second generator (Studies B–D) | [Report data and figures](ESTR4998/interim_report/) and [API study code](ESTR4998/api_study/) | Inspect Study B's aggregate arm and task counts. Recompute 30 C/D contrasts and their intervals from task means. Regenerate the tables and plots. Inspect the grouping, scoring and task-paired analysis code. |
| Two-task automatic search and subsequent check | [Offline search](ESTR4998/offline_search_prototype/), [adaptive demo](ESTR4998/prompt_search_demo/), [continuation](ESTR4998/prompt_search_continuation/), [completion](ESTR4998/prompt_search_completion/) and [screen](ESTR4998/screen_completion/) | Replay synthetic runs to inspect the selection procedure. The completion notes give the historical result on fresh responses to the two development tasks. |
| Smaller-generator screen (Study F) | [Screen summary](ESTR4998/interim_report/data/small_generator_screen_20261117.json) | Inspect coverage and missing-data bounds for Llama 3.1 8B, Llama 3.2 3B and Qwen3.5 9B. |
| Opening reuse (Studies A, C and D) | [Opening-reuse analysis](ESTR4998/interim_report/supplements/opening_reuse/) | Recompute the post-hoc table from set-level counts. |
| Human evaluation | [Rating editor](ESTR4998/human_evaluation_editor/) | Inspect the annotation interface and try it with invented examples. |

The public replay starts at recorded numerical measurements or report summaries, as specified for each study. Repeating generation, independently scoring the original texts or repeating the human rating exercise requires inputs beyond this release. The task-level intervals describe the sampled development tasks. The [second-semester directory](ESTR4999/) is reserved for the final submission.

For a version-specific citation, use the GitHub permalink to the commit shown by `git rev-parse HEAD`. The first submission snapshot is archived under `estr4998-interim-2026-11-25`. The [MIT license](LICENSE) and [third-party notices](THIRD_PARTY.md) describe attribution.
