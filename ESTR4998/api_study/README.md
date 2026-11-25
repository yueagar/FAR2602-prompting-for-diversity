# API study implementation

This directory contains the implementation used to generate response sets, ask independent evaluators to group or score them, and compute the development-study analyses. The report uses the number of groups in a ten-response set as an automatic diversity measure. A group count measures the evaluator's grouping decision. Its repeatability and correspondence with human judgments are separate questions.

| Research step | Implementation |
|---|---|
| Generation, sampling settings, request budgets and provenance | [API request builder](api_runner.py), [client](research_client.py), [run journal](api_runtime.py), [scheduling foundation](runner_foundation_v1.py) |
| Task-sensitive prompt rendering, anonymized order, parsing, judge selection and quality checks | [Evaluator](api_evaluation.py), [adjudication schema](adjudication.py) |
| Study A eligibility checks, development screen and exploratory comparisons | [Study runner](study_runner.py), [headroom checks](fresh_headroom.py), [exploratory comparison](exploratory_main.py), [report calculations](api_report.py) |
| Study B repeat grouping and pooled-set sensitivity | [Rejudging design](rejudge_saved.py), [partition analysis](rejudge_analysis.py) |
| Studies C and D temperature-by-instruction designs and task bootstrap | [First generator](factorial_experiment.py), [second generator](second_generator_experiment.py), [shared analysis](factorial_analysis.py) |

## Offline verification

From the repository root, run the synthetic checks with Python 3.11 or later and NumPy installed. Verification used Python 3.13.7 and NumPy 2.1.3. The [shared environment pins](../local_model_study/requirements.txt) also cover the local study and plotting tools. The fake client constructs temporary study records and returns fixed synthetic completions. The tests cover grouping-label invariance, shared task resampling, missing observations, source binding, budget stops and retry behavior.

```sh
python3 -B -m unittest discover -s ESTR4998/api_study/tests -q
```

The 46 tests passed in the submission environment. To reproduce results on published study data, use the [report analysis and figure instructions](../interim_report/README.md). That numeric replay uses Python's standard library. The [local study](../local_model_study/README.md) supplies the shared constraint and metric definitions.

## Scope and credit

The two JSON configuration templates illustrate schemas and placeholder settings. The frozen design for each factorial is recorded in the corresponding aggregate's `analysis_protocol`. Route discovery, live execution and the original authoring and qualification workflows require source materials outside the submission package. Private development fixtures, response texts, individual human sheets, execution journals and credentials are excluded.

The sharing edition uses public study paths and role names for historical approval fields. The scheduling foundation adapts the second contributor's runner. The report provides the authors' contributions. The upstream NoveltyBench evaluator templates and response-rendering logic retain their [MIT license and attribution](LICENSE-NOVELTYBENCH). The original aggregate source and plan digests identify the scientific inputs independently of these packaging edits.
