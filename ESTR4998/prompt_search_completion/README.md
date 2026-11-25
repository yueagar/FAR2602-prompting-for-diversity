# Fresh-display completion

## Recorded result

The amended study completed on 17 November 2026. Each row combines two previously used development tasks with five fresh responses per task. Group count is the task mean on a 1–5 scale. Quality is a judge score on a 1–10 scale. Constraint passes count individual responses.

| Instruction | Mean groups | Mean quality | Constraint passes |
| --- | ---: | ---: | ---: |
| Default | 4.5 | 9.9 | 10/10 |
| Fixed creative instruction | 5.0 | 9.3 | 9/10 |
| Selected instruction, round 2 slot 2 | 4.5 | 9.7 | 9/10 |

The selected instruction tied the default on fresh-display group count. Its constraint pass rate fell by 0.10, exceeding the allowed drop of 0.05. It remained the frozen choice. These measurements demonstrate execution on familiar tasks and give no estimate of transfer to unseen tasks.

The last completion step added 18 successful attempts and US$0.023787258. Across the original search, amendment and completion there were **193 distinct request IDs, 196 attempts and US$0.21012400896**, with all charges resolved. These are historical aggregates. The raw observations required to reproduce the historical judgments are outside this code release.

## Offline completion rules

[completion.py](completion.py) checks that inherited requests and prior attempts remain intact, allows a single specified inherited retry and returns only newly covered attempts for incremental accounting. It also exposes the historical parser for server retry hints. The release supplies snapshots in memory and replaces the deployment-specific request identity with an explicit argument. Account-bound orchestration and private file bindings are omitted.

```sh
python3 -B -m unittest discover -s ESTR4998/prompt_search_completion -v
```

All test costs and requests are fabricated. The [two-task example](../prompt_search_demo/README.md) generates a separate synthetic search and replay.
