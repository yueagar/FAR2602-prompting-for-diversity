# Continuing a stopped search

A response reached the output limit during the original service-backed selection. The subsequent amendment excluded the entire candidate from ranking and retained the observation and attempts. The continuation reconstructs both proposal rounds and completed cells, evaluates the remaining eligible candidates, freezes the winner and runs the fresh display. An incomplete display cell has no aggregate. A poor display outcome cannot trigger another proposal or a change of winner.

[workflow.py](workflow.py) contains the original continuation engine. It verifies the parent plan, engine and saved exchanges, checks a truncation receipt, then reuses the observed history. Its public demonstration builds a fictional interrupted parent and its amendment entirely inside a temporary directory.

```sh
python3 -B -m unittest discover -s ESTR4998/prompt_search_continuation -v
```

The tests exercise whole-candidate exclusion, immutable history, guard failures, fresh-display omissions and exact replay. For a complete two-task run from the command line, use the [adaptive search example](../prompt_search_demo/README.md). The [completed study summary](../prompt_search_completion/README.md) gives the observed fresh-display results.
