# Two-task adaptive instruction search

The controller evaluates an empty default instruction and a fixed creative instruction. Two rounds of two proposal slots use the earlier score history. On each of two development tasks, the evaluator groups five responses and scores quality and task constraints. Selection maximizes the mean number of groups among candidates within 0.5 quality points and 0.05 constraint-pass proportion of the default. The controller freezes the choice before drawing five fresh samples per task for that choice and the controls.

## Run and replay the synthetic example

Use Python 3.10 or newer on a Unix-like system. The controller uses standard-library file locks. From the repository root, run the commands below. The example supplies its own tasks, proposals, samples and scores. It needs no service, key, dataset or model download.

```sh
work=$(mktemp -d)
python3 ESTR4998/prompt_search_demo/adaptive_demo.py demo --output "$work/demo"
python3 ESTR4998/prompt_search_demo/adaptive_demo.py replay \
  --source "$work/demo" --output "$work/replay"
```

Open `$work/demo/trace.md` to see the proposal history, rejections, selection and fresh display. Both commands should report `proposal-r2-s1` and 18 callback requests. One proposal violates the quality guard and one duplicates an earlier proposal. The winner obtains 5 groups per task in selection and 2 on fresh display, where it fails the quality guard. The fixture deliberately produces these outcomes. Replaying the generated files makes zero new service calls.

[request_plan.py](request_plan.py) constructs the request bodies used in the historical study. After all five generations meet the observation checks, it builds one blinded grouping request and five individual quality requests. The module uses the adjacent [API study](../api_study/) and performs no transport. [test_request_plan.py](test_request_plan.py) exercises the plan with fictional models and answers.

Use `python3 -B -m unittest discover -s ESTR4998/prompt_search_demo` for the controller and request-planning tests. The completed service-backed demonstration used a post-observation amendment and different tasks. Its [aggregate result](../prompt_search_completion/README.md) is reported separately.
