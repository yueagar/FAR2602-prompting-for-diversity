# Fixed-library instruction search

The selector ranks a predeclared library on supplied response groupings and per-response quality scores. It weights tasks equally and selects the highest-diversity instruction that meets the paired point-estimate margins. Exact ties use instruction ID. The winner is written before confirmation begins. A separate evaluator can check the frozen candidate on the confirmation samples.

Run from the repository root with Python 3.10 or newer. The selector creates each output directory and refuses to overwrite an existing one.

```sh
work=$(mktemp -d)
python3 ESTR4998/offline_search_prototype/search.py demo --output "$work/library"
python3 ESTR4998/offline_search_prototype/search.py replay \
  --config "$work/library/config.json" --records "$work/library/replay.json" \
  --output "$work/library-replay"
```

Both commands should print `"winner": "focused"` and 32 completed callback evaluations. The fabricated group counts for that instruction are 3 in selection, 2 in confirmation and 1 under the check evaluator. The displayed US$0.032 is a test value. Both commands make zero service calls. For the experiment with generated proposals, see the [two-task adaptive search](../prompt_search_demo/README.md).

To run the tests, use `python3 -B -m unittest discover -s ESTR4998/offline_search_prototype`. They cover replay identity, budgets, incomplete cells, frozen selection and tie-breaking.
