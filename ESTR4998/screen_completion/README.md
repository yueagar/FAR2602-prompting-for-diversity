# Completing the default-generator screen

The exploratory screen compared generators using the default instruction. It was separate from the adaptive search. The final schedule recorded 584/600 generations, 584/600 individual quality scores and 156/180 groupings, with 1,360 attempts costing US$1.7057639313. The schedule exhausted all eligible jobs. Its denominators include the planned observations that remained missing.

[schedule.py](schedule.py) reconstructs the scheduler decision from a supplied plan and attempts. It distinguishes future eligible requests from observations made permanently unavailable by exhausted or ineligible generations. An exhausted generation also blocks its dependent quality and grouping jobs. The cost summary counts every attempt and retains the reserve against an unresolved charge. The release accepts the one specifically authorized retry ID as a parameter so a synthetic plan can exercise the same rule.

```sh
python3 -B -m unittest discover -s ESTR4998/screen_completion -v
```

The tests create a small fictional schedule and check dependency handling, the extended retry allowance, missing costs and failed-attempt spending. They use the [API study](../api_study/) retry predicate. The production restart bindings and account-specific supervision are outside this release.
