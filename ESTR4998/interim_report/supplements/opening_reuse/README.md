# Opening reuse across generated responses

The post hoc analysis asks how often two responses to the same task start with the same words. [Set counts](sets.csv) cover 440 sets and 4,400 responses across Studies A, C and D. Each set of ten responses supplies 45 unordered pairs. Its numeric record identifies the study, arm, task and repetition.

The [extractor and analysis](analyze.py) lowercases the first eight tokens recognized by its regular expression. It includes English letter sequences, contractions, hyphenated words and digit sequences. A pair matches on the complete token sequence. A nonempty sequence shorter than eight tokens can match a sequence of the same length. Empty openings never match. Rates average the two repetitions within a task and weight each of ten tasks equally. The paired task bootstrap uses 5,000 draws with seed 20261116. Lexical reuse captures a surface feature and cannot identify semantic equivalence.

From this directory the following commands reproduce the [published table](RESULTS.md) and run five synthetic checks. Python 3.11 or later is sufficient.

```sh
python3 -BS analyze.py | diff -u RESULTS.md -
python3 -BS -m unittest -q test_opening_reuse
```

The table compares all four Study A arms and the nine conditions for each generator in Studies C and D. For example, Study A had 285 shared-opening pairs out of 900 under default and 107 out of 900 under the naive instruction.
