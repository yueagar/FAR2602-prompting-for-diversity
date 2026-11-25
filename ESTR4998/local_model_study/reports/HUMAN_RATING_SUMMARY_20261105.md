# First-round human rating aggregates

Two raters independently assessed 100 response pairs for functional equivalence and 100 responses for quality. The protocol required a Krippendorff agreement coefficient of at least 0.667 before drawing a verdict about the corresponding automatic instrument. Every point estimate fell below the gate.

## Pair ratings

Rows and columns represent the two raters. Equivalent means the pair communicates the same central content under the written rubric.

| First rater | Second equivalent | Second different | Total |
|---|---:|---:|---:|
| Equivalent | 7 | 1 | 8 |
| Different | 65 | 27 | 92 |
| Total | 72 | 28 | 100 |

The raters agreed on 34 of 100 pairs. Nominal Krippendorff’s α was −0.368 with a 95% bootstrap interval of [−0.534, −0.198]. The study record reports that item alignment and label polarity were checked. The large asymmetry leaves the classifier’s agreement with a reliable human reference unresolved.

## Quality ratings

| Dimension | Ordinal Krippendorff’s α | Reported 95% interval |
|---|---:|---:|
| Relevance | 0.443 | [0.261, 0.588] |
| Coherence | 0.398 | [0.189, 0.561] |
| Constraints | 0.620 | [0.494, 0.705] |
| Effectiveness | 0.390 | [0.206, 0.549] |

The constraints interval spans the threshold. The protocol used the point estimate to apply the gate.

This reviewer extract transcribes the anonymous totals and reported intervals from the 5 November analysis and Section 4 of the interim report at thesis commit `674b954`. Item-level judgments and pairing keys are outside this release. These totals support a direct check of the contingency table and its agreement count.
