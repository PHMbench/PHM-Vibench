# Query validation component — 2026-09-19

The authoritative offline entry is
`src.task_factory.Components.tii_evaluation.evaluate_query_predictions`.
It takes prediction, original-record, support-split, query-split and predeclared
window CSV paths plus an explicit target-to-ordered-local-class map. The five
files must exist and be distinct; inventing an inventory from predictions is not
an admissible caller workflow. File checks alone do not establish when an
inventory was frozen or whether physical group provenance was qualified.

The module requires all declared window fields, including method/seed/checkpoint
and both split IDs, and compares the entire population before any NLL. Every
declared method/seed has the same complete query windows and one checkpoint.
Original records are unique by `(dataset, recording_id)`; predictions are unique
by the frozen composite key. Multiple legitimate windows and multiple records
inside a physical group remain in the estimator. CSV identity strings preserve
leading zeros; labels, seeds, episode and window boundaries are nonnegative
integers. Class-map order determines the corresponding logits column.

After full validation, finite JSON logits/probabilities are checked for class
count, nonnegativity, unit sum and softmax consistency. NLL uses centered logits
and logsumexp, with no probability floor or probability renormalization. Output
tables contain each window NLL and each group's mean over windows then seeds.

Executed from the child repository with the requested environment:

```bash
conda run -n LQ_signal --no-capture-output python -m pytest test/test_tii_evaluation.py -q
```

PASS: **17 tests and 35 subtests**, 7.21 s; five existing import-time
pkg_resources/namespace deprecation warnings. The first run had a test-message
expectation mismatch for CSV `null` logits; the input already failed before NLL.
That assertion was corrected and the command above passed after the change.

Negative tests prove the NLL function is never called for missing/nonfinite/blank
record or group identifiers, support rows, another recording in a support group,
missing/unknown role, missing split file, both arms missing a query group or
window, duplicate composite prediction keys, inconsistent original provenance,
wrong split/checkpoint/run/window identities, and malformed/nonfinite or
inconsistent probability vectors. Positive tests retain multiwindow records,
recompute losses independently, verify class-map ordering and averaging, and
retain NLL=1000 when the true-class probability underflows to zero.

These are generated CSV fixture tests, not industrial predictions or a J1 GO.
No real query statistics, paired bootstrap, episode sensitivity, macro-F1, cost,
plots or training were run by this component task. Those analyses remain outside
this bounded implementation until valid real outputs exist.
