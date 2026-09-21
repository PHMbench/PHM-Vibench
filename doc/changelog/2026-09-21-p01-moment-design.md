# P01: diagnose impossible nonzero adoption before observing probabilities

The existing independent `moments` preflight now writes `assessment_design.csv`
and explains its count-only conclusion in `decision.md`. It uses the declared
candidate count, distinct assessment groups per source condition and
`delta_total - delta_shift`, through the same radius function as assessment.

If the zero-variance b-radius reaches 2 in any required condition, that condition
makes positive-coefficient selection impossible under the existing maximum-risk
envelope. Otherwise nonzero adoption is merely not ruled out by counts: power,
candidate quality, independence and target scope remain unestablished. No
candidate, coefficient, radius family or data partition is changed. Source
fitting can still proceed when its other preflight requirements hold.

The two-group minimum now agrees with the existing sample-moment estimator for
both Hoeffding and Bernstein. Empirical or paired plans do not receive a
moment-specific feasibility claim. This is not a new bound or an automatic
alternative rule.

Existing D1/G07 artifacts and sealed test bearings are untouched. The original
D1 did not run this reporting addition. Focused tests cover the count boundary,
its agreement with the unchanged envelope on constructed moments, and report
integration with explicitly stubbed record/support readers; they do not run PHM
training or re-open real data. The original loader/model tests remain separate.
