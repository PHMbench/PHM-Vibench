# 2026-09-22: freeze adaptation protocol before TTA algorithms

PHMFactory now validates the scientific protocol vocabulary for source-only controls,
episodic/online/continual TTA, offline/continual SFDA, delayed-label adaptation and
online-supervised continual controls.

B00 adds no adaptation algorithm and no runnable TTA Task. The base TTA YAML is a
protocol fragment only. Target labels are split into an evaluator-only view; ordinary
adaptation code receives no `y`. Physical metadata is positive-allowlisted so repository
label aliases such as `Label_Description`, `fault_type`, or `condition_id` cannot
enter adaptation through a free-form metadata key. Hidden domain boundaries suppress
`domain_id`, online streams cannot be replayed as extra epochs, SFDA requires distinct
adaptation/evaluation populations, and delayed labels cannot be released before their
declared availability.

A small dependency-light protocol helper makes predict-before-update and update-before-predict
observably distinct without importing the maintained classification runtime or implementing
Tent, SAR, CoTTA, SHOT, buffers or a new Trainer. The real adaptation runtime remains B01 work. Focused protocol tests run inside the existing Core quality workflow.

Reviewer follow-up tightened B00 fail-closed behavior: `file_id` is evaluator-only because repository file-number ranges can encode fault class, and reset-dependent protocols are not executable until a real reset lifecycle exists. `episodic_tta` and `domain_reset` therefore remain validated schema choices but cannot be passed through the B00 step executor. These changes prevent a protocol helper from silently behaving as continual adaptation or exposing a target-label proxy.

Final review also keeps `source_only` schema-only in B00. A generic adapter `predict()` call can mutate stateful inference layers when model mode is not guaranteed, so frozen-source execution is deferred to B01, where equality with ordinary frozen inference and state preservation can be tested explicitly.
