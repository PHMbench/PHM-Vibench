# 2026-09-22: freeze adaptation protocol before TTA algorithms

PHMFactory now validates the scientific protocol vocabulary for source-only controls,
episodic/online/continual TTA, offline/continual SFDA, delayed-label adaptation and
online-supervised continual controls.

B00 adds no adaptation algorithm and no runnable TTA Task. The base TTA YAML is a
protocol fragment only. Target labels are split into an evaluator-only view; ordinary
adaptation code receives no `y`. Hidden domain boundaries suppress `domain_id`, online
streams cannot be replayed as extra epochs, SFDA requires distinct adaptation/evaluation
populations, and delayed labels cannot be released before their declared availability.

A small protocol runtime makes predict-before-update and update-before-predict
observably distinct without implementing Tent, SAR, CoTTA, SHOT, buffers or a new
Trainer. Focused protocol tests run inside the existing Core quality workflow.
