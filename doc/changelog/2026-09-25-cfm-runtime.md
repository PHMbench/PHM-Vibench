# CFM: exact sampler steps and current public-stage regression

The Euler sampler previously accepted `num_steps=2.5`, executed two calls, but used a
step size divided by 2.5. With zero initial state and constant unit velocity on [0,1],
it returned 0.8 rather than 1.0. Non-integer/boolean counts and non-finite time bounds
now fail before model evaluation. Valid integer-step arithmetic is unchanged.

Seventeen analytic PyTorch cases cover step counts, exact constant-field endpoints,
float32/float64, invalid inputs, and input/mode preservation. The isolated source-slice
run changed from 9 failures / 8 passes to 17 passes. This is not full-checkout evidence.

The existing three-stage CPU regression now uses `trainer.devices` rather than the
rejected `trainer.gpus` alias, fixes its subprocess working directory, and checks sample
shape/dtype and condition lengths. The existing CFM workflow includes this actual
train/checkpoint/sample/eval test and the compiled-config adapter. Path filters include
the demo and its shared runtime dependencies. No test is skipped to recover a pass.

The first full-install CI attempt stopped in collection because CPU torch was combined
with an incompatible torchvision wheel. The workflow now installs the matching pair
from the official CPU index before the normal editable install. Current-head CI remains
the authority for full integration; the first failure is retained in Actions.

No new model family, normalization algorithm, scientific metric, hash/ledger framework,
registry promotion, real PHM data run, release, or main-branch change is included.
