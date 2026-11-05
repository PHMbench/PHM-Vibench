# Paper implementation ownership

P06 certificate/symbolic execution, P07 bounded-DAG execution and the retired
XOAN protocol have moved to their AI4Engineering-L paper repositories.
The runtime no longer ships their methods, study configuration or tests;
`phmfactory.p06` and the retired model/task import paths have no forwarding
compatibility layer. Current scientific results and protocols are unchanged.

Paper configurations use the existing explicit `task.module` interface.
Run from the paper root with the declared framework dependency installed.
Generic readers, standard baselines, Task/Trainer boundaries and public
configuration remain here. Study callers must migrate before adopting this
runtime revision; the coordinated paper PRs stay Draft until verified.

CORE, contributor guidance, the PR template and the existing migration
checker now enforce ownership. The existing workflow runs on every PR.
Identifiable new paper paths/imports and known returned implementations fail;
reviewers must also check semantic ownership behind generic filenames.
Historical unrelated paper material has not been silently deleted or approved.

This commit records source migration, not scientific performance or a new
baseline qualification. Exact tests and limitations are reported in the PR.
