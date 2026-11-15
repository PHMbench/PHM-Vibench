# Config-driven scientific task execution

The ordinary `phmfactory --config <yaml>` command now dispatches
`task.execution: research` to the existing scientific execution path using the
already resolved configuration. `task.phase` selects an explicit supported phase;
`task.options` provides keyword inputs to that task. The existing research command
reuses the same function, retains explicit command-option precedence, and rejects
conflicting phase declarations.

Task module selection, signature binding, failure results, direct outputs and
installed-build reporting stay in one owner. Experimental Pipeline authorization
is checked before executing either entrypoint. Ordinary Factory tasks still use
the same compiled Pipeline lifecycle. Preflight does not execute a task.

The symbolic software fixture now declares its phase and runs with the ordinary
command. Its outputs are software fixture evidence, not industrial results.
No dataset, numerical method, dependency source, result writer, baseline status,
or model-native explanation path is changed.

Focused local checks: 93 tests passed; 13 registered configs passed strict validation;
documentation validation passed. Ordinary and legacy commands produced identical
files/arrays for the existing symbolic CPU software fixture. The maintained Dummy
preflight and demo completed; invalid phase execution failed without creating output.
These checks do not qualify MFPT or any other real-data benchmark.
