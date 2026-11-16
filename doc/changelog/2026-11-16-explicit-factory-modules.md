# Explicit Factory modules

Pipeline 01 can import a caller-selected Task and dataset registration module before
Data Factory construction, and Model Factory can import a caller-selected model module.
The existing registries, constructors, data loaders and Trainer remain the owners of
their contracts. Missing explicit modules fail; omitting the fields preserves the
built-in paths. A scientific `task.module` remains a separate research execution form.
