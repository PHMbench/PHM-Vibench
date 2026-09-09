# Keep run controls independent of experiment editing

The workspace renders existing runs and batch history before loading the editor's
catalogue or resolving a new template. An empty category, missing template or invalid
configuration can block a new experiment without hiding existing logs and cancellation.
A damaged historical run or unreadable log likewise cannot hide batch controls or the
editor; its original error is displayed separately.

Batch history is rendered separately from batch planning and no longer requires a valid
base YAML. Both views reuse the existing run and batch services. No second scheduler,
backend configuration parser or result authority is introduced.

Regression tests cover editor-catalogue failure, empty groups and rejected templates with
an existing run; cancellation remains addressed to the original run. A paused batch can
still be continued or cancelled while the editor is unusable. Rendering and page reruns
do not submit experiments. No training backend or experiment protocol is changed.
