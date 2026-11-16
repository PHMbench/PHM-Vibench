# Explicit checkpoint prediction

Pipeline 01 accepts an existing Lightning checkpoint and an explicit source-training,
validation or test population for a single-device prediction run. It reuses the Data
Factory's selected windows and evaluation sampler, does not call fit or choose a new
checkpoint, and returns the Task's batch predictions as a direct output file. Invalid or
unavailable populations and missing checkpoints fail rather than changing the request.

The Task owns its prediction schema and any required identity fields. This software
capability does not establish access permission, data independence or benchmark validity.
