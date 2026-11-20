# Explicit Factory extensions in Pipeline 06

Pipeline 06 now imports the caller's existing `task.factory_module` before Data
Factory construction in train, sample and eval stages. The same field was already
implemented for Pipeline 01; Pipeline 06 use was rejected by the configuration
schema and its runtime did not import the module. The schema now accepts both
pipelines while preserving the separate
`task.module` research execution boundary.

The change reuses `register_task`, `register_dataset_adapter` and `model.module`.
It adds no reader, trainer, registry, pipeline or model family. Import errors remain
errors before data construction. Built-in CFM configurations retain their existing
resolution when `factory_module` is absent.

The generative demo README records the actual Model/Task sampling contract and the
first four comparison candidates. This hook change adds none of the GAN, VAE or
DDPM model implementations. DDPM already existed in historical branches; its
current-runtime adaptation belongs to separate model work, now proposed with GAN
and VAE in [PR #348](https://github.com/PHMbench/PHM-Vibench/pull/348). The existing
CFM eval stage still uses training data; independent real-data fidelity and diagnosis
utility are not established by this interface extension.
