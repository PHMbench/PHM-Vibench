# Data Model: PHM 2025+ Literature Integration

## LiteratureEntry

Fields:

- `id`: Stable repository-local identifier, unique.
- `year`: Publication year, integer, must be >= 2025.
- `title`: Paper or work title, unique after case/space normalization.
- `authors`: Short author string suitable for README references.
- `venue`: Journal, conference, preprint server, or publisher source.
- `url`: Primary source URL or DOI URL.
- `doi`: DOI when available; may be empty when the source is arXiv or a publisher
  page without a DOI in the searched metadata.
- `task_family`: PHM task family such as `fault_diagnosis`, `rul`,
  `domain_generalization`, `few_shot`, `anomaly_detection`,
  `health_indicator`, `explainability`, or `phm_agent`.
- `method_family`: Coarse method family such as `cnn`, `transformer`, `mamba`,
  `contrastive`, `few_shot`, `domain_adaptation`, `diffusion`,
  `physics_informed`, `llm`, `distillation`, `signal_processing`,
  `graph`, or `uncertainty`.
- `repo_surface`: Existing or candidate PHM-Vibench surface, e.g.
  `task_factory.DG`, `task_factory.FS`, `Components.contrastive_losses`,
  `model_factory.Transformer`, `model_factory.X_model`, or
  `unsupported.runtime`.
- `support_status`: One of `represented`, `candidate-baseline`,
  `literature-only`, `dependency-blocked`, or `unsupported`.
- `notes`: Short mapping note.

Validation rules:

- `id`, `year`, `title`, `venue`, `url`, `task_family`, `method_family`,
  `repo_surface`, and `support_status` are required.
- `year >= 2025`.
- `id`, normalized `title`, and `url` must be unique.
- `support_status` must use the controlled vocabulary.

## InventoryReport

Fields:

- `total_entries`
- `min_year`
- `max_year`
- `counts_by_task_family`
- `counts_by_method_family`
- `counts_by_support_status`

Validation rules:

- `total_entries >= 50`.
- At least five task families are represented.
- At least eight method families are represented.
