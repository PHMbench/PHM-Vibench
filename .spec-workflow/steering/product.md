# Product Overview

## Product Purpose
PHM-Vibench delivers a reproducible benchmark platform for industrial vibration-based prognostics and health management. It unifies datasets, models, and evaluation workflows so researchers and engineers can accelerate fault diagnosis and predictive maintenance research without rebuilding infrastructure.

## Target Users
- Industrial AI researchers studying vibration signal diagnostics and prognostics
- Reliability and maintenance engineers who need comparable benchmarks across assets
- Data scientists evaluating algorithm performance for Industry 4.0 deployments
- Academic teams preparing publishable experiments with transparent baselines

## Key Features
1. **Configuration-driven experimentation**: YAML-first experiment definitions remove code changes from routine benchmarking.
2. **Modular factory architecture**: Pluggable data, model, task, and trainer factories enable rapid prototyping of new pipelines.
3. **Comprehensive dataset coverage**: Curated library of 15+ industrial vibration datasets with consistent preprocessing and metadata management.
4. **Foundation and few-shot pipelines**: Built-in industrial signal foundation models, HSE contrastive learning, and few-shot toolkits for low-data regimes.
5. **Unified metric and visualization suite**: Standardized evaluation metrics, logging, and visual analysis artifacts for fair comparisons.

## Business Objectives
- Establish the de facto open benchmark for industrial vibration fault diagnosis
- Shorten the iteration cycle for PHM research by providing turnkey baselines and scripts
- Grow an engaged community that contributes datasets, models, and reproducible results
- Support commercialization pathways by demonstrating reliability and performance gains

## Success Metrics
- **Benchmark adoption**: >= 10 active partner labs or companies running PHM-Vibench pipelines each quarter
- **Reproducibility rate**: >= 95% of published baselines successfully re-run from configuration files
- **Coverage growth**: Add >= 3 new datasets or model backbones per release cycle
- **Community contributions**: >= 1 externally contributed pipeline or factory module per quarter

## Product Principles
1. **Reproducibility first**: Every experiment must be runnable from a single configuration entry point with documented outputs.
2. **Composable by default**: New capabilities plug into factories without modifying existing pipelines.
3. **Industry-grounded evaluation**: Metrics, visualizations, and reporting align with maintenance decision needs.
4. **Transparency over convenience**: Preserve explicit configuration and logging even when automation could hide complexity.

## Monitoring & Visibility (if applicable)
- **Dashboard Type**: Web-based Streamlit UI for experiment orchestration and monitoring
- **Real-time Updates**: Manual refresh with optional logging hooks; extendable to event-based updates through Lightning callbacks
- **Key Metrics Displayed**: Training/validation accuracy, domain generalization scores, latency, and resource usage
- **Sharing Capabilities**: Exportable logs, metrics JSON, and figure artifacts stored under `save/` for team review

## Future Vision
PHM-Vibench will evolve into a full PHMbench ecosystem hub, spanning data stewardship, foundation model training, and collaborative evaluation.

### Potential Enhancements
- **Remote Access**: Provide secure tunneling or hosted deployment options so stakeholders can launch Streamlit dashboards without local setup.
- **Analytics**: Embed historical trend analysis, ablation dashboards, and automated reporting for submission-ready summaries.
- **Collaboration**: Introduce multi-user experiment tracking, comment threads on runs, and artifact sharing across partner organizations.
