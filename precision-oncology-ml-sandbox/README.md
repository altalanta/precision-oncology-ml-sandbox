# Precision Oncology ML Sandbox

Deterministic, end-to-end ML sandbox for precision oncology across modalities:

- Genomics (cfDNA methylation) with batch/age confounding
- Imaging (H&E-like tiles) with benign vs malignant labels
- NLP (synthetic literature + synthetic EHR notes) with BM25 RAG
- RL/Active-learning toy environment for cost-aware assay selection

Non‑negotiables satisfied:
- No external datasets; all data generated with fixed seeds
- Reproducible Hydra configs; MLflow logs to `./artifacts/mlruns`
- Clean Python 3.11, docstrings, type hints, CLI entry points
- Fast tests (`pytest`), CI via GitHub Actions, pre‑commit (`black`, `ruff`)
- Dockerfile builds a CPU image for data gen + baseline training

## Quickstart

Prereqs: Python 3.11, pip, optionally Docker.

- Install: `pip install -e .[dev]`
- Pre-commit: `pre-commit install`
- Generate all data: `make data`
- Train baselines: `make train`
- Run tests: `make test`

Hydra configs live in `configs/`. MLflow tracks to `file:./artifacts/mlruns` (created on demand).

## CLI Examples

- Genomics: `python -m pon_ml.genomics.train`
- Imaging: `python -m pon_ml.imaging.train`
- NLP RAG: `python -m pon_ml.nlp.rag --query "CAR-T intraperitoneal delivery" --top_k 3`
- RL: `python -m pon_ml.rl.train`

## AI Affordances → Oncology Workflows

- Biomarker modeling (cfDNA methylation) → feature confounding control, group K-Fold by batch
- Digital pathology (H&E-like tiles) → CNN training + Grad-CAM inspection
- Literature/EHR synthesis → RAG pipeline with BM25 and negation handling
- Adaptive testing → cost-aware assay selection via ε-greedy Q-learning

## Repo Structure

- `src/pon_ml/` — code organized by modality and utilities
- `configs/` — Hydra configs per task
- `tests/` — fast tests for shapes, leakage, RAG relevance, RL env
- `artifacts/` — MLflow runs and saved figures (gitignored)

## Reproducibility

`pon_ml.seeds` fixes seeds for `random`, `numpy`, and `torch` (if available).
Hydra ensures configs are logged; MLflow captures params/metrics/artifacts.

## Docker

Build CPU image and run jobs inside: `docker build -t pon-sandbox .`

## License

MIT

