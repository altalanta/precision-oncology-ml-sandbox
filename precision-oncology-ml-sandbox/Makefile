.PHONY: setup data train test lint format precommit

setup:
	pip install -e .[dev]
	pre-commit install

data:
	pon-generate-genomics
	pon-generate-imaging
	pon-generate-nlp
	pon-generate-ehr

train:
	pon-train-genomics
	pon-train-imaging
	pon-rl-train

qa:
	pon-nlp-qa --query "CAR-T intraperitoneal delivery" --top_k 3

test:
	pytest --cov-fail-under=85

lint:
	ruff check .

format:
	black .

precommit:
	pre-commit run --all-files

