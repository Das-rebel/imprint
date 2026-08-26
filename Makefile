.PHONY: install test lint collect mine

install:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check imprint/ tests/

collect:  ## Phase 0 step 1 — point at your A3M request log (JSONL)
	python -m imprint.collector $(LOG)

mine:  ## Phase 0 step 2 — rank signatures by savings potential
	python -m imprint.miner
