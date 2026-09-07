# Common tasks. `make help` lists them.
#
# Every target here is something that otherwise gets retyped from memory with a
# slightly different flag each time, which is how a "passing" test run turns out
# to have skipped the live tests.

.DEFAULT_GOAL := help
VENV := .venv/bin
SYMBOLS ?= SPY

.PHONY: help install test test-live lint dry-run train backtest publish dashboard deploy clean

help:  ## Show this help
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  \033[35m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Create the venv and install everything, Python and Node
	python3 -m venv .venv
	$(VENV)/pip install -q --upgrade pip
	$(VENV)/pip install -q -r requirements.txt
	cd dashboard && npm install

test:  ## Run the full suite (offline tests only skip without credentials)
	$(VENV)/python -m pytest -q

test-live:  ## Run only the tests that hit the live Alpaca paper API
	$(VENV)/python -m pytest -q -m alpaca

lint:  ## Ruff over the Python, tsc over the dashboard
	$(VENV)/python -m ruff check .
	cd dashboard && npx tsc --noEmit

dry-run:  ## Full pipeline against your account, no orders possible
	$(VENV)/python main.py --dry-run --once --publish --symbols $(SYMBOLS)

train:  ## Fit the HMM and exit
	$(VENV)/python main.py --train-only --symbols $(SYMBOLS)

backtest:  ## Walk-forward with all three benchmarks
	$(VENV)/python main.py --mode backtest --symbols $(SYMBOLS) --compare --export

publish:  ## Write demo data for the dashboard
	$(VENV)/python main.py --publish-demo

dashboard:  ## Terminal dashboard from the saved snapshot
	$(VENV)/python main.py --dashboard

deploy:  ## Build, push to GitHub, deploy to Vercel
	./scripts/deploy.sh

clean:  ## Remove build artefacts and caches, never state or models
	rm -rf .pytest_cache **/__pycache__ dashboard/.next dashboard/out
