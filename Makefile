PY ?= python

.PHONY: help all data reproduce fast test lint app smoke clean pacing intervention monitor ab api

help:
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/ —/'

data:  ## download the 91 MB DataCo dataset into data/
	$(PY) -c "import kagglehub; print(kagglehub.dataset_download('evilspirit05/datasupplychain'))"
	$(PY) -c "import pathlib, shutil; src = next(pathlib.Path.home().glob('.cache/kagglehub/datasets/evilspirit05/datasupplychain/**/DataCoSupplyChainDataset.csv')); pathlib.Path('data').mkdir(exist_ok=True); shutil.copy(src, 'data/DataCoSupplyChainDataset.csv'); print('copied', src)"

all: reproduce drift intervention pacing ab uplift  ## regenerate every analysis

reproduce:  ## full run: models -> cost table -> results/ -> model bundle
	$(PY) scripts/refresh_cost_analysis.py

fast:  ## same pipeline with tiny models (~30 s), for a quick check
	$(PY) scripts/refresh_cost_analysis.py --fast --no-cache

test:  ## unit tests (no dataset needed)
	$(PY) -m unittest discover -s tests -v

lint:  ## ruff + byte-compile
	$(PY) -m ruff check . || ruff check .
	$(PY) -m compileall -q src scripts app tests

app:  ## run the Streamlit demo (requires `make reproduce` first)
	$(PY) -m streamlit run app/app.py

INPUT ?= /tmp/orders_sample.csv
OUT ?= results/decisions.csv

score:  ## batch-score a CSV (INPUT=orders.csv OUT=plan.csv BUDGET=500000)
	$(PY) scripts/score_orders.py --input $(INPUT) --out $(OUT) $(if $(BUDGET),--budget $(BUDGET),)

drift:  ## monthly PSI drift report (requires data)
	$(PY) scripts/drift_report.py

pacing:  ## multi-period budget pacing simulation
	$(PY) scripts/pacing_simulation.py

intervention:  ## break-even effect and A/B sample size
	$(PY) scripts/intervention_analysis.py

monitor:  ## PSI monitoring (REF=old.csv CUR=new.csv)
	$(PY) scripts/monitor.py --reference $(REF) --current $(CUR)

ab:  ## A/B analysis (simulated if no --input is given)
	$(PY) scripts/ab_analysis.py

uplift:  ## heterogeneous effects + targeting (Qini / cost curve)
	$(PY) scripts/uplift_analysis.py

api:  ## run the HTTP service (requires `make reproduce` first)
	$(PY) -m uvicorn api.main:app --reload --port 8000

smoke:  ## end-to-end on synthetic data, exactly what CI runs
	$(PY) tests/fixtures/make_synthetic_dataco.py /tmp/dataco_small.csv
	$(PY) scripts/refresh_cost_analysis.py --data /tmp/dataco_small.csv --fast --no-cache \
		--outdir /tmp/dataco_results --images /tmp/dataco_images \
		--bundle /tmp/dataco_bundle.joblib

clean:
	rm -rf __pycache__ */__pycache__ .pytest_cache .ruff_cache
