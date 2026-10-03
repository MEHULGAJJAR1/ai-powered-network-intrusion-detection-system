PYTHON ?= python
DATA ?= data/raw/kddcup.data_10_percent.gz

.PHONY: install download train evaluate run test clean
install:
	$(PYTHON) -m pip install -r requirements.txt

download:
	$(PYTHON) scripts/download_kdd.py --output $(DATA)

train:
	$(PYTHON) train.py --data $(DATA)

evaluate:
	$(PYTHON) evaluate.py --data $(DATA)

run:
	$(PYTHON) app.py

test:
	$(PYTHON) -m pytest -q

clean:
	rm -rf .pytest_cache __pycache__ nids/**/__pycache__ tests/__pycache__
