.PHONY: help setup check demo-infer demo-app

PYTHON ?= python3
PROJECT_DIR := ccpd_alpr

help:
	@echo "Available targets:"
	@echo "  make setup      # install dependencies and package"
	@echo "  make check      # quick syntax check"
	@echo "  make demo-infer # run inference on demo cases"
	@echo "  make demo-app   # launch web UI"

setup:
	cd $(PROJECT_DIR) && $(PYTHON) -m pip install -U pip
	cd $(PROJECT_DIR) && $(PYTHON) -m pip install -r requirements.txt
	cd $(PROJECT_DIR) && $(PYTHON) -m pip install -e .

check:
	$(PYTHON) -m compileall $(PROJECT_DIR)/ccpd_alpr

demo-infer:
	cd $(PROJECT_DIR) && $(PYTHON) scripts/infer.py \
		--source demo_cases \
		--output-dir runs/infer_demo

demo-app:
	cd $(PROJECT_DIR) && $(PYTHON) app.py --host 0.0.0.0 --port 7860
