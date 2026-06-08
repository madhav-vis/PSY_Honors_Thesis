.PHONY: setup preprocess train evaluate vision annotator help

help:          ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*## "}; {printf "  make %-14s %s\n", $$1, $$2}'

setup:         ## Create venv and install all dependencies
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install -r requirements_vision.txt

preprocess:    ## Run full preprocessing pipeline (steps 1-6)
	.venv/bin/python src/main.py

train:         ## Train all model phases (1-10)
	.venv/bin/python src/train.py

evaluate:      ## Run full evaluation pipeline
	.venv/bin/python src/evaluate.py

vision:        ## Run vision pipeline (set RUN=runs/<name>)
	@test -n "$(RUN)" || (echo "Usage: make vision RUN=runs/<run_name>" && exit 1)
	.venv/bin/python src/vision/vision_main.py --run-dir $(RUN)

annotator:     ## Launch Streamlit annotation tool (port 8502)
	.venv/bin/streamlit run src/vision/stream_annotator.py --server.port 8502
