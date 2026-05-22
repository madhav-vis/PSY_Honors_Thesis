run:
	.venv/bin/streamlit run src/dashboard.py

annotator:
	.venv/bin/streamlit run src/vision/stream_annotator.py --server.port 8502
