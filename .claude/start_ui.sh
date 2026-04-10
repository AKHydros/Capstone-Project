#!/bin/sh
cd "/Users/alexkatzighera/Documents/Capstone Project"
PYTHONPATH=src API_BASE_URL=http://127.0.0.1:8000 exec .venv/bin/python3 -m streamlit run src/ui/app.py --server.port 8501 --server.address 127.0.0.1
