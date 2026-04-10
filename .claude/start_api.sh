#!/bin/sh
cd "/Users/alexkatzighera/Documents/Capstone Project"
PYTHONPATH=src exec .venv/bin/python3 -m uvicorn backend.api.app:app --host 127.0.0.1 --port 8000 --reload
