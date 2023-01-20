#!/bin/bash

streamlit run demo.py &
uvicorn api:app --host 0.0.0.0 --port 8000 --reload