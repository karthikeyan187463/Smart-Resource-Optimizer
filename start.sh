#!/bin/bash
# Script to start application

mkdir -p logs

# Run flask app
nohup python3 predictor.py > logs/flask.log 2>&1 &

# Run streamlit app
nohup streamlit run streamlit_app.py --server.port 8501 > logs/streamlit.log 2>&1 &

echo "Flask and Streamlit apps started in background."
echo "Logs are available in ./logs/"
