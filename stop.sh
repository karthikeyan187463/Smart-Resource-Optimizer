#!/bin/bash
# Script to stop application

echo "Stopping Flask and Streamlit apps..."

# Kill flask app
pkill -f "python3 predictor.py"

# Kill streamlit app
pkill -f "streamlit run streamlit_app.py"

echo "All apps stopped."
