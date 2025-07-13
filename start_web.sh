#!/bin/bash
# LabelForge Web Interface Launcher

echo "🚀 Starting LabelForge Web Interface..."
echo "🌐 Access it at: http://localhost:8501"
echo "💡 Press Ctrl+C to stop"
echo

export PYTHONPATH="/Users/sonishsivarajkumar/Library/Mobile Documents/com~apple~CloudDocs/Personal/code/LabelForge - Snorkal - data labelling/src:$PYTHONPATH"

"/Users/sonishsivarajkumar/Library/Mobile Documents/com~apple~CloudDocs/Personal/code/LabelForge - Snorkal - data labelling/.venv/bin/python" -m streamlit run src/labelforge/web/app.py

echo "👋 LabelForge Web Interface stopped."
