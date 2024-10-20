#!/bin/bash

# Create a new virtual environment
python3 -m venv test_env

# Activate the virtual environment
source test_env/bin/activate

# Install the dependencies from requirements.txt
pip install -r requirements.txt

# Test installation by running a simple Python script to verify key modules are installed
python -c "
try:
    import langchain
    import chromadb
    import openai
    import PyPDF2
    import numpy
    print('All dependencies are installed correctly.')
except ImportError as e:
    print(f'Missing module: {e}')
"

# Keep the environment active for further use
exec $SHELL
