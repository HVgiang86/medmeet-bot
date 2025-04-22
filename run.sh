#!/bin/bash

echo "=================================="
echo "MedBot - Setup and Launch Script"
echo "=================================="

# Check if Poetry is available
if ! command -v poetry &> /dev/null; then
    echo "Poetry is not found. Please install Poetry before running this script."
    echo "You can install it using: curl -sSL https://install.python-poetry.org | python3 -"
    exit 1
else
    echo "Poetry is available."
fi

# Check for .env file
if [ ! -f ".env" ]; then
    echo "Warning: .env file not found."
    echo "Creating a basic .env file..."
    touch .env
    echo "# Add your environment variables here" > .env
fi

# Install dependencies using Poetry
echo "Installing project dependencies..."
poetry install
if [ $? -ne 0 ]; then
    echo "Failed to install dependencies. Exiting..."
    exit 1
fi
echo "Dependencies installed successfully."

# Check if Ollama is running
echo "Checking if Ollama is running..."
if ! nc -z localhost 11434 &> /dev/null; then
    echo "Ollama is not running. Please start Ollama before proceeding."
    exit 1
else
    echo "Ollama is running."
fi

# Check if required models are available
echo "Checking if required models are available..."
poetry run python -c "import os; os.system('ollama list | grep \"llama3.2\\|nomic-embed-text\"')"

# Start the Streamlit application
echo "Starting Streamlit application..."
poetry run streamlit run app.py

echo "Application has stopped."