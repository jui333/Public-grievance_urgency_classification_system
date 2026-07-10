#!/bin/bash
set -e

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Creating virtual environment..."
rm -rf public_grievance_urgency_classification_system_venv
python3 -m venv public_grievance_urgency_classification_system_venv

echo "Installing dependencies..."
public_grievance_urgency_classification_system_venv/bin/pip install --upgrade pip
public_grievance_urgency_classification_system_venv/bin/pip install -r requirements.txt

echo "Configuring Streamlit to skip email prompt..."
mkdir -p ~/.streamlit
if [ ! -f ~/.streamlit/credentials.toml ]; then
    echo -e "[general]\nemail = \"\"" > ~/.streamlit/credentials.toml
fi

echo "Dependencies installed successfully!"
