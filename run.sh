#!/bin/bash
# ==============================================================================
# CreditLens AI - Run Script
# Explainable Credit Risk Analysis
# ==============================================================================

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
PYTHON="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"
STREAMLIT="$VENV_DIR/bin/streamlit"

echo "============================================"
echo "  CreditLens AI - Startup"
echo "  Explainable Credit Risk Analysis"
echo "============================================"
echo ""

# ------------------------------------------
# Step 1: Check Python
# ------------------------------------------
echo "[1/6] Checking Python installation..."
if command -v python3 &>/dev/null; then
    SYSTEM_PYTHON="python3"
elif command -v python &>/dev/null; then
    SYSTEM_PYTHON="python"
else
    echo "ERROR: Python not found. Please install Python 3.9+."
    exit 1
fi

PYTHON_VERSION=$($SYSTEM_PYTHON --version 2>&1)
echo "  Found: $PYTHON_VERSION"

# ------------------------------------------
# Step 2: Create virtual environment
# ------------------------------------------
echo ""
echo "[2/6] Setting up virtual environment..."
if [ ! -d "$VENV_DIR" ]; then
    echo "  Creating virtual environment at $VENV_DIR..."
    $SYSTEM_PYTHON -m venv "$VENV_DIR"
    echo "  Virtual environment created."
else
    echo "  Virtual environment already exists."
fi

# ------------------------------------------
# Step 3: Activate and install dependencies
# ------------------------------------------
echo ""
echo "[3/6] Installing dependencies..."
source "$VENV_DIR/bin/activate"

# Upgrade pip quietly
"$PIP" install --upgrade pip --quiet 2>/dev/null

# Install requirements if needed
if [ -f "$PROJECT_DIR/requirements.txt" ]; then
    "$PIP" install -r "$PROJECT_DIR/requirements.txt" --quiet 2>/dev/null
    echo "  Dependencies installed."
else
    echo "  WARNING: requirements.txt not found."
fi

# ------------------------------------------
# Step 4: Configure matplotlib
# ------------------------------------------
echo ""
echo "[4/6] Configuring matplotlib..."
export MPLBACKEND=Agg
echo "  Matplotlib backend set to Agg."

# ------------------------------------------
# Step 5: Verify model artifacts
# ------------------------------------------
echo ""
echo "[5/6] Verifying model artifacts..."
MODELS_DIR="$PROJECT_DIR/models"

if [ -f "$MODELS_DIR/xgb_model.joblib" ] && \
   [ -f "$MODELS_DIR/mlp_model.joblib" ] && \
   [ -f "$MODELS_DIR/preprocessor.joblib" ] && \
   [ -f "$MODELS_DIR/feature_names.joblib" ]; then
    echo "  Model artifacts found. Skipping training."
else
    echo "  Model artifacts missing. Running training..."
    "$PYTHON" "$PROJECT_DIR/train.py"
    echo "  Training complete."
fi

# ------------------------------------------
# Step 6: Start Streamlit
# ------------------------------------------
echo ""
echo "[6/6] Starting CreditLens AI..."
echo ""
echo "============================================"
echo "  CreditLens AI is starting!"
echo "  Open: http://localhost:8501"
echo "============================================"
echo ""

"$STREAMLIT" run "$PROJECT_DIR/app.py" \
    --server.port 8501 \
    --server.headless true \
    --browser.gatherUsageStats false
