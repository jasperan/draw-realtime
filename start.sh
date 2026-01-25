#!/bin/bash
# StreamDiffusion Real-Time Demo Launcher

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}StreamDiffusion Real-Time Demo${NC}"
echo "================================"

# Check conda environment
if [[ -z "${CONDA_PREFIX}" ]] || [[ "${CONDA_DEFAULT_ENV}" != "streamdiffusion" ]]; then
    echo -e "${YELLOW}Activating streamdiffusion environment...${NC}"
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate streamdiffusion
fi

# Create directories if needed
mkdir -p videos engines

# Check if frontend is built
if [ ! -d "frontend/build" ]; then
    echo -e "${YELLOW}Frontend not built. Building now...${NC}"
    cd frontend
    if [ ! -d "node_modules" ]; then
        npm install
    fi
    npm run build
    cd ..
fi

echo ""
echo -e "${GREEN}Starting server...${NC}"
echo -e "URL: ${GREEN}http://localhost:7860${NC}"
echo ""
echo "First run will download models and compile TensorRT engines."
echo "This may take 5-10 minutes. Subsequent runs will be instant."
echo ""
echo "Press Ctrl+C to stop"
echo ""

python -m app.main
