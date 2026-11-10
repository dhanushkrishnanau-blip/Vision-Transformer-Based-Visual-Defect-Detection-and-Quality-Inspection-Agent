# Vision Transformer-Based Visual Defect Detection and Quality Inspection Agent

This capstone project explores industrial visual quality inspection: it learns the appearance of normal products from good images, detects and highlights possible defects, and may later use an AI agent to prepare inspection tickets and reports. The first baseline uses Anomalib's PatchCore model with the bottle category of the MVTec AD dataset; later AI-agent features are not included yet.

## How to set up

Run these commands from the project root in PowerShell. The dataset is not included or downloaded automatically; place the MVTec AD files so that `data\mvtec\bottle` exists before running the baseline.

```powershell
# Create an isolated Python environment for this project.
py -3.13 -m venv .venv

# Activate the environment in this PowerShell window.
.\.venv\Scripts\Activate.ps1

# Install the Python packages listed for the project.
python -m pip install -r requirements.txt

# Train and evaluate the PatchCore baseline.
python detector\baseline_anomalib.py
```

The first package installation may take a while because PyTorch is large. PatchCore can run without a GPU, but training and testing will be slower on CPU.
