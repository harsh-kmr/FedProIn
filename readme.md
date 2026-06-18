# FedProIn

This repository accompanies our MICCAI 2026 submission. It provides the full training pipeline used in our experiments on the HAM10000 and MATEK19 benchmarks.

---

## Repository Structure

```
FedProIn/
├── fedProIn/          # Main source package
│   ├── data/          # Dataset loaders (HAM10000, MATEK19)
│   ├── configs.py     # All hyperparameters and paths — edit this before running
│   ├── main.py        # Single-experiment entry point
│   ├── modify_configs.py  # Batch experiment config generator
│   ├── run.sh         # Batch experiment runner
│   └── ...
├── dataset/           # Place downloaded datasets here (see below)
├── output/            # Experiment results are written here (auto-created)
├── hyperparameter.md  # Full hyperparameter reference
└── requirements.txt
```

---

## Installation

**Step 1.** Create and activate a Python environment:

```bash
conda create -n fedproin python=3.10
conda activate fedproin
```

**Step 2.** Install PyTorch matching your CUDA version — **run this first** so it is not overwritten by other packages.  
See [pytorch.org/get-started](https://pytorch.org/get-started/locally/) for the right command.  
Example for CUDA 13.0:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

**Step 3.** Install the two required packages:

```bash
# Bhaang: logging, and utilities
git clone https://github.com/harsh-kmr/Bhaang
pip install -e Bhaang/

# Federated learning framework
git clone https://github.com/harsh-kmr/sleeping_fed
pip install -e sleeping_fed/
```

**Step 4.** Install remaining dependencies:

```bash
pip install -r requirements.txt
```

---

## Datasets

Datasets are **not included** in this repository. Download them separately and place them as described below. All dataset files go under `dataset/` in the repo root.

Dowload from the below links, then place the files in the dataset directory.
for HAM10000: https://www.kaggle.com/datasets/blueloki/ham10000
For MATEK19: https://www.kaggle.com/datasets/blueloki/matek19


## Configuration

All experiment settings live in [fedProIn/configs.py](fedProIn/configs.py). Open it and set:

| Variable | Description |
|---|---|
| `Data_Source_Id` | `"HAM10000"` or `"matek19"` |
| `EXPERIMENT_NAME` | Human-readable label for the run |
| `NUM_CLIENTS` | Number of federated clients |
| `NUM_ROUNDS` | Global federation rounds |
| `SLICER_NAME` | `"IID"` or `"Dirichlet"` |

See [hyperparameter.md](hyperparameter.md) for the full reference of all tunable parameters.

---

## Running Experiments

### Single run

```bash
cd fedProIn
python main.py
```

### Batch sweep

Edit the parameter grid in [fedProIn/modify_configs.py](fedProIn/modify_configs.py), then:

```bash
cd fedProIn
./run.sh
```

`run.sh` generates all parameter combinations defined in `modify_configs.py` and runs each configuration three times sequentially. Console output for each run is saved under `fedProIn/out/`.

Optional arguments for `run.sh`:

```bash
./run.sh [start_index] [end_index] [num_repeats]
# Example: run experiments 0–4, 2 repeats each
./run.sh 0 4 2
```

---

## Output Structure

All results are written to `output/` in the repo root. Each experiment creates:

```
output/
├── Experiment{ID}_{NAME}/
│   ├── {ID}_{NAME}/
│   │   ├── client_0/
│   │   │   └── metrics_{date}.csv
│   │   ├── client_1/
│   │   │   └── metrics_{date}.csv
│   │   ├── ...
│   │   └── model/
│   │       └── model.pt
│   ├── {ID}_loss_history.png
│   ├── {ID}_metrics_history.png
│   ├── label_distribution.png
├── experiment_metadata.json
└── results.xlsx
```

