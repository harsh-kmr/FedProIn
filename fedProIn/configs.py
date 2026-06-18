import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sleeping_fed.sleep_slicer.dirichlet_slicer import DirichletSlicer
from sleeping_fed.sleep_slicer.iid_slicer import IIDSlicer
from pathlib import Path
import os
import json
import torch

from sleeping_fed.nightmares.MDL import MarginDistanceLoss
from sleeping_fed.nightmares.MCEL import MinimumClassificationErrorLoss

base_path = Path(__file__).resolve().parent.parent / "output"
os.makedirs(base_path, exist_ok=True)
Metadata_file = "experiment_metadata.json"
Metadata_filepath = base_path / Metadata_file

Data_Source_Id = "MATEK19"
num_classes = 15

if os.path.exists(Metadata_filepath):
    with open(Metadata_filepath, 'r') as f:
        metadata = json.load(f)
    done_experiments = len(metadata)
    EXPERIMENT_ID = done_experiments + 1
else:
    EXPERIMENT_ID = 1

EXPERIMENT_NAME = "FedProIn MCEL, IID"

WORKSPACE = Path("workspace")
if not os.path.exists(WORKSPACE):
    os.makedirs(WORKSPACE)

SAVE_DIR = base_path / f"Experiment{EXPERIMENT_ID}_{EXPERIMENT_NAME}"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

NUM_CLIENTS = 10
KING_PORTION = False
NUM_ROUNDS = 2
num_prototypes_per_class = 1

Fraction_fit = 0.5  # Fraction of clients to be sampled for training in each round
Fraction_evaluate = 1.0  # Fraction of clients to be sampled for evaluation in each round

SLICER_NAME = "IID" 
# "IID" or "Dirichlet"

IID_SLICER = IIDSlicer(
    num_slices=NUM_CLIENTS,
    min_slice_size=10,
    King_portion=KING_PORTION,
    shuffle=True,
    seed=42,
)

DIRICHLET_SLICER = DirichletSlicer(
    num_slices=NUM_CLIENTS,
    min_slice_size=10,
    alpha=0.5,
    King_portion=KING_PORTION,
    shuffle=True,
    seed=42,
)

# slicer = IID_SLICER
slicer = IID_SLICER if SLICER_NAME == "IID" else DIRICHLET_SLICER

EXCEL_FILE = "results.xlsx"

EXCEL_PATH = base_path / EXCEL_FILE

DEBUG = False

dataloader_args = {
    'batch_size': 128,
    'shuffle': True,
    'num_workers': 4,
    'persistent_workers': False,
}

train_args={
    "epochs": 5,
    "optimizer_class": torch.optim.Adam,
    "optimizer_args": {"lr": 0.0001},
    "loss_class": MinimumClassificationErrorLoss,
    "loss_args": {"aggregation": 'mean', "distance_norm": 2},
    "metric_calculation_mode": "weighted",
}

# FedProIn-specific parameters
contrastive_margin = 0.5
feature_divergence_weight = 1.0
proto_contrastive_weight = 0.1

EXPERIMENT_METADATA = {
    "experiment_id": EXPERIMENT_ID,
    "experiment_name": EXPERIMENT_NAME,
    "num_clients": NUM_CLIENTS,
    "slicer_name": SLICER_NAME,
    "king_portion": KING_PORTION,
    "workspace": str(WORKSPACE),
    "num_rounds": NUM_ROUNDS,
    "data_source_id": Data_Source_Id,
    "loss_class": train_args["loss_class"].__name__,
    "loss_args": train_args["loss_args"],
    "optimizer_class": train_args["optimizer_class"].__name__,
    "learning_rate": train_args["optimizer_args"].get("lr", None),
    "num_prototypes_per_class": num_prototypes_per_class,
    "contrastive_margin": contrastive_margin,
    "feature_divergence_weight": feature_divergence_weight,
    "proto_contrastive_weight": proto_contrastive_weight,
}

if os.path.exists(Metadata_filepath):
    with open(Metadata_filepath, 'r') as f:
        metadata = json.load(f)
else:
    metadata = []
metadata.append(EXPERIMENT_METADATA)
with open(Metadata_filepath, 'w') as f:
    json.dump(metadata, f, indent=4)