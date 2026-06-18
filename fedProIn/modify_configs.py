import os
from pathlib import Path
import argparse as ap
import re

experiments = []

NUM_PROTOS = [1]
loss_classes = ["MinimumClassificationErrorLoss"]
loss_args = {
    "MarginDistanceLoss": [
        {"margin": 0.5, "aggregation": "mean", "norm": 2},
    ],
    "MinimumClassificationErrorLoss": [
        {"aggregation": "mean", "distance_norm": 2},
    ],
}

optimizers = ["Adam"]
lrs = [0.0001]

slicers = ["IID", "Dirichlet"]
datasets = ["HAM10000", "matek19"]
num_classes_map = {"HAM10000": 7, "matek19": 15}

# FedProIn-specific parameters
contrastive_margins = [0.5]
feature_divergence_weights = [1.0]
proto_contrastive_weights = [0.1]

# Generate experiments based on combinations of parameters
for dataset in datasets:
    for num_protos in NUM_PROTOS:
        for loss_class in loss_classes:
            for loss_arg in loss_args[loss_class]:
                for optimizer in optimizers:
                    for lr in lrs:
                        for slicer in slicers:
                            for contrastive_margin in contrastive_margins:
                                for feature_divergence_weight in feature_divergence_weights:
                                    for proto_contrastive_weight in proto_contrastive_weights:
                                        margin_text = f"margin {loss_arg['margin']}" if 'margin' in loss_arg else ""
                                        experiment = {
                                            "name": f"FedProIn {'MDL' if loss_class == 'MarginDistanceLoss' else 'MCEL'}, {dataset}, {slicer}, Protos {num_protos} {margin_text}, PCW {proto_contrastive_weight}, FDW {feature_divergence_weight}, CM {contrastive_margin}",
                                            "dataset": dataset,
                                            "fed_mode": True,
                                            "optimizer": optimizer,
                                            "lr": lr,
                                            "slicer": slicer,
                                            "loss_class": loss_class,
                                            "loss_args": loss_arg,
                                            "num_prototypes": num_protos,
                                            "contrastive_margin": contrastive_margin,
                                            "feature_divergence_weight": feature_divergence_weight,
                                            "proto_contrastive_weight": proto_contrastive_weight
                                        }
                                        experiments.append(experiment)

# Sort experiments by name
experiments.sort(key=lambda x: x["name"])
# Add global experiment ID
for i, experiment in enumerate(experiments):
    experiment["EXPERIMENT_ID"] = str(i + 1)


def modify_config(experiment):
    """Modify configs.py based on experiment parameters"""
    
    with open("configs.py", "r") as f:
        config_content = f.read()
    
    with open("configs_backup.py", "w") as f:
        f.write(config_content)
    
    # Update experiment name
    config_content = re.sub(
        r'EXPERIMENT_NAME = ".*?"',
        f'EXPERIMENT_NAME = "{experiment["name"]}"',
        config_content
    )
    
    # Update optimizer class
    if experiment["optimizer"] == "SGD":
        optimizer_class = "torch.optim.SGD"
    elif experiment["optimizer"] == "Adam":
        optimizer_class = "torch.optim.Adam"
    else:
        optimizer_class = "torch.optim.Adam" 
    
    config_content = re.sub(
        r'"optimizer_class":\s*torch\.optim\.\w+,',
        f'"optimizer_class": {optimizer_class},',
        config_content
    )
    
    # Update learning rate
    config_content = re.sub(
        r'"optimizer_args":\s*\{"lr":\s*[0-9.eE+-]+\}',
        f'"optimizer_args": {{"lr": {experiment["lr"]}}}',
        config_content
    )
    
    # Update dataset and matching num_classes
    if "dataset" in experiment:
        config_content = re.sub(
            r'Data_Source_Id = ".*?"',
            f'Data_Source_Id = "{experiment["dataset"]}"',
            config_content
        )
        config_content = re.sub(
            r'num_classes = \d+',
            f'num_classes = {num_classes_map[experiment["dataset"]]}',
            config_content
        )

    # Override rounds for test run
    config_content = re.sub(
        r'NUM_ROUNDS = \d+',
        'NUM_ROUNDS = 2',
        config_content
    )

    # Update slicer
    if "slicer" in experiment:
        config_content = re.sub(
            r'SLICER_NAME = ".*?"',
            f'SLICER_NAME = "{experiment["slicer"]}"',
            config_content
        )
    
    # Update loss class
    config_content = re.sub(
        r'"loss_class":\s*\w+,',
        f'"loss_class": {experiment["loss_class"]},',
        config_content
    )
    
    # Update loss args
    if "loss_args" in experiment:
        loss_args = experiment["loss_args"]
        
        # Format loss_args as a string that matches the config format (with single quotes for strings)
        loss_args_parts = []
        for key, value in loss_args.items():
            if isinstance(value, str):
                loss_args_parts.append(f'"{key}": \'{value}\'')
            else:
                loss_args_parts.append(f'"{key}": {value}')
        
        loss_args_str = "{" + ", ".join(loss_args_parts) + "}"
        
        config_content = re.sub(
            r'"loss_args":\s*\{[^}]*\}',
            f'"loss_args": {loss_args_str}',
            config_content
        )

    # Update number of prototypes
    if "num_prototypes" in experiment:
        config_content = re.sub(
            r'num_prototypes_per_class = \d+',
            f'num_prototypes_per_class = {experiment["num_prototypes"]}',
            config_content
        )
    
    # Update FedProIn-specific parameters
    if "contrastive_margin" in experiment:
        config_content = re.sub(
            r'contrastive_margin = [0-9.eE+-]+',
            f'contrastive_margin = {experiment["contrastive_margin"]}',
            config_content
        )
    
    if "feature_divergence_weight" in experiment:
        config_content = re.sub(
            r'feature_divergence_weight = [0-9.eE+-]+',
            f'feature_divergence_weight = {experiment["feature_divergence_weight"]}',
            config_content
        )
    
    if "proto_contrastive_weight" in experiment:
        config_content = re.sub(
            r'proto_contrastive_weight = [0-9.eE+-]+',
            f'proto_contrastive_weight = {experiment["proto_contrastive_weight"]}',
            config_content
        )

    with open("configs.py", "w") as f:
        f.write(config_content)


def restore_config():
    """Restore original config from backup"""
    try:
        with open("configs_backup.py", "r") as f:
            original_config = f.read()
        with open("configs.py", "w") as f:
            f.write(original_config)
        os.remove("configs_backup.py")
    except FileNotFoundError:
        print("Warning: Could not restore original config - backup not found")


def main():
    """Run all experiments sequentially"""
    
    print("Starting batch experiment runner for FedProIn...")
    print(f"Total experiments to run: {len(experiments)}")
    
    # Create experiments directory if it doesn't exist
    Path("experiments").mkdir(exist_ok=True)
    
    arg_parser = ap.ArgumentParser(description="Batch run FedProIn experiments with modified configs")
    # get index from command line argument
    arg_parser.add_argument("--index", type=int, default=0, help="Index of the experiment to run")
    args = arg_parser.parse_args()
    index = args.index
    
    if index < -1 or index >= len(experiments):
        print(f"Invalid index {index}. Must be between -1 and {len(experiments) - 1}.")
        return
        
    if index == -1:
        restore_config()
        return
        
    experiment = experiments[index]

    print(f"Running experiment: {experiment['name']}")
    modify_config(experiment)


if __name__ == "__main__":
    main()
