import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sleeping_fed.sleep_slicer.knife import PyTorchDatasetSlicer
from sleeping_fed.angels.agni import TorchServer

from pathlib import Path
import os
import shutil
from copy import deepcopy
import numpy as np
import time

from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from tqdm import tqdm
import json
import pandas as pd
from datetime import datetime, timedelta

from bhaang.Medical_imaging.metrics import get_metrics_classification

from configs import EXPERIMENT_ID, EXPERIMENT_NAME, NUM_CLIENTS, KING_PORTION, train_args, dataloader_args, Fraction_evaluate, Data_Source_Id
from configs import WORKSPACE, NUM_ROUNDS, slicer, EXCEL_PATH, SLICER_NAME, DEBUG, SAVE_DIR, Metadata_filepath, Fraction_fit

from client import LocalClient
from model import Resnet18_model
from utils import process_and_plot_history, plot_client_label_distribution

from sleeping_fed.meghdoot.meghdoot import AcknowledgementHandler, CommunicationHandler
from sleeping_fed.nightmares.MDL import MarginDistanceLoss
from sleeping_fed.angels.agni import TorchServer
from sleeping_fed.sleep_slicer.knife import PyTorchDatasetSlicer

from FedProIn import FedProIn

from data.ham10000 import pytorch_dataset as ham10000_pytorch_dataset
from data.matek19 import pytorch_dataset as matek19_pytorch_dataset
from data.make_clients import make_client

def run_fed(train_dataset, val_dataset, test_dataset, dataloader_args, train_labels):
    try:
        if os.path.exists(EXCEL_PATH):
            df_1 = pd.read_excel(EXCEL_PATH, sheet_name="Client_Data")
            df_2 = pd.read_excel(EXCEL_PATH, sheet_name="Client_Metrics")
            df_3 = pd.read_excel(EXCEL_PATH, sheet_name="Server_Metrics")
        else:
            raise FileNotFoundError
    except (FileNotFoundError, ValueError):
        df_1 = pd.DataFrame()
        df_2 = pd.DataFrame()
        df_3 = pd.DataFrame()
        print("Excel file not found or empty. Creating a new one.")
    
    train_pytorchslicer = PyTorchDatasetSlicer(
        dataset=train_dataset,
        slicer_obj=slicer,
        labels=train_labels,
        num_slices=NUM_CLIENTS,
    )
    val_pytorchslicer = PyTorchDatasetSlicer(
            dataset=val_dataset,
        num_slices=NUM_CLIENTS,
        min_slice_size=10,
        king_portion=KING_PORTION,
        shuffle=False,
        seed=42,
    )

    test_pytorchslicer = PyTorchDatasetSlicer(
        dataset=test_dataset,
        num_slices=NUM_CLIENTS,
        min_slice_size=10,
        king_portion=KING_PORTION,
        shuffle=False,
        seed=42,
    )

    def eval_fun(model, selected_clients, test_loader, criterion, metric_calculation_mode):
        model.eval()
        all_labels = []
        all_preds = []
        test_loader = test_loader
        total_loss = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Testing global model"):
                images, labels = images.to(device), labels.to(device)
                if labels.dim() > 1:
                    labels = labels.squeeze(1) 
                features, prototypes = model(images)
                loss, influence, predicted = criterion(features, prototypes, labels)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                
                total_loss += loss.item()
        
        loss = total_loss / len(test_loader)
        metrics = get_metrics_classification(all_labels, all_preds, mode=metric_calculation_mode)
        metrics["test_loss"] = loss

        return metrics
    
    eval_fn_arg = {
        'test_loader': DataLoader(test_dataset, **dataloader_args),
        'criterion': train_args['loss_class'](**train_args['loss_args']),
        'metric_calculation_mode': "weighted",
    }

    fed_pro_in = FedProIn(
        fraction_fit=Fraction_fit,
        fraction_eval=Fraction_evaluate,
        min_fit_clients=2,
        min_eval_clients=2,
        eval_fn=eval_fun,
        eval_fn_arg=eval_fn_arg,
        accept_failure=True,
        model=Resnet18_model,
        seed=42,
        aggregation_mode="bn1_bn2",
        debug=DEBUG,
    )

    server = TorchServer(
        num_clients=NUM_CLIENTS,
        num_rounds=NUM_ROUNDS,
        strategy=fed_pro_in,
        workspace=WORKSPACE,  
        train_data=train_dataset,
        val_data=val_dataset,
        train_args= train_args,
        model=Resnet18_model,
        test_args=None,
        client_class= LocalClient,
        train_data_slicer= train_pytorchslicer,
        val_data_slicer= val_pytorchslicer,
        data_loader_args= dataloader_args,
        debug=DEBUG,
        load_server_weight_before_training_round=True,
        wandb_log=False,
        Experiment_Name=EXPERIMENT_NAME,
        Experiment_ID=EXPERIMENT_ID,
        make_client_fn= make_client,
    )

    clients = server.clients

    for client in clients:
        client.test_data = test_pytorchslicer.get_loader(client.client_id, **dataloader_args)

    client_dist = plot_client_label_distribution(clients, experiment_path=SAVE_DIR)
    server.fit()

    df_1 = pd.concat([df_1, client_dist], ignore_index=True)

    history = server.history.history

    server_df, client_df = process_and_plot_history(history, experiment_path=SAVE_DIR)
    df_2 = pd.concat([df_2, client_df], ignore_index=True)
    df_3 = pd.concat([df_3, server_df], ignore_index=True)

    client_0 = server.clients[0]
    client_0.load_model(server.model_dir)

    client_0.val_data = DataLoader(val_dataset, **dataloader_args)
    #final_loss, final_metric = client_0.test_local()

    client_0.calculate_metrics()
    val_metric = client_0.last_eval_metrics

    print(f"Client 0 validation metrics: {val_metric}")
    server_val_row = {
        "experiment": f"{EXPERIMENT_NAME}_Val",
        "slicer": SLICER_NAME,
        "experiment_id": EXPERIMENT_ID,
        "round": NUM_ROUNDS,
        **val_metric
    }
    df_3 = pd.concat([df_3, pd.DataFrame([server_val_row])], ignore_index=True)

    client_0.val_data = DataLoader(test_dataset, **dataloader_args)
    #final_loss, final_metric = client_0.test_local()

    client_0.calculate_metrics()
    test_metric = client_0.last_eval_metrics

    # add val metrics to Metadata file
    with open(Metadata_filepath, 'r') as f:
        metadata = json.load(f)

    metadata[-1]['val_metrics'] = val_metric
    metadata[-1]['test_metrics'] = test_metric

    with pd.ExcelWriter(EXCEL_PATH) as writer:
        df_1.to_excel(writer, sheet_name="Client_Data", index=False)
        df_2.to_excel(writer, sheet_name="Client_Metrics", index=False)
        df_3.to_excel(writer, sheet_name="Server_Metrics", index=False)
    
    with open(Metadata_filepath, 'w') as f:
        json.dump(metadata, f, indent=4)

def main():
    """
    Main function to run the FedAvg experiment.
    Initializes the server and clients, runs the training rounds, and logs the results.
    """    
    start_time = time.time()
    if Data_Source_Id.lower() == "ham10000":
        pytorch_dataset = ham10000_pytorch_dataset
    elif Data_Source_Id.lower() == "matek19":
        pytorch_dataset = matek19_pytorch_dataset
    else:
        raise ValueError(f"Unsupported dataset: {Data_Source_Id}")
    train_dataset = pytorch_dataset(split="train", as_rgb=True, image_size=128)
    val_dataset = pytorch_dataset(split="val", as_rgb=True, image_size=128)
    test_dataset = pytorch_dataset(split="test", as_rgb=True, image_size=128)

    train_labels = []
    for i in range(len(train_dataset)):
        _, label = train_dataset[i]
        train_labels.append(label)
    
    run_fed(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        dataloader_args=dataloader_args,
        train_labels=train_labels,
    )

    clients_dir = os.listdir(WORKSPACE)
    today_date = datetime.today().strftime("%Y-%m-%d")
    yesterday_date = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")

    log_csv_today = f"metrics_{today_date}.csv"
    log_csv_yesterday = f"metrics_{yesterday_date}.csv"

    for client_dir in clients_dir:
        if "client" in client_dir:
            client_path = WORKSPACE / client_dir

            # Check today's log first, else yesterday
            for log_csv_file in [log_csv_today, log_csv_yesterday]:
                log_csv_path = client_path / log_csv_file
                save_csv_path = SAVE_DIR / f"{EXPERIMENT_ID}_{EXPERIMENT_NAME}" / client_dir / log_csv_file

                if os.path.exists(log_csv_path):
                    os.makedirs(save_csv_path.parent, exist_ok=True)  # ensure directory exists
                    shutil.move(log_csv_path, save_csv_path)
                    print(f"Moved: {log_csv_path} → {save_csv_path}")
                    break  # stop checking after first match
            else:
                print(f"No log file found for {client_dir}")
        else:
            # server directory
            # move model.pt to SAVE_DIR
            model_dir = WORKSPACE/ client_dir / "model.pt"
            if os.path.exists(model_dir):
                os.makedirs(SAVE_DIR / f"{EXPERIMENT_ID}_{EXPERIMENT_NAME}" / "model", exist_ok=True)
                save_model_path = SAVE_DIR / f"{EXPERIMENT_ID}_{EXPERIMENT_NAME}" / "model" / "model.pt"
                shutil.move(model_dir, save_model_path)
                print(f"Moved: {model_dir} → {save_model_path}")
            else:
                print("No model checkpoint found in server directory.")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")
    print("Experiment completed successfully.")
    # # clean up workspace
    # if os.path.exists(WORKSPACE):
    #     shutil.rmtree(WORKSPACE)
    #     print(f"Cleaned up workspace: {WORKSPACE}")
    

if __name__ == "__main__":
    main()