# utils.py
import matplotlib.pyplot as plt
import numpy as np
from configs import EXPERIMENT_ID, EXPERIMENT_NAME,  SLICER_NAME, num_classes
import pandas as pd
import seaborn as sns
from pathlib import Path



def plot_client_label_distribution(clients, experiment_path):
    """
    Generates and saves a figure showing the label distribution for training
    and validation data for each client using bar plots. It also returns a DataFrame with
    detailed statistics.

    Args:
        clients (list): A list of client objects, each with label count attributes.
        experiment_path (str): The directory path to save the output plot.

    Returns:
        pd.DataFrame: A DataFrame containing detailed statistics for each
                      client's training and validation data.
    """
    training_label_counts = [client.train_label_counts for client in clients]
    validation_label_counts = [client.val_label_counts for client in clients]
    client_ids = [client.client_id for client in clients]
    num_clients = len(client_ids)

    client_stats = []
    for i, client in enumerate(clients):
        client_id = client.client_id
        train_counts = training_label_counts[i]
        val_counts = validation_label_counts[i]

        train_labels, train_counts_data = train_counts[0], train_counts[1]
        train_num_samples = sum(train_counts_data)
        train_unique_labels = len(train_labels)
        train_count_dict = dict(zip(train_labels, train_counts_data))
        train_row = {
            'experiment': EXPERIMENT_NAME,
            'slicer': SLICER_NAME,
            'experiment_id': EXPERIMENT_ID,
            'client_id': client_id,
            'num_samples': train_num_samples,
            'unique_labels': train_unique_labels,
            'data_type': 'training'
        }
        for label_id in range(num_classes):
            train_row[f'label_{label_id}_cnt'] = train_count_dict.get(label_id, 0)
        client_stats.append(train_row)

        val_labels, val_counts_data = val_counts[0], val_counts[1]
        val_num_samples = sum(val_counts_data)
        val_unique_labels = len(val_labels)
        val_count_dict = dict(zip(val_labels, val_counts_data))
        val_row = {
            'experiment': EXPERIMENT_NAME,
            'slicer': SLICER_NAME,
            'experiment_id': EXPERIMENT_ID,
            'client_id': client_id,
            'num_samples': val_num_samples,
            'unique_labels': val_unique_labels,
            'data_type': 'validation'
        }
        for label_id in range(num_classes):
            val_row[f'label_{label_id}_cnt'] = val_count_dict.get(label_id, 0)
        client_stats.append(val_row)

    # Create subplots with bar charts
    fig, axes = plt.subplots(num_clients, 2, figsize=(12, 6 * num_clients))

    if num_clients == 1:
        axes = np.array([axes])

    for i, (train_counts, val_counts, client_id) in enumerate(zip(training_label_counts, validation_label_counts, client_ids)):
        # Training data plot
        train_labels, train_values = train_counts[0], train_counts[1]
        train_total_samples = sum(train_values)
        
        # Create full count array for all possible labels
        train_counts_full = np.zeros(num_classes)
        for label, count in zip(train_labels, train_values):
            train_counts_full[label] = count
        
        classes = np.arange(num_classes)
        ax_train = axes[i, 0]
        
        # Use seaborn barplot with explicit hue to match reference style
        sns.barplot(x=classes, y=train_counts_full, hue=classes, dodge=False, 
                   ax=ax_train, legend=False)
        
        # Add counts & percentages on top of bars
        for j, c in enumerate(train_counts_full):
            if c > 0:  # Only show text for non-zero counts
                ax_train.text(j, c + train_total_samples*0.01, f"{int(c)}\n({c/train_total_samples:.1%})", 
                            ha="center", va="bottom", fontsize=9)
        
        ax_train.set_title(f"Training Data - Client {client_id}\n(Total Samples: {train_total_samples})", 
                          fontsize=12, weight="bold")
        ax_train.set_xlabel("Labels")
        ax_train.set_ylabel("Count")

        # Validation data plot
        val_labels, val_values = val_counts[0], val_counts[1]
        val_total_samples = sum(val_values)
        
        # Create full count array for all possible labels
        val_counts_full = np.zeros(num_classes)
        for label, count in zip(val_labels, val_values):
            val_counts_full[label] = count
        
        ax_val = axes[i, 1]
        
        # Use seaborn barplot with explicit hue to match reference style
        sns.barplot(x=classes, y=val_counts_full, hue=classes, dodge=False, 
                   ax=ax_val, legend=False)
        
        # Add counts & percentages on top of bars
        for j, c in enumerate(val_counts_full):
            if c > 0:  # Only show text for non-zero counts
                ax_val.text(j, c + val_total_samples*0.01, f"{int(c)}\n({c/val_total_samples:.1%})", 
                           ha="center", va="bottom", fontsize=9)
        
        ax_val.set_title(f"Validation Data - Client {client_id}\n(Total Samples: {val_total_samples})", 
                        fontsize=12, weight="bold")
        ax_val.set_xlabel("Labels")
        ax_val.set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(f"{experiment_path}/label_distribution.png", dpi=200)
    plt.close()

    return pd.DataFrame(client_stats)



def process_and_plot_history(history_dict: dict, experiment_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parses server.history.history into server/client DataFrames and plots metrics.

    Args:
        history_dict: Dictionary of {round_id: {server, clients}} data from FL training.

    Returns:
        Tuple of (server_df, client_df) with metrics per round.
    """
    server_rows = []
    client_rows = []

    for round_id, round_data in history_dict.items():
        if "server" in round_data:
            srv_metrics = round_data["server"].get("eval_metrics", {})
            server_row = {
                "experiment": EXPERIMENT_NAME,
                "slicer": SLICER_NAME,
                "experiment_id": EXPERIMENT_ID,
                "round": round_id
            }
            for metric_name, metric_val in srv_metrics.items():
                server_row[metric_name] = metric_val
            server_rows.append(server_row)

        if "clients" in round_data:
            for client_id, client_data in round_data["clients"].items():
                client_row = {
                    "experiment": EXPERIMENT_NAME,
                    "slicer": SLICER_NAME,
                    "experiment_id": EXPERIMENT_ID,
                    "round": round_id,
                    "client_id": client_id,
                    "train_loss": client_data.get("train_loss"),
                    "train_metric": client_data.get("train_metric"),
                    "val_loss": client_data.get("val_loss"),
                    "val_metric": client_data.get("val_metric")
                }
                client_rows.append(client_row)

    server_df = pd.DataFrame(server_rows).sort_values("round").reset_index(drop=True)
    client_df = pd.DataFrame(client_rows).sort_values(["round", "client_id"]).reset_index(drop=True)

    # Plotting section
    if server_df.empty and client_df.empty:
        print("No data to plot")
        return server_df, client_df

    num_clients = len(client_df['client_id'].unique()) if not client_df.empty else 0
    has_server = not server_df.empty
    total_plots = num_clients + (1 if has_server else 0)

    if total_plots == 0:
        return server_df, client_df
        
    n_cols = min(3, total_plots)
    n_rows = (total_plots + n_cols - 1) // n_cols
        
    # Plot 1: loss plotting client and server
    fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    # Plot 2: Metric plotting client and server
    fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if total_plots == 1:
        axes1 = [axes1]
        axes2 = [axes2]
    elif n_rows == 1:
        axes1 = axes1 if hasattr(axes1, '__len__') else [axes1]
        axes2 = axes2 if hasattr(axes2, '__len__') else [axes2]
    else:
        axes1 = axes1.flatten()
        axes2 = axes2.flatten()

    plot_idx = 0

    if has_server:
        ax = axes1[plot_idx]
        metric_ax = axes2[plot_idx]
        plot_idx += 1

        loss_metric = "Val_loss"
        metric_cols = ["accuracy", "f1"]

        if server_df[loss_metric].notna().any():
            ax.plot(server_df["round"], server_df[loss_metric], label="sum of val loss of all clients", color="C0")
            ax.set_title("Client Val sum V/S Round")
            ax.set_xlabel("Round")
            ax.set_ylabel("Validation Loss")
            ax.legend()
            ax.grid(True, alpha=0.3)

        for col in metric_cols:
            if col in server_df.columns and server_df[col].notna().any():
                metric_ax.plot(server_df["round"], server_df[col], label=f"Server {col}", color="C0")
        metric_ax.set_title("Server Metrics V/S Round")
        metric_ax.set_xlabel("Round")
        metric_ax.set_ylabel("Metric Value")
        metric_ax.legend()
        metric_ax.grid(True, alpha=0.3)

    if not client_df.empty:
        unique_clients = sorted(
            client_df['client_id'].unique()
        )

        for client_id in unique_clients:
            client_ax = axes1[plot_idx]
            metric_ax = axes2[plot_idx]
            plot_idx += 1

            val_loss_metric = "val_loss"
            train_loss_metric = "train_loss"

            client_data = client_df[client_df['client_id'] == client_id]
            
            if 'train_loss' in client_data.columns and client_data['train_loss'].notna().any():
                client_ax.plot(client_data["round"], client_data[train_loss_metric], label="Train Loss", color="C1")
            if 'val_loss' in client_data.columns and client_data['val_loss'].notna().any():
                client_ax.plot(client_data["round"], client_data[val_loss_metric], label="Val Loss", color="C2")

            client_ax.set_title(f"Client {client_id} Loss V/S Round")
            client_ax.set_xlabel("Round")
            client_ax.set_ylabel("Loss")
            client_ax.legend()
            client_ax.grid(True, alpha=0.3)

            if 'train_metric' in client_data.columns and client_data['train_metric'].notna().any():
                metric_ax.plot(client_data["round"], client_data["train_metric"], 
                              label=f"Train Metric", color="C1", marker='o')
            if 'val_metric' in client_data.columns and client_data['val_metric'].notna().any():
                metric_ax.plot(client_data["round"], client_data["val_metric"], 
                              label=f"Val Metric", color="C2", marker='^')

            metric_ax.set_title(f"Client {client_id} Metrics V/S Round")
            metric_ax.set_xlabel("Round")
            metric_ax.set_ylabel("Metric Value")
            metric_ax.legend()
            metric_ax.grid(True, alpha=0.3)

    for i in range(plot_idx, len(axes1)):
        axes1[i].set_visible(False)
    for i in range(plot_idx, len(axes2)):
        axes2[i].set_visible(False)


    fig1.tight_layout()
    fig2.tight_layout()


    fig1.savefig(f"{experiment_path}/{EXPERIMENT_ID}_loss_history.png", dpi=300, bbox_inches='tight')
    fig2.savefig(f"{experiment_path}/{EXPERIMENT_ID}_metric_history.png", dpi=300, bbox_inches='tight')

    plt.close(fig1)
    plt.close(fig2)

    return server_df, client_df