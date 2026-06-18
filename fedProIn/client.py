from sleeping_fed.humans.torchee import TorchClient
from sleeping_fed.nightmares.MDL import MarginDistanceLoss

from tqdm import tqdm
import numpy as np
import time
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import json

from bhaang.Medical_imaging.logger import TextLogger, CSVLogger
from bhaang.Medical_imaging.metrics import get_metrics_classification

from configs import Metadata_file, Data_Source_Id, EXPERIMENT_ID, EXPERIMENT_NAME, dataloader_args
from configs import contrastive_margin, feature_divergence_weight, proto_contrastive_weight

class FeatureDivergenceLoss(nn.Module):
    """
    Calculate the mean squared error loss between local and global features.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, local_features: torch.Tensor, global_features: torch.Tensor) -> torch.Tensor:
        """
        Calculate the mean squared error loss between local and global features.
        
        :param local_features: Local features from the local model.
        :param global_features: Global features from the global model.
        :return: Mean squared error loss.
        local_features and global_features should be of shape (batch_size, feature_dim)
        """
        if local_features.shape != global_features.shape:
            raise ValueError("Local and global features must have the same shape.")
        
        loss = F.mse_loss(local_features, global_features)
        return loss

class PrototypeDivergenceLoss(nn.Module):
    """
    Calculate the prototype divergence loss.
    intuition : global prototypes have more knowledge than prev local prototypes.
    margin: float
        Margin for the contrastive loss.
    aggregation: str
        Aggregation method for the loss ('mean' or 'sum').
    """
    def __init__(self, margin: float = 1.0, aggregation: str = 'mean') -> None:
        super().__init__()
        self.margin = margin
        self.aggregation = aggregation
        if aggregation not in ['mean', 'sum']:
            raise ValueError("Aggregation must be either 'mean' or 'sum'.")
    
    def forward(self, neg_prototypes: torch.Tensor, pos_prototypes: torch.Tensor, local_prototypes: torch.Tensor) -> torch.Tensor:
        """
        Calculate the contrastive loss.
        
        :param neg_prototypes: Negative prototypes prev (t-1) local prototypes.
        :param pos_prototypes: Positive prototypes global prototypes.
        :param local_prototypes: Local prototypes from the current batch.
        prototype shape: ( num_classes, num_prototypes_per_class, feature_dim)
        """
        change_from_prev = neg_prototypes - local_prototypes
        change_from_global = pos_prototypes - local_prototypes

        # make it in eclidean distance
        change_from_prev = torch.linalg.norm(change_from_prev, dim=-1)
        change_from_global = torch.linalg.norm(change_from_global, dim=-1)

        normalized_change = (change_from_global - change_from_prev) / (change_from_global + change_from_prev + 1e-8)
        
        # Log statistics for debugging zero loss
        # print(f"[PrototypeDivergenceLoss Debug]")
        # print(f"  change_from_prev: mean={change_from_prev.mean().item():.6f}, min={change_from_prev.min().item():.6f}, max={change_from_prev.max().item():.6f}")
        # print(f"  change_from_global: mean={change_from_global.mean().item():.6f}, min={change_from_global.min().item():.6f}, max={change_from_global.max().item():.6f}")
        # print(f"  normalized_change: mean={normalized_change.mean().item():.6f}, min={normalized_change.min().item():.6f}, max={normalized_change.max().item():.6f}")
        # print(f"  normalized_change + margin: mean={(normalized_change + self.margin).mean().item():.6f}, min={(normalized_change + self.margin).min().item():.6f}, max={(normalized_change + self.margin).max().item():.6f}")
        
        loss = F.relu(normalized_change + self.margin)
        
        # Count how many elements are zero after ReLU
        zero_count = (loss == 0).sum().item()
        total_count = loss.numel()
        # print(f"  Zero elements after ReLU: {zero_count}/{total_count} ({100*zero_count/total_count:.2f}%)")
        
        if self.aggregation == 'mean':
            final_loss = loss.mean()
            # print(f"  Final loss (mean): {final_loss.item():.6f}")
            return final_loss
        elif self.aggregation == 'sum':
            final_loss = loss.sum()
            # print(f"  Final loss (sum): {final_loss.item():.6f}")
            return final_loss
        else:
            raise ValueError("Aggregation must be either 'mean' or 'sum'.")

class LocalClient(TorchClient):
    """
    A client for non-yet-named algorithm.
    This client inherits from TorchClient and implements required methods.
    """
    def __init__(self, client_id, model, train_data, val_data, train_args=None, test_args=None, workspace_path=None, ACK=None, Debug=False):
        """
        Initializes the LocalClient with the given parameters.
        
        :param client_id: Unique identifier for the client.
        :param model: The model to be trained.
        :param train_data: Training dataset.
        :param val_data: Validation dataset.
        :param train_args: Arguments for training.
        :param test_args: Arguments for testing.
        :param workspace_path: Path to the workspace.
        :param ACK: Acknowledgment handler for managing messages.
        """

        super().__init__(client_id, workspace_path, model, train_data, val_data, ACK, train_args, test_args)
        self.Debug = Debug
        self.logger = TextLogger(file_name=f"{self.workspace_path}/client_{self.client_id}/log.txt")
        self.csv_logger = CSVLogger(file_name=f"{self.workspace_path}/client_{self.client_id}/metrics.csv")
        
        self.source_dataloader_name = Data_Source_Id

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.global_prototypes = None
        self.previous_prototypes = None
        self.global_model = deepcopy(self.model)
        self.global_model.to(self.device)

        self.optimizer = self.train_args["optimizer_class"](self.model.parameters(), **self.train_args["optimizer_args"])
        self.criterion = self.train_args['loss_class'](**self.train_args['loss_args'])

        self.feature_div_mse_loss = FeatureDivergenceLoss()
        self.proto_div_contrastive_loss = PrototypeDivergenceLoss(margin=contrastive_margin, aggregation='mean')

        if self.criterion is None:
            self.logger.log("Warning: No loss function provided, using default MarginDistanceLoss")
            self.criterion = MarginDistanceLoss(margin=0.2, aggregation='mean', norm=2)

        self.metric_calculation_mode = self.train_args['metric_calculation_mode']

        if self.Debug:
            self.logger.log(f"[DEBUG] Client {client_id} initialized in DEBUG mode.")
            self.logger.log(f"[DEBUG] Using device: {self.device}")

        self.train_label_counts, self.val_label_counts = self._get_labels_counts_()
        self.num_samples = len(train_data.dataset)

        
        if self.Debug:
            self.logger.log(f"[DEBUG] Total training samples: {self.num_samples}")
            self.logger.log(f"[DEBUG] Label distribution: {self.train_label_counts}")


        self.logger.log(f"Client {client_id} has been initialized")
        self._initialize_csv_columns()
        self.trained_model = False

        self.test_data = None

    def _initialize_csv_columns(self):
        """Initialize all CSV columns that will be used for logging."""
        self.csv_logger.create_column('epoch')
        self.csv_logger.create_column('loss') 
        self.csv_logger.create_column('accuracy')
        self.csv_logger.create_column('f1')
        self.csv_logger.create_column('recall')
        self.csv_logger.create_column('precision')
        self.csv_logger.create_column('confusion_matrix')
        self.csv_logger.create_column('source')
        self.csv_logger.create_column('mode')

    def _get_labels_counts_(self):
        if self.Debug:
            self.logger.log(f"[DEBUG] Getting label counts for client {self.client_id}.")

        train_labels = []
        for i in range(len(self.train_data.dataset)):
            _, label = self.train_data.dataset[i]
            train_labels.append(label.cpu().numpy())
        
        train_labels = np.array(train_labels)
        train_label_count = np.unique(train_labels, return_counts=True)

        if self.Debug:
            self.logger.log(f"[DEBUG] Client {self.client_id} - Total training samples: {len(train_labels)}")
            self.logger.log(f"[DEBUG] Client {self.client_id} - Label counts: {train_label_count}")
        
        val_labels = []
        for i in range(len(self.val_data.dataset)):
            _, label = self.val_data.dataset[i]
            val_labels.append(label.cpu().numpy())
        val_labels = np.array(val_labels)
        val_label_count = np.unique(val_labels, return_counts=True)
        if self.Debug:
            self.logger.log(f"[DEBUG] Client {self.client_id} - Total validation samples: {len(val_labels)}")
            self.logger.log(f"[DEBUG] Client {self.client_id} - Validation label counts: {val_label_count}")

        return train_label_count, val_label_count
    
    
    def train_local(self):
        """
        Train the model on the local dataset.
        """
        self.trained_model = True
        if self.Debug:
            self.logger.log("Starting training")


        for param in self.global_model.parameters():
            param.requires_grad = False
        
        for epoch in tqdm(range(self.train_args['epochs'])):
            self.model.train()

            train_loss, train_metric, influence = self._train_one_epoch()

            if self.Debug:
                self.logger.log(f'Epoch: {epoch+1}/{self.train_args["epochs"]} | Loss: {train_loss:.4f} | Metric: {train_metric} | source: {self.source_dataloader_name}')

            log_dict = {
                'epoch': epoch + 1,
                'loss': train_loss,
                'accuracy': train_metric["accuracy"],
                'f1': train_metric["f1"],
                'recall': train_metric["recall"],
                'precision': train_metric["precision"],
                'confusion_matrix': train_metric["confusion_matrix"],
                'source': self.source_dataloader_name,
                'mode': "train"
            }
            self.csv_logger.log(log_dict)

        
        self.csv_logger.save()
        
        # Send completion message to server
        self.communication_handler.send_message(
            receiver_id="server", 
            content={
                    "success": True, 
                    "message": "Training completed", 
                    "num_samples": self.num_samples,
                    "influence": influence
                    }, 
            message_type="train_completion", 
            metadata={"client_id": self.client_id}
        )

        return train_loss, train_metric["f1"]
    
    def test_local(self):
        if self.Debug:
            self.logger.log("Starting testing")
        
        val_loss, val_metric = self._validate_one_epoch()
        test_loss, test_metric = self._validate_one_epoch(mode="test")

        self.csv_logger.log({
            'epoch': "eval", 
            'loss': val_loss, 
            'accuracy': val_metric["accuracy"], 
            'f1': val_metric["f1"],
            'recall': val_metric["recall"], 
            'precision': val_metric["precision"], 
            'confusion_matrix': val_metric['confusion_matrix'],
            'source': self.source_dataloader_name,
            "mode": "val"
        })

        self.csv_logger.log({
            'epoch': "eval", 
            'loss': test_loss, 
            'accuracy': test_metric["accuracy"], 
            'f1': test_metric["f1"],
            'recall': test_metric["recall"], 
            'precision': test_metric["precision"], 
            'confusion_matrix': test_metric['confusion_matrix'],
            'source': self.source_dataloader_name,
            "mode": "test"
        })

        self.communication_handler.send_message(
            receiver_id="server",
            message_type="test_completion",
            content={
                "message" : "testing_complete",
                "success": True,
                "num_samples": len(self.val_data.dataset)
            }
        )

        self.csv_logger.save()
        return val_loss, val_metric["f1"]
    
    def _train_one_epoch(self):
        """
        Trains the model for one epoch.
        
        :return: The training loss and metrics.
        """
        self.global_model.to(self.device)
        self.model.to(self.device)

        total_loss = 0
        y_true = np.array([])
        y_pred = np.array([])
        total_influence = None

        for imgs, labels in self.train_data:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()

            local_features, local_prototypes = self.model(imgs)

            need_feature_div = feature_divergence_weight > 0
            need_proto_div = proto_contrastive_weight > 0 and self.previous_prototypes is not None

            global_features = None
            global_prototypes = None

            if need_feature_div or need_proto_div:
                with torch.no_grad():
                    global_features, global_prototypes = self.global_model(imgs)

            loss, influence, predicted = self.criterion(local_features, local_prototypes, labels)

            total_batch_loss = loss

            if need_feature_div and global_features is not None:
                feature_divergence_loss = self.feature_div_mse_loss(local_features, global_features)
                total_batch_loss = total_batch_loss + feature_divergence_weight * feature_divergence_loss
                if self.Debug:
                    self.logger.log(f"Feature divergence loss: {feature_divergence_loss.item()}")
            else:
                feature_divergence_loss = None

            if need_proto_div:
                if global_prototypes is None:
                    with torch.no_grad():
                        _, global_prototypes = self.global_model(imgs)
                proto_divergence_loss = self.proto_div_contrastive_loss(
                    neg_prototypes=self.previous_prototypes.detach(),
                    pos_prototypes=global_prototypes.detach(),
                    local_prototypes=local_prototypes
                )
                total_batch_loss = total_batch_loss + proto_contrastive_weight * proto_divergence_loss
                if self.Debug:
                    self.logger.log(f"Prototype divergence loss: {proto_divergence_loss.item()}")
            else:
                proto_divergence_loss = None

            if self.Debug:
                self.logger.log(f"Loss: {loss.item()}")

            total_batch_loss.backward()
            self.optimizer.step()

            total_loss += total_batch_loss.item()
            
            y_true = np.concatenate((y_true, labels.cpu().numpy()), axis=0)
            y_pred = np.concatenate((y_pred, predicted.cpu().numpy()), axis=0)
            
            if total_influence is None:
                total_influence = influence.clone()
            else:
                total_influence += influence
        
        train_loss = total_loss / len(self.train_data)
        train_metric = get_metrics_classification(y_true, y_pred, mode=self.metric_calculation_mode)
        
        return train_loss, train_metric, total_influence
    
    def _validate_one_epoch(self, mode="val"):
        """
        Validates the model for one epoch.
        
        :param mode: 'val' for validation, 'test' for testing.
        :return: The validation loss and metrics.
        """
        self.model.eval()

        total_loss = 0
        y_true = np.array([])
        y_pred = np.array([])
        total_influence = None
        data_loader = self.val_data if mode == "val" else self.test_data
        for imgs, labels in data_loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            with torch.no_grad():
                local_features, local_prototypes = self.model(imgs)
                
                loss, influence, predicted = self.criterion(local_features, local_prototypes, labels)

                y_true = np.concatenate((y_true, labels.cpu().numpy()), axis=0)
                y_pred = np.concatenate((y_pred, predicted.cpu().numpy()), axis=0)
                total_loss += loss.item()
                if total_influence is None:
                    total_influence = influence.clone()
                else:
                    total_influence += influence
        val_loss = total_loss / len(data_loader)
        val_metric = get_metrics_classification(y_true, y_pred, mode=self.metric_calculation_mode)
        return val_loss, val_metric
        
    def calculate_metrics(self, mode="val"):
        _, test_metric = self._validate_one_epoch(mode=mode)
        self.last_eval_metrics = test_metric
    
    def load_model(self, model_path):
        #check if "server" is in model_path
        global_flag = False
        if "server" in str(model_path):
            global_flag = True
            self.logger.log(f"Loading model from server path: {model_path}")
        
        if global_flag:
            if self.trained_model:
                for imgs, _ in self.train_data:
                    # take the first image and get the local prototypes
                    img = imgs[0].to(self.device)
                    with torch.no_grad():
                        _, local_prototypes = self.model(img.unsqueeze(0))
                    self.previous_prototypes = local_prototypes.clone().detach()
                    break
                self.logger.log("Stored previous local prototypes before loading global model.")
                self.trained_model = False
                
            self.global_model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.global_model.to(self.device)
            self.logger.log(f"Global model updated with the model from server path: {model_path}")
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)