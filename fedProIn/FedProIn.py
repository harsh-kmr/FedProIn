# FedProIn.py
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from sleeping_fed.dream_weaver.strategy import BaseStrategy
from sleeping_fed.humans.mango_man import BaseClient
from sleeping_fed.meghdoot.meghdoot import CommunicationHandler
from sleeping_fed.pills.state_dict_aggregator import Aggregator


class FedProIn(BaseStrategy):
    """
    Federated Prototype with influence (FedProIn) strategy.
    Combines traditional FedAvg model parameter averaging with prototype influence aggregation.
    Prototypes are updated based on aggregated influence scores from clients.
    """

    def __init__(self,
                 fraction_fit: float = 0.5,
                 fraction_eval: float = 0.5,
                 min_fit_clients: int = 2,
                 min_eval_clients: int = 2,
                 eval_fn: Optional[callable] = None,
                 eval_fn_arg: Optional[Dict[str, Any]] = None,
                 accept_failure: bool = True,
                 model: Any = None,
                 seed: Optional[int] = 42,
                 aggregation_mode: Optional[str] = None,
                 debug: bool = False) -> None:
        """
        Initialize the FedProIn strategy.

        Args:
            fraction_fit (float): Fraction of clients to use for training.
            fraction_eval (float): Fraction of clients to use for evaluation.
            min_fit_clients (int): Minimum number of clients for training.
            min_eval_clients (int): Minimum number of clients for evaluation.
            eval_fn (Optional[callable]): Optional function to evaluate the model on the server side.
            eval_fn_arg (Optional[Dict[str, Any]]): Optional arguments for the evaluation function.
            accept_failure (bool): Whether to accept client failures during training.
            model (Any): The global model with trainable prototypes to be trained.
            seed (Optional[int]): Seed for random number generation.
            aggregation_mode (Optional[str]): Aggregation mode for the Aggregator ('bn1', 'bn2').
            debug (bool): If True, print detailed logs for debugging.
        """
        super().__init__(model=model)
        self.fraction_fit = fraction_fit
        self.fraction_eval = fraction_eval
        self.min_fit_clients = min_fit_clients
        self.min_eval_clients = min_eval_clients
        self.eval_fn = eval_fn
        self.eval_fn_args = eval_fn_arg if eval_fn_arg is not None else {}
        self.accept_failure = accept_failure
        self.seed = seed
        self.aggregation_mode = aggregation_mode
        self.debug = debug
        
        np.random.seed(self.seed)

        # Initialize aggregator for FedAvg
        self.aggregator = Aggregator()
        self.communication_handler: Optional[CommunicationHandler] = None

        # Store prototype dimensions for validation
        if hasattr(model, 'num_classes') and hasattr(model, 'num_prototypes_per_class'):
            self.num_classes = model.num_classes
            self.num_prototypes_per_class = model.num_prototypes_per_class
        else:
            self.num_classes = None
            self.num_prototypes_per_class = None
            if self.debug:
                print("Warning: Model doesn't have prototype dimensions. Will infer from first client update.")

        if self.debug:
            print(f"FedProIn initialized with:\n"
                  f"  fraction_fit={self.fraction_fit}, fraction_eval={self.fraction_eval}\n"
                  f"  min_fit_clients={self.min_fit_clients}, min_eval_clients={self.min_eval_clients}\n"
                  f"  accept_failure={self.accept_failure}, seed={self.seed}, debug={self.debug}\n"
                  f"  aggregation_mode={self.aggregation_mode}\n"
                  f"  num_classes={self.num_classes}, num_prototypes_per_class={self.num_prototypes_per_class}\n")

    def initialize_communication_handler(self, communication_handler: CommunicationHandler) -> None:
        """Initializes the communication handler for the strategy."""
        self.communication_handler = communication_handler
        if self.debug:
            print("FedProIn: Communication handler initialized successfully.")

    def send_test_info(self, client: BaseClient) -> None:
        """
        Sends test request to the client.

        Args:
            client (BaseClient): The client to which the test request will be sent.
        """
        if self.communication_handler is None:
            raise RuntimeError("Communication handler is not initialized.")
        
        self.communication_handler.send_message(
            receiver_id=f"client_{client.client_id}",
            message_type="test_request",
            content={
                "test_request" : True
            }
        )

    def send_train_info(self, client: BaseClient) -> None:
        """
        Sends train request to the client.

        Args:
            client (BaseClient): The client to which the train request will be sent.
        """
        if self.communication_handler is None:
            raise RuntimeError("Communication handler is not initialized.")
        
        self.communication_handler.send_message(
            receiver_id=f"client_{client.client_id}",
            message_type="train_request",
            content={
                "train_request": True,
            }
        )

    def select_clients(self, clients: List[BaseClient], selection_mode: str = "train") -> List[BaseClient]:
        """
        Selects a random subset of clients for training or evaluation.

        Args:
            clients (List[BaseClient]): The list of all available clients.
            selection_mode (str): The mode of selection, either "train" or "eval".

        Returns:
            List[BaseClient]: A list of selected clients.
        """
        if selection_mode not in ["train", "eval"]:
            raise ValueError(f"Invalid selection_mode: {selection_mode}. Must be 'train' or 'eval'.")

        fraction = self.fraction_fit if selection_mode == "train" else self.fraction_eval
        min_clients = self.min_fit_clients if selection_mode == "train" else self.min_eval_clients
        
        num_clients_to_select = max(int(len(clients) * fraction), min_clients)

        if len(clients) < num_clients_to_select:
            raise ValueError(f"Not enough clients ({len(clients)}) to select {num_clients_to_select} for {selection_mode}.")

        selected_clients = np.random.choice(clients, size=num_clients_to_select, replace=False).tolist()

        if self.debug:
            selected_ids = [c.client_id for c in selected_clients]
            print(f"select_clients: Selected {len(selected_clients)} clients for {selection_mode}: {selected_ids}")

        return selected_clients
    
    def aggregate_prototypes(self, prototypes: List[torch.Tensor], influences: List[torch.Tensor]) -> torch.Tensor:
        """
        Aggregates prototypes based on their influence scores.

        Args:
            prototypes (List[torch.Tensor]): List of prototype tensors from clients.
                Shape: [num_clients] -> each tensor is [num_classes, num_prototypes_per_class, feature_dim]
            influences (List[torch.Tensor]): List of influence scores corresponding to each prototype.
                Shape: [num_clients] -> each tensor is [num_classes, num_prototypes_per_class]

        Returns:
            torch.Tensor: The aggregated prototype tensor.
                Shape: [num_classes, num_prototypes_per_class, feature_dim]
        """
        # get the device where the first prototype is located
        if not prototypes:
            raise ValueError("Prototypes must not be empty.")
        device = prototypes[0].device

        for p in range(len(prototypes)):
            if not isinstance(prototypes[p], torch.Tensor):
                raise TypeError(f"Prototype {p} is not a torch.Tensor.")
            prototypes[p] = prototypes[p].to(device)
            
        for i in range(len(influences)):
            if not isinstance(influences[i], torch.Tensor):
                raise TypeError(f"Influence {i} is not a torch.Tensor.")
            influences[i] = influences[i].to(device)


        if not prototypes or not influences:
            raise ValueError("Prototypes and influences must not be empty.")

        if len(prototypes) != len(influences):
            raise ValueError("Number of prototypes and influences must match.")

        # Validate shapes
        prototype_shape = prototypes[0].shape
        influence_shape = influences[0].shape
        
        if len(prototype_shape) != 3 or len(influence_shape) != 2:
            raise ValueError("Prototypes should be 3D [num_classes, num_prototypes_per_class, feature_dim] "
                           "and influences should be 2D [num_classes, num_prototypes_per_class]")
        
        if prototype_shape[:2] != influence_shape:
            raise ValueError("Prototype and influence dimensions don't match for classes and prototypes per class")

        num_clients = len(prototypes)
        num_classes, num_prototypes_per_class, feature_dim = prototype_shape

        # Update stored dimensions if not set
        if self.num_classes is None:
            self.num_classes = num_classes
            self.num_prototypes_per_class = num_prototypes_per_class
            if self.debug:
                print(f"Inferred prototype dimensions: {num_classes} classes, {num_prototypes_per_class} prototypes per class")

        # Stack all influences and prototypes
        stacked_influences = torch.stack(influences, dim=0)  # [num_clients, num_classes, num_prototypes_per_class]
        stacked_prototypes = torch.stack(prototypes, dim=0)  # [num_clients, num_classes, num_prototypes_per_class, feature_dim]

        # Sum influences across clients
        influence_sum = torch.sum(stacked_influences, dim=0)  # [num_classes, num_prototypes_per_class]

        # Initialize aggregated prototypes
        aggregated_prototype = torch.zeros_like(prototypes[0])  # [num_classes, num_prototypes_per_class, feature_dim]

        # Aggregate prototypes weighted by normalized influences
        for i in range(num_clients):
            # Normalize influence scores (add small epsilon to avoid division by zero)
            normalized_influence = influences[i] / (influence_sum + 1e-8)  # [num_classes, num_prototypes_per_class]
            
            # Expand normalized influence to match prototype dimensions
            normalized_influence_expanded = normalized_influence.unsqueeze(-1).expand_as(prototypes[i])
            # [num_classes, num_prototypes_per_class, feature_dim]
            
            # Weighted aggregation of prototypes
            aggregated_prototype += normalized_influence_expanded * prototypes[i]

        if self.debug:
            total_influence = torch.sum(influence_sum)
            print(f"Prototype aggregation completed. Total influence: {total_influence.item():.2f}")

        return aggregated_prototype

    def aggregate_updates(self, clients: List[BaseClient], client_model_dir: Dict[int, str]) -> None:
        """
        Aggregates model updates from clients using FedAvg for model parameters 
        and influence-based aggregation for prototypes.

        Args:
            clients (List[BaseClient]): The list of clients selected for the training round.
            client_model_dir (Dict[int, str]): A dictionary mapping client IDs to their saved model paths.
        """
        if self.debug:
            print(f"aggregate_updates: Starting aggregation for {len(clients)} clients.")
        
        if not clients:
            if self.accept_failure:
                print("Warning: No clients provided for aggregation. Skipping update.")
                return
            else:
                raise ValueError("No clients for aggregation and accept_failure is False.")

        if not self.communication_handler:
            raise RuntimeError("Communication handler is not initialized.")

        # Initialize containers for model and prototype aggregation
        state_dicts_to_aggregate = []
        aggregation_weights = []
        prototype_tensors = []
        influence_tensors = []

        # Process each client's updates
        for client in clients:
            sender_id = f"client_{client.client_id}"
            completion_msgs = self.communication_handler.receive_messages(
                sender_id_filter=sender_id,
                message_type_filter="train_completion"
            )

            if not completion_msgs:
                message = f"No 'train_completion' message from client {client.client_id}."
                if self.accept_failure:
                    if self.debug: print(f"Warning: {message} Skipping client.")
                    continue
                else:
                    raise RuntimeError(message)

            content = completion_msgs[0].content
            if not content.get('success', False):
                if self.debug: print(f"Client {client.client_id} reported training failure. Skipping.")
                continue

            # Process model parameters and influence data
            model_path = client_model_dir.get(client.client_id)
            num_samples = content.get('num_samples')
            influence = content.get('influence')

            if model_path and os.path.isfile(model_path) and num_samples is not None and influence is not None:
                try:
                    # Load client model state dict
                    client_state_dict = torch.load(model_path, map_location='cpu')
                    
                    # Extract prototypes from state dict
                    if 'prototypes' in client_state_dict:
                        prototype_tensor = client_state_dict['prototypes']
                        prototype_tensors.append(prototype_tensor)
                        
                        # Convert influence to tensor if it's not already
                        if not isinstance(influence, torch.Tensor):
                            influence = torch.tensor(influence, dtype=torch.float32)
                        influence_tensors.append(influence)
                        
                        # Add to regular aggregation
                        state_dicts_to_aggregate.append(client_state_dict)
                        aggregation_weights.append(num_samples)
                        
                        if self.debug: 
                            print(f"Successfully processed updates from client {client.client_id}. "
                                  f"Prototype shape: {prototype_tensor.shape}, Influence shape: {influence.shape}")
                    else:
                        message = f"No 'prototypes' found in state dict from client {client.client_id}"
                        if self.accept_failure:
                            print(f"Warning: {message} Skipping prototype aggregation for this client.")
                            # Still add to regular model aggregation
                            state_dicts_to_aggregate.append(client_state_dict)
                            aggregation_weights.append(num_samples)
                        else:
                            raise ValueError(message)
                            
                except Exception as e:
                    message = f"Failed to process updates from client {client.client_id}: {e}"
                    if self.accept_failure:
                        print(f"Warning: {message} Skipping updates for this client.")
                    else:
                        raise RuntimeError(message) from e
            else:
                missing_items = []
                if not model_path or not os.path.isfile(model_path):
                    missing_items.append("model_path")
                if num_samples is None:
                    missing_items.append("num_samples")
                if influence is None:
                    missing_items.append("influence")
                    
                message = f"Missing data for client {client.client_id}: {', '.join(missing_items)}"
                if self.accept_failure:
                    if self.debug: print(f"Warning: {message} Skipping updates.")
                    continue
                else:
                    raise ValueError(message)

        # Aggregate model parameters using FedAvg
        if state_dicts_to_aggregate:
            aggregated_state_dict = self.aggregator.aggregate_state_dicts(
                state_dicts=state_dicts_to_aggregate,
                weights=aggregation_weights,
                mode=self.aggregation_mode,
                model=self.model
            )
            
            # Aggregate prototypes separately if we have them
            if prototype_tensors and influence_tensors:
                try:
                    aggregated_prototypes = self.aggregate_prototypes(prototype_tensors, influence_tensors)
                    # Replace the prototypes in the aggregated state dict
                    aggregated_state_dict['prototypes'] = aggregated_prototypes
                    
                    if self.debug:
                        print(f"Prototypes aggregated successfully from {len(prototype_tensors)} clients.")
                except Exception as e:
                    message = f"Failed to aggregate prototypes: {e}"
                    if self.accept_failure:
                        print(f"Warning: {message} Using FedAvg aggregated prototypes instead.")
                    else:
                        raise RuntimeError(message) from e
            elif self.debug:
                print("No prototype data available for aggregation. Using FedAvg aggregated prototypes.")
            
            # Load the final aggregated state dict
            self.model.load_state_dict(aggregated_state_dict)
            
            if self.debug: 
                print(f"Model updated with aggregation from {len(state_dicts_to_aggregate)} clients.")
                if prototype_tensors:
                    print(f"Prototypes updated using influence-based aggregation from {len(prototype_tensors)} clients.")
        else:
            print("Warning: No valid model updates found for aggregation.")

        if self.debug:
            print(f"FedProIn aggregation completed.")

    def evaluate_model(self, clients: List[BaseClient]) -> Dict[str, Any]:
        """
        Evaluates the global model on a selection of clients.

        Args:
            clients (List[BaseClient]): The list of all available clients for evaluation.

        Returns:
            Dict[str, Any]: A dictionary containing evaluation results, including aggregated loss.
        """
        if self.debug:
            print(f"evaluate_model: Starting evaluation on a selection of {len(clients)} clients.")

        selected_clients = self.select_clients(clients, selection_mode="eval")
        eval_results = {}

        if self.eval_fn:
            if self.debug: print("Using provided server-side evaluation function.")
            eval_results = self.eval_fn(self.model, selected_clients, **self.eval_fn_args)
        
        total_loss = 0
        successful_evaluations = 0
        
        for client in selected_clients:
            try:
                # Send test request to client
                self.send_test_info(client)

                loss, _ = client.test_local()
                
                # Receive the test completion message
                sender_id = f"client_{client.client_id}"
                completion_msgs = self.communication_handler.receive_messages(
                    sender_id_filter=sender_id,
                    message_type_filter="test_completion"
                )
                
                if not completion_msgs:
                    if self.debug: print(f"Warning: No 'test_completion' message from client {client.client_id}.")
                    continue
                    
                if not completion_msgs[0].content.get('success', False):
                    if self.debug: print(f"Client {client.client_id} reported test failure. Skipping.")
                    continue
                    
                total_loss += loss
                successful_evaluations += 1
                
            except Exception as e:
                if self.debug: print(f"Error evaluating client {client.client_id}: {e}")
                continue

        # Calculate total loss (sum) to match FedAvg and utils.py expectations
        eval_results["Val_loss"] = total_loss
        eval_results["num_evaluated_clients"] = successful_evaluations
        
        if successful_evaluations == 0:
            if self.debug: print("Warning: No successful evaluations completed.")

        if self.debug:
            print(f"evaluate_model: Aggregated loss from {successful_evaluations} clients: {eval_results['Val_loss']}")

        return eval_results

    def save_model(self, model_path: str) -> None:
        """
        Saves the current global model's state dictionary to a file.

        Args:
            model_path (str): The path where the model will be saved.
        """
        if self.debug:
            print(f"Saving global model to {model_path}")
        torch.save(self.model.state_dict(), model_path)

    def get_prototype_info(self) -> Dict[str, Any]:
        """
        Returns information about the current prototypes.

        Returns:
            Dict[str, Any]: Dictionary containing prototype statistics.
        """
        if hasattr(self.model, 'prototypes'):
            prototypes = self.model.prototypes
            return {
                "shape": list(prototypes.shape),
                "num_classes": prototypes.shape[0] if len(prototypes.shape) >= 1 else 0,
                "num_prototypes_per_class": prototypes.shape[1] if len(prototypes.shape) >= 2 else 0,
                "feature_dim": prototypes.shape[2] if len(prototypes.shape) >= 3 else 0,
                "mean_norm": torch.norm(prototypes, dim=-1).mean().item(),
                "std_norm": torch.norm(prototypes, dim=-1).std().item()
            }
        else:
            return {"error": "Model does not have prototypes attribute"}