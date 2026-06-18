# To-do :
# 1. Add support for transforms in the data loader. based on label distribution.
# 2. Add support for oversampling in the data loader. based on label distribution.

from sleeping_fed.humans.mango_man import ClientProtocol
from typing import Any, Dict, List, Optional, Tuple, Union
from sleeping_fed.sleep_slicer.iid_slicer import IIDSlicer
from sleeping_fed.sleep_slicer.knife import PyTorchDatasetSlicer
from copy import deepcopy
from pathlib import Path
from sleeping_fed.meghdoot.meghdoot import AcknowledgementHandler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from data.sampler import UniformSampler
from configs import num_classes
import numpy as np

np.random.seed(42)  # Set a random seed for reproducibility

class TransformDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.num_classes = num_classes
        self.label_distribution = [0] * self.num_classes
        for _, label in dataset:
            self.label_distribution[label.item()] += 1
        self.transform = {}

        for label in range(self.num_classes):
            # p_aug_c_j = m_max - m_class / m_max 
            p_aug = 1.0 - (self.label_distribution[label] / max(self.label_distribution))
            if p_aug > 0:
                transforms = T.RandomApply([
                    T.RandomRotation(degrees=15),
                    T.RandomHorizontalFlip(p=p_aug),
                    T.RandomVerticalFlip(p = p_aug),
                    T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    T.RandomAdjustSharpness(sharpness_factor=2, p= p_aug),
                    T.RandomAutocontrast(p= p_aug),
                ], p=p_aug)

                self.transform[label] = T.Compose([
                    transforms,
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transform[label] = T.Compose([
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        label_idx = label.item()
        if self.transform:
            image = self.transform[label_idx](image)
        return image, label
    


def make_client(num_clients: int, 
                model: Any, 
                train_data: Any, 
                val_data: Any, 
                workspace_path: Union[Path, str],
                train_args: Optional[Dict[str, Any]] = None, 
                test_args: Optional[Dict[str, Any]] = None,
                train_data_slicer: Optional[PyTorchDatasetSlicer] = None, 
                val_data_slicer: Optional[PyTorchDatasetSlicer] = None,
                client_class: Optional[ClientProtocol] = None, 
                data_loader_args: Optional[Dict[str, Any]] = None,
                ack_handler: Optional[AcknowledgementHandler] = None) -> List[ClientProtocol]:
    
    """
    Create a list of clients for federated learning.
    
    Args:
        num_clients (int): Number of clients to create.
        model (Any): The model to be used by the clients.
        train_data (Any): The training data for the clients.
        val_data (Any): The validation data for the clients.
        workspace_path (Union[Path, str]): Path to the workspace directory.
        train_args (Optional[Dict[str, Any]]): Additional arguments for training.
        test_args (Optional[Dict[str, Any]]): Additional arguments for testing.
        train_data_slicer (Optional[PyTorchDatasetSlicer]): Slicer object to slice the training data.
        val_data_slicer (Optional[PyTorchDatasetSlicer]): Slicer object to slice the validation data.
        client_class (Optional[ClientProtocol]): Client class to be used.
        data_loader_args (Optional[Dict[str, Any]]): Arguments for the data loader.
        ack_handler (Optional[AcknowledgementHandler]): The acknowledgement handler for communications.
        
    Returns:
        List[ClientProtocol]: List of clients.
    """

    if client_class is None:
        raise ValueError("Client class must be provided.")
    if not issubclass(client_class, ClientProtocol):
        raise TypeError("Client class must be a subclass of ClientProtocol.")
    
    # Ensure workspace_path is a Path object
    workspace_path = Path(workspace_path) if isinstance(workspace_path, str) else workspace_path
    
    # Create slicers if not provided
    if train_data_slicer is None:
        slicer = IIDSlicer( 
            num_slices=num_clients,
            min_slice_size=1,
            King_portion=False,
            shuffle=True,
            seed=42
        )

        train_data_slicer = PyTorchDatasetSlicer(
            dataset=train_data,
            slicer_obj=slicer,
            num_slices=num_clients,
            min_slice_size=1,
            king_portion=False,
            shuffle=True,
            seed=42
        )

        val_data_slicer = PyTorchDatasetSlicer(
            dataset=val_data,
            slicer_obj=slicer,
            num_slices=num_clients,
            min_slice_size=1,
            king_portion=False,
            shuffle=True,
            seed=42
        )
    
    if data_loader_args is None:
        data_loader_args = {
            "batch_size": 32,
            "shuffle": True,
            "num_workers": 4
        }

    clients = []
    for i in range(num_clients):

        train_data_client = train_data_slicer.get_slice(i)
        val_data_client = val_data_slicer.get_slice(i)

        client_label  = [label.item() for _, label in train_data_client]
        unique_labels, counts = np.unique(client_label, return_counts=True)
        num_samples = max(counts) * len(unique_labels)

        client_sampler = UniformSampler(
            dataset=train_data_client,
            num_samples=num_samples,
            seed=42
        )

        data_loader_args['sampler'] = client_sampler
        if "shuffle" in data_loader_args:
            del data_loader_args["shuffle"]

        train_data_transformed = TransformDataset(
            dataset=train_data_client)
        
        train_data_loader = DataLoader(
            train_data_transformed,
            **data_loader_args
        )
        # remove sampler from data_loader_args for validation loader
        if 'sampler' in data_loader_args:
            del data_loader_args['sampler']

        val_data_loader = DataLoader(
            val_data_client,
            **data_loader_args
        )



        
        client = client_class(
            client_id=i,
            workspace_path=workspace_path,
            model=deepcopy(model),
            train_data= train_data_loader,
            val_data=val_data_loader,
            ACK=ack_handler,
            train_args=train_args,
            test_args=test_args
        )
        clients.append(client)
    
    return clients