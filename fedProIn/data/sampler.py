from typing import Union
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset, Subset, Sampler

class UniformSampler(Sampler):
    def __init__(self, dataset : Union[Dataset, Subset] , num_samples, seed=42):
        self.dataset = dataset # is a subset of the original dataset
        self.num_samples = num_samples
        self.length = len(dataset)

        self.class_to_indices = defaultdict(list)
        for i in range(self.length):
            label = dataset[i][1].item()
            self.class_to_indices[label].append(i)
        self.num_classes = len(self.class_to_indices)

        self.sample_weights = torch.zeros(self.length, dtype=torch.float)
        
        for class_label, local_indices in self.class_to_indices.items():
            class_size = len(local_indices)
            weight = 1.0 / (self.num_classes * class_size)
            #print(f"Class {class_label} has {class_size} samples, weight: {weight}")
            
            for local_idx in local_indices:
                self.sample_weights[local_idx] = weight
        
        self.rng = np.random.default_rng(seed)

            
    
    def __iter__(self):
        for _ in range(self.num_samples):
            idx = self.rng.choice(self.length, p=self.sample_weights.numpy())
            yield idx
    
    def __len__(self):
        return self.num_samples