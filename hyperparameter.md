# FedProIn Hyperparameter Reference

---

## 1. Federated Learning

| Parameter | Description |
|---|---|
| `NUM_CLIENTS` | Total number of federated clients |
| `NUM_ROUNDS` | Number of global federation rounds |
| `Fraction_fit` | Fraction of clients sampled for training per round |
| `Fraction_evaluate` | Fraction of clients sampled for evaluation per round |
---

## 2. Data Partitioning

| Parameter | Description |
|---|---|
| `SLICER_NAME` | Partitioning strategy: `"IID"` or `"Dirichlet"` |

### IID Slicer

| Parameter | Description |
|---|---|
| `num_slices` | Number of data partitions -- same as `NUM_CLIENTS` |
| `min_slice_size` | Minimum samples per partition |
| `shuffle` | Shuffle dataset before slicing |
| `seed` | Random seed |

### Dirichlet Slicer

| Parameter | Description |
|---|---|
| `num_slices` | Number of data partitions -- same as `NUM_CLIENTS` |
| `min_slice_size` | Minimum samples per partition |
| `alpha` | Concentration parameter; lower → more class imbalance across clients |
| `shuffle` | Shuffle dataset before slicing |
| `seed` | Random seed |

---

## 3. Model Architecture

| Parameter | Value | Description |
|---|---|---|
| `num_classes` |  | Number of output classes |
| `num_prototypes_per_class` |  | Number of learnable prototype vectors per class |
| `feature_dim` |  | Backbone output dimensionality; prototype shape is `[num_classes, num_prototypes_per_class, feature_dim]` |

---

## 4. Local Training

| Parameter |  Description |
|---|---|
| `epochs` | Local training epochs per round per client |
| `batch_size` | Mini-batch size |
| `shuffle` | Shuffle training data each epoch |
| `num_workers` | Dataloader worker processes |
| `persistent_workers` | Keep worker processes alive between epochs |

---

## 5. Optimizer

| Parameter | Description |
|---|---|
| `optimizer_class` |  Optimizer |
| `lr` | Learning rate |

---



## 7. FedProIn Regularization

| Parameter  | Description |
|---|---|
| `feature_divergence_weight`  | Weight on MSE loss between local and global feature embeddings |
| `proto_contrastive_weight`  | Weight on prototype contrastive loss |
| `contrastive_margin` | margin in the prototype divergence ReLU |

---

## 8. Experiment Settings

| Parameter | Description |
|---|---|
| `EXPERIMENT_NAME` | Human-readable experiment label |
| `Data_Source_Id` | Dataset identifier logged in metadata |
| `DEBUG` | Enables verbose per-batch logging |
| `metric_calculation_mode` | Averaging mode for F1/precision/recall (`weighted`, `macro`, `micro`) |
| `base_path` | Base directory for saving models, training logs, and metadata |
| `Metadata_file` | Filename for experiment metadata JSON |
