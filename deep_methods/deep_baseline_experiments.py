"""Unified DEC, DSC-Net, and SDCN experiments on MAT datasets.

Expected project layout (algorithm folder names are case-insensitive):
    project_root/
    |-- deep_methods/
    |   |-- deep_baseline_experiments.py
    |   |-- deep_baseline_parameter_search.py
    |   |-- SDCN/sdcn.py (plus GNN.py, utils.py, and evaluation.py)
    |   |-- DEC/dec.py (plus the ptdec package)
    |   `-- DSC/main.py and DSC/yaleb.py
    `-- datasets/*.mat

The examples below assume that the current working directory is ``deep_methods``.
Therefore, the sibling dataset directory is referenced as ``../datasets``.
By default, each row of the MAT variable ``fea`` is one sample and ``gnd``
contains the labels. All metrics are computed from the final-epoch predictions;
the script never selects the best metric observed during training.

Examples:
    python deep_baseline_experiments.py --data ../datasets/ORL_32x32.mat         --algorithms dec sdcn --normalize none --runs 1

    python deep_baseline_experiments.py --data ../datasets/ORL_32x32.mat         --algorithms dsc --normalize global_max --dsc-image-size 32 32         --dsc-channels 1,3,3,5 --dsc-kernels 3,3,3

    python deep_baseline_experiments.py --data ../datasets/YaleBCrop025.mat         --algorithms dsc --dsc-variant yaleb         --dsc-pretrain-weights pretrained_weights_original/yaleb.pkl

Run all three preprocessing protocols in isolated output subdirectories:
    python deep_baseline_experiments.py --data ../datasets/ORL_32x32.mat         --algorithms dec dsc sdcn --normalizations none l2 global_max

Protocol notes:
1. Pretraining and clustering learning rates are configured separately. Cached
   weights are keyed by data, algorithm, preprocessing, architecture,
   pretraining protocol, learning rate, epochs, batch size, and random seed.
2. DEC uses the built-in layer-wise denoising autoencoder by default
   (``--dec-pretrain-mode sdae``). The ``joint`` mode is retained for diagnostic
   comparisons. Exact reproduction of another repository still requires checking
   its optimizer, epoch schedule, and end-to-end fine-tuning protocol.
3. Generic DSC requires the image height and width unless they can be inferred
   from a dataset profile or a square feature count. The original 2016-dimensional
   YaleBCrop025 data uses ``--dsc-variant yaleb``. Official pretrained weights
   should be supplied when strict reproduction of the original paper is required.
4. DSC constructs A=(|C|+|C|^T)/2 and runs spectral clustering directly. It does
   not apply thrC, SVD embedding, graph reconstruction, or affinity powers.
   LOBPCG is the default eigensolver; use ``--dsc-eigen-solver arpack`` to match
   the legacy L2 script explicitly.
5. SDCN constructs its symmetric KNN graph, adds self-loops, and row-normalizes
   the adjacency before training. This graph is a model input, not post-processing.
6. With multiple preprocessing methods, outputs are written to
   ``OUTPUT/NORMALIZATION/`` so predictions, caches, and summaries cannot overlap.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import random
import re
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset


ALGORITHMS = ("dec", "dsc", "sdcn")

# Dataset statistics from the experiment table. image_size=None denotes vector data;
# never infer an image solely from a square dimension. DSC profiles come from main.py.
DATASET_PROFILES = {
    "zoo": {"samples": 101, "dimensions": 16, "clusters": 7, "image_size": None},
    "yale": {"samples": 165, "dimensions": 1024, "clusters": 15, "image_size": (32, 32)},
    "jaffe": {"samples": 213, "dimensions": 4096, "clusters": 10, "image_size": (64, 64)},
    "orl": {
        "samples": 400, "dimensions": 1024, "clusters": 40, "image_size": (32, 32),
        "dsc": {"channels": "1,3,3,5", "kernels": "3,3,3", "epochs": 700,
                "weight_coef": 2.0, "weight_self_exp": 0.2},
    },
    "umist": {"samples": 575, "dimensions": 1024, "clusters": 20, "image_size": (32, 32)},
    "feret": {"samples": 1400, "dimensions": 1024, "clusters": 200, "image_size": (32, 32)},
    "coil20": {
        "samples": 1440, "dimensions": 1024, "clusters": 20, "image_size": (32, 32),
        "dsc": {"channels": "1,15", "kernels": "3", "epochs": 40,
                "weight_coef": 1.0, "weight_self_exp": 75.0},
    },
    "ar": {"samples": 1680, "dimensions": 1024, "clusters": 120, "image_size": (32, 32)},
    # This is the 2414x1024 YaleB table dataset, not yaleb.py's 2432x2016 version.
    "yaleb": {"samples": 2414, "dimensions": 1024, "clusters": 38, "image_size": (32, 32)},
    "coil100": {
        "samples": 7200, "dimensions": 1024, "clusters": 100, "image_size": (32, 32),
        "dsc": {"channels": "1,50", "kernels": "5", "epochs": 120,
                "weight_coef": 1.0, "weight_self_exp": 15.0},
    },
    "mnist": {"samples": 10000, "dimensions": 784, "clusters": 10, "image_size": (28, 28)},
    "pendigits": {"samples": 10992, "dimensions": 16, "clusters": 10, "image_size": None},
}


def get_dataset_profile(dataset_name: str) -> Optional[Dict[str, Any]]:
    normalized = re.sub(r"[^a-z0-9]", "", dataset_name.lower())
    for key in sorted(DATASET_PROFILES, key=len, reverse=True):
        if normalized == key or normalized.startswith(key):
            return DATASET_PROFILES[key]
        if len(key) >= 4 and key in normalized:
            return DATASET_PROFILES[key]
    return None


@dataclass
class DataBundle:
    name: str
    x: np.ndarray                 # [N, D], float32
    y: np.ndarray                 # [N], zero-based integer labels
    label_values: np.ndarray      # Original label values
    source: Path

    @property
    def n_samples(self) -> int:
        return self.x.shape[0]

    @property
    def n_features(self) -> int:
        return self.x.shape[1]

    @property
    def n_clusters(self) -> int:
        return len(self.label_values)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_name(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", text)


def torch_load(path: Path, map_location: Any = "cpu") -> Any:
    """Support the weights_only argument across PyTorch versions."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _read_mat(path: Path) -> Dict[str, Any]:
    try:
        return {k: v for k, v in sio.loadmat(path).items() if not k.startswith("__")}
    except NotImplementedError:
        # MATLAB v7.3 uses HDF5. h5py dimension order may differ from MATLAB,
        # so specify --sample-axis explicitly and verify the printed N and D.
        try:
            import h5py
        except ImportError as inner:
            raise RuntimeError("Reading v7.3 MAT files requires h5py: pip install h5py") from inner
        with h5py.File(path, "r") as handle:
            return {key: np.asarray(handle[key]) for key in handle.keys()}
    except Exception as exc:
        raise RuntimeError(f"Cannot read MAT file {path}: {exc}") from exc


def _as_dense_2d(array: Any, name: str) -> np.ndarray:
    if sp.issparse(array):
        array = array.toarray()
    array = np.asarray(array)
    if array.dtype == object:
        array = array.astype(np.float64)
    if array.ndim != 2:
        raise ValueError(f"Feature variable {name!r} must be a two-dimensional matrix; got shape {array.shape}")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"Feature variable {name!r} is not a numeric matrix")
    return array


def _choose_feature_key(mat: Dict[str, Any]) -> str:
    preferred = ("fea", "X", "x", "data", "features", "feature")
    for key in preferred:
        if key in mat:
            return key
    candidates = []
    for key, value in mat.items():
        try:
            shape = value.shape if hasattr(value, "shape") else np.asarray(value).shape
        except Exception:
            continue
        if len(shape) == 2 and min(shape) > 1:
            candidates.append((int(np.prod(shape)), key))
    if not candidates:
        raise KeyError("Cannot infer the feature matrix; specify its MAT variable with --x-key")
    return max(candidates)[1]


def _choose_label_key(mat: Dict[str, Any]) -> str:
    preferred = ("gnd", "y", "Y", "labels", "label", "truth", "gt")
    for key in preferred:
        if key in mat:
            return key
    vectors = []
    for key, value in mat.items():
        try:
            array = value.toarray() if sp.issparse(value) else np.asarray(value)
        except Exception:
            continue
        if array.ndim == 2 and 1 in array.shape:
            vectors.append((array.size, key))
    if len(vectors) == 1:
        return vectors[0][1]
    raise KeyError("Cannot infer the label vector; specify its MAT variable with --y-key")


def _flatten_labels(array: Any) -> np.ndarray:
    if sp.issparse(array):
        array = array.toarray()
    array = np.asarray(array).squeeze()
    if array.ndim != 1:
        raise ValueError(f"Labels must be reducible to a one-dimensional vector; got shape {array.shape}")
    return array


def normalize_features(x: np.ndarray, method: str) -> np.ndarray:
    x = x.astype(np.float32, copy=True)
    if method == "none":
        return x
    if method == "l2":
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.maximum(norm, 1e-12)
    if method == "minmax":
        low = x.min(axis=0, keepdims=True)
        high = x.max(axis=0, keepdims=True)
        return (x - low) / np.maximum(high - low, 1e-12)
    if method == "standard":
        mean = x.mean(axis=0, keepdims=True)
        std = x.std(axis=0, keepdims=True)
        return (x - mean) / np.maximum(std, 1e-12)
    if method == "global_max":
        scale = float(np.max(np.abs(x)))
        return x / max(scale, 1e-12)
    raise ValueError(f"Unknown normalization method: {method}")


def load_yaleb_dataset(path: Path, mat: Dict[str, Any], args: argparse.Namespace) -> DataBundle:
    """Flatten the 3-D Y array in yaleb.py subject-then-image order and create subject labels."""
    key = args.x_key or "Y"
    if key not in mat:
        raise KeyError(f"The raw YaleB data does not contain variable {key!r}")
    images = np.asarray(mat[key])
    if images.ndim != 3:
        raise ValueError(f"YaleB {key} must have shape [pixels, images_per_subject, subjects]; got {images.shape}")
    n_pixels, images_per_subject, n_subjects = images.shape
    if n_pixels != 42 * 48:
        raise ValueError(f"yaleb.py requires 2016 pixels per image; got {n_pixels}")
    # Image j of a subject is images[:, j, subject], matching the original nested loop.
    raw_x = np.empty((n_subjects * images_per_subject, n_pixels), dtype=np.float32)
    for subject in range(n_subjects):
        start = subject * images_per_subject
        raw_x[start:start + images_per_subject] = images[:, :, subject].T
    raw_y = np.repeat(np.arange(n_subjects, dtype=np.int64), images_per_subject)
    if not np.isfinite(raw_x).all():
        raise ValueError("YaleB features contain NaN or Inf; clean the data first")
    label_values, y = np.unique(raw_y, return_inverse=True)
    raw_x = normalize_features(raw_x, args.normalize)
    print(
        f"[Data] {path.name}: raw YaleB Y={images.shape}; flattened X={raw_x.shape}; "
        f"K={len(label_values)}; the DSC branch reshapes to (42,48) and transposes to 48x42"
    )
    return DataBundle(path.stem, raw_x, y.astype(np.int64), label_values, path)


def load_dataset(path: Path, args: argparse.Namespace) -> DataBundle:
    mat = _read_mat(path)
    if args.dsc_variant == "yaleb" and args.y_key is None:
        yaleb_key = args.x_key or "Y"
        if yaleb_key in mat and np.asarray(mat[yaleb_key]).ndim == 3:
            return load_yaleb_dataset(path, mat, args)
    x_key = args.x_key or _choose_feature_key(mat)
    y_key = args.y_key or _choose_label_key(mat)
    if x_key not in mat:
        raise KeyError(f"The MAT file has no feature variable {x_key!r}; available variables: {list(mat)}")
    if y_key not in mat:
        raise KeyError(f"The MAT file has no label variable {y_key!r}; available variables: {list(mat)}")

    raw_x = _as_dense_2d(mat[x_key], x_key)
    raw_y = _flatten_labels(mat[y_key])
    axis = args.sample_axis
    if axis == "auto":
        row_match = raw_x.shape[0] == raw_y.size
        col_match = raw_x.shape[1] == raw_y.size
        if row_match and not col_match:
            axis = "rows"
        elif col_match and not row_match:
            axis = "columns"
        else:
            axis = "rows"  # Use the confirmed convention that rows are samples
    x = raw_x.T if axis == "columns" else raw_x
    x = np.ascontiguousarray(x, dtype=np.float32)

    if x.shape[0] != raw_y.size:
        raise ValueError(
            f"Sample and label counts differ after conversion: X.shape={x.shape}, labels={raw_y.size}. "
            "Check whether --sample-axis columns/rows is correct."
        )
    if x.shape[0] < 2:
        raise ValueError("At least two samples are required")
    if not np.isfinite(x).all():
        raise ValueError("Features contain NaN or Inf; clean the data first")

    # Encode labels as 0..K-1; ACC/NMI/ARI do not depend on the original label offset.
    label_values, y = np.unique(raw_y, return_inverse=True)
    x = normalize_features(x, args.normalize)
    print(
        f"[Data] {path.name}: X={x.shape} (N={x.shape[0]}, D={x.shape[1]}), "
        f"K={len(label_values)}, sample axis={axis}, X variable={x_key}, Y variable={y_key}"
    )
    profile = get_dataset_profile(path.stem)
    if profile:
        image_text = (
            "non-image vector" if profile["image_size"] is None
            else f"image={profile['image_size'][0]}×{profile['image_size'][1]}"
        )
        print(
            f"  [Table profile] reference N={profile['samples']}, D={profile['dimensions']}, "
            f"K={profile['clusters']}, {image_text}"
        )
        mismatches = []
        if x.shape[0] != profile["samples"]:
            mismatches.append(f"N actual{x.shape[0]}/table{profile['samples']}")
        if x.shape[1] != profile["dimensions"]:
            mismatches.append(f"D actual{x.shape[1]}/table{profile['dimensions']}")
        if len(label_values) != profile["clusters"]:
            mismatches.append(f"K actual{len(label_values)}/table{profile['clusters']}")
        if mismatches:
            print("  [Warning] Data differ from the table profile: " + ", ".join(mismatches))
    return DataBundle(path.stem, x, y.astype(np.int64), label_values, path)


def cluster_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.size != y_pred.size:
        raise ValueError("True and predicted label lengths differ")
    true_ids, true_encoded = np.unique(y_true, return_inverse=True)
    pred_ids, pred_encoded = np.unique(y_pred, return_inverse=True)
    size = max(len(true_ids), len(pred_ids))
    matrix = np.zeros((size, size), dtype=np.int64)
    for t, p in zip(true_encoded, pred_encoded):
        matrix[t, p] += 1
    row, col = linear_sum_assignment(-matrix)
    return float(matrix[row, col].sum() / y_true.size)


def evaluate_clustering(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "acc": cluster_accuracy(y_true, y_pred),
        "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
        "ari": float(adjusted_rand_score(y_true, y_pred)),
    }


def report_prediction(tag: str, y_pred: np.ndarray, y_true: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Print and return prediction diagnostics; evaluation is not used for model selection."""
    y_pred = np.asarray(y_pred).reshape(-1)
    if y_pred.size == 0:
        raise ValueError(f"{tag} has empty predictions")
    cluster_ids, counts = np.unique(y_pred, return_counts=True)
    details: Dict[str, Any] = {
        "nonempty_clusters": int(len(cluster_ids)),
        "min_cluster_size": int(counts.min()),
        "max_cluster_size": int(counts.max()),
        "cluster_sizes": {int(k): int(v) for k, v in zip(cluster_ids, counts)},
    }
    message = (
        f"  [{tag}] nonempty clusters={len(cluster_ids)}, smallest cluster={counts.min()}, "
        f"largest cluster={counts.max()}, cluster sizes={details['cluster_sizes']}"
    )
    if y_true is not None:
        metrics = evaluate_clustering(y_true, y_pred)
        details.update(metrics)
        message += f", ACC={metrics['acc']:.4f}, NMI={metrics['nmi']:.4f}, ARI={metrics['ari']:.4f}"
    print(message)
    return details


def diagnose_embedding(tag: str, embedding: Any) -> Dict[str, float]:
    """Check whether embeddings are constant or non-finite; these diagnostics do not use labels."""
    if isinstance(embedding, torch.Tensor):
        array = embedding.detach().cpu().numpy()
    else:
        array = np.asarray(embedding)
    array = np.asarray(array, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 1:
        raise ValueError(f"{tag} must have shape [samples, latent_dimensions]; got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{tag} contains NaN or Inf")
    sample_std = array.std(axis=1)
    dimension_std = array.std(axis=0)
    details = {
        "mean_sample_std": float(sample_std.mean()),
        "max_sample_std": float(sample_std.max()),
        "active_dimensions": int(np.count_nonzero(dimension_std > 1e-12)),
    }
    print(
        f"  [{tag}] mean_sample_std={details['mean_sample_std']:.6e}, "
        f"max_sample_std={details['max_sample_std']:.6e}, "
        f"active latent dimensions={details['active_dimensions']}/{array.shape[1]}"
    )
    return details


@contextmanager
def isolated_algorithm_import(folder: Path):
    """Isolate local imports for one algorithm so same-named helper modules do not collide."""
    old_path = list(sys.path)
    local_names = {path.stem for path in folder.glob("*.py")}
    local_names.update(
        path.name for path in folder.iterdir()
        if path.is_dir() and (path / "__init__.py").exists()
    )
    saved_modules = {
        name: sys.modules[name] for name in local_names if name in sys.modules
    }
    for name in local_names:
        sys.modules.pop(name, None)
        for loaded_name in list(sys.modules):
            if loaded_name.startswith(name + "."):
                sys.modules.pop(loaded_name, None)
    sys.path.insert(0, str(folder))
    try:
        yield
    finally:
        # Algorithm classes and functions are bound, so local import caches can be restored.
        for name in local_names:
            sys.modules.pop(name, None)
            for loaded_name in list(sys.modules):
                if loaded_name.startswith(name + "."):
                    sys.modules.pop(loaded_name, None)
        sys.modules.update(saved_modules)
        sys.path[:] = old_path


def find_child(root: Path, name: str, directory: bool) -> Path:
    candidate = root / name
    if candidate.exists():
        return candidate
    if root.exists():
        for child in root.iterdir():
            if child.name.lower() == name.lower():
                if directory and child.is_dir():
                    return child
                if not directory and child.is_file():
                    return child
    kind = "directory" if directory else "file"
    raise FileNotFoundError(f"Cannot find {kind} {name} under {root}")


_MODULE_CACHE: Dict[Tuple[str, str], Any] = {}


def load_algorithm_module(
    root: Path,
    algorithm: str,
    dsc_variant: str = "generic",
):
    """Load one original algorithm without leaking its local module imports."""
    folder_name = {"sdcn": "SDCN", "dec": "DEC", "dsc": "DSC"}[algorithm]
    folder = find_child(root, folder_name, directory=True)
    if algorithm == "dsc":
        filename = "yaleb.py" if dsc_variant == "yaleb" else "main.py"
    else:
        filename = {"sdcn": "sdcn.py", "dec": "dec.py"}[algorithm]
    file_path = find_child(folder, filename, directory=False)
    cache_key = (algorithm, str(file_path.resolve()))
    if cache_key in _MODULE_CACHE:
        return _MODULE_CACHE[cache_key]

    variant_name = dsc_variant if algorithm == "dsc" else "base"
    module_name = f"user_algorithm_{algorithm}_{variant_name}"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        with isolated_algorithm_import(folder):
            if algorithm == "dsc":
                # Final clustering is implemented here, so this optional import is removed.
                source = file_path.read_text(encoding="utf-8")
                source = source.replace(
                    "from post_clustering import spectral_clustering, acc, nmi", ""
                )
                exec(compile(source, str(file_path), "exec"), module.__dict__)
                module.acc = cluster_accuracy
                module.nmi = normalized_mutual_info_score
            else:
                spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Missing dependency {exc.name!r} while loading {algorithm.upper()}. "
            f"Keep all original helper files in {folder} and install its dependencies."
        ) from exc
    _MODULE_CACHE[cache_key] = module
    return module

class MLPAutoEncoder(nn.Module):
    """Plain MLP autoencoder for DEC; its encoder can be passed directly to the original DEC class."""

    def __init__(self, n_input: int, n_z: int, hidden: Sequence[int] = (500, 500, 2000)):
        super().__init__()
        dims = [n_input, *hidden, n_z]
        encoder_layers = []
        for index, (left, right) in enumerate(zip(dims[:-1], dims[1:])):
            encoder_layers.append(nn.Linear(left, right))
            if index < len(dims) - 2:
                encoder_layers.append(nn.ReLU(inplace=True))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_dims = list(reversed(dims))
        decoder_layers = []
        for index, (left, right) in enumerate(zip(decoder_dims[:-1], decoder_dims[1:])):
            decoder_layers.append(nn.Linear(left, right))
            if index < len(decoder_dims) - 2:
                decoder_layers.append(nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return self.decoder(z), z


def pretrain_reconstruction(
    model: nn.Module,
    x: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    log_interval: int,
) -> float:
    """Joint reconstruction pretraining. Model output may be a tensor or a (reconstruction, ...) tuple."""
    start = time.time()
    if epochs <= 0:
        return 0.0
    model.train()
    dataset = TensorDataset(torch.from_numpy(np.asarray(x, dtype=np.float32)))
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)
    optimizer = Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        total_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            reconstructed = model(batch)
            if isinstance(reconstructed, (tuple, list)):
                reconstructed = reconstructed[0]
            loss = F.mse_loss(reconstructed, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("Joint reconstruction pretraining produced NaN or Inf; stopping this run")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        if epoch == 0 or (epoch + 1) % log_interval == 0 or epoch + 1 == epochs:
            print(f"  [Joint reconstruction pretraining] {epoch + 1}/{epochs}, MSE={total_loss / len(loader):.6f}")
    return time.time() - start


def _apply_sdae_prefix(
    model: MLPAutoEncoder,
    x: np.ndarray,
    layer_index: int,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Use trained layers through layer_index to generate clean inputs for the next layer."""
    loader = DataLoader(
        TensorDataset(torch.from_numpy(np.asarray(x, dtype=np.float32))),
        batch_size=min(batch_size, len(x)), shuffle=False,
    )
    linear = model.encoder[2 * layer_index]
    is_last = 2 * layer_index == len(model.encoder) - 1
    outputs = []
    with torch.no_grad():
        for (batch,) in loader:
            hidden = linear(batch.to(device))
            if not is_last:
                hidden = F.relu(hidden)
            outputs.append(hidden.cpu())
    return torch.cat(outputs).numpy()


def pretrain_sdae(
    model: MLPAutoEncoder,
    x: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    log_interval: int,
    noise: float,
    finetune_epochs: int,
) -> float:
    """Layer-wise denoising pretraining; each layer reconstructs clean samples from fixed preceding inputs."""
    start = time.time()
    layer_count = len(model.encoder) // 2 + 1
    current = np.asarray(x, dtype=np.float32)
    for layer_index in range(layer_count):
        encoder_layer = model.encoder[2 * layer_index]
        decoder_layer = model.decoder[2 * (layer_count - 1 - layer_index)]
        is_last = layer_index == layer_count - 1
        pair = nn.Sequential(
            encoder_layer,
            nn.Identity() if is_last else nn.ReLU(),
            decoder_layer,
        ).to(device)
        optimizer = Adam(pair.parameters(), lr=lr)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(current)),
            batch_size=min(batch_size, len(current)), shuffle=True,
        )
        pair.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for (clean,) in loader:
                clean = clean.to(device)
                keep_probability = 1.0 - noise
                mask = torch.bernoulli(torch.full_like(clean, keep_probability))
                corrupted = clean * mask
                reconstructed = pair(corrupted)
                loss = F.mse_loss(reconstructed, clean)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"SDAE {layer_index + 1} layer produced NaN or Inf; stopping this run"
                    )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
            if epoch == 0 or (epoch + 1) % log_interval == 0 or epoch + 1 == epochs:
                print(
                    f"  [SDAE layer{layer_index + 1}/{layer_count}] "
                    f"{epoch + 1}/{epochs}, MSE={total_loss / len(loader):.6f}"
                )
        if not is_last:
            current = _apply_sdae_prefix(model, current, layer_index, device, batch_size)
    if finetune_epochs > 0:
        print(f"  [SDAE] Starting end-to-end reconstruction fine-tuning for {finetune_epochs} epochs")
        pretrain_reconstruction(
            model, x, device, finetune_epochs, batch_size, lr, log_interval,
        )
    return time.time() - start


def weight_file(
    output: Path,
    dataset: str,
    algorithm: str,
    n_input: int,
    n_z: int,
    suffix: str = "",
    config: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
) -> Path:
    """Hash the configuration into the filename so incompatible protocols or seeds cannot reuse weights."""
    config_suffix = ""
    if config is not None:
        payload = {
            "dataset": dataset, "algorithm": algorithm, "n_input": n_input,
            "n_z": n_z, "suffix": suffix, "seed": seed, "config": config,
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()[:12]
        mode = safe_name(str(config.get("mode", "pretrain")))
        normalization = safe_name(str(config.get("normalize", "none")))
        config_suffix = f"_{mode}_{normalization}_{digest}"
    name = f"{safe_name(dataset)}_{algorithm}_D{n_input}_Z{n_z}{suffix}{config_suffix}.pt"
    return output / "pretrained" / name


def prepare_weights(
    model: nn.Module,
    x: np.ndarray,
    weight_path: Path,
    device: torch.device,
    args: argparse.Namespace,
    pretrain_lr: float,
    pretrain_epochs: Optional[int] = None,
    pretrain_mode: str = "joint",
    denoise: float = 0.2,
    finetune_epochs: int = 0,
    config: Optional[Dict[str, Any]] = None,
) -> float:
    """Load weights only when configurations match exactly; otherwise pretrain and save both weights and configuration."""
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = args.pretrain_epochs if pretrain_epochs is None else pretrain_epochs
    if weight_path.exists() and not args.force_pretrain:
        model.load_state_dict(torch_load(weight_path, map_location="cpu"))
        print(f"  [Pretraining] Reusing configuration-matched weights {weight_path}")
        return 0.0
    if pretrain_mode == "joint":
        seconds = pretrain_reconstruction(
            model, x, device, epochs, args.batch_size, pretrain_lr, args.log_interval,
        )
    elif pretrain_mode == "sdae":
        if not isinstance(model, MLPAutoEncoder):
            raise TypeError("SDAE pretraining supports only this file's MLPAutoEncoder")
        seconds = pretrain_sdae(
            model, x, device, epochs, args.batch_size, pretrain_lr,
            args.log_interval, denoise, finetune_epochs,
        )
    else:
        raise ValueError(f"Unknown pretraining mode: {pretrain_mode}")
    torch.save(model.state_dict(), weight_path)
    metadata = {
        "pretrain_mode": pretrain_mode,
        "pretrain_lr": pretrain_lr,
        "pretrain_epochs": epochs,
        "batch_size": args.batch_size,
        "denoise": denoise if pretrain_mode == "sdae" else None,
        "finetune_epochs": finetune_epochs if pretrain_mode == "sdae" else None,
        "config": config or {},
    }
    with weight_path.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, default=str)
    print(f"  [Pretraining] Weights saved to {weight_path}")
    return seconds


def target_distribution(q: torch.Tensor) -> torch.Tensor:
    weight = q ** 2 / q.sum(dim=0)
    return (weight.t() / weight.sum(dim=1)).t()


def _graph_to_sparse_tensor(adjacency: Any, n_samples: int, device: torch.device) -> torch.Tensor:
    if isinstance(adjacency, torch.Tensor):
        adj = adjacency.to(device)
        if tuple(adj.shape) != (n_samples, n_samples):
            raise ValueError(f"Expected adjacency shape {(n_samples, n_samples)}; got {tuple(adj.shape)}")
        return adj
    if sp.issparse(adjacency):
        adj = adjacency.tocoo().astype(np.float32)
    else:
        adj = sp.coo_matrix(np.asarray(adjacency, dtype=np.float32))
    if adj.shape != (n_samples, n_samples):
        raise ValueError(f"Expected adjacency shape {(n_samples, n_samples)}; got {adj.shape}")
    indices = np.vstack((adj.row, adj.col))
    tensor = torch.sparse_coo_tensor(
        torch.from_numpy(indices.astype(np.int64)),
        torch.from_numpy(adj.data),
        size=adj.shape,
        dtype=torch.float32,
    ).coalesce()
    return tensor.to(device)


def load_adjacency(path: Path, n_samples: int, device: torch.device) -> torch.Tensor:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        adjacency = np.load(path)
    elif suffix == ".npz":
        try:
            adjacency = sp.load_npz(path)
        except Exception:
            loaded = np.load(path)
            if len(loaded.files) != 1:
                raise ValueError("The graph NPZ must contain one array or be a scipy.sparse.save_npz file")
            adjacency = loaded[loaded.files[0]]
    elif suffix == ".mat":
        mat = _read_mat(path)
        for key in ("adj", "adjacency", "graph", "A"):
            if key in mat:
                adjacency = mat[key]
                break
        else:
            raise KeyError("The graph MAT file must contain adj, adjacency, graph, or A")
    elif suffix in (".txt", ".csv"):
        adjacency = np.loadtxt(path, delimiter="," if suffix == ".csv" else None)
        if adjacency.ndim == 2 and adjacency.shape[1] == 2 and adjacency.shape[0] != n_samples:
            edges = adjacency.astype(np.int64)
            adjacency = sp.coo_matrix(
                (np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n_samples, n_samples)
            )
    else:
        raise ValueError("Graph files support only .npy/.npz/.mat/.txt/.csv")
    return _graph_to_sparse_tensor(adjacency, n_samples, device)


def build_knn_adjacency(x: np.ndarray, k: int, device: torch.device) -> torch.Tensor:
    n_samples = x.shape[0]
    if k < 1 or k >= n_samples:
        raise ValueError(f"KNN k must be in [1, N-1]; got k={k}, N={n_samples}")
    # Pretraining input: symmetric binary KNN graph + self-loops + row normalization.
    # This graph is an SDCN/GCN input, not post-training graph processing.
    neighbors = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(x)
    indices = neighbors.kneighbors(x, return_distance=False)[:, 1:]
    rows = np.repeat(np.arange(n_samples), k)
    cols = indices.reshape(-1)
    adjacency = sp.coo_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n_samples, n_samples)
    )
    adjacency = adjacency.maximum(adjacency.T)
    adjacency = adjacency + sp.eye(n_samples, format="coo", dtype=np.float32)
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    adjacency = sp.diags(1.0 / np.maximum(degree, 1e-12)).dot(adjacency).tocoo()
    return _graph_to_sparse_tensor(adjacency, n_samples, device)


def run_sdcn(
    data: DataBundle,
    args: argparse.Namespace,
    device: torch.device,
    output: Path,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    module = load_algorithm_module(args.root, "sdcn")
    n_z = args.latent_dim or 10
    n_clusters = args.n_clusters or data.n_clusters
    lr = args.learning_rate if args.learning_rate is not None else 1e-3
    pretrain_lr = args.sdcn_pretrain_lr
    pretrain_epochs = (
        args.pretrain_epochs if args.sdcn_pretrain_epochs is None
        else args.sdcn_pretrain_epochs
    )
    epochs = args.epochs or 200
    pretrain_config = {
        "mode": "joint", "normalize": args.normalize, "lr": pretrain_lr,
        "epochs": pretrain_epochs, "batch_size": args.batch_size,
        "architecture": [data.n_features, 500, 500, 2000, n_z],
    }
    weights = weight_file(
        output, data.name, "sdcn", data.n_features, n_z,
        config=pretrain_config, seed=seed,
    )
    ae = module.AE(
        n_enc_1=500, n_enc_2=500, n_enc_3=2000,
        n_dec_1=2000, n_dec_2=500, n_dec_3=500,
        n_input=data.n_features, n_z=n_z,
    ).to(device)
    pretrain_seconds = prepare_weights(
        ae, data.x, weights, device, args, pretrain_lr,
        pretrain_epochs=pretrain_epochs, config=pretrain_config,
    )
    # The GCN is created after pretraining; cache reuse must not alter its random initialization.
    set_seed(seed)

    # Original SDCN.__init__ loads the same AE weights from module args.pretrain_path.
    module.args = SimpleNamespace(pretrain_path=str(weights))
    model = module.SDCN(
        n_enc_1=500, n_enc_2=500, n_enc_3=2000,
        n_dec_1=2000, n_dec_2=500, n_dec_3=500,
        n_input=data.n_features, n_z=n_z, n_clusters=n_clusters, v=1.0,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    all_x = torch.from_numpy(data.x).to(device)
    if args.graph_path:
        adjacency = load_adjacency(args.graph_path, data.n_samples, device)
        graph_description = str(args.graph_path)
    else:
        adjacency = build_knn_adjacency(data.x, args.knn_k, device)
        graph_description = f"symmetric-knn(k={args.knn_k})"

    model.eval()
    with torch.no_grad():
        initial_z = model.ae(all_x)[-1]
    embedding_diagnostic = diagnose_embedding("SDCN-pretrain-z", initial_z)
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed)
    initial_kmeans_pred = kmeans.fit_predict(initial_z.cpu().numpy())
    initial_kmeans_diagnostic = report_prediction("SDCN-initial-KMeans", initial_kmeans_pred, data.y)
    model.cluster_layer.data = torch.as_tensor(
        kmeans.cluster_centers_, dtype=torch.float32, device=device,
    )
    with torch.no_grad():
        _, initial_q, initial_gcn, _ = model(all_x, adjacency)
        initial_p = module.target_distribution(initial_q)
        dist_sq = torch.sum(
            (initial_z.unsqueeze(1) - model.cluster_layer.detach().unsqueeze(0)) ** 2,
            dim=2,
        )
        nearest_two = torch.topk(dist_sq, k=2, dim=1, largest=False).values
        initial_q_pred = initial_q.argmax(dim=1).cpu().numpy()
        initial_gcn_pred = initial_gcn.argmax(dim=1).cpu().numpy()
        initial_p_pred = initial_p.argmax(dim=1).cpu().numpy()
        nearest_pred = dist_sq.argmin(dim=1).cpu().numpy()
        q_diagnostics = {
            "mean_row_std": float(initial_q.std(dim=1).mean().item()),
            "mean_max_probability": float(initial_q.max(dim=1).values.mean().item()),
            "nearest_dist_median": float(nearest_two[:, 0].median().item()),
            "second_dist_median": float(nearest_two[:, 1].median().item()),
            "q_vs_nearest_mismatch": float(np.mean(initial_q_pred != nearest_pred)),
        }
    print(f"  [SDCN-initial-Q] {q_diagnostics}")
    report_prediction("SDCN-initial-Q", initial_q_pred, data.y)
    report_prediction("SDCN-initial-GCN", initial_gcn_pred, data.y)
    report_prediction("SDCN-initial-P", initial_p_pred, data.y)

    gradient_diagnostics: Dict[str, float] = {}
    for epoch in range(epochs):
        model.train()
        x_bar, q, gcn_pred, _ = model(all_x, adjacency)
        p = module.target_distribution(q.detach())
        kl_loss = F.kl_div(q.clamp_min(1e-12).log(), p, reduction="batchmean")
        ce_loss = F.kl_div(gcn_pred.clamp_min(1e-12).log(), p, reduction="batchmean")
        re_loss = F.mse_loss(x_bar, all_x)
        loss = (
            args.sdcn_kl_weight * kl_loss
            + args.sdcn_ce_weight * ce_loss
            + re_loss
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("SDCN clustering produced NaN or Inf")
        optimizer.zero_grad()
        loss.backward()
        if epoch == 0:
            for name in ("gnn_1", "gnn_2", "gnn_3", "gnn_4", "gnn_5"):
                layer = getattr(model, name, None)
                if layer is not None and layer.weight.grad is not None:
                    grad = layer.weight.grad
                    gradient_diagnostics[f"{name}_norm"] = float(grad.norm().item())
                    gradient_diagnostics[f"{name}_absmax"] = float(grad.abs().max().item())
            print(f"  [SDCN-first-gradient] {gradient_diagnostics}")
        optimizer.step()
        if epoch == 0 or (epoch + 1) % args.log_interval == 0 or epoch + 1 == epochs:
            print(
                f"  [SDCN] {epoch + 1}/{epochs}, loss={float(loss.item()):.6f}, "
                f"KL={float(kl_loss.item()):.6f}, CE={float(ce_loss.item()):.6f}, "
                f"RE={float(re_loss.item()):.6f}"
            )

    model.eval()
    with torch.no_grad():
        _, q, gcn_pred, _ = model(all_x, adjacency)
        p = module.target_distribution(q)
        q_pred = q.argmax(dim=1).cpu().numpy()
        gcn_final_pred = gcn_pred.argmax(dim=1).cpu().numpy()
        p_pred = p.argmax(dim=1).cpu().numpy()
        selected = {"q": q_pred, "gcn": gcn_final_pred, "p": p_pred}[args.sdcn_output]
        y_pred = selected.copy()
    report_prediction("SDCN-final-Q", q_pred, data.y)
    report_prediction("SDCN-final-GCN", gcn_final_pred, data.y)
    report_prediction("SDCN-final-P", p_pred, data.y)
    final_diagnostic = report_prediction(f"SDCN-selected-{args.sdcn_output}", y_pred, data.y)
    return y_pred, {
        "pretrain_seconds": pretrain_seconds, "epochs": epochs,
        "latent_dim": n_z, "graph": graph_description,
        "output_distribution": args.sdcn_output,
        "pretrain_type": "joint_reconstruction_mlp",
        "pretrain_lr": pretrain_lr, "pretrain_epochs": pretrain_epochs,
        "cluster_lr": lr, "normalize": args.normalize,
        "kl_weight": args.sdcn_kl_weight, "ce_weight": args.sdcn_ce_weight,
        "initial_embedding": embedding_diagnostic,
        "initial_kmeans": initial_kmeans_diagnostic,
        "initial_q_numeric": q_diagnostics,
        "first_gcn_gradient": gradient_diagnostics,
        "final_selected": final_diagnostic,
    }


# ---------------------------------------------------------------------------
# DEC
# ---------------------------------------------------------------------------

def dec_predict(
    model: nn.Module,
    x: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=min(batch_size, len(x)), shuffle=False)
    predictions = []
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            output = model(batch.to(device))
            predictions.append(output.argmax(dim=1).cpu())
    return torch.cat(predictions).numpy()


def run_dec(
    data: DataBundle,
    args: argparse.Namespace,
    device: torch.device,
    output: Path,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    module = load_algorithm_module(args.root, "dec")
    n_z = args.latent_dim or 10
    n_clusters = args.n_clusters or data.n_clusters
    epochs = args.epochs or 200
    pretrain_lr = args.dec_pretrain_lr
    pretrain_epochs = (
        args.pretrain_epochs if args.dec_pretrain_epochs is None
        else args.dec_pretrain_epochs
    )
    pretrain_config = {
        "mode": args.dec_pretrain_mode, "normalize": args.normalize,
        "lr": pretrain_lr, "epochs_per_layer": pretrain_epochs,
        "finetune_epochs": args.dec_sdae_finetune_epochs,
        "noise": args.dec_sdae_noise, "batch_size": args.batch_size,
        "architecture": [data.n_features, 500, 500, 2000, n_z],
    }
    weights = weight_file(
        output, data.name, "dec", data.n_features, n_z,
        config=pretrain_config, seed=seed,
    )
    autoencoder = MLPAutoEncoder(data.n_features, n_z, hidden=(500, 500, 2000)).to(device)
    pretrain_seconds = prepare_weights(
        autoencoder, data.x, weights, device, args, pretrain_lr,
        pretrain_epochs=pretrain_epochs, pretrain_mode=args.dec_pretrain_mode,
        denoise=args.dec_sdae_noise,
        finetune_epochs=args.dec_sdae_finetune_epochs,
        config=pretrain_config,
    )
    # Cache reuse must produce the same clustering batches as fresh pretraining.
    set_seed(seed)
    model = module.DEC(
        cluster_number=n_clusters, hidden_dimension=n_z,
        encoder=autoencoder.encoder, alpha=1.0,
    ).to(device)

    # Original model.train uses SGD(lr=0.01, momentum=0.9) for clustering.
    lr = args.learning_rate if args.learning_rate is not None else 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model.eval()
    with torch.no_grad():
        features = model.encoder(torch.from_numpy(data.x).to(device)).cpu().numpy()
    embedding_diagnostic = diagnose_embedding("DEC-pretrain-z", features)
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed)
    previous_pred = kmeans.fit_predict(features)
    initial_kmeans_diagnostic = report_prediction("DEC-initial-KMeans", previous_pred, data.y)
    centers = torch.as_tensor(kmeans.cluster_centers_, dtype=torch.float32, device=device)
    with torch.no_grad():
        model.state_dict()["assignment.cluster_centers"].copy_(centers)

    all_x = torch.from_numpy(data.x).to(device)
    model.eval()
    with torch.no_grad():
        initial_q = model(all_x)
        initial_center_distances = torch.pdist(model.assignment.cluster_centers.detach())
        initial_optimization_state = {
            "q_mean_row_std": float(initial_q.std(dim=1).mean().item()),
            "q_mean_max_probability": float(initial_q.max(dim=1).values.mean().item()),
            "min_center_distance": float(initial_center_distances.min().item()),
            "median_center_distance": float(initial_center_distances.median().item()),
        }
    print(f"  [DEC-initial-state] {initial_optimization_state}")

    loader = DataLoader(
        TensorDataset(torch.from_numpy(data.x)),
        batch_size=min(args.batch_size, data.n_samples), shuffle=True,
    )
    stopped_epoch = epochs
    training_history = []
    for epoch in range(epochs):
        model.train()
        losses = []
        for (batch,) in loader:
            batch = batch.to(device)
            output = model(batch)
            target = target_distribution(output).detach()
            loss = F.kl_div(output.clamp_min(1e-12).log(), target, reduction="batchmean")
            if not torch.isfinite(loss):
                raise FloatingPointError("DEC clustering produced NaN or Inf")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            current_q = model(all_x)
            current_z = model.encoder(all_x)
            current_pred = current_q.argmax(dim=1).cpu().numpy()
            center_distances = torch.pdist(model.assignment.cluster_centers.detach())
            optimization_state = {
                "q_mean_row_std": float(current_q.std(dim=1).mean().item()),
                "q_mean_max_probability": float(
                    current_q.max(dim=1).values.mean().item()
                ),
                "z_mean_sample_std": float(current_z.std(dim=1).mean().item()),
                "min_center_distance": float(center_distances.min().item()),
                "median_center_distance": float(center_distances.median().item()),
            }
        delta = float(np.mean(current_pred != previous_pred))
        detailed_epoch = (
            epoch < 20 or epoch == 0
            or (epoch + 1) % args.log_interval == 0 or epoch + 1 == epochs
        )
        if detailed_epoch:
            print(
                f"  [DEC] {epoch + 1}/{epochs}, loss={np.mean(losses):.6f}, "
                f"label_delta={delta:.6f}"
            )
            print(
                "  [DEC-state] "
                f"q_row_std={optimization_state['q_mean_row_std']:.6e}, "
                f"q_max={optimization_state['q_mean_max_probability']:.6e}, "
                f"z_std={optimization_state['z_mean_sample_std']:.6e}, "
                f"center_min={optimization_state['min_center_distance']:.6e}, "
                f"center_median={optimization_state['median_center_distance']:.6e}"
            )
            prediction_diagnostic = report_prediction(f"DEC-epoch-{epoch + 1}", current_pred)
            training_history.append({
                "epoch": epoch + 1, "loss": float(np.mean(losses)),
                "label_delta": delta, **optimization_state,
                "nonempty_clusters": prediction_diagnostic["nonempty_clusters"],
                "max_cluster_size": prediction_diagnostic["max_cluster_size"],
            })
        if args.dec_stopping_delta is not None and delta < args.dec_stopping_delta:
            stopped_epoch = epoch + 1
            print(f"  [DEC] Label-change ratio fell below {args.dec_stopping_delta}; stopping early")
            break
        previous_pred = current_pred

    y_pred = dec_predict(model, data.x, device, args.batch_size)
    final_diagnostic = report_prediction("DEC-final", y_pred, data.y)
    return y_pred, {
        "pretrain_seconds": pretrain_seconds, "epochs": stopped_epoch,
        "latent_dim": n_z,
        "pretrain_type": (
            "stacked_denoising_mlp" if args.dec_pretrain_mode == "sdae"
            else "joint_reconstruction_mlp"
        ),
        "pretrain_lr": pretrain_lr, "pretrain_epochs_per_layer": pretrain_epochs,
        "sdae_noise": args.dec_sdae_noise,
        "sdae_finetune_epochs": args.dec_sdae_finetune_epochs,
        "cluster_lr": lr, "normalize": args.normalize,
        "initial_embedding": embedding_diagnostic,
        "initial_kmeans": initial_kmeans_diagnostic,
        "initial_optimization_state": initial_optimization_state,
        "final": final_diagnostic,
        "training_history": training_history,
    }


# ---------------------------------------------------------------------------
# DSC / DSCNet
# ---------------------------------------------------------------------------

def parse_int_list(text: str, option: str) -> Sequence[int]:
    try:
        values = [int(item.strip()) for item in text.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"{option} must be comma-separated integers, for example 1,15") from exc
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"{option} must contain positive integers")
    return values


def resolve_image_shape(
    n_features: int,
    args: argparse.Namespace,
    dataset_name: str = "",
) -> Tuple[int, int]:
    if args.dsc_image_size:
        height, width = args.dsc_image_size
        if height * width != n_features:
            raise ValueError(
                f"Image height times width must equal the feature count: {height}*{width} != {n_features}"
            )
        return height, width
    profile = get_dataset_profile(dataset_name)
    if profile:
        if profile["image_size"] is None:
            raise ValueError(
                f"Dataset {dataset_name} has no image size in the table and is a plain {n_features}-dimensional vector; convolutional DSC is not applicable."
            )
        height, width = profile["image_size"]
        if height * width != n_features:
            raise ValueError(
                f"Dataset {dataset_name} has table image shape {height}×{width}, "
                f"but the actual feature count is {n_features}. Check the data version or specify --dsc-image-size explicitly."
            )
        print(f"  [DSC] Restoring {dataset_name} to {height}×{width}")
        return height, width
    side = math.isqrt(n_features)
    if side * side != n_features:
        raise ValueError(
            f"DSC is convolutional and cannot treat a plain {n_features}-dimensional vector as an image."
            "Specify --dsc-image-size HEIGHT WIDTH, or run only sdcn/dec for non-image data."
        )
    print(f"  [DSC] No table profile matched; inferring a square image of {side}×{side}")
    return side, side


def dsc_direct_spectral_clustering(
    coefficient: np.ndarray,
    n_clusters: int,
    random_state: Optional[int] = None,
    eigen_solver: str = "lobpcg",
) -> np.ndarray:
    """Construct A=(|C|+|C|^T)/2 directly from C and run spectral clustering without post-processing."""
    from sklearn.cluster import SpectralClustering

    coefficient = np.asarray(coefficient, dtype=np.float64)
    if coefficient.ndim != 2 or coefficient.shape[0] != coefficient.shape[1]:
        raise ValueError(f"DSC self-expression coefficients must be square; got {coefficient.shape}")
    if not np.isfinite(coefficient).all():
        raise ValueError("The DSC self-expression matrix contains NaN or Inf")
    affinity = 0.5 * (np.abs(coefficient) + np.abs(coefficient.T))
    if float(affinity.max()) <= 1e-15:
        raise ValueError("The DSC self-expression matrix is all zero; spectral clustering is impossible")
    spectral = SpectralClustering(
        n_clusters=n_clusters,
        # ARPACK may change the eigenvector basis near repeated eigenvalues because of
        # memory allocation or floating-point differences, even with the same random_state.
        # LOBPCG uses a fixed seed for consistent selection and final reruns. Only the
        # solver changes; the affinity remains A=(|C|+|C|^T)/2 without graph post-processing.
        eigen_solver=eigen_solver,
        affinity="precomputed",
        assign_labels="discretize",
        random_state=random_state,
    )
    return spectral.fit_predict(affinity)

def run_dsc(
    data: DataBundle,
    args: argparse.Namespace,
    device: torch.device,
    output: Path,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    variant = args.dsc_variant
    module = load_algorithm_module(args.root, "dsc", dsc_variant=variant)
    n_clusters = args.n_clusters or data.n_clusters

    if variant == "yaleb":
        if data.n_features != 42 * 48:
            raise ValueError(f"Each YaleB sample must contain 2016 pixels; got {data.n_features}")
        if args.dsc_image_size and tuple(args.dsc_image_size) != (48, 42):
            raise ValueError("yaleb.py fixes the final image size to 48x42; do not specify another --dsc-image-size")
        height, width = 48, 42
        # Match the original nested loop: reshape(v, [42,48]) and transpose; directly
        # reshaping to [48,42] would produce a different pixel order.
        x_image = data.x.reshape(data.n_samples, 1, 42, 48).transpose(0, 1, 3, 2)
        x_image = np.ascontiguousarray(x_image, dtype=np.float32)
        channels_text = args.dsc_channels or "1,10,20,30"
        kernels_text = args.dsc_kernels or "5,3,3"
        epochs = args.epochs or (50 + n_clusters * 25)
        weight_self_exp = args.dsc_weight_self_exp
        if weight_self_exp is None:
            weight_self_exp = 1.0 * 10 ** (n_clusters / 10.0 - 3.0)
        weight_coef = args.dsc_weight_coef if args.dsc_weight_coef is not None else 1.0
    else:
        height, width = resolve_image_shape(data.n_features, args, data.name)
        x_image = data.x.reshape(data.n_samples, 1, height, width)
        profile = get_dataset_profile(data.name) or {}
        dsc_defaults = profile.get("dsc", {})
        channels_text = args.dsc_channels or dsc_defaults.get("channels", "1,15")
        kernels_text = args.dsc_kernels or dsc_defaults.get("kernels", "3")
        epochs = args.epochs or dsc_defaults.get("epochs", 200)
        weight_self_exp = args.dsc_weight_self_exp
        if weight_self_exp is None:
            weight_self_exp = dsc_defaults.get("weight_self_exp", 150.0)
        weight_coef = args.dsc_weight_coef
        if weight_coef is None:
            weight_coef = dsc_defaults.get("weight_coef", 1.0)

    channels = list(parse_int_list(channels_text, "--dsc-channels"))
    kernels = list(parse_int_list(kernels_text, "--dsc-kernels"))
    if channels[0] != 1:
        raise ValueError("MAT vectors are restored as single-channel grayscale images, so the first --dsc-channels value must be 1")
    if len(kernels) != len(channels) - 1:
        raise ValueError("DSC requires len(kernels) = len(channels) - 1")
    lr = args.learning_rate if args.learning_rate is not None else 1e-3
    pretrain_lr = args.dsc_pretrain_lr
    pretrain_epochs = (
        args.pretrain_epochs if args.dsc_pretrain_epochs is None
        else args.dsc_pretrain_epochs
    )
    model = module.DSCNet(channels=channels, kernels=kernels, num_sample=data.n_samples).to(device)

    architecture = "_C" + "-".join(map(str, channels)) + "_K" + "-".join(map(str, kernels))
    pretrain_config = {
        "mode": "joint", "normalize": args.normalize, "lr": pretrain_lr,
        "epochs": pretrain_epochs, "batch_size": args.batch_size,
        "channels": channels, "kernels": kernels, "image_shape": [height, width],
    }
    weights = weight_file(
        output, data.name, f"dsc_{variant}", data.n_features, data.n_features,
        architecture, config=pretrain_config, seed=seed,
    )
    if args.dsc_pretrain_weights:
        if not args.dsc_pretrain_weights.exists():
            raise FileNotFoundError(f"DSC pretrained weights do not exist: {args.dsc_pretrain_weights}")
        model.ae.load_state_dict(torch_load(args.dsc_pretrain_weights, map_location="cpu"))
        pretrain_seconds = 0.0
        pretrain_source = f"user_supplied:{args.dsc_pretrain_weights}"
        print(f"  [DSC-{variant}] Loading specified AE weights {args.dsc_pretrain_weights}")
    else:
        pretrain_source = "self_joint_reconstruction"
        pretrain_seconds = prepare_weights(
            model.ae, x_image, weights, device, args, pretrain_lr,
            pretrain_epochs=pretrain_epochs, config=pretrain_config,
        )

    set_seed(seed)
    optimizer = Adam(model.parameters(), lr=lr)
    all_x = torch.from_numpy(x_image).to(device)
    for epoch in range(epochs):
        model.train()
        x_recon, z, z_recon = model(all_x)
        loss = model.loss_fn(
            all_x, x_recon, z, z_recon,
            weight_coef=weight_coef,
            weight_selfExp=weight_self_exp,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch == 0 or (epoch + 1) % args.log_interval == 0 or epoch + 1 == epochs:
            print(f"  [DSC-{variant}] {epoch + 1}/{epochs}, loss={float(loss.item()):.6f}")

    model.eval()
    coefficient = model.self_expression.Coefficient.detach().cpu().numpy()
    coefficient_path = None
    if args.dsc_save_coefficients:
        coefficient_path = args.dsc_save_coefficients.expanduser().resolve()
        coefficient_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(coefficient_path, coefficient)
        print(f"  [DSC-{variant}] Self-expression coefficients saved to {coefficient_path}")
    y_pred = dsc_direct_spectral_clustering(
        coefficient, n_clusters, random_state=seed,
        eigen_solver=args.dsc_eigen_solver,
    )
    return np.asarray(y_pred, dtype=np.int64), {
        "pretrain_seconds": pretrain_seconds,
        "pretrain_source": pretrain_source,
        "pretrain_lr": pretrain_lr,
        "pretrain_epochs": pretrain_epochs,
        "cluster_lr": lr,
        "normalize": args.normalize,
        "epochs": epochs,
        "variant": variant,
        "final_clustering": "direct_abs_symmetric_adjacency_spectral",
        "eigen_solver": args.dsc_eigen_solver,
        "image_shape": f"{height}x{width}",
        "channels": channels,
        "kernels": kernels,
        "weight_coef": weight_coef,
        "weight_self_exp": weight_self_exp,
        "self_expression_size": f"{data.n_samples}x{data.n_samples}",
        "coefficient_path": str(coefficient_path) if coefficient_path else None,
    }


RUNNERS = {
    "dec": run_dec,
    "dsc": run_dsc,
    "sdcn": run_sdcn,
}


# ---------------------------------------------------------------------------
# Experiment scheduling and result storage
# ---------------------------------------------------------------------------

def save_predictions(
    path: Path,
    data: DataBundle,
    y_pred: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_index", "true_label", "true_label_encoded", "predicted_cluster"])
        for index, (true_encoded, predicted) in enumerate(zip(data.y, y_pred)):
            writer.writerow([
                index,
                data.label_values[true_encoded],
                int(true_encoded),
                int(predicted),
            ])


def append_summary(path: Path, row: Dict[str, Any], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_mean_std_summary(summary_path: Path) -> Path:
    """Aggregate successful runs by normalization, dataset, and algorithm; use sample standard deviation and zero for one run."""
    groups: Dict[Tuple[str, str, str], Dict[str, list]] = {}
    if summary_path.exists():
        with summary_path.open("r", newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if row.get("status") != "ok":
                    continue
                key = (row.get("normalize", ""), row["dataset"], row["algorithm"])
                group = groups.setdefault(key, {"acc": [], "nmi": [], "ari": []})
                for metric in ("acc", "nmi", "ari"):
                    group[metric].append(float(row[metric]))
    output_path = summary_path.with_name("summary_mean_std.csv")
    fields = [
        "dataset", "algorithm", "normalize", "runs", "acc_mean", "acc_std",
        "nmi_mean", "nmi_std", "ari_mean", "ari_std",
    ]
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for (normalize, dataset, algorithm), group in sorted(groups.items()):
            row: Dict[str, Any] = {
                "dataset": dataset, "algorithm": algorithm, "normalize": normalize,
                "runs": len(group["acc"]),
            }
            for metric in ("acc", "nmi", "ari"):
                values = np.asarray(group[metric], dtype=np.float64)
                row[f"{metric}_mean"] = f"{values.mean():.8f}"
                row[f"{metric}_std"] = f"{values.std(ddof=1) if len(values) > 1 else 0.0:.8f}"
            writer.writerow(row)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified MAT-data experiments for DEC, DSC-Net, and SDCN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", nargs="+", type=Path, required=True, help="One or more .mat files")
    parser.add_argument("--algorithms", nargs="+", choices=ALGORITHMS, default=list(ALGORITHMS))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent,
                        help="Root directory containing the three algorithm folders")
    parser.add_argument("--output", type=Path, default=Path("experiment_results"), help="Output directory")
    parser.add_argument("--x-key", default=None, help="Feature matrix variable such as fea; inferred when omitted")
    parser.add_argument("--y-key", default=None, help="Label variable such as gnd; inferred when omitted")
    parser.add_argument("--sample-axis", choices=("columns", "rows", "auto"), default="rows",
                        help="Sample axis in fea: rows keeps orientation, columns transposes, auto uses label length")
    parser.add_argument(
        "--normalize", choices=("none", "l2", "global_max"), default="none",
        help="none: no scaling; l2: unit L2 norm per sample; global_max: divide the matrix by its largest absolute value",
    )
    parser.add_argument(
        "--normalizations", nargs="+", choices=("none", "l2", "global_max"),
        default=None,
        help="Optional list of preprocessing methods; overrides --normalize and isolates outputs",
    )
    parser.add_argument("--n-clusters", type=int, default=None, help="Defaults to the number of distinct true labels")
    parser.add_argument("--runs", type=int, default=1, help="Independent runs per algorithm")
    parser.add_argument("--seed", type=int, default=42, help="Run r uses seed+r")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic CUDNN operations where possible")
    parser.add_argument("--epochs", type=int, default=None, help="Clustering epochs; defaults to each algorithm's original setting")
    parser.add_argument("--pretrain-epochs", type=int, default=50)
    parser.add_argument("--force-pretrain", action="store_true", help="Ignore cached autoencoder weights in the output directory")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="Clustering learning rate; defaults to each algorithm's original setting")
    parser.add_argument("--latent-dim", type=int, default=None,
                        help="Latent dimension; SDCN and DEC default to 10")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--fail-fast", action="store_true", help="Stop immediately when an experiment fails")

    sdcn = parser.add_argument_group("SDCN")
    sdcn.add_argument("--knn-k", type=int, default=3, help="Construct a symmetric KNN graph when no graph file is supplied")
    sdcn.add_argument("--graph-path", type=Path, default=None,
                      help="Optional adjacency .npy/.npz/.mat/.txt/.csv; do not share it across datasets")
    sdcn.add_argument("--sdcn-output", choices=("q", "gcn", "p"), default="q",
                      help="Distribution used for final labels; q is recommended for formal comparisons")
    sdcn.add_argument("--sdcn-pretrain-lr", type=float, default=1e-4)
    sdcn.add_argument(
        "--sdcn-pretrain-epochs", type=int, default=None,
        help="Defaults to --pretrain-epochs when omitted",
    )
    sdcn.add_argument("--sdcn-kl-weight", type=float, default=0.1)
    sdcn.add_argument("--sdcn-ce-weight", type=float, default=0.01)

    dec = parser.add_argument_group("DEC")
    dec.add_argument("--dec-stopping-delta", type=float, default=0.001,
                     help="Early-stopping threshold for label changes; a negative value disables it")
    dec.add_argument(
        "--dec-pretrain-mode", choices=("sdae", "joint"), default="sdae",
        help="sdae: layer-wise denoising pretraining; joint: joint-reconstruction diagnostic",
    )
    dec.add_argument("--dec-pretrain-lr", type=float, default=1e-4)
    dec.add_argument(
        "--dec-pretrain-epochs", type=int, default=None,
        help="Epochs per SDAE layer; defaults to --pretrain-epochs",
    )
    dec.add_argument("--dec-sdae-noise", type=float, default=0.2,
                     help="Random input-feature dropout ratio during layer-wise pretraining")
    dec.add_argument("--dec-sdae-finetune-epochs", type=int, default=0,
                     help="End-to-end reconstruction fine-tuning epochs after layer-wise pretraining; 0 disables it")

    dsc = parser.add_argument_group("DSC")
    dsc.add_argument("--dsc-eigen-solver", choices=("lobpcg", "arpack"), default="lobpcg",
                     help="Spectral eigensolver; use arpack to reproduce the legacy L2 script")
    dsc.add_argument("--dsc-variant", choices=("generic", "yaleb"), default="generic",
                     help="generic calls main.py; yaleb is only for original 2016-D YaleBCrop025, not 1024-D table YALEB")
    dsc.add_argument("--dsc-image-size", nargs=2, type=int, metavar=("HEIGHT", "WIDTH"), default=None,
                     help="Image height and width for generic DSC; yaleb fixes them to 48 42")
    dsc.add_argument("--dsc-channels", default=None,
                     help="Defaults: generic 1,15; yaleb 1,10,20,30")
    dsc.add_argument("--dsc-kernels", default=None,
                     help="Defaults: generic 3; yaleb 5,3,3")
    dsc.add_argument("--dsc-pretrain-weights", type=Path, default=None,
                     help="Original convolutional AE weights; intended for single-dataset runs")
    dsc.add_argument("--dsc-save-coefficients", type=Path, default=None,
                     help="Optionally save final self-expression coefficients C as .npy for inspection or reuse")
    dsc.add_argument("--dsc-pretrain-lr", type=float, default=1e-3)
    dsc.add_argument(
        "--dsc-pretrain-epochs", type=int, default=None,
        help="Defaults to --pretrain-epochs when omitted",
    )
    dsc.add_argument("--dsc-weight-coef", type=float, default=None)
    dsc.add_argument("--dsc-weight-self-exp", type=float, default=None,
                     help="yaleb default: 10^(K/10-3); generic default: 150")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.runs < 1:
        raise ValueError("--runs must be at least 1")
    if args.batch_size < 1 or args.pretrain_epochs < 0 or args.log_interval < 1:
        raise ValueError("batch-size/log-interval must be positive; pretrain-epochs cannot be negative")
    if args.epochs is not None and args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.n_clusters is not None and args.n_clusters < 2:
        raise ValueError("--n-clusters must be at least 2")
    if args.latent_dim is not None and args.latent_dim < 1:
        raise ValueError("--latent-dim must be positive")
    for name in (
        "sdcn_pretrain_lr", "dec_pretrain_lr", "dsc_pretrain_lr",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.learning_rate is not None and args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    for name in (
        "sdcn_pretrain_epochs", "dec_pretrain_epochs", "dsc_pretrain_epochs",
    ):
        value = getattr(args, name)
        if value is not None and value < 0:
            raise ValueError(f"--{name.replace('_', '-')} cannot be negative")
    if not 0.0 <= args.dec_sdae_noise < 1.0:
        raise ValueError("--dec-sdae-noise must be in [0, 1)")
    if args.dec_sdae_finetune_epochs < 0:
        raise ValueError("--dec-sdae-finetune-epochs cannot be negative")
    if args.sdcn_kl_weight < 0 or args.sdcn_ce_weight < 0:
        raise ValueError("SDCN loss weights cannot be negative")
    args.root = args.root.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.data = [path.expanduser().resolve() for path in args.data]
    if args.graph_path:
        args.graph_path = args.graph_path.expanduser().resolve()
    if args.dsc_pretrain_weights:
        args.dsc_pretrain_weights = args.dsc_pretrain_weights.expanduser().resolve()


def run_single_protocol(args: argparse.Namespace, device: torch.device) -> int:
    """Run one preprocessing protocol; ``args.output`` is already isolated."""
    args.output.mkdir(parents=True, exist_ok=True)
    summary_path = args.output / "summary.csv"
    if summary_path.exists():
        summary_path.unlink()
    with (args.output / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, ensure_ascii=False, indent=2, default=str)

    fields = [
        "dataset", "algorithm", "normalize", "seed", "n_samples", "n_features", "n_clusters",
        "acc", "nmi", "ari", "runtime_seconds", "pretrain_seconds", "status",
        "error", "details",
    ]
    # Create the summary at startup even if every data path is invalid.
    with summary_path.open("w", newline="", encoding="utf-8-sig") as handle:
        csv.DictWriter(handle, fieldnames=fields).writeheader()
    print(f"[Environment] device={device}, root={args.root}, output={args.output}")
    experiment_count = 0
    success_count = 0
    failure_count = 0

    for data_path in args.data:
        try:
            data = load_dataset(data_path, args)
        except Exception as exc:
            failure_count += 1
            print(f"[Data error] {data_path}: {exc}")
            if args.fail_fast:
                raise
            continue

        for algorithm in args.algorithms:
            profile = get_dataset_profile(data.name)
            if (
                algorithm == "dsc" and profile and profile["image_size"] is None
                and not args.dsc_image_size and args.dsc_variant == "generic"
            ):
                reason = "The table profile has no image size; DSC is not applicable to plain vector data"
                print(f"\n=== {data.name} | DSC: skipped ({reason}) ===")
                append_summary(summary_path, {
                    "dataset": data.name, "algorithm": algorithm,
                    "normalize": args.normalize, "seed": "",
                    "n_samples": data.n_samples, "n_features": data.n_features,
                    "n_clusters": args.n_clusters or data.n_clusters,
                    "acc": "", "nmi": "", "ari": "", "runtime_seconds": "0",
                    "pretrain_seconds": "0", "status": "skipped", "error": "",
                    "details": json.dumps({"reason": reason}, ensure_ascii=False),
                }, fields)
                continue
            if algorithm == "dsc" and data.n_samples > 5000:
                print(
                    f"  [Warning] The DSC self-expression matrix contains {data.n_samples ** 2:,} elements; "
                    "Large datasets may exceed GPU or host memory."
                )
            for run_index in range(args.runs):
                seed = args.seed + run_index
                experiment_count += 1
                set_seed(seed)
                print(f"\n=== {data.name} | {algorithm.upper()} | seed={seed} ===")
                start = time.time()
                try:
                    y_pred, details = RUNNERS[algorithm](data, args, device, args.output, seed)
                    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
                    if y_pred.size != data.n_samples:
                        raise RuntimeError(
                            f"Predicted label count is {y_pred.size}, but sample count is {data.n_samples}"
                        )
                    metrics = evaluate_clustering(data.y, y_pred)
                    success_count += 1
                    runtime = time.time() - start
                    pretrain_seconds = float(details.get("pretrain_seconds", 0.0))
                    row = {
                        "dataset": data.name,
                        "algorithm": algorithm,
                        "normalize": args.normalize,
                        "seed": seed,
                        "n_samples": data.n_samples,
                        "n_features": data.n_features,
                        "n_clusters": args.n_clusters or data.n_clusters,
                        "acc": f"{metrics['acc']:.8f}",
                        "nmi": f"{metrics['nmi']:.8f}",
                        "ari": f"{metrics['ari']:.8f}",
                        "runtime_seconds": f"{runtime:.4f}",
                        "pretrain_seconds": f"{pretrain_seconds:.4f}",
                        "status": "ok",
                        "error": "",
                        "details": json.dumps(details, ensure_ascii=False, default=str),
                    }
                    prediction_name = (
                        f"{safe_name(data.name)}_{algorithm}_seed{seed}_predictions.csv"
                    )
                    save_predictions(args.output / "predictions" / prediction_name, data, y_pred)
                    print(
                        f"[Result] ACC={metrics['acc']:.4f}, NMI={metrics['nmi']:.4f}, "
                        f"ARI={metrics['ari']:.4f}, total time={runtime:.2f}s"
                    )
                except Exception as exc:
                    failure_count += 1
                    runtime = time.time() - start
                    row = {
                        "dataset": data.name,
                        "algorithm": algorithm,
                        "normalize": args.normalize,
                        "seed": seed,
                        "n_samples": data.n_samples,
                        "n_features": data.n_features,
                        "n_clusters": args.n_clusters or data.n_clusters,
                        "acc": "", "nmi": "", "ari": "",
                        "runtime_seconds": f"{runtime:.4f}",
                        "pretrain_seconds": "",
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                        "details": "",
                    }
                    print(f"[Experiment failed] {type(exc).__name__}: {exc}")
                    traceback.print_exc()
                    if args.fail_fast:
                        append_summary(summary_path, row, fields)
                        raise
                append_summary(summary_path, row, fields)

    mean_std_path = write_mean_std_summary(summary_path)
    print(
        f"\nExperiment finished: attempted {experiment_count} algorithm runs; succeeded {success_count}; "
        f"failed records {failure_count} (including data-loading errors)."
        f"\nPer-run results: {summary_path}\nMean and standard deviation: {mean_std_path}"
    )
    return 0 if success_count > 0 else 1


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access CUDA")
    if args.device == "cuda":
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    normalizations = list(args.normalizations or [args.normalize])
    if len(set(normalizations)) != len(normalizations):
        parser.error("--normalizations contains duplicate methods")
    base_output = args.output
    exit_codes = []
    for method in normalizations:
        protocol_args = argparse.Namespace(**vars(args))
        protocol_args.normalize = method
        protocol_args.output = (
            base_output if len(normalizations) == 1 else base_output / method
        )
        print(
            f"\n[Protocol] normalize={method}; "
            f"output={protocol_args.output}"
        )
        exit_codes.append(run_single_protocol(protocol_args, device))
    return 0 if exit_codes and all(code == 0 for code in exit_codes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
