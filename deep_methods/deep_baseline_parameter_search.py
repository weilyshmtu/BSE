"""Parameter-influence search for DEC, DSC-Net, and SDCN.

The script reuses the runners in ``deep_baseline_experiments.py`` and preserves the
original per-normalization grids. Both Python files and the DEC/DSC/SDCN folders
belong in ``deep_methods``. The default data directory is the sibling ``datasets``
folder of that directory (equivalently, the sibling of a custom --root).
- DEC: 40 core configurations and up to 3 refinements.
- DSC-Net: 80 core configurations and pretraining-epoch refinement when no
  external DSC autoencoder weights are supplied.
- SDCN: 288 core configurations and 4 latent-dimension refinements.

All three preprocessing methods (none, l2, and global_max) are searched by
default. Each method has an isolated output directory. Configuration selection
uses seed 42, ranks by mean ACC, and breaks ties by mean NMI. The selected
configuration is rerun once with seed 42. Final labels and metrics always come
from the final epoch, not from the best training epoch.

Quick pipeline check on ORL with one preprocessing method:
    python deep_baseline_parameter_search.py --quick --datasets ORL_32x32         --normalizations none

Full search with the default algorithms, datasets, and preprocessing methods:
    python deep_baseline_parameter_search.py --device cuda

Resume by repeating the same command. A successful record is reused only when
its dataset, algorithm, configuration ID, seed, and complete configuration JSON
all match. Use --restart to clear result records while retaining cached weights.
Start a new output directory when changing the experimental protocol.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


SEED = 42
SUPPORTED_NORMALIZATIONS = ("none", "l2", "global_max")

DATASET_FILES = [
    "zoo.mat",
    "Yale_32x32.mat",
    "JAFFE_64x64.mat",
    "ORL_32x32.mat",
    "Umist_575_32x32.mat",
    "FERET_32x32.mat",
    "COIL20_32x32.mat",
    "AR_32x32.mat",
    "YALEB_32x32.mat",
    "pendigits.mat",
    "COIL100_32x32.mat",
    "MNIST_10k_Train_28x28.mat",
]

RESULT_FIELDS = [
    "dataset", "algorithm", "stage", "config_id", "seed",
    "acc", "nmi", "ari", "nonempty_clusters", "max_cluster_size",
    "runtime_seconds", "pretrain_seconds", "status", "error", "config_json",
]


def load_entry(root: Path):
    entry_path = root / "deep_baseline_experiments.py"
    if not entry_path.exists():
        raise FileNotFoundError(f"Cannot find the unified runner: {entry_path}")
    spec = importlib.util.spec_from_file_location("deep_methods_tuning_entry", entry_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {entry_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def product_configs(grid: Dict[str, Sequence[Any]]) -> List[Dict[str, Any]]:
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def with_ids(configs: List[Dict[str, Any]], prefix: str, stage: str) -> List[Dict[str, Any]]:
    output = []
    for index, config in enumerate(configs, 1):
        item = dict(config)
        item["_config_id"] = f"{prefix}-{index:03d}"
        item["_stage"] = stage
        output.append(item)
    return output


def public_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in config.items() if not k.startswith("_")}


def config_digest(config: Dict[str, Any]) -> str:
    payload = json.dumps(
        public_config(config), ensure_ascii=False, sort_keys=True, default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def dsc_defaults(base: Any, dataset_name: str) -> Dict[str, Any]:
    profile = base.get_dataset_profile(dataset_name) or {}
    values = profile.get("dsc", {})
    return {
        "epochs": values.get("epochs", 200),
        "weight_coef": values.get("weight_coef", 1.0),
        "weight_self_exp": values.get("weight_self_exp", 150.0),
    }


def core_configs(
    base: Any,
    algorithm: str,
    dataset_name: str,
    n_samples: int,
    dsc_pretrain_weights: Optional[Path],
    normalize: str,
    dsc_eigen_solver: str,
) -> List[Dict[str, Any]]:
    if algorithm == "dec":
        common = {
            "dec_pretrain_mode": "sdae", "dec_pretrain_lr": 1e-4,
            "pretrain_epochs": 50, "dec_sdae_finetune_epochs": 0,
            "dec_stopping_delta": 0.001, "epochs": 200,
            "normalize": normalize,
        }
        grid = {
            "latent_dim": [10, 20, 30, 40, 50],
            "dec_sdae_noise": [0.1, 0.2],
            "learning_rate": [1e-3, 1e-4],
            "batch_size": [256, n_samples],
        }
        configs = [dict(common, **c) for c in product_configs(grid)]
        return with_ids(configs, "DEC-C", "core")

    if algorithm == "dsc":
        defaults = dsc_defaults(base, dataset_name)
        common = {
            "epochs": defaults["epochs"],
            "dsc_pretrain_weights": dsc_pretrain_weights,
            "dsc_pretrain_lr": 1e-4, "dsc_pretrain_epochs": 50,
            "normalize": normalize, "dsc_eigen_solver": dsc_eigen_solver,
        }
        grid = {
            "learning_rate": [1e-3, 1e-4],
            "_coef_factor": [0.1, 0.5, 1.0, 2.0, 5, 10.0, 20, 50],
            "_self_factor": [0.1, 1.0, 5, 10.0, 50],
        }
        configs = []
        for item in product_configs(grid):
            item["dsc_weight_coef"] = defaults["weight_coef"] * item.pop("_coef_factor")
            item["dsc_weight_self_exp"] = defaults["weight_self_exp"] * item.pop("_self_factor")
            configs.append(dict(common, **item))
        return with_ids(configs, "DSC-T", "core")

    if algorithm == "sdcn":
        common = {
            "normalize": normalize, "sdcn_pretrain_lr": 1e-4,
            "pretrain_epochs": 50, "batch_size": 256, "latent_dim": 10,
            "sdcn_output": "q", "epochs": 200, "graph_path": None,
        }
        grid = {
            "learning_rate": [1e-3, 1e-4],
            # Reconstruction weight remains 1; KL and CE weights vary independently.
            "sdcn_kl_weight": [0.001, 0.01, 0.1, 1.0, 5, 10],
            "sdcn_ce_weight": [0.0001, 0.001, 0.01, 0.1, 1, 10],
            "knn_k": [3, 5, 10, 20],
        }
        return with_ids(
            [dict(common, **item) for item in product_configs(grid)],
            "SDCN-C", "core",
        )

    raise ValueError(f"Unknown algorithm: {algorithm}")


def refine_configs(best: Dict[str, Any], algorithm: str) -> List[Dict[str, Any]]:
    base_config = public_config(best)
    configs: List[Dict[str, Any]] = []
    if algorithm == "dec":
        for key, value in (
            ("pretrain_epochs", 100),
            ("dec_sdae_finetune_epochs", 50),
            ("dec_sdae_noise", 0.05),
        ):
            item = dict(base_config)
            item[key] = value
            configs.append(item)
    elif algorithm == "dsc":
        if base_config.get("dsc_pretrain_weights") is None:
            for epochs in [50, 100]:
                item = dict(base_config)
                item["dsc_pretrain_epochs"] = epochs
                configs.append(item)
    if algorithm == "sdcn":
        for latent_dim in (20, 30, 40, 50):
            item = dict(base_config)
            item["latent_dim"] = latent_dim
            configs.append(item)
    # Do not repeat the core optimum or emit duplicate refinement configurations.
    unique_configs = []
    seen = set()
    for item in configs:
        signature = config_digest(item)
        if public_config(item) == base_config or signature in seen:
            continue
        seen.add(signature)
        unique_configs.append(item)
    prefix = {"dec": "DEC-R", "dsc": "DSC-R", "sdcn": "SDCN-R"}[algorithm]
    return with_ids(unique_configs, prefix, "refine")


def read_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def append_row(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in RESULT_FIELDS})


def result_key(row: Dict[str, Any]) -> Tuple[str, str, str, int, str]:
    # Include the full configuration in resume keys so quick, full, or edited grids cannot be mixed.
    return (
        row["dataset"], row["algorithm"], row["config_id"], int(row["seed"]),
        row.get("config_json", ""),
    )


def completed_keys(rows: Iterable[Dict[str, Any]]) -> set:
    return {result_key(row) for row in rows if row.get("status") == "ok"}


def make_namespace(
    base: Any,
    cli: argparse.Namespace,
    config: Dict[str, Any],
    data_path: Path,
    cache_dir: Path,
):
    algorithm = config["_algorithm"]
    namespace = base.build_parser().parse_args([
        "--data", str(data_path), "--algorithms", algorithm,
        "--root", str(cli.root), "--output", str(cache_dir),
        "--device", cli.device, "--log-interval", str(cli.log_interval),
    ])
    base.validate_args(namespace)
    for key, value in public_config(config).items():
        if not hasattr(namespace, key):
            raise KeyError(f"The unified runner does not support parameter {key}")
        if key in {"graph_path", "dsc_pretrain_weights", "dsc_save_coefficients"} and value is not None:
            value = Path(value)
        setattr(namespace, key, value)
    namespace.output = cache_dir.resolve()
    namespace.device = cli.device
    namespace.deterministic = cli.deterministic
    namespace.force_pretrain = cli.force_pretrain
    namespace.runs = 1
    return namespace


def execute_config(
    base: Any,
    cli: argparse.Namespace,
    config: Dict[str, Any],
    data_path: Path,
    data: Any,
    device: Any,
    cache_dir: Path,
    seed: int,
) -> Tuple[Dict[str, Any], Optional[np.ndarray], Dict[str, Any]]:
    algorithm = config["_algorithm"]
    namespace = make_namespace(base, cli, config, data_path, cache_dir)
    base.set_seed(seed)
    start = time.time()
    try:
        y_pred, details = base.RUNNERS[algorithm](
            data, namespace, device, cache_dir, seed,
        )
        y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
        metrics = base.evaluate_clustering(data.y, y_pred)
        _, counts = np.unique(y_pred, return_counts=True)
        row = {
            "dataset": data.name, "algorithm": algorithm,
            "stage": config.get("_stage", "core"),
            "config_id": config["_config_id"], "seed": seed,
            **{key: f"{metrics[key]:.8f}" for key in ("acc", "nmi", "ari")},
            "nonempty_clusters": len(counts), "max_cluster_size": int(counts.max()),
            "runtime_seconds": f"{time.time() - start:.4f}",
            "pretrain_seconds": f"{float(details.get('pretrain_seconds', 0.0)):.4f}",
            "status": "ok", "error": "",
            "config_json": json.dumps(public_config(config), ensure_ascii=False, default=str),
        }
        return row, y_pred, details
    except Exception as exc:
        row = {
            "dataset": data.name, "algorithm": algorithm,
            "stage": config.get("_stage", "core"),
            "config_id": config["_config_id"], "seed": seed,
            "acc": "", "nmi": "", "ari": "",
            "nonempty_clusters": "", "max_cluster_size": "",
            "runtime_seconds": f"{time.time() - start:.4f}",
            "pretrain_seconds": "", "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "config_json": json.dumps(public_config(config), ensure_ascii=False, default=str),
        }
        return row, None, {}


def is_quick_row(row: Dict[str, Any]) -> bool:
    """Identify quick records in old result files; new default output directories are also separated."""
    try:
        config = json.loads(row.get("config_json", "{}"))
        pretrain_epochs = config.get("pretrain_epochs", config.get("dsc_pretrain_epochs"))
        return int(config.get("epochs", -1)) == 2 and int(pretrain_epochs) == 1
    except (TypeError, ValueError, json.JSONDecodeError):
        return False


def select_best(
    rows: List[Dict[str, Any]],
    dataset: str,
    algorithm: str,
    selection_seeds: Sequence[int],
    stage: Optional[str] = None,
    quick: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    # config_id may repeat between quick and full searches, so group by configuration content too.
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        if row["dataset"] != dataset or row["algorithm"] != algorithm:
            continue
        if is_quick_row(row) != quick:
            continue
        if stage is not None and row["stage"] != stage:
            continue
        key = (row["config_id"], row.get("config_json", ""))
        grouped.setdefault(key, []).append(row)
    candidates = []
    required = set(selection_seeds)
    for (config_id, _), items in grouped.items():
        successful = [
            row for row in items
            if row["status"] == "ok" and int(row["seed"]) in required
        ]
        # Require exactly one successful row per selection seed; never average different configurations.
        if len(successful) != len(required) or {int(row["seed"]) for row in successful} != required:
            continue
        successful.sort(key=lambda row: int(row["seed"]))
        acc = np.asarray([float(row["acc"]) for row in successful])
        nmi = np.asarray([float(row["nmi"]) for row in successful])
        ari = np.asarray([float(row["ari"]) for row in successful])
        config = json.loads(successful[0]["config_json"])
        config["_config_id"] = config_id
        config["_stage"] = successful[0]["stage"]
        config["_algorithm"] = algorithm
        score = {
            "selection_mean_acc": float(acc.mean()),
            "selection_std_acc": float(acc.std(ddof=1)) if len(acc) > 1 else 0.0,
            "selection_mean_nmi": float(nmi.mean()),
            "selection_mean_ari": float(ari.mean()),
        }
        candidates.append((score["selection_mean_acc"], score["selection_mean_nmi"], config_id, config, score))
    if not candidates:
        raise RuntimeError(f"{dataset} / {algorithm} has no configuration that completed every selection seed")
    _, _, _, best_config, best_score = max(candidates, key=lambda item: (item[0], item[1], item[2]))
    return best_config, best_score


def get_data(base: Any, namespace: Any, data_path: Path, cache: Dict[Tuple[str, str], Any]):
    key = (str(data_path), namespace.normalize)
    if key not in cache:
        cache[key] = base.load_dataset(data_path, namespace)
    return cache[key]


def evaluate_generic_configs(
    base: Any,
    cli: argparse.Namespace,
    configs: List[Dict[str, Any]],
    data_path: Path,
    data_cache: Dict[Tuple[str, str], Any],
    device: Any,
    cache_dir: Path,
    results_path: Path,
    all_rows: List[Dict[str, Any]],
    done: set,
) -> None:
    for config in configs:
        namespace = make_namespace(base, cli, config, data_path, cache_dir)
        data = get_data(base, namespace, data_path, data_cache)
        for seed in cli.selection_seeds:
            probe = {
                "dataset": data.name, "algorithm": config["_algorithm"],
                "config_id": config["_config_id"], "seed": seed,
                "config_json": json.dumps(public_config(config), ensure_ascii=False, default=str),
            }
            if result_key(probe) in done:
                continue
            print(f"\n[Search] {data.name} | {config['_algorithm']} | {config['_config_id']} | seed={seed}")
            print(f"  Parameters: {json.dumps(public_config(config), ensure_ascii=False, default=str)}")
            row, _, _ = execute_config(
                base, cli, config, data_path, data, device, cache_dir, seed,
            )
            append_row(results_path, row)
            all_rows.append(row)
            done.add(result_key(row))
            print(f"  Status={row['status']}, ACC={row['acc']}, error={row['error']}")
            if row["status"] != "ok" and cli.fail_fast:
                raise RuntimeError(row["error"])


def run_final_repetitions(
    base: Any,
    cli: argparse.Namespace,
    best: Dict[str, Any],
    score: Dict[str, Any],
    data_path: Path,
    data_cache: Dict[Tuple[str, str], Any],
    device: Any,
    cache_dir: Path,
    final_path: Path,
    all_final_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    algorithm = best["_algorithm"]
    best = dict(best)
    best["_config_id"] = "BEST"
    best["_stage"] = "final"
    best_config_json = json.dumps(public_config(best), ensure_ascii=False, default=str)
    namespace = make_namespace(base, cli, best, data_path, cache_dir)
    data = get_data(base, namespace, data_path, data_cache)
    done = completed_keys(all_final_rows)
    prediction_dir = cli.output / data.name / algorithm / "predictions"
    for seed in range(cli.final_seed_start, cli.final_seed_start + cli.final_runs):
        probe = {
            "dataset": data.name, "algorithm": algorithm,
            "config_id": "BEST", "seed": seed,
            "config_json": json.dumps(public_config(best), ensure_ascii=False, default=str),
        }
        if result_key(probe) in done:
            continue
        print(f"\n[Final rerun] {data.name} | {algorithm} | seed={seed}")
        row, y_pred, _ = execute_config(
            base, cli, best, data_path, data, device, cache_dir, seed,
        )
        append_row(final_path, row)
        all_final_rows.append(row)
        done.add(result_key(row))
        if row["status"] == "ok" and y_pred is not None:
            base.save_predictions(
                prediction_dir / f"seed{seed}_predictions.csv", data, y_pred,
            )
        elif cli.fail_fast:
            raise RuntimeError(row["error"])

    successful = [
        row for row in all_final_rows
        if row["dataset"] == data.name and row["algorithm"] == algorithm
        and row["config_id"] == "BEST" and row["status"] == "ok"
        and row.get("config_json", "") == best_config_json
        and cli.final_seed_start <= int(row["seed"]) < cli.final_seed_start + cli.final_runs
    ]
    final_seeds = [int(row["seed"]) for row in successful]
    if len(final_seeds) != len(set(final_seeds)):
        raise RuntimeError(f"{data.name} / {algorithm} has duplicate final seeds; use a new output directory and rerun")
    summary: Dict[str, Any] = {
        "dataset": data.name, "algorithm": algorithm,
        "normalize": public_config(best).get("normalize", ""),
        "runs": len(successful), "expected_runs": cli.final_runs,
        **score, "best_config": public_config(best),
    }
    for metric in ("acc", "nmi", "ari"):
        values = np.asarray([float(row[metric]) for row in successful], dtype=np.float64)
        if values.size:
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        else:
            summary[f"{metric}_mean"] = None
            summary[f"{metric}_std"] = None
    method_dir = cli.output / data.name / algorithm
    method_dir.mkdir(parents=True, exist_ok=True)
    with (method_dir / "best_config.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, default=str)
    return summary


def write_final_summary(path: Path, summaries: List[Dict[str, Any]]) -> None:
    fields = [
        "dataset", "algorithm", "normalize", "runs", "expected_runs",
        "selection_mean_acc", "selection_mean_nmi",
        "acc_mean", "acc_std", "nmi_mean", "nmi_std", "ari_mean", "ari_std",
        "best_config",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            row = {key: summary.get(key, "") for key in fields}
            row["best_config"] = json.dumps(summary["best_config"], ensure_ascii=False, default=str)
            writer.writerow(row)


def resolve_dataset_path(name: str, data_dir: Path) -> Path:
    candidate = Path(name).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    if candidate.suffix.lower() != ".mat":
        candidate = candidate.with_suffix(".mat")
    path = (data_dir / candidate).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Cannot find dataset: {path}")
    return path


def build_cli() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Dataset-specific parameter search for DEC, DSC-Net, and SDCN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root", type=Path, default=root,
                        help="Directory containing the unified runner and algorithm folders")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Dataset directory; defaults to the sibling datasets directory of --root")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output directory; quick and full searches use different defaults")
    parser.add_argument("--datasets", nargs="+", default=DATASET_FILES,
                        help="Dataset names or MAT paths; defaults to all 12 datasets")
    parser.add_argument("--all-datasets", action="store_true",
                        help="Use the built-in list of 12 dataset filenames, including MNIST")
    parser.add_argument("--algorithms", nargs="+", choices=("dec", "dsc", "sdcn"),
                        default=["dec", "dsc", "sdcn"])
    parser.add_argument("--normalizations", nargs="+",
                        choices=SUPPORTED_NORMALIZATIONS,
                        default=list(SUPPORTED_NORMALIZATIONS),
                        help="Preprocessing protocols to search; each protocol is isolated")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--deterministic", action="store_true",
                        help="Enable deterministic CUDNN operations where possible")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--dsc-pretrain-weights", type=Path, default=None,
                        help="Optional original DSC autoencoder weights for single-dataset searches")
    parser.add_argument("--dsc-eigen-solver", choices=("lobpcg", "arpack"),
                        default="lobpcg",
                        help="Use arpack only to reproduce the legacy L2 search protocol")
    parser.add_argument("--force-pretrain", action="store_true")
    parser.add_argument("--skip-refine", action="store_true")
    parser.add_argument("--restart", action="store_true",
                        help="Clear search and final records but retain reusable pretrained weights")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--quick", action="store_true",
                        help="Two core configurations per algorithm, 2 epochs, 1 pretraining epoch, no refinement")
    cli = parser.parse_args()
    if len(set(cli.normalizations)) != len(cli.normalizations):
        parser.error("--normalizations contains duplicate methods")
    cli.selection_seeds = [SEED]
    cli.final_runs = 1
    cli.final_seed_start = SEED
    cli.root = cli.root.expanduser().resolve()
    cli.data_dir = (
        (cli.root.parent / "datasets").resolve()
        if cli.data_dir is None else cli.data_dir.expanduser().resolve()
    )
    if cli.output is None:
        folder = (
            "hyperparameter_search_quick_all_methods_seed42"
            if cli.quick else "hyperparameter_search_all_methods_seed42"
        )
        cli.output = root / folder
    cli.output = cli.output.expanduser().resolve()
    if cli.dsc_pretrain_weights:
        cli.dsc_pretrain_weights = cli.dsc_pretrain_weights.expanduser().resolve()
    if cli.log_interval < 1:
        parser.error("--log-interval must be positive")
    if cli.quick:
        cli.skip_refine = True
    return cli

def run_protocol(
    cli: argparse.Namespace,
    base: Any,
    device: Any,
    normalization: str,
) -> List[Dict[str, Any]]:
    """Run search and final reruns for one isolated preprocessing protocol."""
    cli.output.mkdir(parents=True, exist_ok=True)
    results_path = cli.output / "tuning_results.csv"
    final_path = cli.output / "final_runs.csv"
    if cli.restart:
        for path in (results_path, final_path, cli.output / "final_mean_std.csv"):
            if path.exists():
                path.unlink()
    all_rows = read_rows(results_path)
    all_final_rows = read_rows(final_path)
    for row in [*all_rows, *all_final_rows]:
        try:
            old_config = json.loads(row.get("config_json", "{}"))
        except json.JSONDecodeError:
            old_config = {}
        wrong_seed = row.get("seed") != str(SEED)
        wrong_normalize = old_config.get("normalize") != normalization
        wrong_algorithm = row.get("algorithm") not in {"dec", "dsc", "sdcn"}
        wrong_sdcn_protocol = row.get("algorithm") == "sdcn" and (
            old_config.get("sdcn_output") != "q"
            or old_config.get("graph_path") is not None
        )
        wrong_dsc_solver = row.get("algorithm") == "dsc" and (
            old_config.get("dsc_eigen_solver", "lobpcg") != cli.dsc_eigen_solver
        )
        legacy_dsc_post = row.get("algorithm") == "dsc" and (
            "-DSC-P-" in row.get("config_id", "") or any(
                key in old_config for key in ("dsc_alpha", "dsc_dim_subspace", "dsc_ro")
            )
        )
        if (wrong_seed or wrong_normalize or wrong_algorithm or legacy_dsc_post
                or wrong_sdcn_protocol or wrong_dsc_solver):
            raise RuntimeError(
                "Detected records from an incompatible protocol (seed, normalization, algorithm, DSC solver, or post-processing)."
                "Use a new --output directory to avoid mixing protocols or normalization results."
            )
    done = completed_keys(all_rows)
    names = DATASET_FILES if cli.all_datasets else cli.datasets
    data_paths = [resolve_dataset_path(name, cli.data_dir) for name in names]
    cache_dir = cli.output / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    final_summaries = []

    print(f"[Environment] root={cli.root}, device={device}, output={cli.output}")
    print(
        f"[Protocol] seeds={cli.selection_seeds}; normalize={normalization}; "
        f"learning_rate=1e-3/1e-4; final runs={cli.final_runs}; "
        "DEC core=40; DSC core=80; SDCN core=288; DSC uses direct symmetric affinity and one spectral clustering; SDCN output=q"
    )
    for data_path in data_paths:
        # Probe once to obtain dataset metadata and the true sample count.
        probe_namespace = base.build_parser().parse_args([
            "--data", str(data_path), "--root", str(cli.root),
            "--output", str(cache_dir), "--device", cli.device,
            "--normalize", normalization,
        ])
        base.validate_args(probe_namespace)
        probe_data = base.load_dataset(data_path, probe_namespace)
        data_cache: Dict[Tuple[str, str], Any] = {}
        for algorithm in cli.algorithms:
            profile = base.get_dataset_profile(probe_data.name)
            if algorithm == "dsc" and profile and profile.get("image_size") is None:
                print(f"\n[Skip] {probe_data.name} | DSC-Net: plain vector data without image structure")
                continue
            print(f"\n{'=' * 72}\n[Dataset-method] {probe_data.name} | {algorithm}\n{'=' * 72}")
            configs = core_configs(
                base, algorithm, probe_data.name, probe_data.n_samples,
                cli.dsc_pretrain_weights, normalization, cli.dsc_eigen_solver,
            )
            if cli.quick:
                configs = configs[:2]
                for config in configs:
                    config["epochs"] = 2
                    config["pretrain_epochs"] = 1
                    if algorithm == "dsc":
                        config["dsc_pretrain_epochs"] = 1
            for config in configs:
                config["_algorithm"] = algorithm
            evaluate_generic_configs(
                base, cli, configs, data_path, data_cache, device, cache_dir,
                results_path, all_rows, done,
            )
            core_best, _ = select_best(
                all_rows, probe_data.name, algorithm, cli.selection_seeds,
                stage="core", quick=cli.quick,
            )
            print(f"[Best core configuration] {core_best['_config_id']}: {json.dumps(public_config(core_best), ensure_ascii=False, default=str)}")
            best, score = core_best, select_best(
                all_rows, probe_data.name, algorithm, cli.selection_seeds,
                stage="core", quick=cli.quick,
            )[1]
            if not cli.skip_refine:
                refinements = refine_configs(core_best, algorithm)
                for config in refinements:
                    config["_algorithm"] = algorithm
                    if algorithm == "dsc":
                        config["_stage"] = "refine"
                if refinements:
                    evaluate_generic_configs(
                        base, cli, refinements, data_path, data_cache, device, cache_dir,
                        results_path, all_rows, done,
                    )
                    best, score = select_best(
                        all_rows, probe_data.name, algorithm, cli.selection_seeds,
                        quick=cli.quick,
                    )
            print(f"[Selected configuration] {best['_config_id']}, score={score}")
            summary = run_final_repetitions(
                base, cli, best, score, data_path, data_cache, device, cache_dir,
                final_path, all_final_rows,
            )
            final_summaries.append(summary)
            write_final_summary(cli.output / "final_mean_std.csv", final_summaries)
            print(
                f"[Final result] {probe_data.name} | {algorithm} | "
                f"ACC={summary['acc_mean']}±{summary['acc_std']}, runs={summary['runs']}"
            )
    print(f"\nProtocol complete. Search records: {results_path}")
    print(f"Final summary: {cli.output / 'final_mean_std.csv'}")
    return final_summaries


def main() -> int:
    cli = build_cli()
    base = load_entry(cli.root)
    if cli.device == "cuda" and not base.torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access CUDA")
    use_cuda = cli.device == "cuda" or (
        cli.device == "auto" and base.torch.cuda.is_available()
    )
    device = base.torch.device("cuda" if use_cuda else "cpu")
    if cli.deterministic:
        base.torch.backends.cudnn.deterministic = True
        base.torch.backends.cudnn.benchmark = False

    base_output = cli.output
    multiple = len(cli.normalizations) > 1
    all_summaries: List[Dict[str, Any]] = []
    for normalization in cli.normalizations:
        protocol_cli = argparse.Namespace(**vars(cli))
        protocol_cli.output = (
            base_output / normalization if multiple else base_output
        )
        print(
            f"\n[Protocol] normalize={normalization}; "
            f"output={protocol_cli.output}"
        )
        summaries = run_protocol(protocol_cli, base, device, normalization)
        all_summaries.extend(summaries)
        if multiple:
            write_final_summary(
                base_output / "final_mean_std_all_normalizations.csv",
                all_summaries,
            )
    if multiple:
        print(
            "Combined summary: "
            f"{base_output / 'final_mean_std_all_normalizations.csv'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
