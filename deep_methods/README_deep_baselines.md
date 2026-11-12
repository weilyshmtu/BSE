# Unified Experiments and Parameter-Influence Search for DEC, DSC-Net, and SDCN

This directory contains two Python files that can be placed directly in the original experimental project:

1. `deep_baseline_experiments.py`: runs DEC, DSC-Net, and SDCN through a unified interface and supports three feature preprocessing methods: `none`, `l2`, and `global_max`.
2. `deep_baseline_parameter_search.py`: reuses the algorithm runners in the first file and independently performs core search, refinement search, best-configuration selection, and final reruns for each dataset-algorithm-preprocessing combination.

All comments, command-line help, exception messages, and runtime logs in the two Python files are in English. This document explains directory preparation, execution, search protocols, output files, and checkpoint-based resumption.

## 1. Integrated Experimental Protocol

### 1.1 Three Preprocessing Methods

Preprocessing is completed before model training and graph construction, and is recorded in configurations, logs, and result files:

| Name | Processing |
| --- | --- |
| `none` | Preserves the original features without scaling. |
| `l2` | Divides each sample by its L2 norm, using `1e-12` as the lower bound for zero norms, so that samples have approximately unit norm. |
| `global_max` | Divides the entire feature matrix by its largest absolute value, using `1e-12` as the lower bound for a zero or extremely small maximum. |

The three preprocessing methods are run separately. Results obtained with different preprocessing methods on the same dataset are never mixed during parameter selection or aggregation.

### 1.2 Key Settings Preserved for the Three Algorithms

- **DEC**: Uses the script's built-in layer-wise denoising autoencoder pretraining by default, with `dec_pretrain_mode="sdae"`. Clustering and pretraining learning rates are configured separately. The `joint` mode is retained for joint-reconstruction diagnostics.
- **DSC-Net**: Fixes the final affinity matrix to \(A=(|C|+|C|^T)/2\), followed directly by one spectral clustering run. It does not use thrC, SVD embedding, nearest-neighbor graph reconstruction, or affinity-matrix powers.
- **SDCN**: Constructs a symmetric KNN graph, adds self-loops, and applies row normalization before training. This graph is an input to the GCN, not post-training graph processing. Formal results always use `argmax` labels from the `q` distribution.

ACC is computed with Hungarian matching, while NMI and ARI are computed with scikit-learn. All metrics are calculated from **final-epoch predictions**; the best metric observed during training is never selected.

### 1.3 Unified Handling of the DSC Eigensolver

The original L2 script used ARPACK, whereas the other two preprocessing versions used LOBPCG. In the integrated scripts:

- `lobpcg` is the unified default;
- add `--dsc-eigen-solver arpack` explicitly to reproduce the original L2 script;
- the solver is recorded in search configurations and run details, preventing old search records from being reused after only the solver has changed.

The preprocessing method no longer implicitly determines the eigensolver.

## 2. Project Directory Preparation

Use the following structure and place both Python files in `deep_methods`:

```text
project_root/
├── deep_methods/
│   ├── deep_baseline_experiments.py
│   ├── deep_baseline_parameter_search.py
│   ├── DEC/
│   │   ├── dec.py
│   │   └── ptdec/                 # Other files required by the original project
│   ├── DSC/
│   │   ├── main.py
│   │   ├── yaleb.py               # Required only for the original YaleBCrop025 data
│   │   └── ...                    # Other helper files from the original project
│   ├── SDCN/
│   │   ├── sdcn.py
│   │   ├── GNN.py
│   │   ├── utils.py
│   │   ├── evaluation.py
│   │   └── ...
│   └── pretrained_weights_original/
│       └── yaleb.pkl              # Optional: original YaleB convolutional AE weights
└── datasets/
    ├── zoo.mat
    ├── Yale_32x32.mat
    ├── JAFFE_64x64.mat
    ├── ORL_32x32.mat
    ├── Umist_575_32x32.mat
    ├── FERET_32x32.mat
    ├── COIL20_32x32.mat
    ├── AR_32x32.mat
    ├── YALEB_32x32.mat
    ├── pendigits.mat
    ├── COIL100_32x32.mat
    └── MNIST_10k_Train_28x28.mat
```

Algorithm folder names are resolved case-insensitively. Keep all helper files from the original projects. The scripts isolate same-named local modules from different algorithm directories, reducing the risk of conflicts between files such as `utils.py`.

The commands below assume that the current working directory is `deep_methods/`. The unified experiment script reads the sibling `datasets/` directory through `--data ../datasets/...`. The search script automatically uses the `datasets/` directory sibling to `--root`, so `--data-dir` is not required. If commands are run from `project_root/`, prefix both script filenames with `deep_methods/` and use `datasets/...` as the unified experiment's data path.

By default, the search script loads `deep_baseline_experiments.py` from the directory specified by `--root`. The default `--root` is the script's own `deep_methods/` directory, so the DEC, DSC, and SDCN folders do not need to be moved. If the first file is moved or renamed, update the corresponding loading location in the search script. The simplest approach is to keep the two filenames and their same-directory relationship unchanged.

## 3. Python Environment and Data Format

### 3.1 Dependencies

Python 3.9 or later is recommended, together with:

- PyTorch;
- NumPy;
- SciPy;
- scikit-learn;
- h5py, which is required for MATLAB v7.3 HDF5 files.

PyTorch should be installed according to the local CUDA environment. The remaining dependencies can be installed with:

```bash
python -m pip install numpy scipy scikit-learn h5py
```

The original DEC, DSC-Net, and SDCN implementations may have additional dependencies or version requirements; follow their respective original projects. An initial environment check is:

```bash
python -c "import torch, numpy, scipy, sklearn; print(torch.__version__); print(torch.cuda.is_available())"
```

### 3.2 Standard MAT Data

The default variables are:

- `fea`: a two-dimensional feature matrix in which each row is a sample, with shape `[N, D]`;
- `gnd`: labels that can be squeezed into a one-dimensional vector.

If the feature matrix has shape `[D, N]`, use `--sample-axis columns`. To use different MAT variable names, specify them explicitly:

```bash
--x-key fea --y-key gnd
```

The script prints the actual sample count, feature count, class count, sample axis, and MAT variable names. Verify this information before running formal experiments.

### 3.3 Original YaleBCrop025

`--dsc-variant yaleb` is intended only for the original three-dimensional `Y` data used by `yaleb.py`, where every sample must contain 2016 pixels. It is not an alternative reader for the 1024-dimensional `YALEB_32x32.mat` dataset in the built-in table.

## 4. File 1: Unified Baseline Experiments

### 4.1 View All Arguments

```bash
python deep_baseline_experiments.py --help
```

First check the environment with one algorithm, one dataset, and very short training schedules. For example:

```bash
python deep_baseline_experiments.py \
  --data ../datasets/ORL_32x32.mat \
  --algorithms dec \
  --normalize none \
  --epochs 2 \
  --pretrain-epochs 1 \
  --runs 1 \
  --device cpu \
  --output smoke_dec
```

This command is only a workflow check and must not be used for paper results.

### 4.2 One Algorithm and One Preprocessing Method

```bash
python deep_baseline_experiments.py \
  --data ../datasets/ORL_32x32.mat \
  --algorithms dec \
  --normalize l2 \
  --runs 1 \
  --seed 42 \
  --device cuda \
  --output results_orl_dec_l2
```

### 4.3 Three Algorithms and One Preprocessing Method

```bash
python deep_baseline_experiments.py \
  --data ../datasets/ORL_32x32.mat \
  --algorithms dec dsc sdcn \
  --normalize global_max \
  --runs 1 \
  --seed 42 \
  --device cuda \
  --output results_orl_global_max
```

### 4.4 Three Algorithms and All Three Preprocessing Methods in One Run

```bash
python deep_baseline_experiments.py \
  --data ../datasets/ORL_32x32.mat \
  --algorithms dec dsc sdcn \
  --normalizations none l2 global_max \
  --runs 1 \
  --seed 42 \
  --device cuda \
  --output results_orl_all
```

`--normalizations` takes precedence over `--normalize`. The example writes results to:

```text
results_orl_all/none/
results_orl_all/l2/
results_orl_all/global_max/
```

When only one preprocessing method is specified, results are written directly to `--output` without an additional preprocessing subdirectory.

### 4.5 Multiple Datasets and Independent Runs

```bash
python deep_baseline_experiments.py \
  --data ../datasets/Yale_32x32.mat ../datasets/ORL_32x32.mat ../datasets/COIL20_32x32.mat \
  --algorithms dec sdcn \
  --normalize none \
  --runs 3 \
  --seed 42 \
  --device cuda \
  --output results_multi
```

The three runs use seeds 42, 43, and 44, respectively. To enable deterministic CUDNN operations where possible, add:

```bash
--deterministic
```

This option cannot guarantee bitwise-identical results across different hardware, drivers, or dependency versions. Record the execution environment in all cases.

### 4.6 DSC Image Arguments

Built-in dataset profiles provide image dimensions for common tabular datasets. Some profiles also provide DSC-specific network or training defaults. These values can also be specified explicitly:

```bash
python deep_baseline_experiments.py \
  --data ../datasets/ORL_32x32.mat \
  --algorithms dsc \
  --normalize global_max \
  --dsc-image-size 32 32 \
  --dsc-channels 1,3,3,5 \
  --dsc-kernels 3,3,3 \
  --device cuda \
  --output results_orl_dsc
```

In the profile table, `zoo` and `pendigits` are plain vector datasets. Generic DSC is skipped for them unless an image size is explicitly supplied. For large datasets, the DSC self-expression matrix contains \(N^2\) elements, so GPU and host memory requirements should be checked in advance.

Example for the original YaleB data:

```bash
python deep_baseline_experiments.py \
  --data ../datasets/YaleBCrop025.mat \
  --algorithms dsc \
  --dsc-variant yaleb \
  --dsc-pretrain-weights pretrained_weights_original/yaleb.pkl \
  --device cuda \
  --output results_yaleb_original
```

To reproduce the DSC eigensolver used by the original L2 script:

```bash
python deep_baseline_experiments.py \
  --data ../datasets/ORL_32x32.mat \
  --algorithms dsc \
  --normalize l2 \
  --dsc-eigen-solver arpack \
  --device cuda \
  --output results_orl_dsc_l2_arpack
```

### 4.7 Pretraining Cache and Common Training Arguments

- `--force-pretrain`: ignores existing weights and performs pretraining again.
- `--pretrain-epochs`: sets the general pretraining epoch count.
- `--learning-rate`: sets the clustering learning rate; if omitted, the algorithm default is used.
- `--dec-pretrain-lr`, `--sdcn-pretrain-lr`, and `--dsc-pretrain-lr`: set algorithm-specific pretraining learning rates.
- `--latent-dim`: sets the latent dimension for DEC or SDCN.
- `--knn-k`: sets the number of KNN neighbors for SDCN when no external graph is supplied.
- `--fail-fast`: stops immediately after an experiment fails, which is useful for debugging. By default, the failure is recorded and the remaining experiments continue.

Cached weights and their configuration metadata are stored under `pretrained/` in the output directory. Weight filenames include the dataset, algorithm, dimensions, preprocessing method, pretraining protocol, random seed, and configuration digest, preventing incompatible experiments from reusing old weights.

### 4.8 Unified Experiment Outputs

With one preprocessing method, outputs are written directly to `--output`. With multiple methods, they are written to the corresponding preprocessing subdirectory:

```text
OUTPUT/
├── run_config.json
├── summary.csv
├── summary_mean_std.csv
├── predictions/
│   └── DATASET_ALGORITHM_seedSEED_predictions.csv
└── pretrained/
    ├── *.pt
    └── *.json
```

- `run_config.json`: the complete argument set for the current command.
- `summary.csv`: ACC, NMI, ARI, runtime, status, and error information for each run; `details` stores algorithm-specific run information.
- `summary_mean_std.csv`: aggregates successful runs by preprocessing method, dataset, and algorithm. It uses sample standard deviation and reports zero for a single run.
- `predictions/`: each sample's original label, encoded label, and predicted cluster.
- `pretrained/`: reusable pretrained weights and configuration metadata.

Each invocation of the unified experiment script recreates `summary.csv` in the current output directory. It does not automatically append per-run results from the previous invocation. Use a different output directory or create a backup when historical results must be preserved.

## 5. File 2: Parameter-Influence Search

### 5.1 Run a Quick Workflow Check First

```bash
python deep_baseline_parameter_search.py \
  --quick \
  --datasets ORL_32x32 \
  --algorithms dec dsc sdcn \
  --normalizations none \
  --device cpu \
  --output quick_search_orl
```

For each dataset-algorithm-preprocessing combination, quick mode performs only:

- the first 2 core configurations;
- 2 clustering epochs;
- 1 pretraining epoch;
- no refinement search;
- 1 final rerun of the selected configuration.

Thus, one dataset, three algorithms, and one preprocessing method require 9 model runs; all three preprocessing methods require 27 runs. Quick mode validates the workflow only and does not provide formal parameter-selection conclusions.

### 5.2 Formal Search on One Dataset with One Preprocessing Method

```bash
python deep_baseline_parameter_search.py \
  --datasets ORL_32x32 \
  --algorithms dec dsc sdcn \
  --normalizations none \
  --device cuda \
  --output search_orl_none
```

### 5.3 Formal Search for One Algorithm with All Three Preprocessing Methods

```bash
python deep_baseline_parameter_search.py \
  --datasets ORL_32x32 \
  --algorithms sdcn \
  --normalizations none l2 global_max \
  --device cuda \
  --output search_orl_sdcn_all
```

### 5.4 Default Full Search

```bash
python deep_baseline_parameter_search.py --device cuda
```

The default scope is:

- the 12 built-in datasets;
- DEC, DSC-Net, and SDCN;
- `none`, `l2`, and `global_max`;
- DSC is skipped on Zoo and Pendigits, leaving the other 10 image datasets for DSC.

The default output directory is `hyperparameter_search_all_methods_seed42` under `deep_methods/`. Quick mode defaults to `hyperparameter_search_quick_all_methods_seed42`. An explicitly supplied `--output` takes precedence.

**Computational-cost warning:** Without external DSC pretrained weights, and assuming that all runs succeed, the default full search is expected to execute 14,592 model runs. Caching can reduce part of the pretraining cost, but it does not reduce the number of configuration evaluations. Split the workload by dataset, algorithm, and preprocessing method first, and verify the protocol and available resources before starting the default full search.

### 5.5 Core Parameter Grids

All counts below refer to **one algorithm on one dataset under one preprocessing method**. The full procedure is repeated separately for the three preprocessing methods; the 40/80/288 grids are not expanded into one mixed grid.

#### DEC: 40 Core Configurations

| Parameter | Search values |
| --- | --- |
| `latent_dim` | 10, 20, 30, 40, 50 |
| `dec_sdae_noise` | 0.1, 0.2 |
| `learning_rate` | 0.001, 0.0001 |
| `batch_size` | 256, current dataset sample count N |

Total: \(5\times2\times2\times2=40\).

Fixed settings include `dec_pretrain_mode=sdae`, pretraining learning rate `1e-4`, 50 pretraining epochs, 0 end-to-end reconstruction fine-tuning epochs, 200 clustering epochs, and an early-stopping label-change threshold of 0.001.

The following refinements are tried independently around the best core configuration:

1. `pretrain_epochs=100`;
2. `dec_sdae_finetune_epochs=50`;
3. `dec_sdae_noise=0.05`.

These refinements do not form a Cartesian product, and duplicate configurations are removed automatically.

#### DSC-Net: 80 Core Configurations

| Parameter | Search values |
| --- | --- |
| `learning_rate` | 0.001, 0.0001 |
| Multiplier for `weight_coef` | 0.1, 0.5, 1, 2, 5, 10, 20, 50 |
| Multiplier for `weight_self_exp` | 0.1, 1, 5, 10, 50 |

Total: \(2\times8\times5=80\). Each actual loss weight is the current dataset's default weight multiplied by the corresponding factor. Clustering epochs and some default weights come from the dataset profiles in the unified script. The pretraining learning rate is `1e-4`, and the initial pretraining epoch count is 50.

When `--dsc-pretrain-weights` is not supplied, the refinement stage examines 50 and 100 pretraining epochs. The 50-epoch configuration duplicates the core configuration, so only the 100-epoch configuration is added. When external DSC weights are supplied, this pretraining-epoch refinement is not performed. External weights should normally be used only for single-dataset searches.

#### SDCN: 288 Core Configurations

| Parameter | Search values |
| --- | --- |
| `learning_rate` | 0.001, 0.0001 |
| `sdcn_kl_weight` | 0.001, 0.01, 0.1, 1, 5, 10 |
| `sdcn_ce_weight` | 0.0001, 0.001, 0.01, 0.1, 1, 10 |
| `knn_k` | 3, 5, 10, 20 |

Total: \(2\times6\times6\times4=288\).

Fixed settings include the original model's reconstruction-loss weight, pretraining learning rate `1e-4`, 50 pretraining epochs, 200 clustering epochs, batch size 256, initial `latent_dim=10`, `sdcn_output=q`, and no external graph file.

The refinement stage independently tries `latent_dim=20, 30, 40, 50`; the core-stage value of 10 is not repeated.

### 5.6 Run Counts per Dataset and Preprocessing Method

| Algorithm | Core search | Refinement search | Final reruns | Total |
| --- | ---: | ---: | ---: | ---: |
| DEC | 40 | 3 | 1 | 44 |
| DSC-Net without external weights | 80 | 1 | 1 | 82 |
| DSC-Net with external weights | 80 | 0 | 1 | 81 |
| SDCN | 288 | 4 | 1 | 293 |

### 5.7 Parameter-Selection and Final-Rerun Rules

1. The parameter-selection seed is fixed at 42.
2. A configuration is eligible for best-configuration comparison only after it has completed every selection seed.
3. Mean ACC is compared first, mean NMI breaks ties, and the configuration ID provides a deterministic final tie-breaker.
4. The best core configuration is selected first and its refinements are then evaluated. The best configuration is subsequently reselected from both core and refinement candidates.
5. The selected configuration is rerun once, again with seed 42.
6. Final means and standard deviations are computed only from final-rerun records and are not mixed with search-stage records.
7. Failed records do not participate in best-configuration selection. If no configuration completes all required selection seeds, the script raises an error instead of selecting from incomplete results.

### 5.8 Search Outputs and Directory Isolation

With one preprocessing method, results are written directly to `--output`. With multiple methods, the layout is:

```text
OUTPUT/
├── none/
│   ├── tuning_results.csv
│   ├── final_runs.csv
│   ├── final_mean_std.csv
│   ├── cache/
│   │   └── pretrained/
│   └── DATASET/
│       └── ALGORITHM/
│           ├── best_config.json
│           └── predictions/
├── l2/
├── global_max/
└── final_mean_std_all_normalizations.csv
```

- `tuning_results.csv`: per-configuration results for core and refinement searches, including the complete `config_json`.
- `final_runs.csv`: final-rerun records for selected configurations.
- `final_mean_std.csv`: final summary for processed datasets and algorithms under the current preprocessing method.
- `best_config.json`: selected configuration, selection metrics, final metrics, and run counts.
- `predictions/`: predicted labels from final reruns.
- `cache/pretrained/`: the pretraining cache used exclusively for the current preprocessing method.
- `final_mean_std_all_normalizations.csv`: combined multi-directory summary for multi-preprocessing searches, including the `normalize` field. It is not generated for a single-preprocessing search.

The same configuration ID may appear under different preprocessing methods, but their output directories and `config_json` values differ, so their groups are never mixed.

### 5.9 Resumption, Restarting, and Failure Handling

Repeat the same command to resume from existing records. The reuse key for a successful record contains:

```text
dataset + algorithm + config_id + seed + config_json
```

Only successful records with exactly matching configuration content are reused. Changing the grid, training epochs, preprocessing method, DSC solver, or any other configuration prevents incompatible old records from being reused. Failed records are not considered complete and will be retried on the next invocation.

Common options:

- `--restart`: clears `tuning_results.csv`, `final_runs.csv`, and `final_mean_std.csv` in the current preprocessing directory while retaining reusable pretrained weights.
- `--force-pretrain`: performs pretraining again even when matching weights exist.
- `--skip-refine`: performs only core search and final reruns.
- `--fail-fast`: stops immediately after any configuration fails, making environment or data problems easier to locate.
- `--normalizations l2`: searches only L2 preprocessing.
- `--algorithms dec`: searches only DEC.
- `--datasets ORL_32x32`: searches only the specified dataset; a full MAT-file path can also be supplied.
- `--data-dir`: explicitly overrides the default dataset directory, which is the `datasets/` directory sibling to `--root`.

`--restart` does not actively delete prediction files or `best_config.json` files in the output directories. For a completely clean experiment, the safest approach is to use a new `--output` directory. A new directory is also recommended whenever the experimental protocol changes. The script stops when it detects incompatible old-protocol records, preventing mixed results.

## 6. Recommended Execution Order

1. Check MAT variable names, sample axis, sample count, feature count, and class count.
2. Use the unified experiment script for a short-epoch environment check with one algorithm.
3. Use `--quick` in the search script to verify that search, selection, final reruns, CSV files, and prediction files are all generated correctly.
4. Run a formal search for one dataset, one algorithm, and one preprocessing method.
5. Inspect `best_config.json` and `final_mean_std.csv` to confirm that the selected configuration matches the paper's experimental protocol.
6. Split large workloads by preprocessing method or algorithm. After an interruption, repeat the original command to resume.
7. To reproduce old L2 DSC results, use `--dsc-eigen-solver arpack` with a separate new output directory; do not mix them with default LOBPCG results.

## 7. Workflow Validation Already Performed

The integrated scripts have passed the following checks without requiring full long-running training:

- Python syntax checks and static checks for undefined names;
- `--help` checks for both scripts;
- numerical checks for all three preprocessing methods;
- core-configuration count checks: 40 for DEC, 80 for DSC, and 288 for SDCN;
- refinement-configuration count checks for DEC, DSC, and SDCN;
- simulated unified experiments with three algorithms and three preprocessing methods, confirming isolation of predictions, cache directories, and summaries;
- simulated quick searches with three algorithms and three preprocessing methods, confirming core search, final reruns of selected configurations, and the cross-preprocessing summary;
- checkpoint-resumption checks confirming that repeating the same command does not invoke model runs that have already succeeded;
- checks with minimal simulated algorithm modules for original-style module loading, same-named local-module isolation, and optional DSC import handling;
- default-path checks confirming that the search script resolves the sibling `datasets/` directory for both the default and a custom `--root`.

These checks validate the script structure and scheduling logic. Formal numerical results still depend on the original algorithm files, datasets, dependency versions, pretrained weights, and actual training environment.
