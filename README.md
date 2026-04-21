# TrajPred Codebase Guide

This repository is a trajectory prediction training/evaluation library from the
Vehicle Trajectory Prediction Library (TPL). It contains several PyTorch models,
a shared HDF5 dataset loader, YAML-driven experiment configuration, evaluation
KPIs, deployment/export helpers, and visualisation utilities.

The most important practical point for NGSIM is this:

**the training code does not read raw NGSIM `.txt` files directly.** It expects
preprocessed `.h5` files with a specific schema. The repository includes a raw
NGSIM sample under `data/archive/`, but it does not include the preprocessing
pipeline that converts NGSIM text files into the `.h5` format used by
`Dataset.LCDataset`.

## Repository Layout

```text
.
├── train.py                  # Main training entry point
├── evaluate.py               # Loads a saved experiment/model and evaluates it
├── deploy.py                 # Runs a trained model and exports predictions
├── Dataset.py                # Shared PyTorch Dataset for all models
├── params.py                 # Loads YAML config and builds runtime parameters
├── top_functions.py          # Generic training/eval/deploy loops
├── kpis.py                   # Shared KPI implementations
├── export.py                 # CSV/MAT prediction export helpers
├── config/
│   ├── hyperparams.yaml      # Global training/problem/model flags
│   ├── constants.yaml        # Output dirs, feature size, task constants
│   ├── datasets/*.yaml       # Dataset locations and train/val/test splits
│   └── models/*.yaml         # Model class/function/loss config
├── TPMs/                     # Model families and their model/function/KPI code
├── visualiser/               # Plotting helpers, including FNGSIM paths
└── data/
    ├── archive/              # Raw NGSIM sample and data dictionary
    └── highD_sample/         # Raw highD sample CSVs/images
```

## How the Code Runs

The code is configuration-driven:

1. `params.ParametersHandler(...)` loads:
   - one model config from `config/models/`
   - one train dataset config from `config/datasets/`
   - optional separate test/deploy dataset configs
   - shared settings from `config/hyperparams.yaml`
   - constants from `config/constants.yaml`
2. `params.match_parameters()` converts YAML strings into actual Python classes
   and functions such as model class, optimizer, loss, train/eval/deploy hooks.
3. `train.py`, `evaluate.py`, or `deploy.py` instantiates `Dataset.LCDataset`.
4. `Dataset.LCDataset` opens the configured `.h5` files, builds or loads sample
   indexes, normalizes features, and returns PyTorch batches.
5. `top_functions.py` runs the common train/eval/deploy loop and calls the
   selected model-specific functions from `TPMs/<model>/functions.py`.

## Main Files

### `params.py`

`ParametersHandler` is the central config object. Example from the current
`train.py`:

```python
p = params.ParametersHandler(
    'POVL_SM.yaml',
    'm40_train.yaml',
    './config',
    seperate_test_dataset='m40_test.yaml',
    seperate_deploy_dataset='m40_deploy.yaml',
)
p.hyperparams['experiment']['debug_mode'] = True
p.hyperparams['dataset']['balanced'] = False
p.match_parameters()
```

After `match_parameters()`, useful runtime attributes include:

- `p.TR`, `p.TE`, `p.DE`: train/test/deploy dataset settings
- `p.MIN_IN_SEQ_LEN`, `p.MAX_IN_SEQ_LEN`, `p.TGT_SEQ_LEN`, `p.FPS`
- `p.BATCH_SIZE`, `p.LR`, `p.NUM_ITRS`, `p.VAL_FREQ`
- `p.model_dictionary`: resolved model class, optimizer, losses, and hooks
- `p.WEIGHTS_DIR`, `p.RESULTS_DIR`, `p.VIS_DIR`: output locations

### `Dataset.py`

`LCDataset` is the only dataset class used by training, evaluation, and deploy.
It expects each configured data file to be an HDF5 file named like `01.h5`,
`02.h5`, etc.

The model config decides which input feature dataset is loaded:

```python
self.state_data_name = 'state_' + state_type
```

For example:

- `POVL.yaml` and `POVL_SM.yaml` use `state_type: 'povl'`, so the HDF5 key must
  be `state_povl`.
- `MMnTP.yaml`, `SMTP.yaml`, and `DMTP.yaml` use `state_type: 'merging'`, so the
  HDF5 key must be `state_merging`.
- `Constant_Parameter.yaml` uses `state_type: 'constantx_data'`, so the HDF5 key
  must be `state_constantx_data`.

Required HDF5 keys:

```text
state_povl or state_merging or state_constantx_data
output_states_data
labels
frame_data
tv_data
```

What the keys mean:

- `state_*`: normalized later by `LCDataset`; shape is `[N, feature_size]`.
- `output_states_data`: target trajectory increments; shape is usually `[N, 2]`
  for lateral/longitudinal displacement per time step.
- `labels`: manoeuvre labels, using `0` for lane keeping and non-zero values for
  lane-change classes. The loader uses `abs(labels)`.
- `frame_data`: frame IDs used for plotting/export.
- `tv_data`: target vehicle ID per row. Samples must stay within a continuous
  run of the same `tv_data` value.

`LCDataset` creates sample windows using:

- input length: `MIN_IN_SEQ_LEN` to `MAX_IN_SEQ_LEN`
- prediction length: `TGT_SEQ_LEN`
- split ratios from the dataset YAML

It writes generated sample indexes as `.npy` files in the dataset directory.
Those files are named with the split, sequence lengths, balanced/unbalanced
flag, split ratios, and dataset name.

### `train.py`

`train_model_dict(p)`:

- requires CUDA as currently written; if CUDA is unavailable it prints
  `Running on CPU!!!` and exits
- creates train, validation, and test datasets
- trains with the selected model-specific training function
- saves the best model to `weights/<experiment_tag>.pt`
- writes TensorBoard logs under `runs/` or `runs(debugging)/`
- optionally evaluates after training in the `__main__` block

For quick smoke tests, set:

```python
p.hyperparams['experiment']['debug_mode'] = True
```

Debug mode validates immediately and breaks after one validation pass.

### `evaluate.py`

`test_model_dict(p)`:

- instantiates the model
- loads the train dataset only to reuse train-set normalization ranges
- loads the test dataset
- loads weights from `weights/<experiment_tag>.pt`
- computes KPIs
- saves an evaluation pickle under `evaluations/`

`evaluate.py` runs on CPU in its current form because of:

```python
if False:  # torch.cuda.is_available() and p.CUDA:
```

### `deploy.py`

`deploy_model_dict(p, export_file_name)`:

- loads trained weights
- runs model-specific deploy inference
- exports predictions through `export.py`

The export target is currently hard-coded in `export.py`:

```python
PREDICTION_DIR = "../../Dataset/Prediction_exid"
```

Change that path before deploying on another dataset.

### `top_functions.py`

This file contains reusable loops:

- `train_top_func(...)`
- `eval_top_func(...)`
- `deploy_top_func(...)`

It does not know model internals. It calls hooks from the selected YAML model
config, such as `TPMs.POVL_SM.functions.POVL_SM_training`.

### `TPMs/`

Each model family typically has:

```text
TPMs/<model>/
├── model.py       # PyTorch module
├── functions.py   # train/eval/deploy logic for that model
├── kpis.py        # model-specific KPIs, if needed
└── utils.py       # helper functions
```

For running experiments, you usually only need to choose a model YAML. The
lowest-friction starting point is `POVL_SM.yaml` because it is single-modal and
has fewer manoeuvre/mode settings than the multi-modal variants.

## Configuration Files

### `config/hyperparams.yaml`

Key fields:

```yaml
dataset:
  balanced: False
  ablation: False

experiment:
  cuda: True
  debug_mode: False
  multi_modal_eval: False

training:
  batch_size: 2000
  lr: 1e-4
  num_itrs: 4e+4
  val_freq: 1000

problem:
  FPS: 5
  MIN_IN_SEQ_LEN: 15
  MAX_IN_SEQ_LEN: 15
  TGT_SEQ_LEN: 25
```

With `FPS: 5`, the default problem is:

- 15 observed frames = 3 seconds of history
- 25 predicted frames = 5 seconds into the future

### `config/datasets/*.yaml`

Dataset YAMLs tell the loader where `.h5` files live and which file IDs to use.
Example:

```yaml
name: m40_train
train: 0.9
abblation_val: 0.1
val: 0.1
test: 0.0
deploy: 0.0
dataset_dir: '../Datasets/Processed_m40/RenderedDataset/'
dataset_ind: 'list(range(1, 157))'
map_dirs: 'None'
```

`dataset_ind` is evaluated in Python and converted into file names:

```python
[str(i).zfill(2) + '.h5' for i in eval(dataset['dataset_ind'])]
```

So `dataset_ind: '[1, 2]'` means `01.h5` and `02.h5`.

### `config/models/*.yaml`

Model YAMLs connect the framework to model-specific code:

```yaml
name: POVL_SM
ref: TPMs.POVL_SM.model.POVL_SM
optimizer: torch.optim.Adam
model training function: TPMs.POVL_SM.functions.POVL_SM_training
model evaluation function: TPMs.POVL_SM.functions.POVL_SM_evaluation
model deploy function: TPMs.POVL_SM.functions.POVL_SM_deploy
model kpi function: TPMs.POVL_SM.kpis.POVL_SM_kpis
data type: 'state'
state type: 'povl'
```

## Installation

The repository includes `pytorch.yml`, not the `environment.yml` mentioned in
the old README.

```bash
conda env create -f pytorch.yml
conda activate pytorch
```

Useful packages used by the code include:

- PyTorch
- NumPy
- pandas
- h5py
- scikit-learn
- matplotlib
- tensorboard
- scipy
- OpenCV, mainly for visualisation

## Running an Existing Experiment

Before running, make sure the configured dataset directory exists and contains
the required `.h5` files. The current configs point outside this repository, for
example:

```text
../Datasets/Processed_m40/RenderedDataset/
../Datasets/Processed_exid/RenderedDataset/
../../Dataset/Processed_highD/RenderedDataset/
```

Train:

```bash
python train.py
```

Evaluate:

```bash
python evaluate.py
```

Deploy/export predictions:

```bash
python deploy.py
```

The `__main__` blocks currently contain hard-coded experiment choices. In normal
use, edit the `ParametersHandler(...)` call in the script you are running.

## Using NGSIM

### What Is Present

The repo includes a raw NGSIM sample:

```text
data/archive/trajectories-0750am-0805am.txt
data/archive/trajectory-data-dictionary.htm
data/archive/data-analysis-report-0750-0805.pdf
```

The raw text columns are the standard NGSIM trajectory fields:

```text
Vehicle_ID, Frame_ID, Total_Frames, Global_Time,
Local_X, Local_Y, Global_X, Global_Y,
v_Length, v_Width, v_Class, v_Vel, v_Acc,
Lane_ID, Preceding, Following, Space_Headway, Time_Headway
```

There are also FNGSIM paths in `visualiser/param.py`, but those are for
visualisation data locations and are not a training preprocessor.

### What Is Missing

The original repository did not include a raw-NGSIM preprocessing pipeline. This
workspace now includes a baseline converter at:

```text
scripts/convert_ngsim_to_h5.py
```

It creates the schema needed by `MMnTP.yaml`: `state_merging`,
`output_states_data`, `labels`, `frame_data`, and `tv_data`.

To train on NGSIM, you need to create processed files such as:

```text
Datasets/Processed_NGSIM/RenderedDataset/01.h5
Datasets/Processed_NGSIM/RenderedDataset/02.h5
...
```

Each `.h5` file must contain the required keys described in `Dataset.py`.

### Recommended NGSIM Setup

1. Convert NGSIM from 10 Hz to 5 Hz, or change `problem.FPS` and sequence
   lengths consistently.
2. Sort rows by target vehicle and frame.
3. Build continuous per-vehicle sequences.
4. Compute input features for the model you want to run.
5. Compute `output_states_data` as per-frame displacement increments.
6. Compute `labels` from lane changes.
7. Write the HDF5 files.
8. Add a dataset YAML under `config/datasets/`.
9. Point `train.py`, `evaluate.py`, or `deploy.py` at that YAML.

Example NGSIM dataset config:

```yaml
name: ngsim_train
train: 0.7
abblation_val: 0.1
val: 0.1
test: 0.2
deploy: 0.0
image_width: 200
image_height: 80
dataset_dir: '../Datasets/Processed_NGSIM/RenderedDataset/'
dataset_ind: '[1, 2, 3, 4, 5, 6]'
map_dirs: 'None'
map_ind: 'None'
```

For your current MMnTP-only path, use `MMnTP.yaml` and produce `state_merging`
features. The included converter creates a transparent baseline 27-column
`state_merging` vector from ego, lead/follow, adjacent-lane, and lane-change
signals.

### Expected HDF5 Shape Checklist

For each file, check:

```python
import h5py

with h5py.File("01.h5", "r") as f:
    print(list(f.keys()))
    for key in f.keys():
        print(key, f[key].shape, f[key].dtype)
```

Minimum expected result for `MMnTP.yaml`:

```text
state_merging        (N, 27)
output_states_data   (N, 2)
labels               (N,)
frame_data           (N,)
tv_data              (N,)
```

The default `FEATURE_SIZE` is `27` in `config/constants.yaml`. If your
preprocessor creates a different number of input features, update:

```yaml
MODELS:
  FEATURE_SIZE: <your_feature_count>
```

## Practical Notes and Gotchas

- `train.py` now falls back to CPU if CUDA is unavailable. For CPU smoke tests,
  keep `debug_mode=True` and use a small batch size.
- `evaluate.py` is currently forced to CPU.
- `export.py` has a hard-coded export directory for eXiD. Change
  `PREDICTION_DIR` for NGSIM.
- Generated `.npy` index files are written into the dataset directory. Delete
  them or set `force_recalc_start_indexes=True` if you change sequence lengths,
  split ratios, labels, or data order.
- Balanced datasets rely on `labels`; if labels are wrong, balanced training
  will also be wrong.
- `drop_last=True` is used in train/eval loaders. If your dataset is smaller
  than the batch size, lower `training.batch_size`.
- The code uses `eval(...)` on YAML fields such as `dataset_ind`, model refs, and
  learning rate. Treat config files as code.

## Suggested First Path for NGSIM

1. Create one small `01.h5` from `data/archive/trajectories-0750am-0805am.txt`.
2. Use `MMnTP.yaml` if you want manoeuvre-aware multimodal training, or
   `POVL_SM.yaml` for a simpler single-modal baseline.
3. Set `debug_mode: True`.
4. Set `batch_size` low enough for the sample, for example `16` or `32`.
5. Create `config/datasets/ngsim_debug.yaml` with `dataset_ind: '[1]'`.
6. Run one debug training pass.
7. Only then scale to all NGSIM files and full training.

This repo now includes a baseline NGSIM converter for MMnTP:

```bash
python3 scripts/convert_ngsim_to_h5.py \
  --input data/archive/trajectories-0750am-0805am.txt \
  --output-dir data/processed_ngsim/RenderedDataset \
  --file-id 1
```

Then run the MMnTP debug pipeline:

```bash
python3 run_mmntp_ngsim.py
```

## Citation

If you use this code, cite the original authors:

```bibtex
@article{mozaffari2023multimodal,
  title={Multimodal manoeuvre and trajectory prediction for automated driving on highways using transformer networks},
  author={Mozaffari, Sajjad and Sormoli, Mreza Alipour and Koufos, Konstantinos and Dianati, Mehrdad},
  journal={IEEE Robotics and Automation Letters},
  year={2023},
  publisher={IEEE}
}

@article{mozaffari2022early,
  title={Early lane change prediction for automated driving systems using multi-task attention-based convolutional neural networks},
  author={Mozaffari, Sajjad and Arnold, Eduardo and Dianati, Mehrdad and Fallah, Saber},
  journal={IEEE Transactions on Intelligent Vehicles},
  volume={7},
  number={3},
  pages={758--770},
  year={2022},
  publisher={IEEE}
}
```
