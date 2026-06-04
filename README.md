# Lagrangian Attention Tensor Networks for Velocity Gradient Statistical Modeling
  Direct numerical simulation of turbulence at realistic Reynolds numbers is still beyond current computational capability, necessitating models that reduce the number of resolved spatial scales. Motivated by phenomenology and recent data-driven works based on universality of the smallest scales in fully developed turbulence, the statistical dynamics of the velocity gradient tensor (VGT) at the Kolmogorov scale become of critical importance in advancing turbulence models. Physics-informed machine learning (PIML) has found considerable success in exploiting large datasets taken from direct numerical simulation (DNS) of Navier-Stokes to improve models for the evolution of the VGT. In this work, we follow the long line of blending physical insight with data analysis to simultaneously advance both the modeling and understanding of the phenomenology of the VGT. Using the intimate connection between VGT evolution and fluid deformation, we develop the Lagrangian attention tensor network (LATN) approach that significantly improves over current physics-informed machine learning methods. We demonstrate state-of-the-art performance in both a-priori and a-posteriori metrics, before interpreting the trained attention mechanisms to discover a surprising connection between the history of the strain-rate-tensor and the pressure Hessian.
  
# Using the Code

This section is written as a gentle introduction. If a step fails, the most common cause is the Python environment — see [Installation](#installation).

## What the code actually does

The model is trained in **three stages**, all driven by the single script
`src/runner.py`:

1. **Pressure-Hessian model (`ph`)** — a LATN network learns the deviatoric
   pressure Hessian as a function of the velocity-gradient tensor (VGT) and its
   recent Lagrangian history.
2. **Viscous-Laplacian model (`vis`)** — a second LATN network learns the
   viscous term the same way.
3. **Neural-ODE polish (`node`)** — the two trained networks are assembled into
   a closed evolution equation for the VGT, `dA/dt = (restricted Euler) +
   (pressure Hessian) + (viscous Laplacian)`, and fine-tuned so that *integrated
   trajectories* of the VGT match the DNS. A stochastic forcing term is then
   calibrated to the residual.

Running `runner.py` once executes all three stages in order and writes every
checkpoint, loss curve, and evaluation tensor to disk for you. You do **not**
call the three stages separately.

## Installation

You need Python 3.10+ and (for training) an NVIDIA GPU. The model is small; the
GPU requirement comes from the distributed-training backend (NCCL), which only
runs on CUDA GPUs. CPU-only machines can read data and run the test suite but
**cannot train**.

```bash
# 1. Get the code
git clone https://github.com/cmhyett/LDM.git
cd LDM

# 2. Make an isolated Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# 3a. On a GPU machine: install the CUDA build of PyTorch first, then the rest
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r python_config/requirements-aws.txt

# 3b. On a laptop (no GPU, for inspecting data / running tests only)
pip install -r python_config/requirements.txt
```

## The data format (read this before bringing your own data)

All data are stored as **raw binary files of 64-bit floats** (`.bin`), with one
file per physical field. A file is just a flat stream of numbers that is
reshaped into a 4-dimensional array:

```
shape = (num_samples, num_timesteps, 3, 3)
         └────┬─────┘ └─────┬──────┘ └─┬─┘
         independent    snapshots     the 3×3 tensor
         Lagrangian     along each    A_ij at one point
         particles      trajectory    in space & time
```

So a single number in the file is "component `(i, j)` of the tensor, for
particle `n`, at time-snapshot `t`." This `[samples, time, 3, 3]` layout is
exactly the stacking the model expects — **you do not flatten or reorder the
tensor yourself**; the dataset loader does all reshaping internally.

The training directory (`--datapath`) must contain three files, matched by prefix:

| File (glob)  | Field                          | Physical meaning |
|--------------|--------------------------------|------------------|
| `aij*.bin`   | velocity-gradient tensor `A_ij = ∂u_i/∂x_j` | the *input* history and the quantity being evolved |
| `pij*.bin`   | pressure Hessian `∂²p/∂x_i∂x_j` | training target for stage 1 (`ph`) |
| `vis*.bin`   | viscous Laplacian `ν ∇²A_ij`    | training target for stage 2 (`vis`) |

All three files must share the **same** `(num_samples, num_timesteps, 3, 3)`
shape. Consecutive time snapshots are separated by a fixed DNS timestep `dt`
(see [temporal spacing](#using-data-with-a-different-temporal-spacing)).

To sanity-check a file in Python:

```python
import numpy as np
A = np.fromfile("aij.bin", dtype=np.float64).reshape(N, T, 3, 3)
print(A.shape, A[0, 0])   # first particle, first snapshot: a 3×3 matrix
```

A tiny example dataset lives in `test/test_data/` (see its `README.txt`).

## Quickstart

### 1. Smoke-test your install on the bundled mini-dataset

```bash
PYTHONPATH=src python -m pytest test/test_latn.py test/test_lagr_dataset.py
```

This trains nothing of scientific value — it just confirms the data loader and
network wire together correctly on the small sample in `test/test_data/`.

### 2. Train on your data

```bash
PYTHONPATH=src python src/runner.py \
    --datapath  /path/to/dir/with/aij_pij_vis_bins \
    --savepath  /path/to/output/dir \
    --history_length   50 \
    --history_timestep 1  \
    --percent_test     0.2 \
    --num_filters      16 \
    --max_epochs       200
```

`PYTHONPATH=src` tells Python where the modules live; always include it.

**Common flags** (run `python src/runner.py -h` for the full list):

| Flag | Meaning |
|------|---------|
| `-dp/--datapath` | directory holding `aij*.bin`, `pij*.bin`, `vis*.bin` |
| `-sp/--savepath` | where checkpoints/results are written |
| `-hl/--history_length` | how far back in Lagrangian time the model looks, **in multiples of `dt`** |
| `-ht/--history_timestep` | subsampling stride inside that window, in multiples of `dt` |
| `-nf/--num_filters` | number of temporal-attention filters ("how the history is summarized") |
| `-pt/--percent_test` | fraction of each trajectory's time axis reserved for testing |
| `-nl`, `-nu` | hidden layers / units in the feed-forward part |
| `-lr`, `-dr` | learning rate / dropout rate |
| `-me`, `-se` | max epochs / epochs between checkpoints |
| `-ns/--num_samples` | cap on number of trajectories used (default: all) — lower it to fit small machines |
| `--resume` | continue from the last checkpoint (safe to always pass) |
| `-rn/--run_name` | label for this run's output folder (default: auto-derived from the flags) |

### 3. Where the results go

Each run is namespaced by its configuration so a parameter sweep never
overwrites itself:

```
<savepath>/<run_name>/
├── config.json                 # exact flags used (provenance)
├── ph/                         # stage-1 (pressure-Hessian) model
│   ├── train_loss.csv, test_loss.csv
│   ├── checkpoint_*.pt, checkpoint_best_model.pt
│   ├── apriori_model_state_dict.pt
│   └── pij_apriori_eval.pt     # predictions vs. ground truth on the test set
├── vis/                        # stage-2 (viscous) model, same layout
└── node/                       # stage-3 neural-ODE model + posteriori trajectories
```

Loss curves are plain CSVs (one value per epoch) and are also logged to
TensorBoard (`tensorboard --logdir <savepath>/<run_name>/ph`).

## Deployment

The *same* `runner.py` runs everywhere; only how you launch it differs.

### Locally (single workstation with a GPU)

Follow [Installation](#installation), then run the Quickstart command directly.
The code auto-detects the number of GPUs (`torch.cuda.device_count()`) and
spreads each training stage across all of them. With one GPU it simply runs as a
single process. Reduce `--num_samples` if you run out of host RAM (the full
`(131072, 1000, 3, 3)` dataset needs ~20 GB of RAM during loading; 16k samples
needs ~3 GB).

### On an HPC cluster (Slurm)

Use the templates in `src/job_scheduling/`:

- `submit_job.slurm` — the batch script. Edit the `cd` path, the `--partition`
  / `--time` / `--gres=gpu:N` directives, and any `module load` lines to match
  your cluster.
- `sample_arg_file.txt` — a plain-text file holding the `runner.py` flags (one
  per line). The job script reads it with `xargs` and forwards everything to
  `runner.py`. Point `--datapath`/`--savepath` at your cluster filesystem.

Submit with:

```bash
cd src/job_scheduling
sbatch submit_job.slurm
```

For a **hyperparameter sweep**, `gen_arg_file.py` writes one arg-file per
configuration in a nested loop over ranges you specify at the bottom of that
file; submit one Slurm job per arg-file (e.g. with a job array or a shell loop).

### On AWS (spot GPU instances)

A complete, memory-aware cloud workflow is documented separately:

- **`deploy/README.md`** — instance sizing, the measured memory footprint,
  output layout, and spot-interruption/checkpointing strategy.
- **`deploy/aws-configuration.org`** — a from-scratch CLI walkthrough (IAM role,
  key pair, AMI, launch, teardown) for users new to AWS.
- **`deploy/bootstrap.sh`** — a one-command launcher that builds the environment,
  pulls the dataset from S3, trains, mirrors checkpoints back to S3 continuously,
  and shuts the instance down when finished. On a fresh instance:

  ```bash
  git clone https://github.com/cmhyett/LDM.git && cd LDM
  S3_OUT=s3://<your-bucket>/ldm-outputs ./deploy/bootstrap.sh -me 200 --resume
  ```

  It is safe to re-run verbatim after a spot interruption — training resumes
  from the last checkpoint synced to S3.

## Using data with a different temporal spacing

Three quantities describe time in this code, all measured in the **same unit:
one DNS snapshot interval `dt`**.

- **`dt`** — the physical time between two consecutive snapshots stored in your
  `.bin` files. In the shipped configuration `dt = 3e-4` (set inside
  `runner.py`'s `DataDesc`). If your DNS was saved with a different interval,
  change this value so the finite-difference time derivatives and the neural-ODE
  integrator use the correct step.
- **`history_length` (`-hl`)** — the length of the Lagrangian window the model
  looks back over, *counted in `dt` units*. `-hl 50` with `dt = 3e-4` means the
  model sees 50·3e-4 = 0.015 time units of history.
- **`history_timestep` (`-ht`)** — how finely that window is sampled, also in
  `dt` units. `-ht 1` uses every snapshot; `-ht 5` uses every fifth. The
  temporal-attention kernel ends up with `history_length // history_timestep + 1`
  taps, so these two flags together set how many past snapshots feed the model.

**If your data has coarser or finer snapshots than the original**, you only need
to (a) set `dt` to your true snapshot interval and (b) choose `-hl`/`-ht` so that
`history_length * dt` covers the physical memory you care about (roughly a
Kolmogorov time). `-ht` must evenly divide `-hl`. Setting `-ht` equal to `-hl`
collapses the history to a single instant, recovering a memoryless (TBNN-style)
model — useful as a baseline.

The VGT is automatically normalized by the empirical Kolmogorov timescale
computed from the strain-rate magnitude of your own data
(`calc_characteristic_timescale` in `src/utils.py`), so you do **not** need to
non-dimensionalize the inputs by hand.

### Pointing the code at a dataset of a different size

`runner.py` hard-codes the array shape for the paper dataset:

```python
(131072, 1000, 3, 3)   # (num_samples, num_timesteps, 3, 3)
```

Edit the three `DataDesc(...)` blocks in `runner.py` (one each for `pij`, `vis`,
`dA`) to match your `(N, T, 3, 3)`, and set `dt` there too. Everything
downstream — train/test splitting along the time axis, history indexing, batch
sizes — derives from these values automatically.

## Architecture

The central object is the **Lagrangian Attention Tensor Network (LATN)**,
defined in `src/latn.py`. For one fluid particle it computes a tensor (the
pressure Hessian or viscous term) in three steps:

1. **Summarize the history (the "attention").** The recent trajectory of the VGT
   — shape `[time, 3, 3]` — is contracted against a set of learnable
   time-dependent tensor kernels, producing a handful of scalar "characteristics"
   (`num_filters` of them). This is the `ConstrainedTensorHistoryConv` module;
   its kernels are split into symmetric and antisymmetric parts so the learned
   attention respects the tensor structure of the strain-rate and rotation-rate
   tensors.
2. **Predict basis coefficients.** Those history characteristics are concatenated
   with the **5 invariants** of the instantaneous VGT (`calcInvariants` in
   `utils.py`) and fed through a small feed-forward network (`FFN`), which outputs
   one scalar coefficient `g_k` per tensor-basis element.
3. **Reconstruct the tensor.** The coefficients multiply an **integrity tensor
   basis** built from the VGT (`calcSymTensorBasis` → 10 elements for the
   symmetric pressure Hessian; `calcFullTensorBasis` → 16 for the viscous term).
   Summing `Σ_k g_k · T_k` guarantees the prediction transforms as a proper
   tensor under rotations — the physical constraint that makes the model
   generalize.

The third stage wraps the two trained LATNs in **`LATN_NODE`** (also in
`latn.py`), which forms the closed VGT evolution equation, integrates it with
Heun's method, and adds a calibrated stochastic forcing term.

Supporting modules:

| File | Responsibility |
|------|----------------|
| `src/lagrdataset.py` | Memory-mapped loading of the `.bin` files; builds train/test `Dataset`s; all the time-window indexing logic (`_create_inds`). |
| `src/utils.py` | Tensor invariants, tensor bases, finite differences, Kolmogorov-timescale normalization, restricted-Euler term. |
| `src/training_utils.py` | The `Trainer` loop, checkpointing/resume, TensorBoard logging. |
| `src/runner.py` | Command-line entry point; assembles the three stages and launches distributed training. |
| `src/distributed.py` | Thin multi-GPU (DDP/NCCL) setup helpers. |
| `src/plotting_utils.py` | Post-processing and figure generation. |

## Making modifications

A few common changes and where to make them:

- **Change a hyperparameter for one run** — pass a command-line flag (see the
  table in [Quickstart](#quickstart)). No code edit needed.
- **Use a different dataset (size, `dt`, field names)** — edit the `DataDesc(...)`
  blocks in `src/runner.py` (shape + `dt`); see
  [the data section](#pointing-the-code-at-a-dataset-of-a-different-size). The
  loader globs `aij*.bin`/`pij*.bin`/`vis*.bin`, so keep those prefixes.
- **Try a different way of summarizing history** — swap the
  `history_conv_type` passed to `LATN` in `training_utils.load_train_objs`. Three
  variants already exist in `latn.py` (`ScalarHistoryConv`, `TensorHistoryConv`,
  `ConstrainedTensorHistoryConv`); add your own `nn.Module` with the same
  `forward(x)` signature.
- **Change the network shape** — `FFN` vs. the residual `Skip_FFN` is selected in
  `load_train_objs`/`load_node_train_objs`; depth/width come from `-nl`/`-nu`.
- **Add or change invariants / tensor-basis elements** — edit `calcInvariants`
  and `calcSymTensorBasis`/`calcSkewSymTensorBasis` in `src/utils.py`. If you
  change the *count* of basis elements, update the `NUM_*` constants the runner
  reads and the network `output_len` accordingly.
- **Change the integration scheme or stochastic forcing** — see
  `LATN_NODE.forward` / `get_forcing` in `latn.py`.

After any change, re-run the test suite to make sure the data/model interface
still holds:

```bash
PYTHONPATH=src python -m pytest test/
```
