# Deploying LDM training to an AWS spot GPU instance

This reproduces the paper's pressure-Hessian (`ph`) and viscous-Laplacian (`vis`)
LATN models on AWS after losing HPC access. The data pipeline was memory-optimized
first (see `src/lagrdataset.py` memmap loaders); this directory covers the cloud run.

> **New to AWS / setting up from scratch?** Follow the step-by-step CLI walkthrough in
> [`../aws-configuration.org`](../aws-configuration.org) (IAM role, key pair, AMI,
> launch, spot, teardown). This README is the reference for *instance sizing, memory,
> checkpointing, and the output layout* that the walkthrough relies on.

**Output namespacing.** `runner.py` writes each run under a config slug derived from its
hyperparameters: `outputs/<slug>/{ph,vis}/` (e.g.
`outputs/nu80_nl3_nf16_lr0.3_dr0_hl50_ht1_pt0.2/`), with a `config.json` manifest. So a
sweep never lets one config overwrite another's checkpoints, and `--resume` is per-config.
Override the slug with `-rn/--run_name`. `bootstrap.sh` syncs this whole tree to/from S3
asynchronously (background loop + spot-interruption watcher), so checkpoints reach S3
without stalling the training loop.

## Measured memory footprint (the constraint that drove this)

Peak host RSS of `LagrDataset.from_file` on the full `(131072, 1000, 3, 3)` dataset
(`-hl 50 -ht 1 -pt 0.2`), measured locally:

| `--num_samples` | train samples | peak host RSS |
|-----------------|---------------|---------------|
| 16384           | 245,760       | **2.9 GB**    |
| all (131072)    | 1,966,080     | **20.3 GB**   |

(Down from the old `np.fromfile` + `torch.tensor` double-copy path, which peaked
~26 GB+ and OOM'd a 30 GB box.) `runner.py` trains `ph` then `vis` in **separate**
`spawn`ed processes, so this is the per-process peak, not additive. The GPU only ever
holds one mini-batch (~290 MB at batch `1<<17`) plus the tiny LATN model — GPU memory
is not the binding constraint.

## Instance selection

Single GPU ⇒ `world_size = torch.cuda.device_count() = 1`; the existing `spawn`/DDP/NCCL
path runs fine with one process. **NCCL requires a GPU — do not run CPU-only.**

| Instance      | GPU            | vCPU / RAM   | Fits full dataset? | Notes |
|---------------|----------------|--------------|--------------------|-------|
| `g6.xlarge`   | L4 24 GB       | 4 / 16 GB    | No (20 GB > 16 GB) | Cheapest; use `--num_samples` ≤ ~80k, or train a subsample. |
| `g5.xlarge`   | A10G 24 GB     | 4 / 16 GB    | No (20 GB > 16 GB) | Same RAM ceiling as above. |
| **`g5.2xlarge`** | **A10G 24 GB** | **8 / 32 GB** | **Yes (20 GB < 32 GB)** | **Recommended** for the full-dataset paper reproduction. |
| `g6.2xlarge`  | L4 24 GB       | 8 / 32 GB    | Yes                | Cheaper L4 alternative to g5.2xlarge. |

Recommendation: **`g5.2xlarge` (or `g6.2xlarge`) on spot** for the full run. Drop to a
`*.xlarge` only if training a subsample via `--num_samples`.

## Data location and region

Data lives in **S3 Standard** at `s3://ua-hpc-archive/groups-data/lagrangian-vgt/`
(the three `.bin` files, ~26 GB total), in **us-east-2**. **Launch every instance in
us-east-2** so the S3→EC2 transfer is free and fast. Standard class has no per-read
retrieval fee, so re-pulling on each launch is free.

## Storage / disk

The instance needs a disk (an EBS root volume) holding the OS, the Python env, the
~26 GB dataset, and outputs — size the **root volume to ~60 GB** at launch (the default
8 GB Ubuntu root is too small; the Deep Learning AMI default is already large enough).
Outputs are small. No *separate* EBS volume is required for a one-off run.

## Amortizing setup across a hyperparameter sweep (~50 runs)

Env install (~2–3 min for the torch wheel) + first data pull is a meaningful fraction of
each short (~5–10 min) run. Re-paying it 50× is wasteful. Two good ways to pay it once:

- **Bake a custom AMI (recommended).** Set up one instance, run `bootstrap.sh` once so the
  venv and dataset are present, then *Create Image* from it. Every future instance — spot
  or on-demand, **and many in parallel** — boots from that AMI with env + data already
  there, ready to train in ~1 min. An AMI is region-wide (no AZ pinning) and lets you fan
  the 50-run grid across N spot instances simultaneously. Storage is ~$1–2/month of EBS
  snapshot; delete the image when the campaign ends.
- **Persistent volume + sequential runs.** Keep `DATA_DIR` and `.venv` on an EBS volume you
  reuse; `bootstrap.sh` is idempotent (skips env install when torch imports, and the data
  `sync` is a no-op when present), so repeat launches skip both costs. Caveat: an EBS volume
  is **pinned to one Availability Zone** and attaches to **one instance at a time** — fine
  for sequential runs on one box, awkward for parallel spot (capacity may land in another
  AZ). Prefer the AMI for parallelism.

Either way, drive the 50 configs from a loop (varying `runner.py` flags) and write each
run's outputs to a distinct S3 prefix.

## Spot setup

1. Launch a spot instance **in us-east-2** from the **AWS Deep Learning AMI (GPU, PyTorch)**
   — or your baked AMI, or Ubuntu 22.04 + NVIDIA driver. Attach an **IAM instance role**
   with `s3:GetObject` on the data bucket (and `PutObject` on the outputs bucket) so no keys
   are baked in.
2. Run the bootstrap (S3_DATA defaults to the path above):
   ```bash
   git clone https://github.com/cmhyett/LDM.git && cd LDM
   S3_OUT=s3://<bucket>/ldm-outputs \
   ./deploy/bootstrap.sh -me 200            # add -ns / -bs to tune
   ```
3. **Interruption handling (implemented).** Every `--save_every` epochs, `Trainer` writes
   both the bare `checkpoint_<epoch>.pt` and a full **`checkpoint_resume.pt`** bundle
   (model + optimizer + scheduler + epoch + best-loss + loss normalization) into each of
   `outputs/{ph,vis}/`. To survive spot reclaim:
   - Lower the checkpoint cadence for spot, e.g. `-se 10`.
   - **Put the resume bundle somewhere AZ-independent — i.e. S3.** Sync `outputs/` to S3
     periodically so the bundle outlives the instance (the bootstrap syncs at the end; add a
     cron/`/loop` for mid-run, or sync on the spot interruption notice). A persistent EBS
     volume can also hold it, but EBS is pinned to one Availability Zone, so if spot capacity
     reappears in a different AZ you cannot re-attach it — S3 has no such constraint and any
     replacement instance in the region can pull the latest checkpoint. Use the persistent
     volume/AMI for the expensive-to-recreate env+data, and S3 for the small, frequently
     updated resume checkpoint.
   - On relaunch, re-run with `--resume`; each model continues from its
     `checkpoint_resume.pt` (the run is numerically continuous — normalization and
     optimizer/scheduler state are restored, not recomputed). If a bundle is absent (fresh
     launch, or the second model never started), that model starts cleanly from epoch 0.

   Example resilient invocation:
   ```bash
   ./deploy/bootstrap.sh -me 200 -se 10 --resume
   ```
   `--resume` is safe to pass on the very first launch too (no bundle yet → fresh start).

## Validation against the original HPC run

The repo ships the original outputs in `outputs/{ph,vis}/` (loss CSVs, checkpoints,
`*_apriori_eval.pt`). After a cloud run, compare loss-curve shape/magnitude and the
`*_apriori_eval.pt` tensor shapes/trends against those references.
