#!/usr/bin/env bash
# Bootstrap an LDM training run on a fresh AWS GPU instance.
#
# Data lives in S3 Standard at s3://ua-hpc-archive/groups-data/lagrangian-vgt/
# in us-east-2 -- LAUNCH THE INSTANCE IN us-east-2 so the transfer is free/fast.
#
# Assumes:
#   - An NVIDIA GPU + driver are present (Deep Learning AMI, or Ubuntu 22.04
#     with the NVIDIA driver installed). Verify with `nvidia-smi`.
#   - The instance role (or `aws configure`) can read the data bucket
#     (and write the outputs bucket, if S3_OUT is set).
#   - This repo is checked out at $REPO_DIR.
#
# Usage:
#   ./deploy/bootstrap.sh [extra runner.py args...]                 # uses defaults
#   S3_OUT=s3://my-bucket/ldm-outputs ./deploy/bootstrap.sh -me 200 # sync results out
#
# Any extra args are forwarded to runner.py (e.g. -ns 65536 -bs 65536 -me 200).
#
# Idempotent for persistent-disk reuse: the venv install is skipped when torch
# already imports, and `aws s3 sync` is a near-no-op when the data is already
# present. So if DATA_DIR + .venv live on a volume you keep across runs, repeat
# launches pay neither the env-install nor the data-download cost again.
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_DIR="${DATA_DIR:-/mnt/data/lagrangian_vgt_240_data}"
OUT_DIR="${OUT_DIR:-$REPO_DIR/outputs}"
S3_DATA="${S3_DATA:-s3://ua-hpc-archive/groups-data/lagrangian-vgt/}"
S3_OUT="${S3_OUT:-}"   # optional: sync outputs back here when done

cd "$REPO_DIR"

# --- 1. Python env -----------------------------------------------------------
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
# torch must come from the CUDA index; everything else from PyPI.
python -c "import torch" 2>/dev/null || \
  pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r python_config/requirements-aws.txt

# --- 2. Sanity: GPU visible to torch ----------------------------------------
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not available -- check NVIDIA driver/AMI"
print("GPUs:", torch.cuda.device_count(), "->", torch.cuda.get_device_name(0))
PY

# --- 3. Pull dataset from S3 (idempotent) -----------------------------------
mkdir -p "$DATA_DIR"
aws s3 sync "$S3_DATA" "$DATA_DIR"
echo "Dataset in $DATA_DIR:"; ls -lh "$DATA_DIR"

# --- 4. Pull existing checkpoints down from S3 (enables --resume) ------------
# A spot replacement re-runs this script verbatim with no "I am a restart"
# flag, so the ONLY way runner.py --resume finds a prior bundle is if we mirror
# the outputs prefix down first. Checkpoints are KB, so syncing the whole tree
# is cheap; runner.py selects this run's config slug under it internally.
mkdir -p "$OUT_DIR"
if [ -n "$S3_OUT" ]; then
  aws s3 sync "$S3_OUT" "$OUT_DIR"
fi

# --- 5. Asynchronous checkpoint -> S3 (decoupled from the training loop) -----
# Training writes checkpoints to local disk (sub-ms); these background helpers
# ship them to S3 so the GPU thread never blocks on S3 latency.
SYNC_PID=""; WATCH_PID=""
sync_up() { [ -n "$S3_OUT" ] && aws s3 sync "$OUT_DIR" "$S3_OUT" >/dev/null 2>&1 || true; }

on_ec2() {
  # True only on a real EC2 instance (IMDS reachable). Fast-fails elsewhere so a
  # local invocation never powers off your workstation.
  local tok
  tok=$(curl -s --connect-timeout 1 --max-time 2 -X PUT \
        "http://169.254.169.254/latest/api/token" \
        -H "X-aws-ec2-metadata-token-ttl-seconds: 60" 2>/dev/null) || return 1
  curl -s --connect-timeout 1 --max-time 2 -H "X-aws-ec2-metadata-token: $tok" \
       http://169.254.169.254/latest/meta-data/instance-id >/dev/null 2>&1
}

cleanup() {
  # Stop the background helpers and flush the final state on ANY exit.
  [ -n "$SYNC_PID" ]  && kill "$SYNC_PID"  2>/dev/null || true
  [ -n "$WATCH_PID" ] && kill "$WATCH_PID" 2>/dev/null || true
  echo "Final sync to S3..."; sync_up
  # Always terminate the instance once artifacts are safely on S3. The launch
  # sets --instance-initiated-shutdown-behavior terminate, so halting the OS
  # tears the instance down (and deletes the DeleteOnTermination root volume).
  if on_ec2; then
    echo "Artifacts on S3; shutting down instance (-> terminate)."
    sudo shutdown -h now
  fi
}
trap cleanup EXIT

if [ -n "$S3_OUT" ]; then
  # Periodic mirror up (default every 120 s; override with SYNC_INTERVAL).
  ( while :; do sync_up; sleep "${SYNC_INTERVAL:-120}"; done ) &
  SYNC_PID=$!

  # Spot-interruption watcher: poll IMDS (v2 token required) for the ~2-minute
  # notice and flush immediately when it arrives.
  ( while :; do
      TOK=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" \
            -H "X-aws-ec2-metadata-token-ttl-seconds: 60" 2>/dev/null)
      ACTION=$(curl -s -H "X-aws-ec2-metadata-token: $TOK" \
               http://169.254.169.254/latest/meta-data/spot/instance-action 2>/dev/null)
      if echo "$ACTION" | grep -q '"action"'; then
        echo "Spot interruption notice received -- flushing to S3."
        sync_up
        break
      fi
      sleep 5
    done ) &
  WATCH_PID=$!
fi

# --- 6. Train ----------------------------------------------------------------
# Defaults mirror the paper run; override hyperparameters via extra args.
# --resume is always on: a replacement instance continues from the synced-down
# bundle, and it is a no-op on a genuinely fresh run (no bundle -> epoch 0).
# To force a clean re-run of a config, give it a new --run_name (-rn).
PYTHONPATH=src python src/runner.py \
  -dp "$DATA_DIR" \
  -sp "$OUT_DIR" \
  -hl 50 -ht 1 -pt 0.2 -nf 16 \
  -me 200 \
  --resume \
  "$@"

# The EXIT trap performs the final S3 flush and stops the background helpers.
