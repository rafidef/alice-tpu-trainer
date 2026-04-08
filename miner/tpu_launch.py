#!/usr/bin/env python3
"""
TPU Pod Launcher for Alice Protocol Miner.

Launches the miner across all TPU cores on the local VM.
Each VM acts as an INDEPENDENT miner — no cross-VM coordination needed.
This means each VM earns rewards separately from the Alice protocol.

Architecture (example: v5litepod-32 = 8 VMs × 4 cores):
  - Run this script on EACH VM
  - Each VM spawns 4 processes (one per local TPU core)
  - The 4 local cores do data-parallel training with gradient all-reduce
  - Each VM submits its own delta to the server at epoch end
  - Total: 8 independent miners, each using 4 cores

Usage:
  # Single VM with 4 TPU cores:
  python tpu_launch.py --ps-url https://... --address a1...

  # Multi-VM pod — run the SAME command on EACH VM:
  # (via gcloud compute tpus tpu-vm ssh --worker=all, or a loop)
  gcloud compute tpus tpu-vm ssh my-tpu-pod --worker=all --command="
    cd /path/to/Alice-Protocol/miner &&
    python tpu_launch.py --ps-url https://... --address a1...
  "

Environment variables (set automatically by GCP TPU VM runtime):
  TPU_ACCELERATOR_TYPE  - e.g. "v5litepod-32"
  TPU_NUM_DEVICES       - local cores per VM (usually 4)
  TPU_WORKER_ID         - this VM's index in the pod (0-based)
"""

from __future__ import annotations

import os
import sys
import argparse


def main():
    # Ensure torch_xla is available before anything else
    try:
        import torch_xla
    except ImportError:
        print("[ERROR] torch_xla is required for TPU training.")
        print("[ERROR] Install with: pip install torch_xla")
        sys.exit(1)

    # Set XLA runtime flags for optimal performance
    # Use PJRT runtime (default in torch_xla >= 2.1)
    os.environ.setdefault("PJRT_DEVICE", "TPU")

    # Enable async data loading to overlap data prep with TPU compute
    os.environ.setdefault("XLA_USE_EAGER_DEBUG_MODE", "0")

    # Optimize XLA compilation cache
    xla_cache_dir = os.path.expanduser("~/.alice/xla_cache")
    os.makedirs(xla_cache_dir, exist_ok=True)
    os.environ.setdefault("XLA_PERSISTENT_CACHE_PATH", xla_cache_dir)
    os.environ.setdefault("XLA_FLAGS", f"--xla_gpu_autotune_level=0")

    # Disable multi-VM discovery for PyTorch XLA PJRT
    # Since each VM operates entirely independently, we must prevent PyTorch XLA
    # from trying to connect to other VMs in the pod.
    for key in [
        "TPU_WORKER_HOSTNAMES",
        "MEGATRON_WORKER_HOSTNAMES",
        "CLOUD_TPU_TASK_ID",
        "TPU_PROCESS_ADDRESSES",
    ]:
        os.environ.pop(key, None)
    # Reset WORKER_ID so PyTorch XLA thinks it's the master of its own single-node cluster
    os.environ["TPU_WORKER_ID"] = "0"
    os.environ.pop("TPU_NAME", None)
    os.environ.pop("TPU_POD_NAME", None)

    # DO NOT import torch_xla.runtime or xla_model here before xmp.spawn!
    # It will initialize the XLA runtime and cause "Runtime is already initialized" error.

    # We can infer configuration from env vars instead of querying XLA directly in the launcher.
    accelerator_type = os.environ.get('TPU_ACCELERATOR_TYPE', 'auto-detected')
    worker_id = int(os.environ.get("TPU_WORKER_ID", "0"))

    # Best-effort inference of cores to log
    local_cores = int(os.environ.get("TPU_NUM_DEVICES", "4"))

    # In newer TPU pods, ACCELERATOR_TYPE might look like v5litepod-32
    global_cores = local_cores
    if '-' in accelerator_type and accelerator_type.split('-')[1].isdigit():
        global_cores = int(accelerator_type.split('-')[1])

    num_vms = max(1, global_cores // max(1, local_cores))

    print(f"[TPU-LAUNCH] TPU Configuration:")
    print(f"  Accelerator type: {accelerator_type}")
    print(f"  Local cores (this VM): ~{local_cores} (estimated before launch)")
    print(f"  Global cores (all VMs): ~{global_cores} (estimated before launch)")
    print(f"  Number of VMs in pod: ~{num_vms}")
    print(f"  This VM worker ID: {worker_id}")
    print(f"  Mode: each VM mines independently (~{num_vms} independent miners)")

    # Force the miner to use TPU device
    sys.argv_backup = list(sys.argv)

    # Parse args meant for the miner, pass them through
    parser = argparse.ArgumentParser(
        description="TPU Pod Launcher for Alice Miner",
        add_help=False,
    )
    parser.add_argument("--ps-url", required=True, help="Parameter server URL")
    parser.add_argument("--address", required=True, help="Miner wallet address")
    # Capture all other args to forward
    known_args, remaining_args = parser.parse_known_args()

    # Build the miner argv — inject --device tpu
    miner_argv = [
        "alice_miner.py",
        "--ps-url", known_args.ps_url,
        "--address", known_args.address,
        "--device", "tpu",
        "--precision", "bf16",
    ] + remaining_args

    # Inject into sys.argv so the miner sees them
    sys.argv = miner_argv

    # Import and run
    print(f"[TPU-LAUNCH] Starting miner on {local_cores} local TPU cores...")
    print(f"[TPU-LAUNCH] Per-VM batch multiplier: {local_cores}x (data parallel across local cores)")
    print(f"[TPU-LAUNCH] Miner args: {' '.join(miner_argv[1:])}")
    print()

    from alice_miner import main as miner_main
    miner_main()


if __name__ == "__main__":
    main()
