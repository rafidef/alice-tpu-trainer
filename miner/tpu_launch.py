#!/usr/bin/env python3
"""
TPU Pod Launcher for Alice Protocol Miner.

Launches the miner across all TPU cores on the local VM.
For multi-VM TPU pods (e.g. v5litepod-32 with 8 VMs), run this script
on EACH VM — the script automatically coordinates via torch_xla distributed.

Usage:
  # Single VM with 4 TPU cores:
  python tpu_launch.py --ps-url https://... --address a1...

  # Multi-VM pod (run on EACH VM, e.g. via gcloud ssh or podrun):
  python tpu_launch.py --ps-url https://... --address a1...

  The script auto-detects the number of local cores and total pod size.

Environment variables (set automatically by GCP TPU VM runtime):
  TPU_ACCELERATOR_TYPE  - e.g. "v5litepod-32"
  TPU_NUM_DEVICES       - local cores per VM (usually 4)
  TPU_WORKER_ID         - this VM's index in the pod (0-based)
  TPU_WORKER_HOSTNAMES  - comma-separated hostnames of all VMs

For manual multi-VM setup, you can also set:
  MASTER_ADDR  - hostname/IP of VM 0
  MASTER_PORT  - coordination port (default 8476)
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

    # Import after env vars are set
    import torch_xla.runtime as xr
    import torch_xla.core.xla_model as xm

    local_cores = xr.local_device_count()
    global_cores = xr.global_device_count()
    worker_id = int(os.environ.get("TPU_WORKER_ID", "0"))
    num_vms = max(1, global_cores // max(1, local_cores))

    print(f"[TPU-LAUNCH] TPU Pod Configuration:")
    print(f"  Accelerator type: {os.environ.get('TPU_ACCELERATOR_TYPE', 'auto-detected')}")
    print(f"  Local cores: {local_cores}")
    print(f"  Global cores: {global_cores}")
    print(f"  Number of VMs: {num_vms}")
    print(f"  This VM worker ID: {worker_id}")

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
    print(f"[TPU-LAUNCH] Effective batch size multiplier: {global_cores}x")
    print(f"[TPU-LAUNCH] Miner args: {' '.join(miner_argv[1:])}")
    print()

    from alice_miner import main as miner_main
    miner_main()


if __name__ == "__main__":
    main()
