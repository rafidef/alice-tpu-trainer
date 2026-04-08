#!/usr/bin/env python3
"""
TPU adapter for Alice Protocol Miner.

Provides TPU detection, multi-core training (all cores on a single VM),
and multi-VM TPU pod support (e.g. v5litepod-32 with 8 VMs × 4 cores).

Requires: torch_xla  (pip install torch_xla)
"""

from __future__ import annotations

import torch_xla.runtime as xr
import contextlib
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# Lazy import helpers – torch_xla may not be installed on non-TPU machines.
# ---------------------------------------------------------------------------

_xla_available: Optional[bool] = None


def is_xla_available() -> bool:
    """Return True if torch_xla is importable and a TPU device is reachable."""
    global _xla_available
    if _xla_available is not None:
        return _xla_available
    try:
        import torch_xla  # noqa: F401
        import torch_xla.core.xla_model as xm  # noqa: F401
        # Try to actually reach a TPU device
        _ = xm.xla_device()
        _xla_available = True
    except Exception:
        _xla_available = False
    return _xla_available


def _require_xla():
    if not is_xla_available():
        raise RuntimeError(
            "torch_xla is required for TPU support. "
            "Install with: pip install torch_xla"
        )


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def xla_device() -> torch.device:
    """Return the default XLA device for this process."""
    _require_xla()
    import torch_xla.core.xla_model as xm
    return xm.xla_device()


def xla_device_count() -> int:
    """Return the number of local TPU cores (devices) on this VM."""
    _require_xla()
    import torch_xla.core.xla_model as xm
    return xr.world_size()


def xla_local_device_count() -> int:
    """Return the number of TPU cores visible to this VM."""
    _require_xla()
    try:
        import torch_xla.runtime as xr
        return xr.local_device_count()
    except (ImportError, AttributeError):
        # Fallback for older torch_xla versions
        return int(os.environ.get("TPU_NUM_DEVICES", 4))


def xla_world_size() -> int:
    """Return total number of TPU cores across all VMs in the pod."""
    _require_xla()
    import torch_xla.core.xla_model as xm
    return xm.xr_world_size()


def xla_ordinal() -> int:
    """Return global ordinal of this process in the pod."""
    _require_xla()
    import torch_xla.core.xla_model as xm
    return xr.global_ordinal()


def xla_local_ordinal() -> int:
    """Return local ordinal of this process within its VM."""
    _require_xla()
    import torch_xla.core.xla_model as xm
    return xr.local_ordinal()


def xla_is_master() -> bool:
    """Return True if this is the master process (ordinal 0)."""
    return xla_ordinal() == 0


# ---------------------------------------------------------------------------
# TPU hardware detection
# ---------------------------------------------------------------------------

def detect_tpu_info() -> Dict[str, Any]:
    """Detect TPU hardware details. Returns a capabilities dict."""
    _require_xla()
    import torch_xla.core.xla_model as xm

    try:
        import torch_xla.runtime as xr
        local_cores = xr.local_device_count()
        global_cores = xr.global_device_count()
    except (ImportError, AttributeError):
        local_cores = int(os.environ.get("TPU_NUM_DEVICES", 4))
        global_cores = xm.xrt_world_size()

    # Determine TPU type from env or metadata
    tpu_type = os.environ.get("TPU_ACCELERATOR_TYPE", "unknown")
    if tpu_type == "unknown":
        tpu_type = os.environ.get("ACCELERATOR_TYPE", "unknown")

    # TPU memory per core (HBM)
    tpu_memory_map = {
        "v2": 8.0,
        "v3": 16.0,
        "v4": 32.0,
        "v5e": 16.0,
        "v5p": 95.0,
        "v5litepod": 16.0,
        "v6e": 32.0,
    }
    hbm_per_core_gb = 16.0  # default
    for prefix, mem in tpu_memory_map.items():
        if prefix in tpu_type.lower():
            hbm_per_core_gb = mem
            break

    total_hbm_gb = hbm_per_core_gb * local_cores
    num_vms = max(1, global_cores // max(1, local_cores))

    return {
        "device_type": "tpu",
        "tpu_type": tpu_type,
        "local_cores": local_cores,
        "global_cores": global_cores,
        "num_vms": num_vms,
        "hbm_per_core_gb": hbm_per_core_gb,
        "total_local_hbm_gb": total_hbm_gb,
        "total_pod_hbm_gb": hbm_per_core_gb * global_cores,
        "ordinal": xm.get_ordinal(),
        "local_ordinal": xm.get_local_ordinal(),
    }


def detect_tpu_device_info() -> Dict[str, Any]:
    """
    Return a capabilities dict compatible with alice_miner.detect_device_info().
    """
    _require_xla()
    import platform as _platform

    tpu = detect_tpu_info()

    try:
        import psutil
        ram_gb = round(psutil.virtual_memory().total / 1e9, 1)
        cpu_count = int(psutil.cpu_count() or 1)
    except Exception:
        ram_gb = 16.0
        cpu_count = int(os.cpu_count() or 1)

    tpu_name = f"TPU {tpu['tpu_type']} ({tpu['local_cores']} cores)"
    memory_gb = tpu["total_local_hbm_gb"]

    return {
        "device": "tpu",
        "device_type": "tpu",
        "device_name": tpu_name,
        "memory_gb": float(memory_gb),
        "runtime_memory_cap_gb": float(memory_gb),
        "system_memory_gb": float(ram_gb),
        "cpu_count": cpu_count,
        "platform": _platform.system().lower(),
        "os": _platform.system(),
        "arch": _platform.machine().lower(),
        "vendor": "google",
        "vram_gb": 0.0,
        "physical_vram_gb": 0.0,
        "unified_memory_gb": 0.0,
        "ram_gb": float(ram_gb),
        "cpu_model": _platform.processor() or "Unknown CPU",
        "gpu_model": tpu_name,
        "gpu_vram_gb": float(memory_gb),
        "gpu_count": tpu["local_cores"],
        "python": _platform.python_version(),
        "torch": torch.__version__,
        # TPU-specific fields
        "tpu_type": tpu["tpu_type"],
        "tpu_local_cores": tpu["local_cores"],
        "tpu_global_cores": tpu["global_cores"],
        "tpu_num_vms": tpu["num_vms"],
        "tpu_hbm_per_core_gb": tpu["hbm_per_core_gb"],
        "tpu_total_pod_hbm_gb": tpu["total_pod_hbm_gb"],
    }


# ---------------------------------------------------------------------------
# Distributed initialization for TPU pods
# ---------------------------------------------------------------------------

def init_tpu_distributed() -> None:
    """
    Initialize torch.distributed with the XLA backend for multi-VM pod training.
    Must be called once per process before any distributed ops.
    """
    _require_xla()
    import torch.distributed as dist

    if dist.is_initialized():
        return

    import torch_xla.distributed.xla_backend  # noqa: F401 — registers 'xla' backend
    import torch_xla.core.xla_model as xm

    dist.init_process_group(backend="xla")


def barrier() -> None:
    """Synchronize all processes in the pod."""
    _require_xla()
    import torch_xla.core.xla_model as xm
    xm.rendezvous("alice_barrier")


def mark_step() -> None:
    """
    Execute pending XLA operations. This is the TPU equivalent of
    torch.cuda.synchronize() / empty_cache(). Must be called periodically
    to flush the XLA computation graph.
    """
    _require_xla()
    import torch_xla.core.xla_model as xm
    xm.mark_step()


def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce (sum) a tensor across all TPU cores in the pod."""
    _require_xla()
    import torch_xla.core.xla_model as xm
    return xm.all_reduce(xm.REDUCE_SUM, tensor)


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce a tensor and divide by world size to get mean."""
    _require_xla()
    import torch_xla.core.xla_model as xm
    reduced = xm.all_reduce(xm.REDUCE_SUM, tensor)
    return reduced / xm.xrt_world_size()


def broadcast_master_param(model: torch.nn.Module) -> None:
    """Broadcast model parameters from master (ordinal 0) to all workers."""
    _require_xla()
    import torch_xla.core.xla_model as xm

    for param in model.parameters():
        xm.all_reduce(xm.REDUCE_SUM, param.data)
        param.data /= xm.xrt_world_size()
    mark_step()


# ---------------------------------------------------------------------------
# TPU-aware training utilities
# ---------------------------------------------------------------------------

def empty_cache_tpu() -> None:
    """
    TPU equivalent of torch.cuda.empty_cache().
    On TPU we flush the graph instead.
    """
    if is_xla_available():
        mark_step()


def optimizer_step_tpu(optimizer: torch.optim.Optimizer, barrier: bool = True) -> None:
    """
    Perform optimizer step with XLA graph sync.
    Use this instead of optimizer.step() on TPU.
    """
    _require_xla()
    import torch_xla.core.xla_model as xm
    xm.optimizer_step(optimizer, barrier=barrier)


@contextlib.contextmanager
def tpu_autocast(enabled: bool = True):
    """
    Autocast context for TPU. Uses bfloat16 which is natively supported
    on all TPU generations.
    """
    if not enabled:
        yield
        return
    with torch.autocast(device_type="xla", dtype=torch.bfloat16):
        yield


def move_model_to_tpu(
    model: torch.nn.Module,
    device: Optional[torch.device] = None,
) -> torch.nn.Module:
    """Move a model to TPU device with proper handling."""
    _require_xla()
    if device is None:
        device = xla_device()
    model = model.to(device)
    mark_step()
    return model


# ---------------------------------------------------------------------------
# Multi-core data parallel training (single VM, all cores)
# ---------------------------------------------------------------------------

class TPUDataParallelTrainer:
    """
    Wraps a training function to run across all local TPU cores using
    data parallelism. Each core gets a different slice of the data.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        world_size: Optional[int] = None,
    ):
        self.model = model
        self.world_size = world_size

    def shard_data(
        self,
        tokens: torch.Tensor,
        ordinal: int,
        world_size: int,
    ) -> torch.Tensor:
        """Split token tensor evenly across cores."""
        total = tokens.numel()
        per_core = total // world_size
        start = ordinal * per_core
        end = start + per_core
        return tokens[start:end]


def get_tpu_batch_size_multiplier() -> int:
    """
    Return the effective batch size multiplier for TPU.
    Each core processes its own micro-batch, so effective batch =
    per_core_batch × num_cores.
    """
    if not is_xla_available():
        return 1
    try:
        import torch_xla.runtime as xr
        return xr.global_device_count()
    except (ImportError, AttributeError):
        return xla_world_size()


# ---------------------------------------------------------------------------
# TPU pod multi-process spawn
# ---------------------------------------------------------------------------

def spawn_on_all_cores(fn, args=(), nprocs: Optional[int] = None):
    """
    Spawn a function on all local TPU cores.
    Each process gets (index, *args) as arguments.

    For multi-VM pods, this should be called on each VM.
    The pod orchestrator (tpu_launch.py) handles running across VMs.
    """
    _require_xla()
    import torch_xla.distributed.xla_multiprocessing as xmp

    if nprocs is None:
        nprocs = xla_local_device_count()

    xmp.spawn(fn, args=args, nprocs=nprocs)


# ---------------------------------------------------------------------------
# Gradient aggregation across TPU cores
# ---------------------------------------------------------------------------

def aggregate_gradients(model: torch.nn.Module) -> None:
    """
    Average gradients across all TPU cores before optimizer step.
    This implements data-parallel gradient synchronization.
    """
    _require_xla()
    import torch_xla.core.xla_model as xm

    gradients = []
    for param in model.parameters():
        if param.grad is not None:
            gradients.append(param.grad)

    if gradients:
        xm.all_reduce(xm.REDUCE_SUM, gradients)
        world_size = xm.xrt_world_size()
        for grad in gradients:
            grad /= world_size


def reduce_scalar(value: float, world_size: Optional[int] = None) -> float:
    """
    Reduce a scalar value (e.g. loss) across TPU cores, returning the mean.
    Within xmp.spawn(), this automatically scopes to local VM cores.
    Pass world_size to divide by a specific count (e.g. local core count).
    """
    _require_xla()
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    tensor = torch.tensor([value], dtype=torch.float32, device=device)
    reduced = xm.all_reduce(xm.REDUCE_SUM, tensor)
    mark_step()
    divisor = world_size if world_size is not None else xm.xrt_world_size()
    return float(reduced.item()) / max(1, divisor)


# ---------------------------------------------------------------------------
# Logging helper – only log from master process
# ---------------------------------------------------------------------------

def tpu_print(*args, **kwargs) -> None:
    """Print only from the master process to avoid duplicated output in pods."""
    _require_xla()
    import torch_xla.core.xla_model as xm
    xm.master_print(*args, **kwargs)
