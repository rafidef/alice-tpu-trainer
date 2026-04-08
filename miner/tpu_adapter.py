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
        # Do not call xm.xla_device() here! It initializes the XLA runtime,
        # which crashes xmp.spawn() later. Just check if the module imports
        # and if we are on a TPU.
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
    """Return the number of TPU cores visible to this VM.

    Warning: Calling this before xmp.spawn will initialize the runtime!
    If you need the count before spawn, use `xla_local_device_count_safe()`.
    """
    _require_xla()
    try:
        import torch_xla.runtime as xr
        return xr.local_device_count()
    except (ImportError, AttributeError):
        # Fallback for older torch_xla versions
        return int(os.environ.get("TPU_NUM_DEVICES", 4))


def xla_local_device_count_safe() -> int:
    """Return the number of TPU cores without initializing the XLA runtime."""
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

    # Determine TPU type from env or metadata
    tpu_type = os.environ.get("TPU_ACCELERATOR_TYPE", "unknown")
    if tpu_type == "unknown":
        tpu_type = os.environ.get("ACCELERATOR_TYPE", "unknown")

    local_cores = xla_local_device_count_safe()
    global_cores = local_cores
    if '-' in tpu_type and tpu_type.split('-')[1].isdigit():
        global_cores = int(tpu_type.split('-')[1])

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
        "ordinal": 0,
        "local_ordinal": 0,
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
    """Synchronize all local processes on this VM."""
    _require_xla()
    import torch_xla.core.xla_model as xm
    # Create a local process group for rendezvous so we don't wait for other VMs
    try:
        import torch_xla.runtime as xr
        local_device_count = xr.local_device_count()
        global_device_count = xr.global_device_count()
        # replicas must be a list of lists representing process groups for ALL global replicas
        local_group = [list(range(i, i + local_device_count)) for i in range(0, global_device_count, local_device_count)]
        # Ensure we only sync local devices
        xm.rendezvous("alice_local_barrier", payload=b'', replicas=local_group)
    except (ImportError, AttributeError):
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
    """All-reduce (sum) a tensor across all local TPU cores."""
    _require_xla()
    import torch_xla.core.xla_model as xm

    try:
        import torch_xla.runtime as xr
        local_device_count = xr.local_device_count()
        global_device_count = xr.global_device_count()
        # groups must be a list of lists representing process groups for ALL global replicas
        local_group = [list(range(i, i + local_device_count)) for i in range(0, global_device_count, local_device_count)]
        return xm.all_reduce(xm.REDUCE_SUM, tensor, groups=local_group)
    except (ImportError, AttributeError):
        return xm.all_reduce(xm.REDUCE_SUM, tensor)


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce a tensor across local cores and divide by world size to get mean."""
    _require_xla()
    import torch_xla.core.xla_model as xm

    try:
        import torch_xla.runtime as xr
        local_device_count = xr.local_device_count()
        global_device_count = xr.global_device_count()
        # groups must be a list of lists representing process groups for ALL global replicas
        local_group = [list(range(i, i + local_device_count)) for i in range(0, global_device_count, local_device_count)]
        reduced = xm.all_reduce(xm.REDUCE_SUM, tensor, groups=local_group)
    except (ImportError, AttributeError):
        reduced = xm.all_reduce(xm.REDUCE_SUM, tensor)

    return reduced / xla_local_device_count()


def broadcast_master_param(model: torch.nn.Module) -> None:
    """Broadcast model parameters from master (ordinal 0) to all workers on the local VM."""
    _require_xla()
    import torch_xla.core.xla_model as xm

    try:
        import torch_xla.runtime as xr
        local_device_count = xr.local_device_count()
        global_device_count = xr.global_device_count()
        # groups must be a list of lists representing process groups for ALL global replicas
        local_group = [list(range(i, i + local_device_count)) for i in range(0, global_device_count, local_device_count)]
    except (ImportError, AttributeError):
        local_group = None

    world_size = xla_local_device_count()
    for param in model.parameters():
        if local_group is not None:
            xm.all_reduce(xm.REDUCE_SUM, param.data, groups=local_group)
        else:
            xm.all_reduce(xm.REDUCE_SUM, param.data)
        param.data /= world_size
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
    Spawn a function on all local TPU cores (one process per core).
    Works on both old and new torch_xla/PJRT runtimes (v5e, v5litepod, etc.).

    Each worker receives (index, *args) where index = local core ordinal (0..N-1).
    """
    _require_xla()
    import torch_xla.distributed.xla_multiprocessing as xmp

    # Disable multi-VM discovery for PyTorch XLA PJRT in case the user didn't use the launcher.
    # Since each VM operates entirely independently, we must prevent PyTorch XLA
    # from trying to connect to other VMs in the pod.
    for key in [
        "TPU_WORKER_HOSTNAMES",
        "MEGATRON_WORKER_HOSTNAMES",
        "CLOUD_TPU_TASK_ID",
        "TPU_PROCESS_ADDRESSES",
    ]:
        os.environ.pop(key, None)
    os.environ["TPU_WORKER_ID"] = "0"
    os.environ.pop("TPU_NAME", None)
    os.environ.pop("TPU_POD_NAME", None)

    # Modern PJRT fix (required on v5 TPUs)
    # Explicit nprocs > 1 is no longer supported.
    # nprocs=None tells it to automatically use ALL available TPU cores.
    xmp.spawn(fn, args=args, nprocs=None)


# ---------------------------------------------------------------------------
# Gradient aggregation across TPU cores
# ---------------------------------------------------------------------------

def aggregate_gradients(model: torch.nn.Module) -> None:
    """
    Average gradients across all local TPU cores before optimizer step.
    This implements data-parallel gradient synchronization within a VM.
    """
    _require_xla()
    import torch_xla.core.xla_model as xm

    try:
        import torch_xla.runtime as xr
        local_device_count = xr.local_device_count()
        global_device_count = xr.global_device_count()
        # groups must be a list of lists representing process groups for ALL global replicas
        local_group = [list(range(i, i + local_device_count)) for i in range(0, global_device_count, local_device_count)]
    except (ImportError, AttributeError):
        local_group = None

    gradients = []
    for param in model.parameters():
        if param.grad is not None:
            gradients.append(param.grad)

    if gradients:
        if local_group is not None:
            xm.all_reduce(xm.REDUCE_SUM, gradients, groups=local_group)
        else:
            xm.all_reduce(xm.REDUCE_SUM, gradients)

        world_size = xla_local_device_count()
        for grad in gradients:
            grad /= world_size


def reduce_scalar(value: float, world_size: Optional[int] = None) -> float:
    """
    Reduce a scalar value (e.g. loss) across local TPU cores, returning the mean.
    """
    _require_xla()
    import torch_xla.core.xla_model as xm

    try:
        import torch_xla.runtime as xr
        local_device_count = xr.local_device_count()
        global_device_count = xr.global_device_count()
        # groups must be a list of lists representing process groups for ALL global replicas
        local_group = [list(range(i, i + local_device_count)) for i in range(0, global_device_count, local_device_count)]
    except (ImportError, AttributeError):
        local_group = None

    device = xm.xla_device()
    tensor = torch.tensor([value], dtype=torch.float32, device=device)

    if local_group is not None:
        reduced = xm.all_reduce(xm.REDUCE_SUM, tensor, groups=local_group)
    else:
        reduced = xm.all_reduce(xm.REDUCE_SUM, tensor)

    mark_step()
    divisor = world_size if world_size is not None else xla_local_device_count()
    return float(reduced.item()) / max(1, divisor)


# ---------------------------------------------------------------------------
# Logging helper – only log from master process
# ---------------------------------------------------------------------------

def tpu_print(*args, **kwargs) -> None:
    """Print only from the master process to avoid duplicated output in pods."""
    _require_xla()
    import torch_xla.core.xla_model as xm
    xm.master_print(*args, **kwargs)
