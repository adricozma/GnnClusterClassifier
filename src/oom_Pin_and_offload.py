"""Memory safety, disk activation offloading, and multi-node DDP helpers for GNN training."""

import os
import tempfile
import uuid
import torch
import torch.distributed as dist


def init_distributed():
    """Initialize PyTorch Distributed (gloo on CPU) from SLURM or torchrun environment variables.

    Returns:
        `(is_distributed, rank, local_rank, world_size)`
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    elif "SLURM_PROCID" in os.environ and int(os.environ.get("SLURM_NTASKS", "1")) > 1:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ.get("SLURM_LOCALID", 0))
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(local_rank)
    else:
        return False, 0, 0, 1

    if world_size <= 1:
        return False, 0, 0, 1

    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://", rank=rank, world_size=world_size)
    return True, rank, local_rank, world_size


def get_process_and_cgroup_rss_gb() -> float:
    """Return current memory usage in GB (maximum of process VmRSS and cgroup memory.current)."""
    rss_gb = 0.0
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    kb = float(line.split()[1])
                    rss_gb = max(rss_gb, kb / (1024.0 * 1024.0))
                    break
    except Exception:
        pass

    for cg_path in (
        "/sys/fs/cgroup/memory.current",
        "/sys/fs/cgroup/memory/memory.usage_in_bytes",
    ):
        if os.path.exists(cg_path):
            try:
                with open(cg_path, "r", encoding="utf-8") as f:
                    b = float(f.read().strip())
                    rss_gb = max(rss_gb, b / (1024.0**3))
            except Exception:
                pass
    return rss_gb


def get_memory_limit_gb(default_gb: float = 492.0) -> float:
    """Detect SLURM or cgroup memory limit in GB."""
    for cg_max in (
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    ):
        if os.path.exists(cg_max):
            try:
                with open(cg_max, "r", encoding="utf-8") as f:
                    val = f.read().strip()
                if val.isdigit():
                    b = int(val)
                    if 0 < b < (1 << 50):
                        return float(b) / (1024.0**3)
            except Exception:
                pass
    slurm_mem = os.environ.get("SLURM_MEM_PER_NODE", "")
    if slurm_mem.isdigit():
        return float(slurm_mem) / 1024.0
    return default_gb


class DiskActivationOffloader:
    """Context manager using `torch.autograd.graph.saved_tensors_hooks` to offload large activations to `/scratch` when RAM usage exceeds a threshold."""

    def __init__(
        self,
        offload_dir: str = "/scratch/cozma/tmp_autograd_offload",
        mem_threshold_fraction: float = 0.72,
        min_tensor_bytes: int = 8 * 1024 * 1024,
        force_enable: bool = False,
    ):
        self.offload_dir = offload_dir
        self.mem_limit_gb = get_memory_limit_gb()
        self.threshold_gb = self.mem_limit_gb * mem_threshold_fraction
        self.min_tensor_bytes = min_tensor_bytes
        self.force_enable = force_enable
        self._hooks_ctx = None
        self._files = []
        self._pack_counter = 0
        self._cached_over = False

    def _should_offload(self, tensor: torch.Tensor) -> bool:
        if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            return False
        if tensor.numel() * tensor.element_size() < self.min_tensor_bytes:
            return False
        if isinstance(tensor, torch.nn.Parameter):
            return False
        if self.force_enable:
            return True
        if self._pack_counter % 16 == 0:
            self._cached_over = get_process_and_cgroup_rss_gb() >= self.threshold_gb
        self._pack_counter += 1
        return self._cached_over

    def _pack_hook(self, tensor: torch.Tensor):
        if not self._should_offload(tensor):
            return tensor
        os.makedirs(self.offload_dir, exist_ok=True)
        fpath = os.path.join(self.offload_dir, f"act_{os.getpid()}_{uuid.uuid4().hex}.bin")
        arr = tensor.detach().cpu().contiguous()
        with open(fpath, "wb") as f:
            f.write(arr.numpy().tobytes())
        self._files.append(fpath)
        return (fpath, tuple(arr.shape), arr.dtype, tensor.device)

    def _unpack_hook(self, packed):
        if isinstance(packed, torch.Tensor):
            return packed
        fpath, shape, dtype, device = packed
        import numpy as np
        np_dtype = {
            torch.float16: np.float16,
            torch.float32: np.float32,
            torch.float64: np.float64,
        }.get(dtype, np.float32)
        raw = np.fromfile(fpath, dtype=np_dtype).reshape(shape)
        tensor = torch.from_numpy(raw).to(device)
        return tensor

    def __enter__(self):
        if not self.force_enable:
            self._cached_over = get_process_and_cgroup_rss_gb() >= self.threshold_gb
        else:
            self._cached_over = True
        if self._cached_over:
            self._hooks_ctx = torch.autograd.graph.saved_tensors_hooks(self._pack_hook, self._unpack_hook)
            self._hooks_ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._hooks_ctx is not None:
            self._hooks_ctx.__exit__(exc_type, exc_val, exc_tb)
            self._hooks_ctx = None
        for fpath in self._files:
            try:
                if os.path.exists(fpath):
                    os.remove(fpath)
            except OSError:
                pass
        self._files.clear()
