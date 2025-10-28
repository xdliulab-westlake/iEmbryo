"""
PyTorch optimization setup for multi-GPU and distributed training
Supports both DataParallel (single-process multi-GPU) and DDP (multi-process distributed)
"""

import os
import torch
import torch.distributed as dist
from typing import Dict, Any, Tuple


def is_main_process() -> bool:
    """Check if current process is main process"""
    return not dist.is_initialized() or dist.get_rank() == 0


def setup_pytorch_optimization(config: Dict[str, Any] = None) -> Tuple[torch.device, bool]:
    """
    Setup PyTorch optimization configuration for multi-GPU training

    Supports two modes:
    1. DataParallel: Single-process multi-GPU (python main.py)
    2. DDP: Multi-process distributed (torchrun --nproc_per_node=4 main.py)

    Args:
        config: Configuration dictionary (optional)

    Returns:
        (device, is_distributed): Device and whether distributed training is enabled
    """
    if config is None:
        config = {}

    opt_config = config.get('optimization', {})

    # Device setup
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True  # Optimize cudnn performance

        if is_main_process():
            print(f"\n{'='*60}")
            print("GPU CONFIGURATION")
            print(f"{'='*60}")
            print(f"CUDA Available: Yes")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"         Memory: {props.total_memory / 1024**3:.1f}GB")
    else:
        device = torch.device('cpu')
        if is_main_process():
            print("\nWarning: No CUDA devices found, using CPU training (much slower)")
        return device, False

    # Distributed training setup
    is_distributed = False
    if torch.cuda.device_count() > 1:
        try:
            # Check if already initialized
            if not dist.is_initialized():
                # Check if launched via torchrun or torch.distributed.launch
                if 'WORLD_SIZE' in os.environ and 'RANK' in os.environ:
                    # Launched with distributed launcher
                    if is_main_process():
                        print(f"\nDetected distributed environment")
                        print(f"  WORLD_SIZE: {os.environ.get('WORLD_SIZE')}")
                        print(f"  RANK: {os.environ.get('RANK')}")
                        print(f"  MASTER_ADDR: {os.environ.get('MASTER_ADDR')}")
                        print(f"  MASTER_PORT: {os.environ.get('MASTER_PORT')}")

                    try:
                        # Initialize DDP
                        dist.init_process_group(backend='nccl')
                        is_distributed = True

                        if is_main_process():
                            world_size = dist.get_world_size()
                            print(f"\nDistributed training initialized successfully")
                            print(f"  Using {world_size} GPUs for DDP training")
                            print(f"  Current rank: {dist.get_rank()}")
                    except Exception as e:
                        if is_main_process():
                            print(f"\nWarning: Failed to initialize DDP: {e}")
                            print(f"Falling back to DataParallel training")
                        is_distributed = False
                else:
                    # Single-process launch with multiple GPUs
                    if is_main_process():
                        print(f"\nDetected {torch.cuda.device_count()} GPUs in single-process mode")
                        print(f"Will use DataParallel for multi-GPU training")
                        print(f"\nTip: For better performance with DDP, launch with:")
                        print(f"  torchrun --nproc_per_node={torch.cuda.device_count()} main.py ...")
                    is_distributed = False
            else:
                is_distributed = True
                if is_main_process():
                    print(f"\nDistributed training already initialized")
                    print(f"  Using {dist.get_world_size()} GPUs")

        except Exception as e:
            if is_main_process():
                print(f"\nWarning: Error during distributed setup: {e}")
            is_distributed = False

    # Mixed precision setup
    if opt_config.get('mixed_precision', True) and device.type == 'cuda':
        if is_main_process():
            print(f"\nMixed Precision: Enabled (FP16 with AMP)")

    return device, is_distributed


def calculate_optimal_batch_size(batch_size_per_gpu: int) -> Tuple[int, int]:
    """
    Calculate batch size for multi-GPU training

    Args:
        batch_size_per_gpu: Batch size per GPU

    Returns:
        (per_gpu_batch_size, global_batch_size)
    """
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    global_batch_size = batch_size_per_gpu * num_gpus

    if is_main_process():
        print(f"\nBatch Size Configuration:")
        print(f"  Per-GPU batch size: {batch_size_per_gpu}")
        print(f"  Number of GPUs: {num_gpus}")
        print(f"  Global batch size: {global_batch_size}")

    return batch_size_per_gpu, global_batch_size


class MemoryMonitor:
    """Monitor GPU memory usage"""

    def __init__(self, device: torch.device):
        self.device = device
        self.enabled = device.type == 'cuda'

    def log_memory_usage(self, stage: str = ""):
        """Log current GPU memory usage"""
        if not self.enabled:
            return

        allocated = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(self.device) / 1024**3   # GB
        max_allocated = torch.cuda.max_memory_allocated(self.device) / 1024**3

        if stage:
            print(f"{stage} GPU Memory: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, Peak={max_allocated:.2f}GB")
        else:
            print(f"GPU Memory: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, Peak={max_allocated:.2f}GB")

    def reset_peak_stats(self):
        """Reset peak memory statistics"""
        if self.enabled:
            torch.cuda.reset_peak_memory_stats(self.device)


def get_device_index() -> int:
    """Get current device index for multi-GPU training"""
    if dist.is_initialized():
        return dist.get_rank() % torch.cuda.device_count()
    else:
        return 0


def synchronize():
    """Synchronize all processes (for distributed training)"""
    if dist.is_initialized():
        dist.barrier()
