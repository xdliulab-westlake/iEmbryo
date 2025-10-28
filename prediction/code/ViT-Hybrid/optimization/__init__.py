"""
Optimization utilities for multi-GPU training and distributed setup
"""

from .pytorch_optimization import (
    setup_pytorch_optimization,
    MemoryMonitor,
    calculate_optimal_batch_size
)

__all__ = [
    'setup_pytorch_optimization',
    'MemoryMonitor',
    'calculate_optimal_batch_size'
]
