"""
STORK优化模块
包含DeepSpeed和TensorFlow内存优化工具
"""

from .deepspeed_config import (
    DeepSpeedManager,
    setup_tensorflow_memory_optimization,
    create_memory_efficient_strategy,
    calculate_batch_sizes,
    initialize_optimization
)

__all__ = [
    'DeepSpeedManager',
    'setup_tensorflow_memory_optimization',
    'create_memory_efficient_strategy',
    'calculate_batch_sizes',
    'initialize_optimization'
]