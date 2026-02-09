"""
PyTorch + DeepSpeed 优化配置
原生支持DeepSpeed的内存和性能优化
"""

import os
import torch
import torch.distributed as dist
from typing import Dict, Any, Optional, Tuple


def is_main_process() -> bool:
    """检查当前进程是否为主进程"""
    return not dist.is_initialized() or dist.get_rank() == 0


try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False


class DeepSpeedOptimizer:
    """DeepSpeed优化器管理类"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化DeepSpeed优化器

        Args:
            config: 优化配置
        """
        self.config = config
        self.opt_config = config.get('optimization', {})
        self.available = DEEPSPEED_AVAILABLE and self.opt_config.get('use_deepspeed', False)

        if self.available:
            if is_main_process():
                print("DeepSpeed可用，将启用ZeRO内存优化")
        else:
            if is_main_process():
                if not DEEPSPEED_AVAILABLE:
                    print("DeepSpeed未安装，使用标准PyTorch优化")
                else:
                    print("使用标准PyTorch优化（DeepSpeed已禁用）")

    def is_available(self) -> bool:
        """检查DeepSpeed是否可用"""
        return self.available

    def create_config(self) -> Dict[str, Any]:
        """创建DeepSpeed配置"""
        if not self.available:
            return {}

        train_config = self.config.get('training', {})
        batch_size = train_config.get('batch_size', 8)
        learning_rate = train_config.get('learning_rate', 1e-4)
        weight_decay = train_config.get('weight_decay', 0.01)
        epochs = train_config.get('epochs', 2000)

        # 计算总批次大小（所有GPU） - 修复分布式环境
        if 'WORLD_SIZE' in os.environ:
            world_size = int(os.environ['WORLD_SIZE'])
        else:
            world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
        global_batch_size = batch_size * world_size

        ds_config = {
            "train_batch_size": global_batch_size,
            "gradient_accumulation_steps": 1,

            # 优化器配置
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": learning_rate,
                    "weight_decay": weight_decay,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8
                }
            },

            # 学习率调度器 - 简化配置避免步数计算错误
            "scheduler": {
                "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": learning_rate,
                    "warmup_num_steps": 1000  # 固定warmup步数
                }
            },

            # ZeRO优化配置 - 减小bucket_size避免内存问题
            "zero_optimization": {
                "stage": self.opt_config.get('zero_stage', 2),
                "allgather_partitions": True,
                "allgather_bucket_size": 50000000,  # 减小到50MB
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 50000000,     # 减小到50MB
                "contiguous_gradients": True,
                "cpu_offload": self.opt_config.get('offload_optimizer', True)
            },

            # 参数卸载（ZeRO-3需要）
            "zero_allow_untested_optimizer": True,

            # 混合精度配置
            "fp16": {
                "enabled": self.opt_config.get('mixed_precision', True),
                "auto_cast": True,
                "loss_scale": 0,  # 动态loss scaling
                "initial_scale_power": 16,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1
            },

            # 梯度裁剪
            "gradient_clipping": 1.0,

            # 内存优化
            "zero_force_ds_cpu_optimizer": self.opt_config.get('offload_optimizer', True),

            # 调试和日志
            "steps_per_print": 100,
            "wall_clock_breakdown": False,
            "dump_state": False,

            # 稳定性配置 - 防止死锁和SIGKILL
            "communication_data_type": "fp16",
            "prescale_gradients": False,
            "gradient_predivide_factor": 1.0,
            "sparse_gradients": False,

            # 内存和性能优化
            "max_live_parameters": 1000000000,
            "max_reuse_distance": 1000000000
        }

        # ZeRO-3特殊配置
        if self.opt_config.get('zero_stage', 2) == 3:
            ds_config["zero_optimization"]["stage3_prefetch_bucket_size"] = 200000000
            ds_config["zero_optimization"]["stage3_param_persistence_threshold"] = 200000
            ds_config["zero_optimization"]["stage3_max_live_parameters"] = 1000000000
            ds_config["zero_optimization"]["stage3_max_reuse_distance"] = 1000000000

            if self.opt_config.get('offload_parameters', False):
                ds_config["zero_optimization"]["offload_param"] = {
                    "device": "cpu",
                    "pin_memory": True,
                    "buffer_count": 5,
                    "buffer_size": 100000000,
                    "max_in_cpu": 1000000000
                }

        return ds_config

    def get_strategy_description(self) -> str:
        """获取优化策略描述"""
        # 实时检查配置状态
        current_use_deepspeed = self.config.get('optimization', {}).get('use_deepspeed', False)

        if not self.available or not current_use_deepspeed:
            gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
            if gpu_count > 1:
                return "DataParallel多GPU训练"
            else:
                return "标准PyTorch训练"

        zero_stage = self.opt_config.get('zero_stage', 2)
        offload_optimizer = self.opt_config.get('offload_optimizer', True)
        offload_params = self.opt_config.get('offload_parameters', False)
        mixed_precision = self.opt_config.get('mixed_precision', True)

        desc_parts = [f"DeepSpeed ZeRO Stage {zero_stage}"]

        if offload_optimizer:
            desc_parts.append("优化器CPU卸载")
        if offload_params:
            desc_parts.append("参数CPU卸载")
        if mixed_precision:
            desc_parts.append("FP16混合精度")

        return " + ".join(desc_parts)


def setup_pytorch_optimization(config: Dict[str, Any]) -> Tuple[torch.device, bool]:
    """
    设置PyTorch优化配置

    Args:
        config: 配置字典

    Returns:
        (device, is_distributed)
    """
    opt_config = config.get('optimization', {})

    # 设备设置
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True  # 优化cudnn性能

        # GPU内存优化
        if opt_config.get('enable_memory_growth', True):
            # PyTorch默认就是按需分配，无需特殊设置
            pass

        if is_main_process():
            print(f"CUDA设备数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")

    else:
        device = torch.device('cpu')
        if is_main_process():
            print("未检测到CUDA设备，使用CPU训练")

    # 分布式训练设置
    is_distributed = False
    if torch.cuda.device_count() > 1:
        try:
            # 检查是否已经初始化了分布式
            if not dist.is_initialized():
                # 检查是否通过torchrun或launch启动
                if 'WORLD_SIZE' in os.environ and 'RANK' in os.environ:
                    # 已经通过分布式启动器启动，直接初始化
                    if is_main_process():
                        print(f"检测到分布式环境，初始化 {os.environ.get('WORLD_SIZE')} 块GPU")
                    try:
                        dist.init_process_group(backend='nccl')
                        is_distributed = True
                        if is_main_process():
                            print(f"分布式训练初始化完成: 使用 {dist.get_world_size()} 块GPU")
                    except Exception as e:
                        if is_main_process():
                            print(f"分布式初始化失败: {e}，回退到单GPU训练")
                        is_distributed = False
                else:
                    # 检查是否要使用DeepSpeed
                    opt_config = config.get('optimization', {})
                    use_deepspeed = opt_config.get('use_deepspeed', False) and DEEPSPEED_AVAILABLE

                    if use_deepspeed:
                        # DeepSpeed需要分布式环境，但单进程启动时禁用DeepSpeed
                        print(f"DeepSpeed需要分布式环境启动，当前为单进程模式")
                        print("自动禁用DeepSpeed，使用DataParallel进行多GPU训练")
                        print("如需使用DeepSpeed，请使用以下命令启动：")
                        print(f"   torchrun --nproc_per_node={torch.cuda.device_count()} main.py --config config/my_config.yaml")

                        # 在配置中临时禁用DeepSpeed
                        config['optimization']['use_deepspeed'] = False
                        is_distributed = False
                    else:
                        # 不使用DeepSpeed，使用DataParallel
                        print(f"检测到 {torch.cuda.device_count()} 块GPU，将使用DataParallel进行多GPU训练")
                        print("提示：如需更高效的分布式训练，请使用以下命令启动：")
                        print(f"   torchrun --nproc_per_node={torch.cuda.device_count()} main.py --config config/my_config.yaml")
                        is_distributed = False
            else:
                is_distributed = True
                if is_main_process():
                    print("分布式训练已初始化")
        except Exception as e:
            if is_main_process():
                print(f"分布式训练初始化失败: {e}")
            is_distributed = False

    # 混合精度设置
    if opt_config.get('mixed_precision', True) and device.type == 'cuda':
        # PyTorch AMP会自动处理
        if is_main_process():
            print("启用混合精度训练(AMP)")

    # 编译优化（PyTorch 2.0+）
    if hasattr(torch, 'compile') and opt_config.get('torch_compile', False):
        if is_main_process():
            print("启用torch.compile优化")

    return device, is_distributed


def calculate_optimal_batch_size(config: Dict[str, Any]) -> Tuple[int, int]:
    """
    计算最优批处理大小

    Args:
        config: 配置字典

    Returns:
        (per_gpu_batch_size, global_batch_size)
    """
    per_gpu_batch_size = config['training']['batch_size']
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

    global_batch_size = per_gpu_batch_size * num_gpus

    print(f"批处理大小配置:")
    print(f"   每GPU批次: {per_gpu_batch_size}")
    print(f"   GPU数量: {num_gpus}")
    print(f"   总批次大小: {global_batch_size}")

    return per_gpu_batch_size, global_batch_size


class MemoryMonitor:
    """GPU内存监控器"""

    def __init__(self, device: torch.device):
        self.device = device
        self.enabled = device.type == 'cuda'

    def log_memory_usage(self, stage: str = ""):
        """记录内存使用情况"""
        if not self.enabled:
            return

        allocated = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(self.device) / 1024**3   # GB
        max_allocated = torch.cuda.max_memory_allocated(self.device) / 1024**3

        if stage:
            print(f"{stage} GPU内存使用: 已分配={allocated:.2f}GB, 已保留={reserved:.2f}GB, 峰值分配={max_allocated:.2f}GB")
        else:
            print(f"GPU内存使用: 已分配={allocated:.2f}GB, 已保留={reserved:.2f}GB, 峰值分配={max_allocated:.2f}GB")

    def reset_peak_stats(self):
        """重置峰值统计"""
        if self.enabled:
            torch.cuda.reset_peak_memory_stats(self.device)


def create_optimization_setup(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    创建完整的优化设置

    Args:
        config: 配置字典

    Returns:
        优化设置字典
    """
    # 基础PyTorch优化
    device, is_distributed = setup_pytorch_optimization(config)

    # 批处理大小计算
    per_gpu_batch_size, global_batch_size = calculate_optimal_batch_size(config)

    # DeepSpeed优化器
    deepspeed_optimizer = DeepSpeedOptimizer(config)

    # 内存监控器
    memory_monitor = MemoryMonitor(device)

    setup_info = {
        'device': device,
        'is_distributed': is_distributed,
        'per_gpu_batch_size': per_gpu_batch_size,
        'global_batch_size': global_batch_size,
        'deepspeed_optimizer': deepspeed_optimizer,
        'memory_monitor': memory_monitor,
        'strategy_description': deepspeed_optimizer.get_strategy_description()
    }

    print(f"\n优化设置完成:")
    print(f"   策略: {setup_info['strategy_description']}")
    print(f"   设备: {device}")
    print(f"   分布式: {is_distributed}")

    return setup_info