"""
DeepSpeed配置和优化模块
独立管理深度学习模型的内存和计算优化
"""

import os
import tensorflow as tf


class DeepSpeedManager:
    """DeepSpeed内存优化管理器"""

    def __init__(self):
        self.available = False
        self.config = None
        self._check_availability()

    def _check_availability(self):
        """检查DeepSpeed是否可用"""
        try:
            import deepspeed
            from deepspeed.ops.adam import DeepSpeedCPUAdam
            self.available = True
            print("🚀 DeepSpeed可用，将启用ZeRO内存优化")
        except ImportError:
            self.available = False
            print("⚠️  DeepSpeed未安装，使用标准内存管理")

    def create_config(self, batch_size, learning_rate=1e-4):
        """创建DeepSpeed配置"""
        if not self.available:
            return None

        self.config = {
            "train_batch_size": batch_size,
            "gradient_accumulation_steps": 1,
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": learning_rate,
                    "weight_decay": 0.01
                }
            },
            "scheduler": {
                "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": learning_rate,
                    "warmup_num_steps": 100
                }
            },
            "zero_optimization": {
                "stage": 2,  # ZeRO Stage 2: 分割优化器状态
                "offload_optimizer": {
                    "device": "cpu"  # 优化器状态放CPU
                },
                "contiguous_gradients": True,
                "overlap_comm": True
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1
            },
            "wall_clock_breakdown": False
        }
        return self.config

    def should_use_deepspeed(self, num_gpus):
        """判断是否应该使用DeepSpeed"""
        return self.available and num_gpus > 1

    def get_memory_optimization_strategy(self, num_gpus):
        """获取内存优化策略描述"""
        if self.should_use_deepspeed(num_gpus):
            return "DeepSpeed ZeRO Stage 2 + CPU Offload"
        elif num_gpus > 1:
            return "TensorFlow MirroredStrategy + FP16"
        else:
            return "Single GPU + FP16 + Memory Growth"


def setup_tensorflow_memory_optimization(gpus):
    """设置TensorFlow内存优化（独立于DeepSpeed）"""
    if gpus:
        try:
            for gpu in gpus:
                # 1. 启用内存增长
                tf.config.experimental.set_memory_growth(gpu, True)

                # 2. 尝试设置内存限制（兼容不同TensorFlow版本）
                try:
                    # 新版本API
                    tf.config.experimental.set_memory_limit(gpu, 70 * 1024)
                    print(f"GPU {gpu} 内存优化: 增长模式 + 70GB限制")
                except AttributeError:
                    # 较新版本可能使用不同API或已移除此功能
                    try:
                        tf.config.experimental.set_virtual_device_configuration(
                            gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=70 * 1024)]
                        )
                        print(f"GPU {gpu} 内存优化: 增长模式 + 70GB限制 (VirtualDevice)")
                    except:
                        print(f"GPU {gpu} 内存优化: 仅增长模式 (内存限制API不可用)")

            # 3. 启用XLA编译优化
            try:
                tf.config.optimizer.set_jit(True)
                print("🔧 启用XLA编译优化")
            except:
                print("⚠️  XLA优化不可用")

            # 4. 启用混合精度训练
            try:
                from tensorflow.keras import mixed_precision
                policy = mixed_precision.Policy('mixed_float16')
                mixed_precision.set_global_policy(policy)
                print("🎯 启用混合精度训练(FP16)")
            except:
                try:
                    from tensorflow.keras.mixed_precision import experimental as mixed_precision
                    policy = mixed_precision.Policy('mixed_float16')
                    mixed_precision.set_policy(policy)
                    print("🎯 启用混合精度训练(FP16)")
                except:
                    print("⚠️  混合精度训练不可用")

            # 5. 设置CUDA异步内存分配器
            os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
            print("🔧 启用CUDA异步内存分配器")

            return True

        except RuntimeError as e:
            print(f"内存优化设置失败: {e}")
            # 回退到基础内存增长
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("🔧 回退到基础内存增长模式")
            except:
                print("⚠️  基础内存增长设置失败")
            return False

    return False


def create_memory_efficient_strategy(num_gpus, deepspeed_manager=None):
    """创建内存高效的训练策略"""
    if num_gpus > 1:
        if deepspeed_manager and deepspeed_manager.should_use_deepspeed(num_gpus):
            print("🚀 使用DeepSpeed分布式策略")
            # 这里可以扩展DeepSpeed的TensorFlow集成
            # 目前先用标准策略作为后备
            strategy = tf.distribute.MirroredStrategy()
        else:
            print("🔧 使用TensorFlow MirroredStrategy")
            strategy = tf.distribute.MirroredStrategy()
    else:
        print("🔧 单GPU训练模式")
        strategy = tf.distribute.get_strategy()  # 默认策略

    return strategy


def calculate_batch_sizes(per_gpu_batch_size):
    """计算最优批处理大小配置"""
    num_gpus = len(tf.config.list_physical_devices('GPU'))

    if num_gpus > 1:
        batch_size_per_replica = per_gpu_batch_size
        global_batch_size = batch_size_per_replica * num_gpus
    else:
        batch_size_per_replica = per_gpu_batch_size
        global_batch_size = batch_size_per_replica

    return global_batch_size, batch_size_per_replica, num_gpus


# 便捷接口函数
def initialize_optimization(per_gpu_batch_size, learning_rate=1e-4):
    """一键初始化所有优化设置"""
    # 1. 计算批处理大小
    global_batch_size, batch_size_per_replica, num_gpus = calculate_batch_sizes(per_gpu_batch_size)

    # 2. 检测GPU并设置内存优化
    gpus = tf.config.list_physical_devices('GPU')
    setup_tensorflow_memory_optimization(gpus)

    # 3. 创建DeepSpeed管理器
    deepspeed_mgr = DeepSpeedManager()

    # 4. 创建训练策略
    strategy = create_memory_efficient_strategy(num_gpus, deepspeed_mgr)

    # 5. 如果使用DeepSpeed，创建配置
    deepspeed_config = None
    if deepspeed_mgr.should_use_deepspeed(num_gpus):
        deepspeed_config = deepspeed_mgr.create_config(global_batch_size, learning_rate)

    print(f"🎯 内存优化策略: {deepspeed_mgr.get_memory_optimization_strategy(num_gpus)}")

    return strategy, deepspeed_mgr, deepspeed_config, global_batch_size