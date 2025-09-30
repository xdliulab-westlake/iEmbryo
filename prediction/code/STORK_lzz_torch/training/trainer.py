"""
训练器主类
统一管理训练过程，支持DeepSpeed优化和分布式训练
"""

import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, Any, Optional, Tuple
import numpy as np
from tqdm import tqdm

# 添加相对路径到sys.path以支持导入
import sys
from pathlib import Path
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

from utils.logger import MetricsMonitor

try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False


class STORKTrainer:
    """STORK胚胎分类训练器"""

    def __init__(self,
                 model: nn.Module,
                 config: Dict[str, Any],
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 device: Optional[torch.device] = None,
                 model_manager=None,
                 logger=None):
        """
        初始化训练器

        Args:
            model: 要训练的模型
            config: 训练配置
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器（可选）
            device: 设备
            model_manager: 模型管理器（可选）
            logger: 日志记录器（可选）
        """
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model_manager = model_manager
        self.logger = logger

        # 设备设置
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.is_distributed = dist.is_available() and dist.is_initialized()

        # 模型设置
        self.model = model.to(self.device)

        # DeepSpeed设置
        self.use_deepspeed = config.get('optimization', {}).get('use_deepspeed', False) and DEEPSPEED_AVAILABLE
        if self.use_deepspeed:
            self._setup_deepspeed()
        else:
            self._setup_standard_training()

        # 损失函数
        self.criterion = self._create_loss_function()

        # 训练状态
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.train_history = {'loss': [], 'acc': []}
        self.val_history = {'loss': [], 'acc': []}

        # 早停配置
        early_stopping_config = config.get('training', {}).get('early_stopping', {})
        self.early_stopping_enabled = early_stopping_config.get('enabled', False)
        if self.early_stopping_enabled:
            monitor_metric = early_stopping_config.get('monitor', 'val_loss')
            patience = early_stopping_config.get('patience', 10)
            min_delta = early_stopping_config.get('min_delta', 0.001)
            self.restore_best_weights = early_stopping_config.get('restore_best_weights', True)

            self.metrics_monitor = MetricsMonitor(
                monitor_metrics=[monitor_metric],
                patience=patience,
                min_delta=min_delta
            )

            # 保存最佳模型权重（用于恢复）
            self.best_model_weights = None

            print(f"早停策略已启用: 监控={monitor_metric}, 耐心值={patience}, 最小改进={min_delta}")

        # 混合精度设置
        self.use_amp = config.get('optimization', {}).get('mixed_precision', True)
        if self.use_amp and not self.use_deepspeed:
            self.scaler = GradScaler()

        print(f"训练器初始化完成: 设备={self.device}, 分布式={self.is_distributed}, DeepSpeed={self.use_deepspeed}, 混合精度={self.use_amp}")

    def _setup_deepspeed(self):
        """设置DeepSpeed训练"""
        # 创建DeepSpeed配置
        ds_config = self._create_deepspeed_config()

        # 初始化DeepSpeed引擎
        self.model_engine, self.optimizer, _, self.lr_scheduler = deepspeed.initialize(
            model=self.model,
            config=ds_config
        )

    def _setup_standard_training(self):
        """设置标准训练"""
        # 分布式训练
        if self.is_distributed:
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.device.index] if self.device.type == 'cuda' else None
            )

        # 优化器
        self.optimizer = self._create_optimizer()

        # 学习率调度器
        self.lr_scheduler = self._create_scheduler()


    def _create_deepspeed_config(self) -> Dict:
        """创建DeepSpeed配置"""
        opt_config = self.config.get('optimization', {})
        train_config = self.config.get('training', {})

        ds_config = {
            "train_batch_size": train_config.get('batch_size', 8) * torch.cuda.device_count(),
            "gradient_accumulation_steps": 1,
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": train_config.get('learning_rate', 1e-4),
                    "weight_decay": train_config.get('weight_decay', 0.01)
                }
            },
            "scheduler": {
                "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": train_config.get('learning_rate', 1e-4),
                    "warmup_num_steps": train_config.get('warmup_epochs', 10) * len(self.train_loader)
                }
            },
            "zero_optimization": {
                "stage": opt_config.get('zero_stage', 2),
                "offload_optimizer": {
                    "device": "cpu" if opt_config.get('offload_optimizer', True) else "none"
                },
                "offload_param": {
                    "device": "cpu" if opt_config.get('offload_parameters', False) else "none"
                },
                "contiguous_gradients": True,
                "overlap_comm": True
            },
            "fp16": {
                "enabled": opt_config.get('mixed_precision', True),
                "loss_scale": 0,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1
            },
            "gradient_clipping": 1.0,
            "wall_clock_breakdown": False
        }

        return ds_config

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """创建优化器"""
        train_config = self.config.get('training', {})
        optimizer_name = train_config.get('optimizer', 'adamw').lower()
        lr = train_config.get('learning_rate', 1e-4)
        weight_decay = train_config.get('weight_decay', 0.01)

        optimizers = {
            'adamw': lambda: torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay),
            'adam': lambda: torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay),
            'sgd': lambda: torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        }

        if optimizer_name not in optimizers:
            raise ValueError(f"不支持的优化器: {optimizer_name}")

        return optimizers[optimizer_name]()

    def _create_scheduler(self):
        """创建学习率调度器"""
        train_config = self.config.get('training', {})
        scheduler_name = train_config.get('scheduler', 'cosine').lower()
        epochs = train_config.get('epochs', 2000)

        schedulers = {
            'cosine': lambda: torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs),
            'step': lambda: torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=epochs // 3, gamma=0.1),
            'exponential': lambda: torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)
        }

        return schedulers.get(scheduler_name, lambda: None)()

    def _create_loss_function(self) -> nn.Module:
        """创建损失函数"""
        train_config = self.config.get('training', {})
        loss_name = train_config.get('loss', 'focal').lower()

        if loss_name == 'cross_entropy':
            class_weights = train_config.get('class_weights', None)
            weight = torch.tensor(class_weights, dtype=torch.float32).to(self.device) if class_weights else None

            # 检查是否使用Label Smoothing
            label_smoothing = train_config.get('label_smoothing', 0.0)
            if label_smoothing > 0:
                return LabelSmoothingCrossEntropy(
                    weight=weight,
                    smoothing=label_smoothing,
                    num_classes=2
                )
            else:
                return nn.CrossEntropyLoss(weight=weight)
        elif loss_name == 'focal':
            return FocalLoss(
                alpha=train_config.get('focal_alpha', 0.25),
                gamma=train_config.get('focal_gamma', 2.0),
                num_classes=2
            )
        else:
            raise ValueError(f"不支持的损失函数: {loss_name}")

    def train_epoch(self) -> Tuple[float, float]:
        """训练一个epoch"""
        if self.use_deepspeed:
            self.model_engine.train()
        else:
            self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)

            if self.use_deepspeed:
                # DeepSpeed训练步骤
                outputs = self.model_engine(data)
                loss = self.criterion(outputs, target)
                self.model_engine.backward(loss)
                self.model_engine.step()
            else:
                # 标准训练步骤
                self.optimizer.zero_grad()

                if self.use_amp:
                    with autocast():
                        outputs = self.model(data)
                        loss = self.criterion(outputs, target)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(data)
                    loss = self.criterion(outputs, target)
                    loss.backward()
                    self.optimizer.step()

            # 统计
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            # 更新进度条
            acc = 100. * correct / total
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{acc:.2f}%'
            })

        epoch_loss = total_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def validate(self) -> Tuple[float, float]:
        """验证模型"""
        if self.val_loader is None:
            return 0.0, 0.0

        if self.use_deepspeed:
            self.model_engine.eval()
            model = self.model_engine.module
        else:
            self.model.eval()
            model = self.model

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc="Validation"):
                data, target = data.to(self.device), target.to(self.device)

                if self.use_amp:
                    with autocast():
                        outputs = model(data)
                        loss = self.criterion(outputs, target)
                else:
                    outputs = model(data)
                    loss = self.criterion(outputs, target)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        val_loss = total_loss / len(self.val_loader)
        val_acc = 100. * correct / total

        return val_loss, val_acc

    def train(self, num_epochs: int) -> Dict[str, list]:
        """完整训练过程"""
        print(f"开始训练 {num_epochs} 个epochs")

        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1
            start_time = time.time()

            # 训练
            train_loss, train_acc = self.train_epoch()
            self.train_history['loss'].append(train_loss)
            self.train_history['acc'].append(train_acc)

            # 验证
            val_loss, val_acc = self.validate()
            if self.val_loader is not None:
                self.val_history['loss'].append(val_loss)
                self.val_history['acc'].append(val_acc)

            # 早停检查
            should_stop = False
            if self.early_stopping_enabled and self.val_loader is not None:
                # 准备监控指标
                metrics_dict = {
                    'val_loss': val_loss,
                    'val_accuracy': val_acc / 100.0,  # 转换为0-1范围
                    'val_acc': val_acc / 100.0
                }

                # 更新监控器
                improvements = self.metrics_monitor.update(metrics_dict)

                # 如果有改进，保存模型权重用于恢复
                monitor_metric = list(self.metrics_monitor.monitor_metrics)[0]
                if improvements.get(monitor_metric, False) and self.restore_best_weights:
                    # 保存最佳模型权重
                    if self.use_deepspeed:
                        self.best_model_weights = self.model_engine.module.state_dict().copy()
                    elif hasattr(self.model, 'module'):  # DataParallel
                        self.best_model_weights = self.model.module.state_dict().copy()
                    else:
                        self.best_model_weights = self.model.state_dict().copy()

                    if self.logger:
                        current_value = metrics_dict.get(monitor_metric, 0)
                        self.logger.log_best_model(epoch, monitor_metric, current_value)

                # 检查早停
                should_stop = self.metrics_monitor.check_early_stopping()
                if should_stop:
                    if self.logger:
                        self.logger.log_early_stopping(epoch, self.metrics_monitor.patience)

                    # 恢复最佳权重
                    if self.restore_best_weights and self.best_model_weights is not None:
                        if self.use_deepspeed:
                            self.model_engine.module.load_state_dict(self.best_model_weights)
                        elif hasattr(self.model, 'module'):
                            self.model.module.load_state_dict(self.best_model_weights)
                        else:
                            self.model.load_state_dict(self.best_model_weights)
                        print("已恢复最佳模型权重")

            # 学习率调度
            if self.lr_scheduler and not self.use_deepspeed:
                self.lr_scheduler.step()

            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc

                # 使用ModelManager保存最佳模型（如果可用）
                if self.model_manager:
                    metrics = {
                        'val_accuracy': val_acc / 100.0,  # 转换为0-1范围
                        'val_loss': val_loss,
                        'train_accuracy': train_acc / 100.0,  # 转换为0-1范围
                        'train_loss': train_loss,
                        'epoch': self.current_epoch
                    }

                    # 获取要保存的模型
                    model_to_save = self.model
                    if self.use_deepspeed:
                        model_to_save = self.model_engine.module
                    elif hasattr(self.model, 'module'):  # DataParallel情况
                        model_to_save = self.model.module

                    is_best = self.model_manager.save_best_model(
                        model=model_to_save,
                        metrics=metrics,
                        metric_name='val_accuracy',
                        mode='max',
                        config=self.config
                    )

                    if is_best:
                        print(f"新的最佳模型已保存! 验证准确率: {val_acc:.2f}%")
                else:
                    # 后备保存方式
                    model_name = self.config.get('model', {}).get('name', 'unknown')
                    checkpoint_filename = f'{model_name}_best_model.pth'
                    self.save_checkpoint(checkpoint_filename)
                    print(f"最佳模型已保存到checkpoints/{checkpoint_filename} (准确率: {val_acc:.2f}%)")

            # 使用ModelManager定期保存检查点
            if self.model_manager and (epoch % self.config.get('logging', {}).get('save_model_every_n_epochs', 50) == 0):
                metrics = {
                    'val_accuracy': val_acc / 100.0,
                    'val_loss': val_loss,
                    'train_accuracy': train_acc / 100.0,
                    'train_loss': train_loss,
                    'epoch': self.current_epoch
                }

                # 获取要保存的模型
                model_to_save = self.model
                if self.use_deepspeed:
                    model_to_save = self.model_engine.module
                elif hasattr(self.model, 'module'):
                    model_to_save = self.model.module

                checkpoint_path = self.model_manager.save_checkpoint(
                    model=model_to_save,
                    optimizer=self.optimizer if not self.use_deepspeed else None,
                    epoch=self.current_epoch,
                    metrics=metrics,
                    config=self.config,
                    scheduler=self.lr_scheduler if hasattr(self, 'lr_scheduler') else None
                )
            elif epoch % 50 == 0:
                # 后备保存方式
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')

            epoch_time = time.time() - start_time

            # 打印训练信息
            print(f"Epoch {self.current_epoch}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            if self.val_loader is not None:
                print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"  Time: {epoch_time:.2f}s")

            # 早停状态提示
            if self.early_stopping_enabled and hasattr(self, 'metrics_monitor'):
                status = self.metrics_monitor.get_status()
                monitor_metric = list(self.metrics_monitor.monitor_metrics)[0]
                wait_count = status['wait_counts'].get(monitor_metric, 0)
                best_value = status['best_values'].get(monitor_metric, 0)
                print(f"  早停监控: {monitor_metric}最佳值={best_value:.4f}, 等待轮数={wait_count}/{self.metrics_monitor.patience}")

            print("-" * 50)

            # 早停检查
            if should_stop:
                print(f"\n🛑 早停触发! 在第 {self.current_epoch} 轮停止训练")
                print(f"连续 {self.metrics_monitor.patience} 轮无改进")
                break

        return {
            'train_history': self.train_history,
            'val_history': self.val_history
        }

    def save_checkpoint(self, filename: str):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict() if not self.use_deepspeed else self.model_engine.module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if not self.use_deepspeed else None,
            'best_val_acc': self.best_val_acc,
            'train_history': self.train_history,
            'val_history': self.val_history,
            'config': self.config
        }

        os.makedirs('checkpoints', exist_ok=True)
        torch.save(checkpoint, os.path.join('checkpoints', filename))

    def load_checkpoint(self, filename: str):
        """加载检查点"""
        checkpoint = torch.load(filename, map_location=self.device)

        if self.use_deepspeed:
            self.model_engine.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict']:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        self.train_history = checkpoint.get('train_history', {'loss': [], 'acc': []})
        self.val_history = checkpoint.get('val_history', {'loss': [], 'acc': []})

        print(f"检查点加载完成: {filename}")


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, num_classes: int = 2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class LabelSmoothingCrossEntropy(nn.Module):
    """Label Smoothing Cross Entropy Loss - 防止过度自信"""

    def __init__(self, weight=None, smoothing=0.1, num_classes=2):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.weight = weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 创建平滑标签
        smooth_targets = self._smooth_labels(targets, inputs.device)

        # 计算log softmax
        log_probs = torch.log_softmax(inputs, dim=1)

        # 计算损失
        if self.weight is not None:
            # 应用类别权重
            nll_loss = -torch.sum(smooth_targets * log_probs, dim=1)
            # 根据真实标签应用权重
            weighted_loss = nll_loss * self.weight[targets]
            return weighted_loss.mean()
        else:
            # 不使用权重
            loss = -torch.sum(smooth_targets * log_probs, dim=1)
            return loss.mean()

    def _smooth_labels(self, targets: torch.Tensor, device: torch.device) -> torch.Tensor:
        """创建平滑标签

        例如：ε=0.1时
        [0, 1] -> [0.05, 0.95]
        [1, 0] -> [0.95, 0.05]
        """
        batch_size = targets.size(0)
        smooth_targets = torch.zeros(batch_size, self.num_classes, device=device)

        # 填充基础平滑值 ε/K
        smooth_targets.fill_(self.smoothing / self.num_classes)

        # 为真实标签位置设置 1-ε+ε/K
        smooth_targets.scatter_(1, targets.unsqueeze(1),
                              1.0 - self.smoothing + self.smoothing / self.num_classes)

        return smooth_targets