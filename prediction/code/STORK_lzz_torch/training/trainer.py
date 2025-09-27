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
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

        # 混合精度设置
        self.use_amp = config.get('optimization', {}).get('mixed_precision', True)
        if self.use_amp and not self.use_deepspeed:
            self.scaler = GradScaler()

        print(f"🚀 训练器初始化完成:")
        print(f"   设备: {self.device}")
        print(f"   分布式训练: {self.is_distributed}")
        print(f"   DeepSpeed: {self.use_deepspeed}")
        print(f"   混合精度: {self.use_amp}")

    def _setup_deepspeed(self):
        """设置DeepSpeed训练"""
        # 创建DeepSpeed配置
        ds_config = self._create_deepspeed_config()

        # 初始化DeepSpeed引擎
        self.model_engine, self.optimizer, _, self.lr_scheduler = deepspeed.initialize(
            model=self.model,
            config=ds_config
        )
        print("✅ DeepSpeed引擎初始化完成")

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

        print("✅ 标准训练设置完成")

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

        if optimizer_name == 'adamw':
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=train_config.get('learning_rate', 1e-4),
                weight_decay=train_config.get('weight_decay', 0.01)
            )
        elif optimizer_name == 'adam':
            return torch.optim.Adam(
                self.model.parameters(),
                lr=train_config.get('learning_rate', 1e-4),
                weight_decay=train_config.get('weight_decay', 0.01)
            )
        elif optimizer_name == 'sgd':
            return torch.optim.SGD(
                self.model.parameters(),
                lr=train_config.get('learning_rate', 1e-4),
                momentum=0.9,
                weight_decay=train_config.get('weight_decay', 0.01)
            )
        else:
            raise ValueError(f"不支持的优化器: {optimizer_name}")

    def _create_scheduler(self):
        """创建学习率调度器"""
        train_config = self.config.get('training', {})
        scheduler_name = train_config.get('scheduler', 'cosine').lower()

        if scheduler_name == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=train_config.get('epochs', 2000)
            )
        elif scheduler_name == 'step':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=train_config.get('epochs', 2000) // 3,
                gamma=0.1
            )
        elif scheduler_name == 'exponential':
            return torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=0.95
            )
        else:
            return None

    def _create_loss_function(self) -> nn.Module:
        """创建损失函数"""
        train_config = self.config.get('training', {})
        loss_name = train_config.get('loss', 'focal').lower()

        if loss_name == 'cross_entropy':
            class_weights = train_config.get('class_weights', None)
            if class_weights:
                weight = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
                return nn.CrossEntropyLoss(weight=weight)
            else:
                return nn.CrossEntropyLoss()

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
        print(f"\n🎯 开始训练 {num_epochs} 个epochs")

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
                        print(f"🏆 新的最佳模型已保存! 验证准确率: {val_acc:.2f}%")
                else:
                    # 后备保存方式
                    model_name = self.config.get('model', {}).get('name', 'unknown')
                    checkpoint_filename = f'{model_name}_best_model.pth'
                    self.save_checkpoint(checkpoint_filename)
                    print(f"💾 最佳模型已保存到checkpoints/{checkpoint_filename} (准确率: {val_acc:.2f}%)")

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
            print("-" * 50)

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

        print(f"✅ 检查点加载完成: {filename}")


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