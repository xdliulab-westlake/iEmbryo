"""
模型保存和加载管理器
支持模型检查点、最佳模型保存、模型恢复等功能
"""

import os
import json
import shutil
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union

import torch
import torch.nn as nn

try:
    import torch.onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


class ModelManager:
    """模型管理器"""

    def __init__(self,
                 model_dir: Union[str, Path],
                 model_name: str = "stork_model"):
        """
        初始化模型管理器

        Args:
            model_dir: 模型保存目录
            model_name: 模型名称前缀
        """
        self.model_dir = Path(model_dir).resolve()
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name

        # 创建子目录
        self.checkpoints_dir = self.model_dir / "checkpoints"
        self.best_models_dir = self.model_dir / "best_models"
        self.exports_dir = self.model_dir / "exports"

        for dir_path in [self.checkpoints_dir, self.best_models_dir, self.exports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # 最佳模型跟踪
        self.best_metrics = {}
        self.metrics_history = []

    def save_checkpoint(self,
                       model: nn.Module,
                       optimizer: torch.optim.Optimizer,
                       epoch: int,
                       metrics: Dict[str, float],
                       config: Optional[Dict[str, Any]] = None,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> str:
        """
        保存训练检查点

        Args:
            model: 模型
            optimizer: 优化器
            epoch: 当前轮数
            metrics: 评估指标
            config: 训练配置
            scheduler: 学习率调度器

        Returns:
            保存路径
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': config,
            'timestamp': datetime.now().isoformat(),
        }

        # 添加调度器状态
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        # 保存路径
        checkpoint_path = self.checkpoints_dir / f"{self.model_name}_epoch_{epoch:04d}.pth"

        torch.save(checkpoint, checkpoint_path)

        print(f"检查点已保存: {checkpoint_path}")
        return str(checkpoint_path)

    def save_best_model(self,
                       model: nn.Module,
                       metrics: Dict[str, float],
                       metric_name: str = 'val_accuracy',
                       mode: str = 'max',
                       config: Optional[Dict[str, Any]] = None) -> bool:
        """
        保存最佳模型

        Args:
            model: 模型
            metrics: 当前指标
            metric_name: 监控的指标名称
            mode: 'max' 或 'min'
            config: 配置信息

        Returns:
            是否为最佳模型
        """
        current_metric = metrics.get(metric_name)
        if current_metric is None:
            warnings.warn(f"指标 '{metric_name}' 不存在于 metrics 中")
            return False

        # 检查是否为最佳
        is_best = False
        if metric_name not in self.best_metrics:
            is_best = True
        else:
            best_value = self.best_metrics[metric_name]['value']
            if mode == 'max' and current_metric > best_value:
                is_best = True
            elif mode == 'min' and current_metric < best_value:
                is_best = True

        if is_best:
            # 更新最佳指标
            self.best_metrics[metric_name] = {
                'value': current_metric,
                'epoch': metrics.get('epoch', 0),
                'timestamp': datetime.now().isoformat(),
                'all_metrics': metrics.copy()
            }

            # 只保存完整模型（包含结构、权重和元信息），用于直接加载
            best_model_path = self.best_models_dir / f"{self.model_name}_best_{metric_name}.pth"

            # 设置模型为评估模式
            model.eval()

            # 在模型对象上附加元信息
            model.training_metrics = metrics
            model.best_metric = {
                'name': metric_name,
                'value': current_metric,
                'mode': mode
            }
            model.training_config = config
            model.save_timestamp = datetime.now().isoformat()

            torch.save(model, best_model_path)
            print(f"完整模型已保存: {best_model_path}")
            print(f"   {metric_name}: {current_metric:.4f}")
            print(f"   包含: 模型结构 + 权重 + 训练信息")

            # 保存独立的元数据文件（可选，用于快速查看信息）
            metadata = {
                'metrics': metrics,
                'best_metric': {
                    'name': metric_name,
                    'value': current_metric,
                    'mode': mode
                },
                'config': config,
                'timestamp': datetime.now().isoformat(),
                'model_path': str(best_model_path)
            }

            metadata_path = self.best_models_dir / f"{self.model_name}_best_{metric_name}_info.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self._make_serializable(metadata), f, indent=2, ensure_ascii=False)

        return is_best

    def load_checkpoint(self,
                       model: nn.Module,
                       checkpoint_path: Union[str, Path],
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                       device: Optional[torch.device] = None) -> Dict[str, Any]:
        """
        加载检查点

        Args:
            model: 模型
            checkpoint_path: 检查点路径
            optimizer: 优化器（可选）
            scheduler: 调度器（可选）
            device: 设备（可选）

        Returns:
            检查点信息
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")

        print(f"加载检查点: {checkpoint_path}")

        # 加载检查点
        if device is not None:
            checkpoint = torch.load(checkpoint_path, map_location=device)
        else:
            checkpoint = torch.load(checkpoint_path)

        # 加载模型状态
        model.load_state_dict(checkpoint['model_state_dict'])

        # 加载优化器状态
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # 加载调度器状态
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"检查点加载完成")
        print(f"   轮数: {checkpoint.get('epoch', 'Unknown')}")
        print(f"   指标: {checkpoint.get('metrics', {})}")

        return checkpoint

    def load_best_model(self,
                       model: nn.Module,
                       metric_name: str = 'val_accuracy',
                       device: Optional[torch.device] = None) -> Dict[str, Any]:
        """
        加载最佳模型

        Args:
            model: 模型
            metric_name: 指标名称
            device: 设备

        Returns:
            模型信息
        """
        best_model_path = self.best_models_dir / f"{self.model_name}_best_{metric_name}.pth"

        if not best_model_path.exists():
            raise FileNotFoundError(f"最佳模型不存在: {best_model_path}")

        print(f"加载最佳模型: {best_model_path}")

        # 加载模型
        if device is not None:
            model_data = torch.load(best_model_path, map_location=device)
        else:
            model_data = torch.load(best_model_path)

        model.load_state_dict(model_data['model_state_dict'])

        print(f"最佳模型加载完成")
        print(f"   {metric_name}: {model_data['best_metric']['value']:.4f}")

        return model_data

    def export_model(self,
                    model: nn.Module,
                    input_shape: tuple,
                    export_format: str = 'onnx',
                    model_name: Optional[str] = None) -> str:
        """
        导出模型

        Args:
            model: 模型
            input_shape: 输入形状 (C, H, W)
            export_format: 导出格式 ('onnx', 'torchscript', 'traced')
            model_name: 模型名称

        Returns:
            导出文件路径
        """
        if model_name is None:
            model_name = self.model_name

        model.eval()

        if export_format.lower() == 'onnx':
            return self._export_onnx(model, input_shape, model_name)
        elif export_format.lower() == 'torchscript':
            return self._export_torchscript(model, input_shape, model_name)
        elif export_format.lower() == 'traced':
            return self._export_traced(model, input_shape, model_name)
        else:
            raise ValueError(f"不支持的导出格式: {export_format}")

    def _export_onnx(self, model: nn.Module, input_shape: tuple, model_name: str) -> str:
        """导出ONNX格式"""
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX导出需要安装onnx库: pip install onnx")

        export_path = self.exports_dir / f"{model_name}.onnx"

        # 创建示例输入，确保与模型在同一设备
        device = next(model.parameters()).device
        dummy_input = torch.randn(1, *input_shape, device=device)

        torch.onnx.export(
            model,
            dummy_input,
            export_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )

        print(f"ONNX模型已导出: {export_path}")
        return str(export_path)

    def _export_torchscript(self, model: nn.Module, input_shape: tuple, model_name: str) -> str:
        """导出TorchScript格式"""
        export_path = self.exports_dir / f"{model_name}_scripted.pt"

        # 使用脚本化导出
        scripted_model = torch.jit.script(model)
        scripted_model.save(export_path)

        print(f"TorchScript模型已导出: {export_path}")
        return str(export_path)

    def _export_traced(self, model: nn.Module, input_shape: tuple, model_name: str) -> str:
        """导出Traced格式"""
        export_path = self.exports_dir / f"{model_name}_traced.pt"

        # 创建示例输入，确保与模型在同一设备
        device = next(model.parameters()).device
        dummy_input = torch.randn(1, *input_shape, device=device)

        # 使用追踪导出
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save(export_path)

        print(f"Traced模型已导出: {export_path}")
        return str(export_path)

    def get_latest_checkpoint(self) -> Optional[str]:
        """获取最新的检查点路径"""
        checkpoints = list(self.checkpoints_dir.glob(f"{self.model_name}_epoch_*.pth"))
        if not checkpoints:
            return None

        # 按修改时间排序
        latest = max(checkpoints, key=lambda x: x.stat().st_mtime)
        return str(latest)

    def clean_old_checkpoints(self, keep_last_n: int = 5):
        """清理旧的检查点，只保留最新的N个"""
        checkpoints = list(self.checkpoints_dir.glob(f"{self.model_name}_epoch_*.pth"))
        if len(checkpoints) <= keep_last_n:
            return

        # 按修改时间排序
        checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        # 删除多余的检查点
        for checkpoint in checkpoints[keep_last_n:]:
            checkpoint.unlink()
            print(f"删除旧检查点: {checkpoint.name}")

    def get_model_summary(self) -> Dict[str, Any]:
        """获取模型目录摘要"""
        summary = {
            'model_dir': str(self.model_dir),
            'model_name': self.model_name,
            'checkpoints': [],
            'best_models': [],
            'exports': [],
            'best_metrics': self.best_metrics
        }

        # 检查点信息
        for checkpoint in self.checkpoints_dir.glob(f"{self.model_name}_epoch_*.pth"):
            try:
                data = torch.load(checkpoint, map_location='cpu')
                summary['checkpoints'].append({
                    'path': str(checkpoint),
                    'epoch': data.get('epoch', 'Unknown'),
                    'metrics': data.get('metrics', {}),
                    'timestamp': data.get('timestamp', 'Unknown'),
                    'size_mb': checkpoint.stat().st_size / (1024 * 1024)
                })
            except Exception as e:
                print(f"无法读取检查点 {checkpoint}: {e}")

        # 最佳模型信息
        for best_model in self.best_models_dir.glob(f"{self.model_name}_best_*.pth"):
            try:
                data = torch.load(best_model, map_location='cpu')
                # 检查是否是完整模型对象（有属性）还是字典格式
                if hasattr(data, 'best_metric'):
                    # 新格式：完整模型对象，从属性获取信息
                    metric_name = getattr(data, 'best_metric', {}).get('name', 'Unknown') if hasattr(data, 'best_metric') else 'Unknown'
                    metric_value = getattr(data, 'best_metric', {}).get('value', 'Unknown') if hasattr(data, 'best_metric') else 'Unknown'
                    timestamp = getattr(data, 'save_timestamp', 'Unknown')
                else:
                    # 旧格式：字典格式，从字典获取信息
                    metric_name = data.get('best_metric', {}).get('name', 'Unknown')
                    metric_value = data.get('best_metric', {}).get('value', 'Unknown')
                    timestamp = data.get('timestamp', 'Unknown')

                summary['best_models'].append({
                    'path': str(best_model),
                    'metric_name': metric_name,
                    'metric_value': metric_value,
                    'timestamp': timestamp,
                    'size_mb': best_model.stat().st_size / (1024 * 1024)
                })
            except Exception as e:
                print(f"无法读取最佳模型 {best_model}: {e}")

        # 导出模型信息
        for export in self.exports_dir.glob(f"{self.model_name}*"):
            summary['exports'].append({
                'path': str(export),
                'format': export.suffix,
                'size_mb': export.stat().st_size / (1024 * 1024)
            })

        return summary

    def _make_serializable(self, obj):
        """将对象转换为JSON可序列化格式"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (torch.Tensor,)):
            return "tensor_removed"  # 移除张量
        elif hasattr(obj, '__dict__'):
            return str(obj)  # 复杂对象转为字符串
        else:
            return obj


def create_model_manager(model_dir: Union[str, Path],
                        model_name: str = "stork_model") -> ModelManager:
    """
    创建模型管理器的便捷函数

    Args:
        model_dir: 模型保存目录
        model_name: 模型名称

    Returns:
        ModelManager实例
    """
    return ModelManager(model_dir, model_name)