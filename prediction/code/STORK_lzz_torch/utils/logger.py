"""
日志和监控模块
提供训练过程的日志记录和监控功能
"""

import logging
import sys
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict, deque


class TrainingLogger:
    """训练日志记录器"""

    def __init__(self,
                 log_dir: Union[str, Path],
                 model_name: str = "stork",
                 log_level: str = "INFO",
                 console_output: bool = True,
                 file_output: bool = True):
        """
        初始化训练日志记录器

        Args:
            log_dir: 日志目录
            model_name: 模型名称
            log_level: 日志级别
            console_output: 是否输出到控制台
            file_output: 是否输出到文件
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name

        # 设置日志记录器
        self.logger = logging.getLogger(f"STORK_{model_name}")
        self.logger.setLevel(getattr(logging, log_level.upper()))

        # 清除现有处理器
        self.logger.handlers.clear()

        # 创建格式器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # 控制台输出
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # 文件输出
        if file_output:
            log_file = self.log_dir / f"{model_name}_training.log"
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        # 训练指标历史
        self.metrics_history = defaultdict(list)
        self.epoch_times = []
        self.start_time = None

    def info(self, message: str):
        """记录信息"""
        self.logger.info(message)

    def warning(self, message: str):
        """记录警告"""
        self.logger.warning(message)

    def error(self, message: str):
        """记录错误"""
        self.logger.error(message)

    def debug(self, message: str):
        """记录调试信息"""
        self.logger.debug(message)

    def log_training_start(self, config: Dict[str, Any]):
        """记录训练开始"""
        self.start_time = time.time()
        self.info("🚀 开始训练")
        self.info(f"📋 训练配置:")
        for key, value in config.items():
            self.info(f"   {key}: {value}")

    def log_epoch_start(self, epoch: int, total_epochs: int):
        """记录轮次开始"""
        self.epoch_start_time = time.time()
        self.info(f"📊 开始第 {epoch + 1}/{total_epochs} 轮训练")

    def log_epoch_end(self, epoch: int, metrics: Dict[str, float]):
        """记录轮次结束"""
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)

        # 记录指标
        for key, value in metrics.items():
            self.metrics_history[key].append(value)

        # 格式化指标信息
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.info(f"✅ 第 {epoch + 1} 轮完成 (用时: {epoch_time:.1f}s) | {metrics_str}")

    def log_training_end(self, final_metrics: Dict[str, float]):
        """记录训练结束"""
        total_time = time.time() - self.start_time
        avg_epoch_time = np.mean(self.epoch_times) if self.epoch_times else 0

        self.info("🎉 训练完成")
        self.info(f"⏱️  总用时: {total_time:.1f}s ({total_time/3600:.2f}h)")
        self.info(f"⏱️  平均每轮: {avg_epoch_time:.1f}s")
        self.info("📊 最终指标:")
        for key, value in final_metrics.items():
            self.info(f"   {key}: {value:.4f}")

        # 保存训练历史
        self._save_training_history()

    def log_best_model(self, epoch: int, metric_name: str, metric_value: float):
        """记录最佳模型"""
        self.info(f"🏆 发现最佳模型! 第 {epoch + 1} 轮, {metric_name}: {metric_value:.4f}")

    def log_early_stopping(self, epoch: int, patience: int):
        """记录早停"""
        self.warning(f"⏹️  早停触发! 第 {epoch + 1} 轮, 耐心值: {patience}")

    def log_learning_rate(self, lr: float):
        """记录学习率变化"""
        self.debug(f"📈 学习率: {lr:.6f}")

    def log_memory_usage(self, gpu_memory_mb: float, cpu_memory_mb: float):
        """记录内存使用"""
        self.debug(f"💾 内存使用 - GPU: {gpu_memory_mb:.1f}MB, CPU: {cpu_memory_mb:.1f}MB")

    def _save_training_history(self):
        """保存训练历史"""
        history_file = self.log_dir / f"{self.model_name}_training_history.json"

        history_data = {
            'metrics_history': dict(self.metrics_history),
            'epoch_times': self.epoch_times,
            'total_epochs': len(self.epoch_times),
            'total_time': sum(self.epoch_times),
            'average_epoch_time': np.mean(self.epoch_times) if self.epoch_times else 0,
            'timestamp': datetime.now().isoformat()
        }

        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False)

        self.info(f"📊 训练历史已保存: {history_file}")


class MetricsMonitor:
    """指标监控器"""

    def __init__(self,
                 monitor_metrics: List[str] = None,
                 patience: int = 10,
                 min_delta: float = 0.001):
        """
        初始化指标监控器

        Args:
            monitor_metrics: 要监控的指标列表
            patience: 早停耐心值
            min_delta: 最小改进阈值
        """
        self.monitor_metrics = monitor_metrics or ['val_loss', 'val_accuracy']
        self.patience = patience
        self.min_delta = min_delta

        # 监控状态
        self.best_values = {}
        self.wait_counts = {}
        self.early_stop = False

        for metric in self.monitor_metrics:
            self.best_values[metric] = None
            self.wait_counts[metric] = 0

    def update(self, metrics: Dict[str, float]) -> Dict[str, bool]:
        """
        更新监控指标

        Args:
            metrics: 当前指标值

        Returns:
            改进状态字典
        """
        improvements = {}

        for metric in self.monitor_metrics:
            if metric not in metrics:
                continue

            current_value = metrics[metric]
            is_loss = 'loss' in metric.lower()

            if self.best_values[metric] is None:
                # 第一次记录
                self.best_values[metric] = current_value
                improvements[metric] = True
                self.wait_counts[metric] = 0
            else:
                best_value = self.best_values[metric]

                # 检查是否有改进
                if is_loss:
                    # 损失函数：越小越好
                    improved = current_value < (best_value - self.min_delta)
                else:
                    # 准确率等：越大越好
                    improved = current_value > (best_value + self.min_delta)

                if improved:
                    self.best_values[metric] = current_value
                    self.wait_counts[metric] = 0
                    improvements[metric] = True
                else:
                    self.wait_counts[metric] += 1
                    improvements[metric] = False

        return improvements

    def check_early_stopping(self) -> bool:
        """检查是否应该早停"""
        # 如果任何一个监控指标超过耐心值，则早停
        for metric in self.monitor_metrics:
            if self.wait_counts[metric] >= self.patience:
                self.early_stop = True
                return True
        return False

    def get_status(self) -> Dict[str, Any]:
        """获取监控状态"""
        return {
            'best_values': self.best_values.copy(),
            'wait_counts': self.wait_counts.copy(),
            'early_stop': self.early_stop
        }


class TensorBoardLogger:
    """TensorBoard日志记录器"""

    def __init__(self, log_dir: Union[str, Path], enabled: bool = True):
        """
        初始化TensorBoard记录器

        Args:
            log_dir: 日志目录
            enabled: 是否启用
        """
        self.enabled = enabled
        self.writer = None

        if enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir)
                print(f"📊 TensorBoard已启用: {log_dir}")
            except ImportError:
                print("⚠️  TensorBoard未安装，跳过TensorBoard日志")
                self.enabled = False

    def log_scalar(self, tag: str, value: float, step: int):
        """记录标量值"""
        if self.enabled and self.writer:
            self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        """记录多个标量值"""
        if self.enabled and self.writer:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_histogram(self, tag: str, values: np.ndarray, step: int):
        """记录直方图"""
        if self.enabled and self.writer:
            self.writer.add_histogram(tag, values, step)

    def log_image(self, tag: str, img_tensor, step: int):
        """记录图像"""
        if self.enabled and self.writer:
            self.writer.add_image(tag, img_tensor, step)

    def log_graph(self, model, input_tensor):
        """记录模型图"""
        if self.enabled and self.writer:
            self.writer.add_graph(model, input_tensor)

    def close(self):
        """关闭记录器"""
        if self.enabled and self.writer:
            self.writer.close()


class PlotGenerator:
    """训练图表生成器"""

    def __init__(self, output_dir: Union[str, Path]):
        """
        初始化图表生成器

        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_training_curves(self,
                           history: Dict[str, List[float]],
                           save_name: str = "training_curves") -> str:
        """
        绘制训练曲线

        Args:
            history: 训练历史数据
            save_name: 保存文件名

        Returns:
            保存路径
        """
        # 确定需要绘制的指标
        metrics = ['loss', 'accuracy']
        available_metrics = []

        for metric in metrics:
            train_key = f'train_{metric}'
            val_key = f'val_{metric}'
            if train_key in history and val_key in history:
                available_metrics.append(metric)

        if not available_metrics:
            print("⚠️  没有可绘制的训练曲线数据")
            return ""

        # 创建子图
        fig, axes = plt.subplots(1, len(available_metrics), figsize=(6*len(available_metrics), 5))
        if len(available_metrics) == 1:
            axes = [axes]

        for i, metric in enumerate(available_metrics):
            train_key = f'train_{metric}'
            val_key = f'val_{metric}'

            epochs = range(1, len(history[train_key]) + 1)

            axes[i].plot(epochs, history[train_key], 'b-', label=f'训练{metric}', linewidth=2)
            axes[i].plot(epochs, history[val_key], 'r-', label=f'验证{metric}', linewidth=2)

            axes[i].set_title(f'训练和验证{metric}')
            axes[i].set_xlabel('轮次')
            axes[i].set_ylabel(metric.capitalize())
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

            # 标记最佳点
            if metric == 'loss':
                best_epoch = np.argmin(history[val_key]) + 1
                best_value = min(history[val_key])
            else:
                best_epoch = np.argmax(history[val_key]) + 1
                best_value = max(history[val_key])

            axes[i].axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7)
            axes[i].text(best_epoch, best_value, f'最佳: {best_value:.4f}',
                        ha='center', va='bottom' if metric != 'loss' else 'top',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

        plt.tight_layout()

        # 保存图片
        save_path = self.output_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"📈 训练曲线已保存: {save_path}")
        return str(save_path)

    def plot_learning_rate_schedule(self,
                                   lr_history: List[float],
                                   save_name: str = "lr_schedule") -> str:
        """
        绘制学习率调度曲线

        Args:
            lr_history: 学习率历史
            save_name: 保存文件名

        Returns:
            保存路径
        """
        if not lr_history:
            return ""

        plt.figure(figsize=(10, 6))
        epochs = range(1, len(lr_history) + 1)
        plt.plot(epochs, lr_history, 'b-', linewidth=2)
        plt.title('学习率调度')
        plt.xlabel('轮次')
        plt.ylabel('学习率')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # 使用对数尺度

        save_path = self.output_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"📉 学习率曲线已保存: {save_path}")
        return str(save_path)


def create_training_logger(log_dir: Union[str, Path],
                          model_name: str = "stork",
                          config: Optional[Dict[str, Any]] = None) -> TrainingLogger:
    """
    创建训练日志记录器的便捷函数

    Args:
        log_dir: 日志目录
        model_name: 模型名称
        config: 配置字典

    Returns:
        TrainingLogger实例
    """
    log_config = config.get('logging', {}) if config else {}

    return TrainingLogger(
        log_dir=log_dir,
        model_name=model_name,
        log_level=log_config.get('log_level', 'INFO'),
        console_output=log_config.get('console_output', True),
        file_output=log_config.get('file_output', True)
    )