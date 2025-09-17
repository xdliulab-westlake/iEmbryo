"""
配置管理器
统一管理所有配置参数，支持YAML文件和命令行参数覆盖
"""

import os
import yaml
import argparse
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigManager:
    """配置管理器类"""

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器

        Args:
            config_path: 配置文件路径，默认使用default.yaml
        """
        self.config_dir = Path(__file__).parent
        self.default_config_path = self.config_dir / "default.yaml"

        # 加载默认配置
        self.config = self._load_yaml(self.default_config_path)

        # 如果指定了自定义配置文件，合并配置
        if config_path and os.path.exists(config_path):
            custom_config = self._load_yaml(config_path)
            self.config = self._merge_configs(self.config, custom_config)

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """加载YAML配置文件"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load config from {path}: {e}")

    def _merge_configs(self, base: Dict, override: Dict) -> Dict:
        """递归合并配置字典"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        return result

    def update_from_args(self, args: argparse.Namespace):
        """从命令行参数更新配置"""
        # 更新常用参数
        if hasattr(args, 'data_dir') and args.data_dir:
            self.config['data']['data_dir'] = args.data_dir

        if hasattr(args, 'model') and args.model:
            self.config['model']['name'] = args.model

        if hasattr(args, 'batch_size') and args.batch_size:
            self.config['training']['batch_size'] = args.batch_size

        if hasattr(args, 'epochs') and args.epochs:
            self.config['training']['epochs'] = args.epochs

        if hasattr(args, 'learning_rate') and args.learning_rate:
            self.config['training']['learning_rate'] = args.learning_rate

        if hasattr(args, 'img_size') and args.img_size:
            self.config['data']['img_size'] = args.img_size

        if hasattr(args, 'fine_tune') and args.fine_tune is not None:
            self.config['model']['fine_tune'] = args.fine_tune

        if hasattr(args, 'resume') and args.resume is not None:
            self.config['training']['resume_training'] = args.resume

        if hasattr(args, 'workers') and args.workers:
            self.config['data']['num_workers'] = args.workers

        if hasattr(args, 'cross_validation') and args.cross_validation is not None:
            self.config['validation']['cross_validation'] = args.cross_validation

        if hasattr(args, 'no_gradcam') and args.no_gradcam:
            self.config['evaluation']['generate_gradcam'] = False

        if hasattr(args, 'gradcam_samples') and args.gradcam_samples:
            self.config['evaluation']['gradcam_samples'] = args.gradcam_samples

    def get(self, key_path: str, default=None):
        """
        获取配置值，支持嵌套键路径

        Args:
            key_path: 键路径，如 'model.name' 或 'training.batch_size'
            default: 默认值

        Returns:
            配置值
        """
        keys = key_path.split('.')
        value = self.config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key_path: str, value: Any):
        """
        设置配置值，支持嵌套键路径

        Args:
            key_path: 键路径，如 'model.name'
            value: 要设置的值
        """
        keys = key_path.split('.')
        config = self.config

        # 导航到最后一层
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        # 设置值
        config[keys[-1]] = value

    def save(self, path: str):
        """保存当前配置到文件"""
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)

    def print_config(self):
        """打印当前配置"""
        print("🔧 当前配置:")
        print("=" * 50)
        self._print_dict(self.config, indent=0)
        print("=" * 50)

    def _print_dict(self, d: Dict, indent: int = 0):
        """递归打印字典"""
        for key, value in d.items():
            prefix = "  " * indent
            if isinstance(value, dict):
                print(f"{prefix}{key}:")
                self._print_dict(value, indent + 1)
            else:
                print(f"{prefix}{key}: {value}")

    @property
    def data(self):
        """数据配置"""
        return self.config['data']

    @property
    def model(self):
        """模型配置"""
        return self.config['model']

    @property
    def training(self):
        """训练配置"""
        return self.config['training']

    @property
    def validation(self):
        """验证配置"""
        return self.config['validation']

    @property
    def optimization(self):
        """优化配置"""
        return self.config['optimization']

    @property
    def evaluation(self):
        """评估配置"""
        return self.config['evaluation']

    @property
    def paths(self):
        """路径配置"""
        return self.config['paths']


def create_arg_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="STORK PyTorch 胚胎分类训练系统",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 基础参数
    parser.add_argument('--config', type=str, default=None,
                       help='自定义配置文件路径')

    # 数据参数
    parser.add_argument('--data-dir', type=str, default=None,
                       help='训练数据目录路径')
    parser.add_argument('--img-size', type=int, default=None,
                       help='输入图像尺寸')
    parser.add_argument('--workers', type=int, default=None,
                       help='数据加载进程数')

    # 模型参数
    parser.add_argument('--model', '-m', type=str, default=None,
                       choices=['vgg16', 'resnet50', 'resnet152', 'inceptionv3', 'efficientnetb0'],
                       help='选择预训练模型')
    parser.add_argument('--fine-tune', action='store_true', default=None,
                       help='启用微调模式')
    parser.add_argument('--no-fine-tune', dest='fine_tune', action='store_false',
                       help='禁用微调模式')

    # 训练参数
    parser.add_argument('--epochs', '-e', type=int, default=None,
                       help='训练轮数')
    parser.add_argument('--batch-size', '-b', type=int, default=None,
                       help='每个GPU的批处理大小')
    parser.add_argument('--learning-rate', '--lr', type=float, default=None,
                       help='学习率')
    parser.add_argument('--resume', action='store_true', default=None,
                       help='恢复训练')
    parser.add_argument('--no-resume', dest='resume', action='store_false',
                       help='不恢复训练')

    # 验证参数
    parser.add_argument('--cross-validation', action='store_true', default=None,
                       help='启用交叉验证')
    parser.add_argument('--no-cross-validation', dest='cross_validation', action='store_false',
                       help='禁用交叉验证')

    # 评估参数
    parser.add_argument('--no-gradcam', action='store_true', default=False,
                       help='跳过Grad-CAM生成')
    parser.add_argument('--gradcam-samples', type=int, default=None,
                       help='Grad-CAM样本数量')

    # 调试参数
    parser.add_argument('--debug', action='store_true', default=False,
                       help='调试模式')
    parser.add_argument('--dry-run', action='store_true', default=False,
                       help='仅验证配置，不实际训练')

    return parser


# 便捷函数
def load_config(config_path: Optional[str] = None,
                args: Optional[argparse.Namespace] = None) -> ConfigManager:
    """
    加载配置的便捷函数

    Args:
        config_path: 配置文件路径
        args: 命令行参数

    Returns:
        ConfigManager实例
    """
    config_manager = ConfigManager(config_path)

    if args:
        config_manager.update_from_args(args)

    return config_manager