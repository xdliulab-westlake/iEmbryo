#!/usr/bin/env python3
"""
STORK PyTorch 快速设置脚本
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, check=True):
    """运行shell命令"""
    print(f"🔧 执行: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout.strip())
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 命令执行失败: {e}")
        if e.stderr:
            print(f"错误输出: {e.stderr}")
        return False


def setup_environment():
    """设置Python环境"""
    print("🐍 设置Python环境...")

    # 检查Python版本
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ 需要Python 3.8或更高版本")
        return False

    print(f"✅ Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")

    # 升级pip
    if not run_command(f"{sys.executable} -m pip install --upgrade pip"):
        return False

    # 安装依赖
    requirements_file = Path(__file__).parent / "requirements.txt"
    if requirements_file.exists():
        print("📦 安装Python依赖...")
        cmd = f"{sys.executable} -m pip install -r {requirements_file}"
        if not run_command(cmd):
            print("❌ 依赖安装失败，请手动安装")
            return False
    else:
        print("⚠️  requirements.txt不存在，跳过依赖安装")

    return True


def check_pytorch():
    """检查PyTorch安装"""
    print("\n🔥 检查PyTorch...")
    try:
        import torch
        print(f"✅ PyTorch版本: {torch.__version__}")

        if torch.cuda.is_available():
            print(f"✅ CUDA可用: {torch.version.cuda}")
            print(f"✅ GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("⚠️  CUDA不可用，将使用CPU训练")

        return True
    except ImportError:
        print("❌ PyTorch未安装")
        return False


def check_deepspeed():
    """检查DeepSpeed"""
    print("\n🚀 检查DeepSpeed...")
    try:
        import deepspeed
        print(f"✅ DeepSpeed版本: {deepspeed.__version__}")
        return True
    except ImportError:
        print("⚠️  DeepSpeed未安装，将使用标准PyTorch优化")
        print("   安装DeepSpeed: pip install deepspeed")
        return False


def create_sample_config():
    """创建示例配置文件"""
    print("\n📝 创建示例配置...")

    sample_config = """# STORK PyTorch 示例配置
# 复制并修改此文件以适应你的数据集

data:
  data_dir: "/path/to/your/data"  # 修改为你的数据路径
  img_size: 224
  train_split: 0.7
  use_cache: true
  num_workers: 4  # 根据你的CPU核心数调整

model:
  name: "vgg16"  # 或 resnet50, efficientnetb0 等
  fine_tune: true

training:
  epochs: 100     # 减少用于测试
  batch_size: 8   # 根据GPU内存调整
  learning_rate: 0.0001

optimization:
  use_deepspeed: true
  mixed_precision: true

paths:
  output_dir: "./outputs"
  model_dir: "./models"
  log_dir: "./logs"
  cache_dir: "./cache"
"""

    config_file = Path(__file__).parent / "config" / "sample.yaml"
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(sample_config)

    print(f"✅ 示例配置已创建: {config_file}")
    print("   请根据你的数据修改配置文件")


def main():
    """主函数"""
    print("🧬 STORK PyTorch 环境设置")
    print("=" * 50)

    # 设置环境
    if not setup_environment():
        print("❌ 环境设置失败")
        return False

    # 检查关键组件
    pytorch_ok = check_pytorch()
    deepspeed_ok = check_deepspeed()

    # 创建示例配置
    create_sample_config()

    print("\n" + "=" * 50)
    if pytorch_ok:
        print("🎉 设置完成！")
        print("\n📚 快速开始:")
        print("1. 修改 config/sample.yaml 中的数据路径")
        print("2. 运行训练: python main.py --config config/sample.yaml")
        print("3. 或使用默认配置: python main.py")

        if deepspeed_ok:
            print("\n🚀 DeepSpeed优化已启用，将获得更好的内存管理")

        return True
    else:
        print("❌ 设置失败，请检查错误信息")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)