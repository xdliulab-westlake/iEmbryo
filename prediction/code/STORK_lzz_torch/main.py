#!/usr/bin/env python3
"""
STORK PyTorch 主程序
简洁的面向对象设计，配置文件驱动
"""

import os
import sys
import warnings
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

# 抑制警告
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import torch
from config import create_arg_parser, load_config
from models import create_model
from data import create_stork_data_loader, create_train_transforms, create_val_transforms
from training import STORKTrainer
from optimization import create_optimization_setup
from evaluation import evaluate_model, create_cross_validator
from visualization import create_gradcam_visualizer
from utils import create_model_manager, create_training_logger


class STORKApplication:
    """STORK应用主类"""

    def __init__(self, args):
        """初始化应用"""
        # 检查是否为主进程
        self.is_main_process = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0

        if self.is_main_process:
            print("🧬 STORK PyTorch 胚胎分类系统")
            print("=" * 50)

        # 加载配置
        self.config_manager = load_config(args.config, args)
        self.config = self.config_manager.config

        # 打印配置（调试模式）
        if args.debug and self.is_main_process:
            self.config_manager.print_config()

        # 创建输出目录
        self._create_directories()

        # 设置优化
        self.optimization_setup = create_optimization_setup(self.config)
        self.device = self.optimization_setup['device']

        # 初始化组件
        self.model = None
        self.data_loader = None
        self.trainer = None
        self.model_manager = None
        self.logger = None

        # 数据集
        self.train_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def _create_directories(self):
        """创建必要的目录"""
        paths = self.config['paths']
        for path_key, path_value in paths.items():
            Path(path_value).mkdir(parents=True, exist_ok=True)

    def setup_data(self):
        """设置数据加载器"""
        if self.is_main_process:
            print("\n📊 设置STORK数据加载器...")

        # 创建STORK数据加载器
        self.data_loader = create_stork_data_loader(self.config)

        # 加载数据集（使用STORK预处理）
        self.train_dataset, self.test_dataset = self.data_loader.load_data()

        # 创建PyTorch数据加载器
        from torch.utils.data import DataLoader, random_split

        # 从训练集中分离验证集
        train_size = int(0.8 * len(self.train_dataset))
        val_size = len(self.train_dataset) - train_size
        train_subset, val_subset = random_split(self.train_dataset, [train_size, val_size])

        # 创建数据加载器
        batch_size = self.config['training']['batch_size']
        num_workers = self.config.get('data', {}).get('num_workers', 4)

        # 检查是否需要分布式采样器
        train_sampler = None
        if self.optimization_setup['is_distributed'] and torch.distributed.is_initialized():
            from torch.utils.data.distributed import DistributedSampler
            train_sampler = DistributedSampler(train_subset)
            if self.is_main_process:
                print("🔄 使用分布式采样器")

        self.train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=(train_sampler is None),  # 如果有sampler就不shuffle
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True
        )
        self.val_loader = DataLoader(
            val_subset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )

        if self.is_main_process:
            print(f"✅ STORK数据加载器设置完成")
            print(f"   训练集: {len(train_subset)} 样本")
            print(f"   验证集: {len(val_subset)} 样本")
            print(f"   测试集: {len(self.test_dataset)} 样本")

    def setup_model(self):
        """设置模型"""
        if self.is_main_process:
            print("\n🤖 设置模型...")

        # 创建模型
        self.model = create_model(self.config['model'])
        if self.is_main_process:
            self.model.print_model_info()

        # 移动到设备
        self.model = self.model.to(self.device)

        # 多GPU设置
        if torch.cuda.device_count() > 1 and not self.optimization_setup['is_distributed']:
            # 使用DataParallel进行多GPU训练
            self.model = torch.nn.DataParallel(self.model)
            if self.is_main_process:
                print(f"🚀 启用DataParallel多GPU训练: {torch.cuda.device_count()} 块GPU")

        if self.is_main_process:
            print("✅ 模型设置完成")

    def setup_trainer(self):
        """设置训练器"""
        if self.is_main_process:
            print("\n🎯 设置训练器...")

        # 创建模型管理器
        self.model_manager = create_model_manager(
            model_dir=self.config['paths']['model_dir'],
            model_name=self.config['model']['name']
        )

        # 创建日志记录器
        self.logger = create_training_logger(
            log_dir=self.config['paths']['log_dir'],
            model_name=self.config['model']['name'],
            config=self.config
        )

        # 创建训练器
        self.trainer = STORKTrainer(
            model=self.model,
            config=self.config,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            device=self.device,
            model_manager=self.model_manager,
            logger=self.logger
        )

        if self.is_main_process:
            print("✅ 训练器设置完成")

    def train(self):
        """执行训练"""
        if self.is_main_process:
            print("\n🚀 开始训练...")

        # 检查是否恢复训练
        if self.config['training']['resume_training']:
            checkpoint_path = Path('checkpoints/best_model.pth')
            if checkpoint_path.exists():
                if self.is_main_process:
                    print(f"📂 恢复训练: {checkpoint_path}")
                self.trainer.load_checkpoint(str(checkpoint_path))

        # 执行训练
        num_epochs = self.config['training']['epochs']
        history = self.trainer.train(num_epochs)

        if self.is_main_process:
            print("✅ 训练完成")
        return history

    def evaluate(self):
        """执行评估"""
        print("\n📈 开始全面评估...")

        # 加载最佳模型
        try:
            self.model_manager.load_best_model(self.model, 'val_accuracy', self.device)
            print("📂 已加载最佳模型")
        except Exception as e:
            print(f"⚠️  无法加载最佳模型: {e}")
            print("使用当前模型进行评估")

        # 在测试集上进行详细评估
        output_dir = self.config['paths']['output_dir']
        model_name = self.config['model']['name']

        metrics = evaluate_model(
            model=self.model,
            data_loader=self.test_loader,
            device=self.device,
            output_dir=output_dir,
            model_name=model_name,
            class_names=['Non-viable', 'Viable']
        )

        print(f"📊 测试集评估完成:")
        print(f"   准确率: {metrics.get('accuracy', 0):.4f}")
        print(f"   精确率: {metrics.get('precision_weighted', 0):.4f}")
        print(f"   召回率: {metrics.get('recall_weighted', 0):.4f}")
        print(f"   F1分数: {metrics.get('f1_weighted', 0):.4f}")

        # 生成Grad-CAM可视化（如果启用）
        if self.config.get('evaluation', {}).get('generate_gradcam', False):
            print("\n🎨 生成Grad-CAM可视化...")
            try:
                gradcam_visualizer = create_gradcam_visualizer(self.model, self.config)
                gradcam_results = gradcam_visualizer.visualize_samples(
                    self.test_loader,
                    num_samples=self.config.get('evaluation', {}).get('gradcam_samples', 10)
                )
                print("✅ Grad-CAM可视化完成")
            except Exception as e:
                print(f"⚠️  Grad-CAM可视化失败: {e}")

        return metrics

    def run(self):
        """运行完整流程"""
        try:
            # 设置数据
            self.setup_data()

            # 设置模型
            self.setup_model()

            # 设置训练器
            self.setup_trainer()

            # 训练
            history = self.train()

            # 评估
            metrics = self.evaluate()

            print("\n🎉 训练和评估完成!")
            print(f"🏆 最终测试准确率: {metrics.get('accuracy', 0)*100:.2f}%")

            # 可选：运行交叉验证
            if self.config.get('validation', {}).get('cross_validation', False):
                print("\n🔄 开始交叉验证...")
                try:
                    cv_validator = create_cross_validator(self.config, self.device)

                    # 创建完整数据集用于交叉验证
                    from torch.utils.data import ConcatDataset
                    full_dataset = ConcatDataset([self.train_dataset, self.test_dataset])

                    # 运行交叉验证
                    cv_results = cv_validator.run_cross_validation(
                        full_dataset,
                        lambda: create_model(self.config['model'])
                    )
                    print("✅ 交叉验证完成")
                except Exception as e:
                    print(f"⚠️  交叉验证失败: {e}")

            # 显示模型摘要
            if hasattr(self, 'model_manager') and self.model_manager:
                summary = self.model_manager.get_model_summary()
                print(f"\n📁 模型文件摘要:")
                print(f"   检查点数量: {len(summary.get('checkpoints', []))}")
                print(f"   最佳模型数量: {len(summary.get('best_models', []))}")
                print(f"   导出模型数量: {len(summary.get('exports', []))}")

            return True

        except KeyboardInterrupt:
            print("\n⏹️  训练被用户中断")
            return False

        except Exception as e:
            print(f"\n❌ 训练过程发生错误: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """主函数"""
    # 解析命令行参数
    parser = create_arg_parser()
    args = parser.parse_args()

    # 调试模式
    if args.debug:
        print("🐛 调试模式已启用")

    # 仅验证配置模式
    if args.dry_run:
        print("🔍 仅验证配置（不实际训练）")
        config_manager = load_config(args.config, args)
        config_manager.print_config()
        print("✅ 配置验证完成")
        return

    # 创建并运行应用
    app = STORKApplication(args)
    success = app.run()

    if success:
        print("\n🎯 程序执行成功")
        sys.exit(0)
    else:
        print("\n❌ 程序执行失败")
        sys.exit(1)


if __name__ == "__main__":
    main()