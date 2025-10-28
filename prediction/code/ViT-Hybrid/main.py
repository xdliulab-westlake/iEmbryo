"""
ViT-Hybrid Main Entry Point
Orchestrates the complete workflow: setup, data loading, model training, evaluation

Before training, generate score_dict.json:
  python generate_score_dict.py --maindb MainDB.csv --image_dir ./image

Author: Liao Zizhuo
Date: 2025-10-27
"""

import sys
import argparse
import torch
import torch.distributed as dist
from pathlib import Path

from config.config import get_model_config, get_training_config, get_data_config, PathConfig
from model.model import create_model
from dataset.dataset import create_data_loaders
from utils.utils import create_directories, count_parameters
from training.train import train_stage1, train_stage2
from optimization import setup_pytorch_optimization, MemoryMonitor


def setup_environment(args):
    """Step 1: Setup - Create directories and verify environment"""
    print("\n" + "="*60)
    print("STEP 1: ENVIRONMENT SETUP")
    print("="*60)

    # Check if this is main process (for distributed training)
    is_main_process = not dist.is_initialized() or dist.get_rank() == 0

    if is_main_process:
        create_directories()
        print("Directories created successfully")

    # Setup PyTorch optimization (multi-GPU support)
    device, is_distributed = setup_pytorch_optimization()

    # Create memory monitor
    memory_monitor = MemoryMonitor(device)

    if is_main_process:
        print(f"\nTraining Mode:")
        print(f"  Device: {device}")
        print(f"  Distributed: {is_distributed}")
        if torch.cuda.is_available():
            print(f"  CUDA Version: {torch.version.cuda}")

    return device, is_distributed, memory_monitor


def load_configuration(args):
    """Step 2: Configuration - Load and merge configurations"""
    is_main_process = not dist.is_initialized() or dist.get_rank() == 0

    if is_main_process:
        print("\n" + "="*60)
        print("STEP 2: CONFIGURATION LOADING")
        print("="*60)

    import json
    from pathlib import Path

    model_config = get_model_config()
    training_config = get_training_config()
    data_config = get_data_config()

    if args.config_file and Path(args.config_file).exists():
        print(f"Loading config from: {args.config_file}")
        with open(args.config_file, 'r') as f:
            json_config = json.load(f)

        if 'model' in json_config:
            for key, value in json_config['model'].items():
                config_key = key.lower()
                if config_key in model_config:
                    model_config[config_key] = value

        if 'training' in json_config:
            for key, value in json_config['training'].items():
                config_key = key.lower()
                if config_key in training_config:
                    training_config[config_key] = value

        if 'data' in json_config:
            for key, value in json_config['data'].items():
                config_key = key.lower()
                if config_key in data_config:
                    data_config[config_key] = value

    if args.data_root:
        data_config['data_root'] = args.data_root
        print(f"Data root: {args.data_root}")

    if args.two_stage:
        training_config['two_stage'] = True
        print("Training mode: Two-stage")
    else:
        print("Training mode: Single-stage")

    if args.use_focal_loss:
        training_config['use_focal_loss'] = True
        if is_main_process:
            print("Loss function: Focal Loss")
    else:
        if is_main_process:
            print("Loss function: Binary Cross-Entropy")

    if is_main_process:
        print(f"Model config: time_steps={model_config['time_steps']}, focal_planes={model_config['focal_planes']}")
        print(f"  num_transformer_layers={model_config['num_transformer_layers']}, num_heads={model_config['num_heads']}")

        # Show batch size info for multi-GPU
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        global_batch_size = training_config['batch_size'] * num_gpus
        print(f"Batch size: {training_config['batch_size']} per GPU, {global_batch_size} global")
        print(f"Epochs: {training_config['epochs']}")

    return model_config, training_config, data_config


def prepare_data(data_config):
    """Step 3: Data Preparation - Load and prepare datasets"""
    is_main_process = not dist.is_initialized() or dist.get_rank() == 0

    if is_main_process:
        print("\n" + "="*60)
        print("STEP 3: DATA PREPARATION")
        print("="*60)
        print("Creating data loaders...")

    try:
        train_loader, val_loader, test_loader = create_data_loaders(
            data_root=data_config['data_root'],
            config=data_config
        )

        if is_main_process:
            print(f"Data loaded successfully")
            print(f"  Training samples: {len(train_loader.dataset)}")
            print(f"  Validation samples: {len(val_loader.dataset)}")
            print(f"  Test samples: {len(test_loader.dataset)}")

        return train_loader, val_loader, test_loader

    except Exception as e:
        print(f"Error loading data: {e}")
        print(f"Please check:")
        print(f"  1. Data path: {data_config['data_root']}")
        print(f"  2. Data format (should be 'image/' + 'score_dict.json' or 'good/'+'poor/')")
        print(f"  3. score_dict.json exists in data root")
        sys.exit(1)


def build_model(model_config, device, is_distributed=False):
    """Step 4: Model Building - Create and initialize model"""
    is_main_process = not dist.is_initialized() or dist.get_rank() == 0

    if is_main_process:
        print("\n" + "="*60)
        print("STEP 4: MODEL BUILDING")
        print("="*60)
        print("Creating model...")

    # Create model
    model = create_model(model_config).to(device)

    # Multi-GPU setup
    if torch.cuda.device_count() > 1 and not is_distributed:
        # Use DataParallel for single-process multi-GPU
        model = torch.nn.DataParallel(model)
        if is_main_process:
            print(f"Enabled DataParallel for {torch.cuda.device_count()} GPUs")
    elif is_distributed:
        # DDP will be wrapped in training script
        if is_main_process:
            print(f"Model ready for DDP distributed training")

    if is_main_process:
        params = count_parameters(model)
        print(f"Model created successfully")
        print(f"  Total parameters: {params['total']:,}")
        print(f"  Trainable parameters: {params['trainable']:,}")
        print(f"  Non-trainable parameters: {params['non_trainable']:,}")

    return model


def train_model(model, train_loader, val_loader, training_config, device, args, is_distributed=False):
    """Step 5: Model Training - Execute training pipeline"""
    is_main_process = not dist.is_initialized() or dist.get_rank() == 0

    if is_main_process:
        print("\n" + "="*60)
        print("STEP 5: MODEL TRAINING")
        print("="*60)

    if args.two_stage:
        if is_main_process:
            print("Executing two-stage training...")
            print("\nStage 1: Training Transformer (VGG16 frozen)")
        model = train_stage1(model, train_loader, val_loader, training_config, device)

        if is_main_process:
            print("\nStage 2: Fine-tuning with VGG16 unfrozen")
        model = train_stage2(model, train_loader, val_loader, training_config, device)
    else:
        if is_main_process:
            print("Executing single-stage training...")
        model = train_stage1(model, train_loader, val_loader, training_config, device)

    return model


def evaluate_model(model, test_loader, device):
    """Step 6: Model Evaluation - Evaluate on test set"""
    is_main_process = not dist.is_initialized() or dist.get_rank() == 0

    if is_main_process:
        print("\n" + "="*60)
        print("STEP 6: MODEL EVALUATION")
        print("="*60)
        print("Running evaluation on test set...")

    from utils.utils import calculate_metrics
    import numpy as np

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    metrics = calculate_metrics(all_labels, all_preds)

    if is_main_process:
        print(f"\nTest Set Results:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        print(f"  AUC:       {metrics['auc']:.4f}")

    return metrics


def save_final_model(model):
    """Step 7: Save Final Model"""
    is_main_process = not dist.is_initialized() or dist.get_rank() == 0

    if is_main_process:
        print("\n" + "="*60)
        print("STEP 7: SAVE FINAL MODEL")
        print("="*60)

        # Handle DataParallel wrapper
        model_to_save = model
        if hasattr(model, 'module'):  # DataParallel or DDP
            model_to_save = model.module

        final_path = Path(PathConfig.MODEL_SAVE_DIR) / 'final_model.pth'
        torch.save(model_to_save.state_dict(), final_path)
        print(f"Final model saved to: {final_path}")

        return final_path
    return None


def summarize_results():
    """Step 8: Summary - Show training summary and next steps"""
    is_main_process = not dist.is_initialized() or dist.get_rank() == 0

    if is_main_process:
        print("\n" + "="*60)
        print("STEP 8: TRAINING COMPLETE - SUMMARY")
        print("="*60)

        checkpoint_dir = Path(PathConfig.MODEL_SAVE_DIR)
        result_dir = Path(PathConfig.RESULT_DIR)

        print("\nGenerated files:")
        if (checkpoint_dir / 'final_model.pth').exists():
            print(f"  ✓ Model: {checkpoint_dir / 'final_model.pth'}")
        if (result_dir / 'stage1_history.json').exists():
            print(f"  ✓ History: {result_dir / 'stage1_history.json'}")
        if (result_dir / 'stage2_history.json').exists():
            print(f"  ✓ History: {result_dir / 'stage2_history.json'}")

        print("\nNext steps:")
        print("  1. Review training history in results/")
        print("  2. Run inference: python main.py infer --model_path checkpoints/final_model.pth --data_path /path/to/data")
        print("  3. Deploy model to production")


def train_command(args):
    """Execute training workflow"""
    try:
        device, is_distributed, memory_monitor = setup_environment(args)
        model_config, training_config, data_config = load_configuration(args)

        # Check if using multi-focal mode
        is_main_process = not dist.is_initialized() or dist.get_rank() == 0

        if args.multi_focal:
            if is_main_process:
                print("\n" + "="*60)
                print("MULTI-FOCAL TRAINING MODE")
                print("="*60)

            from dataset.multi_focal_data_loader import (
                load_score_dict_for_multi_focal,
                MultiFocalDataLoader
            )
            from dataset.dataset import create_multi_focal_data_loaders

            # Load quality labels
            maindb_csv = args.maindb_csv or "/liaozizhuo/scripts/MainDB.csv"
            filtered_csv = args.filtered_csv or "/liaozizhuo/scripts/filtered_clean_embryos.csv"
            score_dict = load_score_dict_for_multi_focal(maindb_csv, filtered_csv)

            # Load and validate multi-focal data
            selected_images_csv = args.selected_images_csv or "/liaozizhuo/selected_image_paths.csv"
            loader = MultiFocalDataLoader(
                verbose=True,
                use_cache=True,
                cache_dir=PathConfig.CACHE_DIR
            )
            multi_focal_df, embryo_to_label = loader.load_multi_focal_data(
                selected_images_csv, score_dict, skip_validation=args.skip_validation
            )

            # Create data loaders
            train_loader, val_loader, test_loader, labels = create_multi_focal_data_loaders(
                multi_focal_df, embryo_to_label, data_config
            )
        else:
            # Standard mode
            train_loader, val_loader, test_loader = prepare_data(data_config)

        model = build_model(model_config, device, is_distributed)
        model = train_model(model, train_loader, val_loader, training_config, device, args, is_distributed)
        metrics = evaluate_model(model, test_loader, device)
        save_final_model(model)
        summarize_results()

        if is_main_process:
            print("\n" + "="*60)
            print("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            print("="*60 + "\n")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def generate_config_command(args):
    """Generate configuration template file"""
    import json

    print("\n" + "="*60)
    print("GENERATING CONFIG TEMPLATE")
    print("="*60)

    config_template = {
        "model": {
            "TIME_STEPS": 33,
            "FOCAL_PLANES": 11,
            "IMAGE_SIZE": 224,
            "NUM_TRANSFORMER_LAYERS": 4,
            "NUM_HEADS": 8,
            "D_MODEL": 4096,
            "DFF": 2048,
            "DROPOUT_RATE": 0.1,
            "FUSION_TYPE": "average"
        },
        "training": {
            "BATCH_SIZE": 4,
            "EPOCHS": 100,
            "LEARNING_RATE": 0.0001,
            "WEIGHT_DECAY": 0.00001,
            "TRAIN_RATIO": 0.7,
            "VAL_RATIO": 0.15,
            "EARLY_STOPPING_PATIENCE": 10,
            "REDUCE_LR_PATIENCE": 5,
            "REDUCE_LR_FACTOR": 0.5,
            "MIN_LR": 0.0000001,
            "MONITOR_METRIC": "val_auc",
            "MONITOR_MODE": "max",
            "NUM_WORKERS": 4,
            "PIN_MEMORY": True
        },
        "data": {
            "DATA_ROOT": "/liaozizhuo/",
            "SCORE_DICT_PATH": "/liaozizhuo/score_dict.json",
            "TEST_RATIO": 0.15,
            "APPLY_PETRI_CROP": True,
            "APPLY_CENTER_CROP": True
        },
        "paths": {
            "MODEL_SAVE_DIR": "/liaozizhuo/scripts/ViT-Hybrid/checkpoints",
            "LOG_DIR": "/liaozizhuo/scripts/ViT-Hybrid/logs",
            "RESULT_DIR": "/liaozizhuo/scripts/ViT-Hybrid/results",
            "CACHE_DIR": "/liaozizhuo/scripts/ViT-Hybrid/cache"
        }
    }

    output_path = args.output if args.output else 'config.json'

    with open(output_path, 'w') as f:
        json.dump(config_template, f, indent=2)

    print(f"\nConfiguration template generated: {output_path}")
    print("\nIMPORTANT: Edit the following fields before training:")
    print("  - data.DATA_ROOT: Set to your data directory path")
    print("  - data.SCORE_DICT_PATH: Set to your score_dict.json path (or null for temporal data)")
    print("  - training.BATCH_SIZE: Adjust based on GPU memory")
    print("  - training.EPOCHS: Set training epochs")
    print("\nUsage with this config:")
    print(f"  python main.py train --data_root /path/to/data --config_file {output_path}")
    print(f"  python main.py train --data_root /path/to/data -c {output_path}")
    print("\nYou can modify these fields in the config file:")
    print("  - model.NUM_TRANSFORMER_LAYERS: Number of Transformer encoder layers (default: 4)")
    print("  - model.NUM_HEADS: Number of attention heads (default: 8)")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='ViT-Hybrid: Vision Transformer + VGG16 for Embryo Quality Assessment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate config template
  python main.py generate-config --output my_config.json

  # Single-stage training with default config
  python main.py train --data_root /path/to/data

  # Training with custom config (change encoder layers, attention heads, etc)
  python main.py train --data_root /path/to/data --config_file my_config.json

  # Two-stage training (recommended)
  python main.py train --data_root /path/to/data --two_stage -c my_config.json

  # With Focal Loss for imbalanced data
  python main.py train --data_root /path/to/data --two_stage --use_focal_loss -c my_config.json
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Generate config command
    config_parser = subparsers.add_parser('generate-config', help='Generate configuration template')
    config_parser.add_argument('--output', '-o', type=str, default='config.json', help='Output config file path')

    # Training command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--data_root', type=str, required=True, help='Path to data root directory')
    train_parser.add_argument('--config_file', '-c', type=str, default=None, help='Path to config JSON file')
    train_parser.add_argument('--two_stage', action='store_true', help='Use two-stage training')
    train_parser.add_argument('--use_focal_loss', action='store_true', help='Use Focal Loss for imbalanced data')
    train_parser.add_argument('--multi_focal', action='store_true', help='Use multi-focal multi-timepoint data mode')
    train_parser.add_argument('--selected_images_csv', type=str, default=None, help='Path to selected_image_paths.csv')
    train_parser.add_argument('--maindb_csv', type=str, default=None, help='Path to MainDB.csv')
    train_parser.add_argument('--filtered_csv', type=str, default=None, help='Path to filtered_clean_embryos.csv')
    train_parser.add_argument('--skip_validation', action='store_true', help='Skip validation and use cached results (faster for repeated runs)')

    args = parser.parse_args()

    if args.command == 'generate-config':
        generate_config_command(args)
    elif args.command == 'train':
        train_command(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
