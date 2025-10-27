"""
ViT-Hybrid Main Entry Point
Orchestrates the complete workflow: setup, data loading, model training, evaluation
Author: Liao Zizhuo
Date: 2025-10-27
"""

import sys
import argparse
import torch
from pathlib import Path

from config.config import get_model_config, get_training_config, get_data_config, PathConfig
from model.model import create_model
from dataset.dataset import create_data_loaders
from utils.utils import create_directories, count_parameters
from training.train import train_stage1, train_stage2


def setup_environment(args):
    """Step 1: Setup - Create directories and verify environment"""
    print("\n" + "="*60)
    print("STEP 1: ENVIRONMENT SETUP")
    print("="*60)

    create_directories()
    print("Directories created successfully")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")

    return device


def load_configuration(args):
    """Step 2: Configuration - Load and merge configurations"""
    print("\n" + "="*60)
    print("STEP 2: CONFIGURATION LOADING")
    print("="*60)

    model_config = get_model_config()
    training_config = get_training_config()
    data_config = get_data_config()

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
        print("Loss function: Focal Loss")
    else:
        print("Loss function: Binary Cross-Entropy")

    print(f"Model config: time_steps={model_config['time_steps']}, focal_planes={model_config['focal_planes']}")
    print(f"Batch size: {training_config['batch_size']}, Epochs: {training_config['epochs']}")

    return model_config, training_config, data_config


def prepare_data(data_config):
    """Step 3: Data Preparation - Load and prepare datasets"""
    print("\n" + "="*60)
    print("STEP 3: DATA PREPARATION")
    print("="*60)

    try:
        print("Creating data loaders...")
        train_loader, val_loader, test_loader = create_data_loaders(
            data_root=data_config['data_root'],
            config=data_config
        )

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


def build_model(model_config, device):
    """Step 4: Model Building - Create and initialize model"""
    print("\n" + "="*60)
    print("STEP 4: MODEL BUILDING")
    print("="*60)

    print("Creating model...")
    model = create_model(model_config).to(device)

    params = count_parameters(model)
    print(f"Model created successfully")
    print(f"  Total parameters: {params['total']:,}")
    print(f"  Trainable parameters: {params['trainable']:,}")
    print(f"  Non-trainable parameters: {params['non_trainable']:,}")

    return model


def train_model(model, train_loader, val_loader, training_config, device, args):
    """Step 5: Model Training - Execute training pipeline"""
    print("\n" + "="*60)
    print("STEP 5: MODEL TRAINING")
    print("="*60)

    if args.two_stage:
        print("Executing two-stage training...")
        print("\nStage 1: Training Transformer (VGG16 frozen)")
        model = train_stage1(model, train_loader, val_loader, training_config, device)

        print("\nStage 2: Fine-tuning with VGG16 unfrozen")
        model = train_stage2(model, train_loader, val_loader, training_config, device)
    else:
        print("Executing single-stage training...")
        model = train_stage1(model, train_loader, val_loader, training_config, device)

    return model


def evaluate_model(model, test_loader, device):
    """Step 6: Model Evaluation - Evaluate on test set"""
    print("\n" + "="*60)
    print("STEP 6: MODEL EVALUATION")
    print("="*60)

    from utils.utils import calculate_metrics
    import numpy as np

    model.eval()
    all_preds = []
    all_labels = []

    print("Running evaluation on test set...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    metrics = calculate_metrics(all_labels, all_preds)
    print(f"\nTest Set Results:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    print(f"  AUC:       {metrics['auc']:.4f}")

    return metrics


def save_final_model(model):
    """Step 7: Save Final Model"""
    print("\n" + "="*60)
    print("STEP 7: SAVE FINAL MODEL")
    print("="*60)

    final_path = Path(PathConfig.MODEL_SAVE_DIR) / 'final_model.pth'
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved to: {final_path}")

    return final_path


def summarize_results():
    """Step 8: Summary - Show training summary and next steps"""
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
        device = setup_environment(args)
        model_config, training_config, data_config = load_configuration(args)
        train_loader, val_loader, test_loader = prepare_data(data_config)
        model = build_model(model_config, device)
        model = train_model(model, train_loader, val_loader, training_config, device, args)
        metrics = evaluate_model(model, test_loader, device)
        save_final_model(model)
        summarize_results()

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


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='ViT-Hybrid: Vision Transformer + VGG16 for Embryo Quality Assessment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single-stage training
  python main.py train --data_root /path/to/data

  # Two-stage training (recommended)
  python main.py train --data_root /path/to/data --two_stage

  # With Focal Loss for imbalanced data
  python main.py train --data_root /path/to/data --two_stage --use_focal_loss
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Training command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--data_root', type=str, required=True, help='Path to data root directory')
    train_parser.add_argument('--two_stage', action='store_true', help='Use two-stage training')
    train_parser.add_argument('--use_focal_loss', action='store_true', help='Use Focal Loss for imbalanced data')

    args = parser.parse_args()

    if args.command == 'train':
        train_command(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
