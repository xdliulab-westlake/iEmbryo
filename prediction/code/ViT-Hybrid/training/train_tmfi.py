"""
Multi-Task TMFI-Net Training Script
完全使用TMFI-Net的原始架构和损失函数

多任务学习：
1. 主任务：分类 (CrossEntropyLoss)
2. 辅助任务：Saliency prediction (KLDiv + CC，如果有GT saliency map)

Author: Liao Zizhuo
Date: 2025-12-22
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
import argparse
import numpy as np
import sys
from tqdm import tqdm

from model import create_tmfi_embryo_model, FocalFusionWrapper
from dataset.dataset import create_multi_focal_data_loaders
from dataset.multi_focal_data_loader import MultiFocalDataLoader, load_score_dict_for_multi_focal
from utils.utils import (
    EarlyStopping,
    calculate_metrics,
    save_checkpoint,
    save_history,
    create_directories,
    count_parameters,
    print_confusion_matrix,
    AverageMeter,
)
from utils.saliency_losses import get_tmfi_loss
from config.config import PathConfig


def is_main_process():
    """Check if current process is main process"""
    return not dist.is_initialized() or dist.get_rank() == 0


def log_print(*args, **kwargs):
    """Print only on main process"""
    if is_main_process():
        print(*args, **kwargs, flush=True)
        sys.stdout.flush()


def train_one_epoch(model, dataloader, cls_criterion, optimizer, device,
                    use_saliency_loss=False, saliency_weight=0.1, print_cm=False):
    """
    Train one epoch with multi-task learning

    Args:
        model: FocalFusionWrapper containing Multi-Task TMFI-Net
        dataloader: Training data loader
        cls_criterion: Classification loss (CrossEntropyLoss)
        optimizer: Optimizer
        device: Device
        use_saliency_loss: Whether to use saliency loss (需要GT saliency map)
        saliency_weight: Weight for saliency loss
        print_cm: Whether to print confusion matrix
    """
    model.train()

    cls_loss_meter = AverageMeter()
    sal_loss_meter = AverageMeter()
    total_loss_meter = AverageMeter()

    all_preds = []
    all_labels = []

    iterator = tqdm(dataloader, disable=not is_main_process(), desc="Training")

    for batch_idx, (inputs, labels) in enumerate(iterator):
        # inputs: (B, T=33, F=11, C=3, H=224, W=224)
        inputs = inputs.to(device)
        labels = labels.to(device).long()

        optimizer.zero_grad()

        # Forward: saliency_map + logits
        saliency_map, logits = model(inputs)  # (B, H, W), (B, 2)

        # Classification loss
        cls_loss = cls_criterion(logits, labels)

        # Total loss
        if use_saliency_loss:
            # TODO: 如果有GT saliency map，计算saliency loss
            # 目前我们没有GT，所以跳过
            sal_loss = torch.tensor(0.0).to(device)
            total_loss = cls_loss
        else:
            sal_loss = torch.tensor(0.0).to(device)
            total_loss = cls_loss

        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Update meters
        cls_loss_meter.update(cls_loss.item())
        sal_loss_meter.update(sal_loss.item())
        total_loss_meter.update(total_loss.item())

        # Get predictions
        probs = torch.softmax(logits, dim=1)[:, 1]
        all_preds.extend(probs.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

        iterator.set_postfix({
            'cls': f'{cls_loss.item():.4f}',
            'total': f'{total_loss.item():.4f}'
        })

    all_preds_flat = np.array(all_preds).flatten()
    all_labels_flat = np.array(all_labels).flatten()
    metrics = calculate_metrics(all_labels_flat, all_preds_flat)

    if print_cm and is_main_process():
        print_confusion_matrix(all_preds_flat, all_preds_flat)

    return total_loss_meter.avg, cls_loss_meter.avg, sal_loss_meter.avg, metrics


def validate(model, dataloader, criterion, device, print_cm=False, save_attention=False, epoch=None):
    """
    Validate model

    Args:
        model: FocalFusionWrapper containing TMFI-Embryo
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to run on
        print_cm: Whether to print confusion matrix
        save_attention: Whether to save attention maps
        epoch: Current epoch (for saving attention maps)
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    attention_maps = []

    iterator = tqdm(dataloader, disable=not is_main_process(), desc="Validating")

    with torch.no_grad():
        for inputs, labels in iterator:
            inputs = inputs.to(device)
            labels = labels.to(device).long()

            # Forward: saliency_map + logits
            saliency_map, logits = model(inputs)

            # Classification loss
            cls_loss = criterion(logits, labels)
            total_loss += cls_loss.item()

            # Collect predictions
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Collect saliency maps if needed
            if save_attention:
                attention_maps.append(saliency_map.cpu())

            iterator.set_postfix({'loss': f'{cls_loss.item():.4f}'})

    avg_loss = total_loss / len(dataloader)
    all_preds_flat = np.array(all_preds).flatten()
    all_labels_flat = np.array(all_labels).flatten()

    if print_cm and is_main_process():
        log_print(f"\n  Probability statistics (class 1 - Good):")
        log_print(f"    Mean: {all_preds_flat.mean():.4f}, Std: {all_preds_flat.std():.4f}")
        log_print(f"    Min: {all_preds_flat.min():.4f}, Max: {all_preds_flat.max():.4f}")

    metrics = calculate_metrics(all_labels_flat, all_preds_flat)

    if print_cm and is_main_process():
        print_confusion_matrix(all_labels_flat, all_preds_flat)

    # Save saliency maps if requested
    if save_attention and is_main_process() and epoch is not None:
        saliency_dir = Path(PathConfig.RESULT_DIR) / 'saliency_maps'
        saliency_dir.mkdir(exist_ok=True, parents=True)

        saliency_all = torch.cat(attention_maps, dim=0)  # (N, H, W)
        save_path = saliency_dir / f'saliency_epoch_{epoch:03d}.pt'
        torch.save(saliency_all, save_path)
        log_print(f"  Saved {len(saliency_all)} saliency maps to: {save_path}")

    return avg_loss, metrics


def train_tmfi(model, train_loader, val_loader, config, device, print_cm=True):
    """
    Train TMFI-Embryo model

    Args:
        model: FocalFusionWrapper containing TMFI-Embryo
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        device: Device
        print_cm: Whether to print confusion matrices
    """
    log_print("\n" + "="*60)
    log_print("Multi-Task TMFI-Net Training")
    log_print("="*60)

    # Loss function: Classification loss (TMFI-Net saliency loss暂不使用，因为没有GT saliency map)
    cls_criterion = nn.CrossEntropyLoss()
    log_print(f"\nLoss functions:")
    log_print(f"  Classification: CrossEntropyLoss")
    log_print(f"  Saliency: Not used (no GT saliency maps)")

    # Optimizer (与TMFI-Net保持一致)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate']
    )
    log_print(f"\nOptimizer: Adam (same as TMFI-Net)")
    log_print(f"  Learning rate: {config['learning_rate']}")

    # Scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=config['reduce_lr_factor'],
        patience=config['reduce_lr_patience'],
        min_lr=config['min_lr']
    )

    # Early stopping
    early_stopping = EarlyStopping(
        patience=config['early_stopping_patience'],
        mode='max'
    )

    history = {
        'train_total_loss': [],
        'train_cls_loss': [],
        'val_total_loss': [],
        'val_cls_loss': [],
        'val_auc': []
    }

    # Training loop
    for epoch in range(config['epochs']):
        # Train
        train_total_loss, train_cls_loss, train_sal_loss, train_metrics = train_one_epoch(
            model, train_loader, cls_criterion, optimizer, device,
            use_saliency_loss=False, print_cm=print_cm
        )

        # Validate (save saliency maps every 10 epochs)
        save_sal = (epoch + 1) % 10 == 0
        val_total_loss, val_cls_loss, val_sal_loss, val_metrics = validate(
            model, val_loader, cls_criterion, device,
            use_saliency_loss=False,
            print_cm=print_cm, save_attention=save_sal, epoch=epoch
        )

        # Update scheduler and early stopping
        scheduler.step(val_metrics['auc'])
        early_stopping(val_metrics['auc'], model)

        # Record history
        history['train_total_loss'].append(train_total_loss)
        history['train_cls_loss'].append(train_cls_loss)
        history['val_total_loss'].append(val_total_loss)
        history['val_cls_loss'].append(val_cls_loss)
        history['val_auc'].append(val_metrics['auc'])

        # Log progress
        log_print(f"\nEpoch [{epoch+1}/{config['epochs']}]")
        log_print(f"  Train Loss: {train_total_loss:.4f} (cls: {train_cls_loss:.4f})")
        log_print(f"  Val Loss: {val_total_loss:.4f} (cls: {val_cls_loss:.4f})")
        log_print(f"  Val AUC: {val_metrics['auc']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
        log_print(f"  Train: Precision={train_metrics['precision']:.4f} | Recall={train_metrics['recall']:.4f} | F1={train_metrics['f1_score']:.4f}")
        log_print(f"  Val:   Precision={val_metrics['precision']:.4f} | Recall={val_metrics['recall']:.4f} | F1={val_metrics['f1_score']:.4f}")

        if early_stopping.early_stop:
            log_print("Early stopping triggered")
            break

    # Load best model
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.load_state_dict(early_stopping.best_model_state)

    # Save checkpoint
    save_path = Path(PathConfig.MODEL_SAVE_DIR) / 'tmfi_embryo_best.pth'
    save_checkpoint({
        'epoch': epoch,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_auc': early_stopping.best_score
    }, save_path)

    # Save history
    history_path = Path(PathConfig.RESULT_DIR) / 'tmfi_embryo_history.json'
    save_history(history, history_path)

    if is_main_process():
        log_print(f"\nTraining completed. Best model saved to: {save_path}")
        log_print(f"Best validation AUC: {early_stopping.best_score:.4f}")

    return model


def main(args):
    """Main training function"""
    create_directories()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_print(f"Using device: {device}")

    # Configuration
    config = {
        # Model
        'num_classes': 2,
        'time_steps': 33,
        'pretrained_path': args.pretrained_swin if args.pretrained_swin else None,

        # Data
        'image_size': 224,
        'batch_size': args.batch_size,
        'num_workers': 4,
        'pin_memory': True,
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'balanced_sampling': True,
        'apply_petri_crop': True,
        'apply_center_crop': True,

        # Training
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'weight_decay': 1e-5,
        'reduce_lr_factor': 0.5,
        'reduce_lr_patience': 5,
        'min_lr': 1e-7,
        'early_stopping_patience': 15,
    }

    log_print("\n" + "="*60)
    log_print("Configuration")
    log_print("="*60)
    for key, value in config.items():
        log_print(f"  {key}: {value}")

    # Load data
    log_print("\n" + "="*60)
    log_print("Loading Multi-Focal Data")
    log_print("="*60)

    data_loader = MultiFocalDataLoader(verbose=True, use_cache=True)
    score_dict = load_score_dict_for_multi_focal(
        maindb_csv=args.maindb,
        filtered_csv=args.filtered if args.filtered else None
    )

    validated_df, embryo_to_label = data_loader.load_multi_focal_data(
        selected_images_csv=args.selected_images,
        score_dict=score_dict,
        skip_validation=args.skip_validation
    )

    # Create data loaders
    train_loader, val_loader, test_loader, labels = create_multi_focal_data_loaders(
        validated_df, embryo_to_label, config, enable_preprocessing=True
    )

    log_print(f"\nData loaders created:")
    log_print(f"  Train: {len(train_loader.dataset)} samples")
    log_print(f"  Val: {len(val_loader.dataset)} samples")
    log_print(f"  Test: {len(test_loader.dataset)} samples")

    # Create model
    log_print("\n" + "="*60)
    log_print("Creating TMFI-Embryo Model")
    log_print("="*60)

    tmfi_model = create_tmfi_embryo_model(config)

    # Wrap with focal fusion
    model = FocalFusionWrapper(
        tmfi_model=tmfi_model,
        fusion_type=args.fusion_type,
        num_focal_planes=11
    ).to(device)

    log_print(f"Model: TMFI-Embryo with FocalFusionWrapper")
    log_print(f"  Fusion type: {args.fusion_type}")

    params = count_parameters(model)
    log_print(f"  Total parameters: {params['total']:,}")
    log_print(f"  Trainable parameters: {params['trainable']:,}")

    # Train
    model = train_tmfi(model, train_loader, val_loader, config, device, print_cm=True)

    # Final test evaluation
    log_print("\n" + "="*60)
    log_print("Final Test Evaluation")
    log_print("="*60)

    test_total_loss, test_cls_loss, test_sal_loss, test_metrics = validate(
        model, test_loader, nn.CrossEntropyLoss(), device,
        use_saliency_loss=False,
        print_cm=True, save_attention=True, epoch=999
    )

    log_print(f"\nTest Results:")
    log_print(f"  Total Loss: {test_total_loss:.4f}")
    log_print(f"  Classification Loss: {test_cls_loss:.4f}")
    log_print(f"  AUC: {test_metrics['auc']:.4f}")
    log_print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    log_print(f"  Precision: {test_metrics['precision']:.4f}")
    log_print(f"  Recall: {test_metrics['recall']:.4f}")
    log_print(f"  F1 Score: {test_metrics['f1_score']:.4f}")

    # Save final model
    model_to_save = model.module if hasattr(model, 'module') else model
    final_path = Path(PathConfig.MODEL_SAVE_DIR) / 'tmfi_embryo_final.pth'
    torch.save(model_to_save.state_dict(), final_path)
    log_print(f"\nFinal model saved to: {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train TMFI-Embryo Model')

    # Data paths
    parser.add_argument('--maindb', type=str, required=True, help='Path to MainDB.csv')
    parser.add_argument('--selected_images', type=str, required=True, help='Path to selected_image_paths.csv')
    parser.add_argument('--filtered', type=str, default=None, help='Path to filtered_clean_embryos.csv (optional)')
    parser.add_argument('--skip_validation', action='store_true', help='Skip validation and use cached results')

    # Model
    parser.add_argument('--pretrained_swin', type=str, default=None, help='Path to pretrained Swin Transformer weights')
    parser.add_argument('--fusion_type', type=str, default='average',
                        choices=['average', 'max', 'middle', 'learned'],
                        help='Focal plane fusion strategy')

    # Training
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')

    args = parser.parse_args()
    main(args)
