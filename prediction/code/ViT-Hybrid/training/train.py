"""
Training Script for ViT-Hybrid Model (PyTorch)
Author: Liao Zizhuo
Date: 2025-10-27
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

from model.model import create_model
from dataset.dataset import create_data_loaders
from utils.utils import (
    EarlyStopping,
    calculate_metrics,
    save_checkpoint,
    save_history,
    create_directories,
    count_parameters,
    print_confusion_matrix,
    print_dataset_distribution
)
from config.config import (
    get_model_config,
    get_training_config,
    get_data_config,
    PathConfig
)


def is_main_process():
    """Check if current process is main process"""
    return not dist.is_initialized() or dist.get_rank() == 0


def log_print(*args, **kwargs):
    """Print only on main process, with flushing"""
    if is_main_process():
        print(*args, **kwargs, flush=True)
        sys.stdout.flush()


def train_one_epoch(model, dataloader, criterion, optimizer, device, print_cm=False):
    """Train one epoch with progress bar

    Args:
        model: The model to train
        dataloader: Training data loader
        criterion: Loss function (CrossEntropyLoss with weights)
        optimizer: Optimizer
        device: Device to run on
        print_cm: Whether to print confusion matrix (default: False)
    """
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    # Only show progress bar on main process
    iterator = tqdm(dataloader, disable=not is_main_process(), desc="Training")

    for batch_idx, (inputs, labels) in enumerate(iterator):
        inputs = inputs.to(device)
        labels = labels.to(device).long()  # Convert to int64 for CrossEntropyLoss

        optimizer.zero_grad()
        logits = model(inputs)  # (batch, 2)
        loss = criterion(logits, labels)

        loss.backward()
        # 梯度裁剪：防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        # Convert logits to probabilities for metrics calculation
        probs = torch.softmax(logits, dim=1)[:, 1]  # Get probability of class 1 (Good)
        all_preds.extend(probs.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

        # Update progress bar with loss
        iterator.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(dataloader)
    all_preds_flat = np.array(all_preds).flatten()
    all_labels_flat = np.array(all_labels).flatten()
    metrics = calculate_metrics(all_labels_flat, all_preds_flat)

    if print_cm and is_main_process():
        print_confusion_matrix(all_labels_flat, all_preds_flat)

    return avg_loss, metrics


def validate(model, dataloader, criterion, device, print_cm=False):
    """Validate with progress bar

    Args:
        model: The model to validate
        dataloader: Validation data loader
        criterion: Loss function (CrossEntropyLoss with weights)
        device: Device to run on
        print_cm: Whether to print confusion matrix (default: False)
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    # Only show progress bar on main process
    iterator = tqdm(dataloader, disable=not is_main_process(), desc="Validating")

    with torch.no_grad():
        for inputs, labels in iterator:
            inputs = inputs.to(device)
            labels = labels.to(device).long()  # Convert to int64 for CrossEntropyLoss

            logits = model(inputs)  # (batch, 2)
            loss = criterion(logits, labels)

            total_loss += loss.item()

            # Convert logits to probabilities for metrics calculation
            probs = torch.softmax(logits, dim=1)[:, 1]  # Get probability of class 1 (Good)
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Update progress bar
            iterator.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(dataloader)
    all_preds_flat = np.array(all_preds).flatten()
    all_labels_flat = np.array(all_labels).flatten()

    # Debug: Print probability statistics
    if print_cm and is_main_process():
        log_print(f"\n  Probability statistics (class 1 - Good):")
        log_print(f"    Mean: {all_preds_flat.mean():.4f}, Std: {all_preds_flat.std():.4f}")
        log_print(f"    Min: {all_preds_flat.min():.4f}, Max: {all_preds_flat.max():.4f}")
        log_print(f"    > 0.5: {(all_preds_flat > 0.5).sum()} / {len(all_preds_flat)}")

    metrics = calculate_metrics(all_labels_flat, all_preds_flat)

    if print_cm and is_main_process():
        print_confusion_matrix(all_labels_flat, all_preds_flat)

    return avg_loss, metrics


def train_stage1(model, train_loader, val_loader, config, device, print_cm=True, train_labels=None, freeze_vgg16=True):
    """Stage 1: Train Transformer (VGG16 frozen or with differential lr)

    Args:
        model: The model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        device: Device to run on
        print_cm: Whether to print confusion matrices (default: True)
        train_labels: Optional tensor of training labels for class weight calculation
        freeze_vgg16: If True, freeze VGG16; if False, use differential learning rates (default: True)
    """
    # Handle DataParallel wrapper
    model_for_params = model.module if hasattr(model, 'module') else model

    if freeze_vgg16:
        log_print("\n" + "="*60)
        log_print("Stage 1: Training Transformer (VGG16 frozen)")
        log_print("="*60)
    else:
        log_print("\n" + "="*60)
        log_print("End-to-End Training (VGG16 with differential learning rates)")
        log_print("="*60)

    # Print data statistics
    if train_labels is not None:
        train_labels_np = train_labels.cpu().numpy() if isinstance(train_labels, torch.Tensor) else train_labels
        num_class_0 = (train_labels_np == 0).sum()
        num_class_1 = (train_labels_np == 1).sum()
        total = len(train_labels_np)

        log_print(f"\nClass distribution in training set:")
        log_print(f"  Class 0 (Poor): {num_class_0} samples ({num_class_0/total*100:.1f}%)")
        log_print(f"  Class 1 (Good): {num_class_1} samples ({num_class_1/total*100:.1f}%)")

    # 初始化损失函数：使用不加权的CrossEntropyLoss（依赖数据加载器的平衡采样）
    log_print(f"\n✓ Using unweighted CrossEntropyLoss with balanced sampling")
    criterion = nn.CrossEntropyLoss()

    # 根据freeze_vgg16决定优化器配置
    if freeze_vgg16:
        # Stage 1策略：冻结VGG16前1/3，解冻后2/3
        log_print(f"  Applying partial freezing strategy to VGG16...")

        # 收集VGG16所有有参数的层
        vgg_features = model_for_params.vgg16_extractor.features
        vgg_layers_with_params = [
            (i, layer) for i, layer in enumerate(vgg_features.children())
            if len(list(layer.parameters())) > 0
        ]
        total_param_layers = len(vgg_layers_with_params)
        split_point = int(total_param_layers * 1/3)  # 前1/3冻结

        # 冻结前1/3的有参数层
        for idx, (i, layer) in enumerate(vgg_layers_with_params):
            if idx < split_point:
                for param in layer.parameters():
                    param.requires_grad = False
            else:
                for param in layer.parameters():
                    param.requires_grad = True

        # VGG16的classifier部分也解冻
        for param in model_for_params.vgg16_extractor.classifier.parameters():
            param.requires_grad = True

        log_print(f"  VGG16 freezing strategy:")
        log_print(f"    Total VGG16 param layers: {total_param_layers}")
        log_print(f"    Frozen param layers: {split_point} (first 1/3)")
        log_print(f"    Trainable param layers: {total_param_layers - split_point} (last 2/3)")
        log_print(f"    VGG16 classifier: Trainable")

        optimizer = optim.AdamW(
            model_for_params.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        log_print(f"  Learning rate: {config['learning_rate']}")
    else:
        # 端到端训练：部分冻结VGG16（只解冻顶层），减少显存占用
        log_print(f"  Partial unfreezing VGG16 for end-to-end training (top layers only)...")

        # VGG16分层：前67%保持冻结，后33%解冻
        vgg_features = model_for_params.vgg16_extractor.features
        vgg_layers = list(vgg_features.children())
        total_vgg_layers = len(vgg_layers)
        split_point = int(total_vgg_layers * 0.67)  # 前67%冻结

        # 冻结VGG16底层（前67%）
        for layer in vgg_layers[:split_point]:
            for param in layer.parameters():
                param.requires_grad = False

        # 解冻VGG16顶层（后33%）
        vgg_top_params = []
        for layer in vgg_layers[split_point:]:
            for param in layer.parameters():
                param.requires_grad = True
            vgg_top_params.extend(layer.parameters())

        # 收集Transformer参数
        transformer_params = []
        for block in model_for_params.transformer_blocks:
            transformer_params.extend(block.parameters())
        transformer_params.extend(model_for_params.temporal_pooling.parameters())

        # 收集分类器参数
        classifier_params = [
            *model_for_params.classifier_norm.parameters(),
            *model_for_params.classifier_fc1.parameters(),
            *model_for_params.classifier_fc2.parameters()
        ]

        # 优化器：只包含可训练参数（VGG16顶层+Transformer+Classifier）
        # 使用更大的学习率避免loss停滞
        optimizer = optim.AdamW([
            {'params': vgg_top_params, 'lr': 5e-5, 'weight_decay': 1e-5},        # VGG16顶层 (增大5倍)
            {'params': transformer_params, 'lr': 3e-4, 'weight_decay': 1e-5},    # Transformer (增大3倍)
            {'params': classifier_params, 'lr': 1e-3, 'weight_decay': 1e-5}      # 分类器 (增大2倍)
        ])

        log_print(f"  Partial freezing strategy:")
        log_print(f"    VGG16 bottom ({split_point} layers): FROZEN (requires_grad=False)")
        log_print(f"    VGG16 top ({total_vgg_layers-split_point} layers): lr=5e-5")
        log_print(f"    Transformer: lr=3e-4")
        log_print(f"    Classifier: lr=1e-3")
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=config['reduce_lr_factor'],
        patience=config['reduce_lr_patience'],
        min_lr=config['min_lr']
    )
    early_stopping = EarlyStopping(
        patience=config['early_stopping_patience'],
        mode='max'
    )

    history = {'train_loss': [], 'val_loss': [], 'val_auc': []}

    # 根据freeze_vgg16决定训练epochs数
    # freeze_vgg16=True: 只训练一半epochs (stage1)
    # freeze_vgg16=False: 训练全部epochs (end-to-end)
    num_epochs = config['epochs'] // 2 if freeze_vgg16 else config['epochs']

    for epoch in range(num_epochs):
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, print_cm=print_cm
        )
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device, print_cm=print_cm
        )

        scheduler.step(val_metrics['auc'])
        early_stopping(val_metrics['auc'], model)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_metrics['auc'])

        log_print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        log_print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        log_print(f"  Val AUC: {val_metrics['auc']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
        log_print(f"  Train Metrics: Precision={train_metrics['precision']:.4f} | Recall={train_metrics['recall']:.4f} | F1={train_metrics['f1_score']:.4f}")
        log_print(f"  Val Metrics:   Precision={val_metrics['precision']:.4f} | Recall={val_metrics['recall']:.4f} | F1={val_metrics['f1_score']:.4f}")

        if early_stopping.early_stop:
            log_print("Early stopping triggered")
            break

    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.load_state_dict(early_stopping.best_model_state)

    # 根据训练模式保存不同的模型文件名
    save_path = Path(PathConfig.MODEL_SAVE_DIR) / ('end_to_end_best.pth' if not freeze_vgg16 else 'stage1_best.pth')
    save_checkpoint({
        'epoch': epoch,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, save_path)

    # 根据训练模式保存不同的历史文件名
    history_path = Path(PathConfig.RESULT_DIR) / ('end_to_end_history.json' if not freeze_vgg16 else 'stage1_history.json')
    save_history(history, history_path)

    if is_main_process():
        if freeze_vgg16:
            log_print(f"\nStage 1 completed. Best model saved to: {save_path}")
        else:
            log_print(f"\nEnd-to-end training completed. Best model saved to: {save_path}")

    return model


def train_stage2(model, train_loader, val_loader, config, device, print_cm=True, train_labels=None):
    """Stage 2: Fine-tune with VGG16 unfrozen

    Args:
        model: The model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        device: Device to run on
        print_cm: Whether to print confusion matrices (default: True)
        train_labels: Optional tensor of training labels for class weight calculation
    """
    log_print("\n" + "="*60)
    log_print("Stage 2: Fine-tuning with VGG16 fully unfrozen")
    log_print("="*60)

    # Handle DataParallel wrapper
    model_for_params = model.module if hasattr(model, 'module') else model

    # 完全解冻VGG16所有层（包括features和classifier）
    for param in model_for_params.vgg16_extractor.parameters():
        param.requires_grad = True

    log_print("  VGG16: All layers unfrozen for fine-tuning")

    # Print data statistics
    if train_labels is not None:
        train_labels_np = train_labels.cpu().numpy() if isinstance(train_labels, torch.Tensor) else train_labels
        num_class_0 = (train_labels_np == 0).sum()
        num_class_1 = (train_labels_np == 1).sum()
        total = len(train_labels_np)

        log_print(f"\nClass distribution in training set:")
        log_print(f"  Class 0 (Poor): {num_class_0} samples ({num_class_0/total*100:.1f}%)")
        log_print(f"  Class 1 (Good): {num_class_1} samples ({num_class_1/total*100:.1f}%)")

    # 初始化损失函数：使用不加权的CrossEntropyLoss（依赖数据加载器的平衡采样）
    log_print(f"\n✓ Using unweighted CrossEntropyLoss with balanced sampling")
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model_for_params.parameters(),
        lr=config['learning_rate'] * 0.1,
        weight_decay=config['weight_decay']
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=config['reduce_lr_factor'],
        patience=config['reduce_lr_patience'],
        min_lr=config['min_lr']
    )
    early_stopping = EarlyStopping(
        patience=config['early_stopping_patience'],
        mode='max'
    )

    history = {'train_loss': [], 'val_loss': [], 'val_auc': []}

    for epoch in range(config['epochs'] // 2):
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, print_cm=print_cm
        )
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device, print_cm=print_cm
        )

        scheduler.step(val_metrics['auc'])
        early_stopping(val_metrics['auc'], model)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_metrics['auc'])

        log_print(f"\nEpoch [{epoch+1}/{config['epochs']//2}]")
        log_print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        log_print(f"  Val AUC: {val_metrics['auc']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
        log_print(f"  Train Metrics: Precision={train_metrics['precision']:.4f} | Recall={train_metrics['recall']:.4f} | F1={train_metrics['f1_score']:.4f}")
        log_print(f"  Val Metrics:   Precision={val_metrics['precision']:.4f} | Recall={val_metrics['recall']:.4f} | F1={val_metrics['f1_score']:.4f}")

        if early_stopping.early_stop:
            log_print("Early stopping triggered")
            break

    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.load_state_dict(early_stopping.best_model_state)

    save_checkpoint({
        'epoch': epoch,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, Path(PathConfig.MODEL_SAVE_DIR) / 'stage2_best.pth')

    save_history(history, Path(PathConfig.RESULT_DIR) / 'stage2_history.json')

    if is_main_process():
        log_print(f"\nStage 2 completed. Best model saved to: {Path(PathConfig.MODEL_SAVE_DIR) / 'stage2_best.pth'}")

    return model


def main(args):
    """Main training function"""
    create_directories()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_print(f"Using device: {device}")

    model_config = get_model_config()
    training_config = get_training_config()
    data_config = get_data_config()

    if args.data_root:
        data_config['data_root'] = args.data_root

    log_print("\nCreating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        data_root=data_config['data_root'],
        config=data_config
    )

    # Print dataset distribution
    if is_main_process():
        print_dataset_distribution(train_loader, val_loader, test_loader)

    # Extract training labels for class weight calculation
    log_print("\nExtracting training labels for class weight calculation...")
    train_labels_list = []
    for _, labels in train_loader:
        train_labels_list.extend(labels.numpy())
    train_labels = torch.tensor(train_labels_list, dtype=torch.float32)
    log_print(f"Extracted {len(train_labels)} training labels")

    log_print("\nBuilding model...")
    model = create_model(model_config).to(device)

    params = count_parameters(model)
    log_print(f"Total parameters: {params['total']:,}")
    log_print(f"Trainable parameters: {params['trainable']:,}")

    if args.two_stage:
        model = train_stage1(model, train_loader, val_loader, training_config, device, train_labels=train_labels)
        model = train_stage2(model, train_loader, val_loader, training_config, device, train_labels=train_labels)
    else:
        log_print("\n" + "="*60)
        log_print("Single Stage Training")
        log_print("="*60)
        model = train_stage1(model, train_loader, val_loader, training_config, device, train_labels=train_labels)

    log_print("\nTraining completed")

    # Handle DataParallel wrapper for saving final model
    model_to_save = model.module if hasattr(model, 'module') else model
    final_path = Path(PathConfig.MODEL_SAVE_DIR) / 'final_model.pth'
    torch.save(model_to_save.state_dict(), final_path)
    log_print(f"Final model saved to: {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ViT-Hybrid Model')
    parser.add_argument('--data_root', type=str, help='Path to data root directory')
    parser.add_argument('--two_stage', action='store_true', help='Use two-stage training')

    args = parser.parse_args()
    main(args)
