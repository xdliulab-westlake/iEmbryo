"""
Training Script for ViT-Hybrid Model (PyTorch)
Author: Liao Zizhuo
Date: 2025-10-27
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
import argparse
import numpy as np

from model import create_model
from dataset import create_data_loaders
from utils import (
    FocalLoss,
    EarlyStopping,
    calculate_metrics,
    save_checkpoint,
    save_history,
    create_directories,
    count_parameters
)
from config import (
    get_model_config,
    get_training_config,
    get_data_config,
    PathConfig
)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(outputs.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    metrics = calculate_metrics(np.array(all_labels), np.array(all_preds))

    return avg_loss, metrics


def validate(model, dataloader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    metrics = calculate_metrics(np.array(all_labels), np.array(all_preds))

    return avg_loss, metrics


def train_stage1(model, train_loader, val_loader, config, device):
    """
    第一阶段训练: 冻结VGG16
    """
    print("\n" + "="*60)
    print("Stage 1: Training Transformer (VGG16 frozen)")
    print("="*60)

    criterion = nn.BCELoss() if not config.get('use_focal_loss') else FocalLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
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
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device
        )

        scheduler.step(val_metrics['auc'])
        early_stopping(val_metrics['auc'], model)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_metrics['auc'])

        print(f"Epoch [{epoch+1}/{config['epochs']//2}]")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Val AUC: {val_metrics['auc']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")

        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    model.load_state_dict(early_stopping.best_model_state)
    save_checkpoint({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, Path(PathConfig.MODEL_SAVE_DIR) / 'stage1_best.pth')

    save_history(history, Path(PathConfig.RESULT_DIR) / 'stage1_history.json')

    return model


def train_stage2(model, train_loader, val_loader, config, device):
    """
    第二阶段训练: 解冻VGG16
    """
    print("\n" + "="*60)
    print("Stage 2: Fine-tuning with VGG16 unfrozen")
    print("="*60)

    model.unfreeze_vgg16()

    criterion = nn.BCELoss()
    optimizer = optim.AdamW(
        model.parameters(),
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
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device
        )

        scheduler.step(val_metrics['auc'])
        early_stopping(val_metrics['auc'], model)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_metrics['auc'])

        print(f"Epoch [{epoch+1}/{config['epochs']//2}]")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Val AUC: {val_metrics['auc']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")

        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    model.load_state_dict(early_stopping.best_model_state)
    save_checkpoint({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, Path(PathConfig.MODEL_SAVE_DIR) / 'stage2_best.pth')

    save_history(history, Path(PathConfig.RESULT_DIR) / 'stage2_history.json')

    return model


def main(args):
    """主训练函数"""
    create_directories()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model_config = get_model_config()
    training_config = get_training_config()
    data_config = get_data_config()

    if args.data_root:
        data_config['data_root'] = args.data_root
    if args.use_focal_loss:
        training_config['use_focal_loss'] = True

    print("\nCreating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        data_root=data_config['data_root'],
        config=data_config
    )

    print("\nBuilding model...")
    model = create_model(model_config).to(device)

    params = count_parameters(model)
    print(f"Total parameters: {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")

    if args.two_stage:
        model = train_stage1(model, train_loader, val_loader, training_config, device)
        model = train_stage2(model, train_loader, val_loader, training_config, device)
    else:
        print("\n" + "="*60)
        print("Single Stage Training")
        print("="*60)
        model = train_stage1(model, train_loader, val_loader, training_config, device)

    print("\nTraining completed")
    final_path = Path(PathConfig.MODEL_SAVE_DIR) / 'final_model.pth'
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved to: {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ViT-Hybrid Model')
    parser.add_argument('--data_root', type=str, help='Path to data root directory')
    parser.add_argument('--two_stage', action='store_true', help='Use two-stage training')
    parser.add_argument('--use_focal_loss', action='store_true', help='Use Focal Loss')

    args = parser.parse_args()
    main(args)
