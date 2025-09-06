# STORK/VGG16 胚胎质量预测工具使用手册

## 工具简介

本工具基于训练好的STORK/VGG16模型，对胚胎图像进行质量分类预测（Good/Poor）。支持批量处理和可选的Grad-CAM可视化功能。

## 功能特点

- 🔬 **专业预处理**: 使用与训练时一致的VGG16标准预处理
- 🚀 **批量处理**: 支持大规模图像批量预测
- 📊 **详细输出**: 提供概率、标签、置信度等完整信息
- 🎯 **可视化分析**: 可选Grad-CAM热力图生成
- 💾 **结果导出**: CSV格式结果，便于后续分析

## 安装依赖

```bash
pip install tensorflow pandas numpy opencv-python
```

## 使用方法

### 基本用法

```bash
python predict_with_stork_vgg16.py \
    --images-dir /path/to/images \
    --model-path /path/to/vgg16_best_model \
    --output-csv predictions.csv
```

### 完整参数

```bash
python predict_with_stork_vgg16.py \
    --images-dir /path/to/images \
    --model-path /path/to/vgg16_best_model.h5 \
    --output-csv stork_predictions.csv \
    --img-size 224 \
    --batch-size 64 \
    --generate-gradcam \
    --gradcam-dir gradcam_results \
    --sample-size 20
```

## 参数说明

### 必需参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `--images-dir` | string | 图像目录路径，支持递归搜索 |
| `--model-path` | string | 训练好的VGG16模型文件路径 |

### 可选参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--output-csv` | string | `stork_vgg16_predictions.csv` | 输出CSV文件路径 |
| `--img-size` | int | 224 | 输入图像尺寸 |
| `--batch-size` | int | 64 | 批处理大小 |
| `--generate-gradcam` | flag | False | 是否生成Grad-CAM可视化 |
| `--gradcam-dir` | string | `gradcam_results` | Grad-CAM保存目录 |
| `--sample-size` | int | 10 | Grad-CAM样本数量 |

## 输入格式

### 目录结构