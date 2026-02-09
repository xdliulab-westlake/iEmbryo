# iEmbryo - Intelligent Embryo Quality Assessment System

[![Website](https://img.shields.io/badge/Website-iEmbryo-blue)](https://iembryo.westlake.edu.cn/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Deep Learning-Based IVF Embryo Quality Analysis Platform

---

## 🔬 Overview

**iEmbryo** is an advanced AI-powered platform for embryo quality assessment and selection in reproductive medicine. Leveraging state-of-the-art computer vision and deep learning technologies, iEmbryo provides precise and efficient embryo quality assessment solutions to support clinical decision-making in IVF procedures.

---

## 🎯 Main Features

### 1. Quality Detection

![Quality Detection](Quality%20Detection.jpg)

An intelligent embryo quality assessment system that combines:

- **VGG16 Deep Network**: CNN-based image feature extraction model for comprehensive morphological analysis
- **BiLSTM Temporal Analysis**: Bidirectional LSTM network for capturing temporal features across embryo development stages

**Capabilities:**
- Single file, batch, or folder upload support
- Multiple model architectures (VGG16, ResNet, DenseNet, InceptionV3)
- Real-time intelligent analysis
- Attention heatmap generation for interpretability
- Local data processing ensuring privacy and security

### 2. Embryo Selection & Pregnancy Rate Prediction

![Embryo Selection](Embryo%20Selection.jpg)

AI-based embryo optimization and clinical decision support system featuring:

**Three-Step Workflow:**
1. **Data Import**: Upload embryo folders with time-lapse images
2. **Intelligent Ranking**: Scientific embryo ranking based on quality scores
3. **Pregnancy Rate Prediction**: Predictive analytics for optimal clinical outcomes

---

## 📁 Project Structure

```
IVF_AI/
├── prediction/
│   └── code/
│       ├── iEmbryo/                  # Main implementation (iEmbryo core)
│       │   ├── config/               # Configuration management
│       │   ├── data/                 # Data loading and preprocessing
│       │   ├── models/               # Neural network architectures
│       │   ├── training/             # Training pipeline
│       │   ├── evaluation/           # Model evaluation and metrics
│       │   ├── visualization/        # GradCAM and feature visualization
│       │   ├── utils/                # Utility functions
│       │   ├── embryo_info/          # Embryo analysis tools
│       │   ├── outputs/              # Training outputs and results
│       │   └── main.py               # Main entry point
│       │
│       ├── BiLSTM/                   # Temporal analysis module
│       │   ├── model.py              # BiLSTM architecture
│       │   ├── preprocessing/        # Time-lapse data preprocessing
│       │   └── main.py               # BiLSTM training
│       │
│       └── iEmbryo.py                # iEmbryo main script
│
├── database/                         # Database management
├── Figures/                          # Result visualizations
└── README.md                         # This file
```

---

## 🏗️ Model Architectures

### Supported Backbones

- **VGG16** (Recommended): Optimal balance of accuracy and speed
- **ResNet50/152**: Deep residual networks for complex feature extraction
- **DenseNet121/169/201**: Dense connections for feature reuse
- **InceptionV3**: Multi-scale feature extraction
- **Vision Transformer (ViT)**: Attention-based architecture

### Training Features

- Multi-GPU distributed training support
- Mixed precision training (FP16)
- Advanced data augmentation
- Class imbalance handling (Focal Loss)
- Cross-validation framework
- Model checkpointing and early stopping

---

## 📊 Evaluation Metrics

The system evaluates models using comprehensive metrics:

- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1-Score**: Per-class performance
- **AUC-ROC**: Area under the receiver operating characteristic curve
- **Confusion Matrix**: Detailed classification breakdown
- **Embryo-level Metrics**: Aggregated predictions per embryo

---

## 🔍 Visualization Tools

### GradCAM Attention Maps

Generate interpretable attention heatmaps to understand model decisions:

```bash
python visualization/gradcam.py --model_path path/to/model.pth --image_path path/to/image.jpg
```

### Feature Space Analysis

Visualize learned embeddings using PCA/t-SNE:

```bash
python unified_pca_analysis.py --backbone vgg16
```

---

## 🌐 Web Platform

Visit our online platform: **[https://iembryo.westlake.edu.cn/](https://iembryo.westlake.edu.cn/)**

The web interface provides:
- User-friendly embryo image upload
- Real-time quality assessment
- Interactive result visualization
- Batch processing capabilities
- Secure local data processing

---

## 📖 Citation

If you use iEmbryo in your research, please cite:

```bibtex
@article{iembryo2026,
  title={iEmbryo: Intelligent Embryo Quality Assessment System},
  author={Your Name},
  journal={Journal Name},
  year={2026}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📧 Contact

For questions or collaboration inquiries:
- Website: [https://iembryo.westlake.edu.cn/](https://iembryo.westlake.edu.cn/)
- Email: liaozizhuo@westlake.edu.cn

---

## 🙏 Acknowledgments

This work was supported by Westlake University, Liuxiaodong Lab.

Special thanks to all contributors and the reproductive medicine community for their valuable feedback.

---

**Note**: This system is designed for research purposes and clinical decision support. All clinical applications should be supervised by qualified medical professionals.
