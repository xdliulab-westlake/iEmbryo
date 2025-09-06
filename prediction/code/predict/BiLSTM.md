---

## 2. BiLSTM 预测脚本

- 工具文件: `predict_with_bilstm.py`
- 用途: 从胚胎图像目录自动构建特征，使用训练好的 BiLSTM 模型预测 BS 分数，并结合年龄信息通过逻辑回归模型预测胚胎质量。

### 输入格式

#### 必需参数
- `--input-csv`: 输入CSV文件路径，包含两列：
  - 第一列：胚胎目录路径（每个胚胎一个文件夹，包含所有焦平面图像）
  - 第二列：病人年龄
- `--scaler-path`: 训练好的 `StandardScaler`（joblib 文件）
- `--logreg-path`: 训练好的逻辑回归模型（joblib 文件）

#### 可选参数
- `--model-path`: BiLSTM 模型路径，默认 `bilstm_quality_model.h5`
- `--output-csv`: 输出CSV路径，默认 `bilstm_predictions.csv`
- `--batch-size`: 推理批量大小，默认 16
- `--img-size`: 图像预处理尺寸，默认 224

### 输入CSV格式示例

```csv
embryo_path,age
/path/to/embryo1_folder,28
/path/to/embryo2_folder,32
/path/to/embryo3_folder,35
/path/to/embryo4_folder,29
```

### 胚胎目录结构要求

每个胚胎目录应包含按以下命名规则的图像文件：
- 文件名格式：`F{focal_plane}_{time_point}_{其他信息}.{扩展名}`
- 焦平面：F-75, F-60, F-45, F-30, F-15, F0, F15, F30, F45, F60, F75
- 时间点：96.0, 96.5, 97.0, ..., 112.0（每0.5小时一个时间点）

示例文件名：
F-15_96.0_D2023.08.05_S00886_I4424_P_WELL6_RUN466.jpeg
F-15_96.5_D2023.08.05_S00886_I4424_P_WELL6_RUN468.jpeg
F-15_97.0_D2023.08.05_S00886_I4424_P_WELL6_RUN471.jpeg
F-15_97.5_D2023.08.05_S00886_I4424_P_WELL6_RUN474.jpeg
F-15_98.0_D2023.08.05_S00886_I4424_P_WELL6_RUN477.jpeg


### 输出格式（CSV）

- `EmbryoID`: 胚胎ID（从目录名提取）
- `Age`: 病人年龄
- `bs_pred`: BiLSTM预测的BS分数（回归）
- `ValidImages`: 有效图像数量
- `prob_good`: 预测为Good胚胎的概率
- `prob_poor`: 预测为Poor胚胎的概率
- `pred_label`: 二分类标签（1=Good, 0=Poor）
- `predicted_quality`: 预测质量（Good/Poor）

### 使用示例

```bash
python predict_with_bilstm.py \
  --input-csv embryos_input.csv \
  --model-path bilstm_quality_model.h5 \
  --scaler-path scaler.joblib \
  --logreg-path logistic_regression.joblib \
  --output-csv bilstm_predictions.csv
```

### 工作流程

1. **加载输入CSV**: 读取胚胎目录路径和年龄信息
2. **图像扫描**: 自动扫描每个胚胎目录，解析图像文件名
3. **特征构建**: 使用多模型特征提取器（EfficientNetV2-M, ResNet152V2, DenseNet201, VGG16）
4. **特征张量**: 构建 [N, 33, 11×2048] 的特征张量
5. **BiLSTM预测**: 预测BS分数
6. **逻辑回归**: 结合年龄和BS分数预测胚胎质量
7. **结果输出**: 保存完整的预测结果到CSV

### 特征提取详情

- **多模型融合**: 4个预训练模型，每个降维到512维
- **总特征维度**: 2048维（4×512）
- **时序长度**: 33个时间点（96-112小时，每0.5小时）
- **焦平面数**: 11个焦平面
- **最终张量**: [N, 33, 11×2048] = [N, 33, 22528]

### 模型要求

- **BiLSTM模型**: 必须与训练时的输入维度匹配（11×feature_dim）
- **StandardScaler**: 用于特征标准化，必须与训练时使用的一致
- **LogisticRegression**: 用于质量分类，必须与训练时使用的一致

### 常见问题

- **问**: 图像数量不足怎么办？
  - **答**: 脚本要求每个胚胎至少50张有效图像，否则跳过该胚胎
  
- **问**: 文件名格式不匹配怎么办？
  - **答**: 脚本会显示警告并跳过无法解析的图像，确保文件名符合格式要求
  
- **问**: 如何确保模型一致性？
  - **答**: 使用训练时保存的scaler和logreg文件，确保特征处理和分类逻辑一致

### 性能优化

- **GPU支持**: 自动检测GPU并启用内存增长
- **批量处理**: 支持自定义批量大小，优化内存使用
- **特征缓存**: 多模型特征提取支持批量处理，提高效率

---

## 3. 模型路径与一致性建议

- **BiLSTM模型**: 训练时使用 `ModelCheckpoint` 保存最佳模型为 `.h5` 格式
- **StandardScaler**: 训练时保存为 `scaler.joblib`
- **LogisticRegression**: 训练时保存为 `logreg.joblib`
- **特征维度**: 确保推理时的特征维度与训练时完全一致

---

## 4. 注意事项

- 脚本会自动跳过无效的胚胎目录和图像文件
- 年龄信息必须为数值类型
- 图像文件支持常见格式：jpg, jpeg, png, bmp, tiff
- 建议在运行前检查输入CSV的格式和路径有效性
- 确保所有必需的模型文件（BiLSTM、scaler、logreg）都存在且可访问

---

## 5. 输出结果分析

脚本运行完成后会显示：

- **预测摘要**: 总胚胎数、BS分数范围、平均分数
- **质量分布**: Good/Poor胚胎的数量和比例
- **年龄相关性**: 年龄与BS分数的相关性分析
- **年龄分组统计**: 按年龄组（<30, 30-35, 35-40, >40）的统计信息

这些信息有助于分析预测结果的合理性和临床应用价值。