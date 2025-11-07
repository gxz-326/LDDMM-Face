# AFLFP (Facial Palsy Classification and Grading) 模型集成指南

## 概述

本指南说明如何将LDDMM-Face模型迁移到AFLFP（面瘫识别和分级）数据集上。

## 修改内容总结

### 1. 新增AFLFP数据集类 (`lib/datasets/aflfp.py`)
- 实现了`AFLFP`数据集类，继承自`torch.utils.data.Dataset`
- 支持CSV格式的标注文件
- 集成了面瘫分级标签（`palsy_grade`）到元数据中
- 支持数据增强（缩放、旋转、翻转）
- 生成关键点的高斯热力图作为目标

### 2. 数据集注册 (`lib/datasets/__init__.py`)
- 导入新的AFLFP类
- 在`get_dataset()`和`get_testset()`函数中添加AFLFP条件判断

### 3. 关键点翻转映射 (`lib/utils/transforms.py`)
- 添加AFLFP的关键点对称对应关系
- 使用68个关键点的标准对应关系（与300W相同）

### 4. 实验配置文件 (`experiments/aflfp/face_alignment_aflfp_hrnet_w18.yaml`)
- 新建AFLFP特定的训练配置
- 配置为68个关键点
- 设置数据路径和训练超参数

## 使用步骤

### 第一步：准备数据

将AFLFP数据组织为以下结构：

```
data/aflfp/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── face_landmarks_aflfp_train.csv
└── face_landmarks_aflfp_test.csv
```

### 第二步：CSV格式说明

CSV文件应包含以下列（以空格或逗号分隔）：

```
image_path  scale  box_size  center_x  center_y  x1  y1  x2  y2  ...  x68  y68  palsy_grade
```

其中：
- `image_path`: 相对于ROOT的图像路径
- `scale`: 图像缩放因子（通常为1.0-2.0）
- `box_size`: 边界框大小
- `center_x, center_y`: 人脸中心坐标
- `x1, y1, ..., x68, y68`: 68个面部关键点坐标
- `palsy_grade`: 面瘫分级标签（0-6，对应House-Brackmann scale等）

示例：
```
img001.jpg 1.5 300 256 256 100 50 120 55 130 60 ... 200 300 2
img002.jpg 1.4 310 260 260 105 55 125 60 135 65 ... 205 305 1
```

### 第三步：训练模型

运行以下命令开始训练：

```bash
python tools/train.py --cfg experiments/aflfp/face_alignment_aflfp_hrnet_w18.yaml
```

或修改配置文件中的数据路径后运行：

```bash
python tools/train.py --cfg experiments/aflfp/face_alignment_aflfp_hrnet_w18.yaml \
    DATASET.ROOT /path/to/data/aflfp/images/ \
    DATASET.TRAINSET /path/to/data/aflfp/face_landmarks_aflfp_train.csv \
    DATASET.TESTSET /path/to/data/aflfp/face_landmarks_aflfp_test.csv
```

### 第四步：测试模型

```bash
python tools/test.py --cfg experiments/aflfp/face_alignment_aflfp_hrnet_w18.yaml \
    TEST.DATASET AFLFP \
    MODEL.PRETRAINED path/to/checkpoint.pth
```

## 配置参数说明

在 `experiments/aflfp/face_alignment_aflfp_hrnet_w18.yaml` 中可以调整：

- `MODEL.NUM_JOINTS`: 关键点数量（默认68）
- `TRAIN.END_EPOCH`: 训练轮数（默认100）
- `TRAIN.LR`: 学习率（默认0.0002）
- `TRAIN.LR_STEP`: 学习率衰减步数
- `TRAIN.BATCH_SIZE_PER_GPU`: 每GPU批大小
- `DATASET.SCALE_FACTOR`: 数据增强缩放范围（默认0.25）
- `DATASET.ROT_FACTOR`: 数据增强旋转范围（默认30）

## 整合面瘫分级分类

若要同时进行面瘫分级分类，可以：

1. **修改模型头部**：在`lib/models/`中的模型文件中添加分级分类分支
2. **修改损失函数**：在`lib/core/evaluation.py`中组合关键点定位损失和分类损失
3. **修改训练循环**：在`tools/train.py`中处理分级标签的监督信号

示例修改思路：
```python
# 在模型输出中添加分级预测头
logits = model(images)  # [batch, 68, H, W] for heatmaps
grade_pred = classifier_head(features)  # [batch, num_grades] for palsy grading

# 组合损失
landmark_loss = criterion_landmark(logits, targets)
grade_loss = criterion_grade(grade_pred, palsy_grades)
total_loss = landmark_loss + alpha * grade_loss
```

## 关键点定义

AFLFP使用68个面部关键点（与300-W相同）：
- 轮廓: 1-17 (17个点)
- 眉毛: 18-27 (10个点)
- 鼻子: 28-36 (9个点)
- 眼睛: 37-48 (12个点)
- 嘴巴: 49-68 (20个点)

## 常见问题

### Q: 数据增强是否会影响面瘫的侧面性判断？
A: 是的，在应用翻转增强时需要谨慎。可以在配置中设置 `DATASET.FLIP: false` 以禁用翻转，或在数据预处理中标记需要特殊处理的样本。

### Q: 如何处理不同的关键点数量？
A: 修改配置中的 `MODEL.NUM_JOINTS` 和对应的CSV列数即可。代码会自动调整。

### Q: 能否结合其他损失函数进行分级？
A: 可以。修改`lib/core/function.py`中的训练函数，添加额外的损失项，然后组合两个损失进行反向传播。

## 相关文件

- AFLFP数据集实现: `lib/datasets/aflfp.py`
- 数据集注册: `lib/datasets/__init__.py`
- 关键点映射: `lib/utils/transforms.py`
- 训练脚本: `tools/train.py`
- 配置文件: `experiments/aflfp/face_alignment_aflfp_hrnet_w18.yaml`

## 预期性能指标

- 关键点定位精度 (NME): 预期 < 5%
- 推理速度: 约 30-50 FPS (GPU)
- 内存占用: 约 2-3 GB (VRAM) per GPU

## 引用

如使用此实现，请参考原始LDDMM-Face论文和AFLFP数据集的相关文献。
