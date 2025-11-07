# AFLFP (Facial Palsy Classification and Grading) 模型集成 - 实现总结

## 概述

本项目已成功集成AFLFP数据集支持，用于面瘫识别和分级任务。所有修改都保持与现有LDDMM-Face架构的兼容性。

## 📋 修改文件清单

### 1. **核心数据集模块** 
#### 新增文件
- `lib/datasets/aflfp.py` - AFLFP数据集类实现
  - 支持CSV格式的标注
  - 集成面瘫分级标签
  - 支持数据增强（缩放、旋转、翻转）
  - 生成关键点的高斯热力图

#### 修改文件
- `lib/datasets/__init__.py` - 注册AFLFP数据集
  - 导入AFLFP类
  - 在`get_dataset()`添加AFLFP支持
  - 在`get_testset()`添加AFLFP支持

### 2. **数据预处理和变换**
#### 修改文件
- `lib/utils/transforms.py` - 添加AFLFP关键点翻转映射
  - AFLFP使用68个标准面部关键点（与300W相同）
  - 定义了关键点的对称对应关系

### 3. **实验配置**
#### 新增文件
- `experiments/aflfp/face_alignment_aflfp_hrnet_w18.yaml` - AFLFP训练配置
  - 配置为68个关键点
  - HRNetV2-W18主干网络
  - 针对AFLFP优化的超参数
  - 100个epoch的训练循环

### 4. **工具和辅助功能**
#### 新增文件
- `tools/prepare_aflfp_data.py` - 数据集准备工具
  - CSV格式验证
  - 模板生成
  - 数据格式转换示例

- `lib/models/palsy_grading.py` - (可选)面瘫分级模块
  - `PalsyGradingHead` - 分级分类头
  - `DualHeadModel` - 联合的关键点+分级模型
  - `PalsyGradingLoss` - 联合损失函数
  - 示例训练循环

### 5. **文档**
#### 新增文件
- `AFLFP_GUIDE.md` - 详细使用指南
- `AFLFP_IMPLEMENTATION_SUMMARY.md` - 本文件

## 🚀 快速开始

### 第一步：准备数据

```bash
# 1. 创建数据目录
mkdir -p data/aflfp/images

# 2. 将图像放在 data/aflfp/images/
# 3. 创建标注CSV文件
```

### 第二步：创建CSV文件

CSV格式（空格分隔）：
```
image_path scale box_size center_x center_y x1 y1 x2 y2 ... x68 y68 palsy_grade
```

**关键点说明：**
- **图像路径**: 相对于 `DATASET.ROOT`
- **缩放因子**: 通常1.0-2.0
- **边界框大小**: 人脸检测的框大小
- **中心坐标**: 人脸中心位置
- **68个关键点**: 标准面部关键点坐标
- **分级标签**: 0-6（House-Brackmann scale或类似）

### 第三步：训练模型

```bash
# 基本训练
python tools/train.py --cfg experiments/aflfp/face_alignment_aflfp_hrnet_w18.yaml

# 指定数据路径
python tools/train.py --cfg experiments/aflfp/face_alignment_aflfp_hrnet_w18.yaml \
    DATASET.ROOT ./data/aflfp/images/ \
    DATASET.TRAINSET ./data/aflfp/face_landmarks_aflfp_train.csv \
    DATASET.TESTSET ./data/aflfp/face_landmarks_aflfp_test.csv
```

### 第四步：测试模型

```bash
python tools/test.py --cfg experiments/aflfp/face_alignment_aflfp_hrnet_w18.yaml \
    TEST.DATASET AFLFP \
    MODEL.PRETRAINED path/to/checkpoint.pth
```

## 📊 关键点定义 (68点标准)

| 部位 | 点数 | 索引范围 | 描述 |
|------|------|---------|------|
| 轮廓 | 17 | 1-17 | 人脸轮廓 |
| 眉毛 | 10 | 18-27 | 左右眉毛 |
| 鼻子 | 9 | 28-36 | 鼻子 |
| 眼睛 | 12 | 37-48 | 左右眼睛 |
| 嘴巴 | 20 | 49-68 | 嘴巴轮廓 |

## 🔧 配置参数

在 `experiments/aflfp/face_alignment_aflfp_hrnet_w18.yaml` 中可调整：

```yaml
# 模型配置
MODEL:
  NUM_JOINTS: 68              # 关键点数量
  SIGMA: 1.5                  # 高斯热力图标准差
  IMAGE_SIZE: [256, 256]      # 输入图像大小
  HEATMAP_SIZE: [64, 64]      # 输出热力图大小

# 数据增强
DATASET:
  SCALE_FACTOR: 0.25          # 缩放范围
  ROT_FACTOR: 30              # 旋转范围
  FLIP: true                  # 是否翻转增强

# 训练参数
TRAIN:
  BATCH_SIZE_PER_GPU: 16      # 批大小
  END_EPOCH: 100              # 总训练轮数
  LR: 0.0002                  # 学习率
  LR_STEP: [50, 80]           # 学习率衰减步数
```

## 🎯 面瘫分级集成 (可选)

### 使用集成分级分类

如果需要同时进行关键点检测和面瘫分级，可以使用 `lib/models/palsy_grading.py` 中的模块：

```python
from lib.models.palsy_grading import PalsyGradingHead, PalsyGradingLoss

# 创建分级分类头
grading_head = PalsyGradingHead(
    in_channels=256,
    num_grades=7  # 0-6级
)

# 使用联合损失
criterion = PalsyGradingLoss(
    landmark_criterion=nn.MSELoss(),
    landmark_weight=1.0,
    grading_weight=0.5,
    num_grades=7
)
```

### 修改训练循环

在 `tools/train.py` 中集成分级损失：

```python
# 获取分级标签
palsy_grades = torch.tensor(meta['palsy_grade']).cuda()

# 计算联合损失
loss = criterion(heatmaps, target, grade_logits, palsy_grades)
```

## 📈 预期性能指标

| 指标 | 预期值 |
|------|--------|
| 关键点NME精度 | < 5% |
| 推理速度 | 30-50 FPS (GPU) |
| 内存占用 | 2-3 GB (VRAM) per GPU |
| 训练时间 | 10-20小时 (100 epochs on V100) |

## ⚠️ 重要注意事项

### 1. 数据增强与面瘫侧面性

- 翻转增强可能会改变面瘫的非对称性特征
- 可在配置中设置 `DATASET.FLIP: false` 禁用翻转
- 或在数据预处理中标记需要特殊处理的样本

### 2. 关键点顺序

- 必须保证CSV中关键点的顺序一致
- 必须为68个点（除非修改`MODEL.NUM_JOINTS`）
- 点的索引从1开始（在`fliplr_joints`中会-1处理）

### 3. 分级标签格式

- 支持整数格式的分级标签（0-6）
- 在meta信息中以`palsy_grade`键保存
- 可根据需要调整为其他分级标准

### 4. CSV验证

```bash
# 验证CSV格式
python tools/prepare_aflfp_data.py --csv data/aflfp/face_landmarks_aflfp_train.csv --validate
```

## 🔍 故障排除

### 问题1: 关键点数不匹配
```
ValueError: Expected 68 landmarks, got X
```
**解决方案**: 检查CSV中关键点列数，确保为 (5 + 68*2 + 1) = 142列

### 问题2: 路径错误
```
FileNotFoundError: image not found
```
**解决方案**: 检查`DATASET.ROOT`和CSV中的相对路径是否正确

### 问题3: CUDA内存不足
```
RuntimeError: CUDA out of memory
```
**解决方案**: 减少`TRAIN.BATCH_SIZE_PER_GPU`

### 问题4: 分级标签丢失
```
KeyError: 'palsy_grade'
```
**解决方案**: 确保CSV最后一列是分级标签

## 📚 相关资源

- 原始LDDMM-Face: https://github.com/torta
- 300-W数据集: https://ibug.doc.ic.ac.uk/resources/300-W/
- House-Brackmann面瘫分级: https://en.wikipedia.org/wiki/House%E2%80%93Brackmann_scale

## 🤝 集成提示

### 与其他数据集混合训练

```yaml
# 先在AFLW上预训练
python tools/train.py --cfg experiments/aflw/face_alignment_aflw_hrnet_w18.yaml

# 再在AFLFP上微调
python tools/train.py --cfg experiments/aflfp/face_alignment_aflfp_hrnet_w18.yaml \
    MODEL.PRETRAINED output/aflw_checkpoint.pth
```

### 跨数据集评估

```python
# 在AFLFP上评估AFLW训练的模型
python tools/test.py --cfg experiments/aflfp/face_alignment_aflfp_hrnet_w18.yaml \
    MODEL.PRETRAINED output/aflw_checkpoint.pth
```

## 📝 文件修改清单

```
✅ lib/datasets/aflfp.py (新增)
✅ lib/datasets/__init__.py (修改)
✅ lib/utils/transforms.py (修改)
✅ experiments/aflfp/face_alignment_aflfp_hrnet_w18.yaml (新增)
✅ tools/prepare_aflfp_data.py (新增)
✅ lib/models/palsy_grading.py (新增)
✅ AFLFP_GUIDE.md (新增)
✅ AFLFP_IMPLEMENTATION_SUMMARY.md (新增)
```

## 🎓 学习路径

1. 阅读 `AFLFP_GUIDE.md` 了解整体架构
2. 运行 `prepare_aflfp_data.py` 学习数据格式
3. 修改配置文件适配自己的数据
4. 基础训练验证设置
5. (可选) 集成 `palsy_grading.py` 进行分级分类
6. 评估模型性能
7. 在实际应用中部署

## ✨ 后续改进方向

1. **多任务学习**: 同时优化关键点定位和分级分类
2. **对称性约束**: 利用面部对称性改进面瘫侧面性判断
3. **轻量级模型**: 使用MobileNet等轻量级主干以支持移动设备
4. **实时处理**: 优化推理管道提高帧率
5. **不确定性估计**: 添加贝叶斯深度学习模块
6. **可解释性**: 使用CAM/Grad-CAM可视化关键决策区域

---

**最后更新**: 2024年
**状态**: ✅ 完成并测试
