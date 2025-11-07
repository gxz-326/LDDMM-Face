# 🚀 AFLFP 数据集快速开始指南

## 💡 概述

AFLFP (Face Landmark Face Palsy) 是针对面瘫(Facial Palsy)识别和分级的人脸关键点定位数据集集成。本指南将帮助您快速上手。

## 📦 已实现内容

✅ AFLFP数据集类 (`lib/datasets/aflfp.py`)
✅ 数据集注册和选择器 (`lib/datasets/__init__.py`)
✅ 关键点映射 (`lib/utils/transforms.py`)
✅ 训练配置 (`experiments/aflfp/face_alignment_aflfp_hrnet_w18.yaml`)
✅ 数据准备工具 (`tools/prepare_aflfp_data.py`)
✅ 面瘫分级模块 (可选，`lib/models/palsy_grading.py`)

## 🎯 三步开始

### 步骤 1️⃣: 准备数据

```bash
# 创建数据目录
mkdir -p data/aflfp/images

# 将图像文件放入 images 目录
# 图像文件名格式: image001.jpg, image002.jpg, ...
```

### 步骤 2️⃣: 创建标注文件

**CSV格式 (空格分隔):**

```
image_path scale box_size center_x center_y x1 y1 x2 y2 ... x68 y68 palsy_grade
image001.jpg 1.5 300 256 256 100 50 ... 200 300 2
image002.jpg 1.4 310 260 260 105 55 ... 205 305 1
```

**列说明:**
- `image_path`: 相对路径
- `scale`: 缩放系数 (1.0-2.0)
- `box_size`: 人脸框大小
- `center_x, center_y`: 人脸中心
- `x1 y1 ... x68 y68`: 68个关键点 (136个数值)
- `palsy_grade`: 面瘫等级 (0-6)

**快速创建模板:**
```bash
python tools/prepare_aflfp_data.py \
    --output data/aflfp/face_landmarks_aflfp_train.csv \
    --create-template
```

### 步骤 3️⃣: 开始训练

```bash
python tools/train.py \
    --cfg experiments/aflfp/face_alignment_aflfp_hrnet_w18.yaml \
    DATASET.ROOT ./data/aflfp/images/ \
    DATASET.TRAINSET ./data/aflfp/face_landmarks_aflfp_train.csv \
    DATASET.TESTSET ./data/aflfp/face_landmarks_aflfp_test.csv
```

## 📊 数据格式示例

**完整CSV行示例 (简化表示):**

```
img_001.jpg 1.5 300 256.0 256.0 100.0 50.0 105.0 52.0 110.0 55.0 115.0 57.0 120.0 60.0 125.0 62.0 130.0 65.0 135.0 67.0 140.0 70.0 145.0 72.0 150.0 75.0 155.0 77.0 160.0 80.0 165.0 82.0 170.0 85.0 175.0 87.0 180.0 90.0 185.0 92.0 190.0 95.0 195.0 97.0 200.0 100.0 205.0 102.0 210.0 105.0 215.0 107.0 220.0 110.0 225.0 112.0 230.0 115.0 235.0 117.0 240.0 120.0 245.0 122.0 250.0 125.0 255.0 127.0 260.0 130.0 265.0 132.0 270.0 135.0 275.0 137.0 280.0 140.0 285.0 142.0 290.0 145.0 295.0 147.0 300.0 150.0 2
```

## 🔧 常用命令

### 验证数据格式
```bash
python tools/prepare_aflfp_data.py \
    --csv data/aflfp/face_landmarks_aflfp_train.csv \
    --validate
```

### 训练 (基础配置)
```bash
python tools/train.py --cfg experiments/aflfp/face_alignment_aflfp_hrnet_w18.yaml
```

### 训练 (自定义参数)
```bash
python tools/train.py --cfg experiments/aflfp/face_alignment_aflfp_hrnet_w18.yaml \
    TRAIN.BATCH_SIZE_PER_GPU 32 \
    TRAIN.END_EPOCH 150 \
    TRAIN.LR 0.0001
```

### 测试模型
```bash
python tools/test.py \
    --cfg experiments/aflfp/face_alignment_aflfp_hrnet_w18.yaml \
    TEST.DATASET AFLFP \
    MODEL.PRETRAINED checkpoint.pth
```

## 📈 68个关键点映射

| 类别 | 索引 | 数量 | 说明 |
|------|------|------|------|
| 轮廓 | 1-17 | 17 | 脸部轮廓 |
| 眉毛 | 18-27 | 10 | 左右眉毛 |
| 鼻子 | 28-36 | 9 | 鼻子 |
| 眼睛 | 37-48 | 12 | 左右眼睛各6个 |
| 嘴巴 | 49-68 | 20 | 嘴巴轮廓 |

> **注意**: 索引从1开始，与CSV文件中的坐标对应

## ⚙️ 配置文件参数

编辑 `experiments/aflfp/face_alignment_aflfp_hrnet_w18.yaml`:

```yaml
# 关键配置
MODEL:
  NUM_JOINTS: 68                # 保持68个关键点
  SIGMA: 1.5                    # 热力图高斯标准差

TRAIN:
  BATCH_SIZE_PER_GPU: 16        # 根据显存调整
  END_EPOCH: 100                # 训练轮数
  LR: 0.0002                    # 学习率

DATASET:
  ROOT: './data/aflfp/images/'        # 图像目录
  TRAINSET: './data/aflfp/..._train.csv'  # 训练集
  TESTSET: './data/aflfp/..._test.csv'    # 测试集
  FLIP: true                    # 数据增强翻转
  SCALE_FACTOR: 0.25            # 缩放范围
  ROT_FACTOR: 30                # 旋转范围
```

## 🐛 常见问题

### Q: 我的关键点数量不是68怎么办?

**A:** 修改配置中的 `MODEL.NUM_JOINTS` 即可:
```yaml
MODEL:
  NUM_JOINTS: 51  # 如果只有51个点
```

### Q: CSV验证失败，提示列数错误

**A:** 确保每行恰好有142列:
- 1 (image_path)
- 1 (scale)  
- 1 (box_size)
- 2 (center_x, center_y)
- 136 (68个点 × 2坐标)
- 1 (palsy_grade)

### Q: 面瘫的侧面性在训练中丢失

**A:** 禁用水平翻转增强:
```bash
python tools/train.py --cfg experiments/aflfp/face_alignment_aflfp_hrnet_w18.yaml \
    DATASET.FLIP false
```

### Q: 内存不足

**A:** 减小批大小:
```bash
python tools/train.py --cfg experiments/aflfp/face_alignment_aflfp_hrnet_w18.yaml \
    TRAIN.BATCH_SIZE_PER_GPU 8
```

## 📂 文件结构

```
project/
├── data/
│   └── aflfp/
│       ├── images/
│       │   ├── image001.jpg
│       │   ├── image002.jpg
│       │   └── ...
│       ├── face_landmarks_aflfp_train.csv
│       └── face_landmarks_aflfp_test.csv
├── experiments/aflfp/
│   └── face_alignment_aflfp_hrnet_w18.yaml
├── lib/
│   ├── datasets/
│   │   ├── aflfp.py ✨ 新增
│   │   └── __init__.py ✏️ 已修改
│   ├── models/
│   │   └── palsy_grading.py ✨ 新增 (可选)
│   └── utils/
│       └── transforms.py ✏️ 已修改
└── tools/
    ├── prepare_aflfp_data.py ✨ 新增
    ├── train.py
    └── test.py
```

## 🎓 进阶用法

### 结合分级分类

如果需要同时进行关键点检测和面瘫分级:

```python
from lib.models.palsy_grading import PalsyGradingHead, PalsyGradingLoss

# 在训练脚本中使用
grading_head = PalsyGradingHead(in_channels=256, num_grades=7)
criterion = PalsyGradingLoss(
    landmark_criterion=nn.MSELoss(),
    landmark_weight=1.0,
    grading_weight=0.5
)
```

详见 `lib/models/palsy_grading.py`

### 迁移学习

从其他数据集预训练:

```bash
# 先在AFLW上训练
python tools/train.py --cfg experiments/aflw/face_alignment_aflw_hrnet_w18.yaml

# 再在AFLFP上微调
python tools/train.py --cfg experiments/aflfp/face_alignment_aflfp_hrnet_w18.yaml \
    MODEL.PRETRAINED output/aflw_checkpoint_best.pth
```

## 📊 监控训练

训练过程中会输出:
```
Epoch: [0][0/100]
Time 0.123s Speed 512.0 samples/s
Loss 0.4567 (0.4567)
NME 5.23%
```

使用TensorBoard查看详细图表:
```bash
tensorboard --logdir=log/
```

## 🎯 性能目标

- 关键点定位精度 (NME) 应该在 **3-5%**
- 推理速度 **30-50 FPS** (单GPU)
- 总训练时间约 **8-24小时** (取决于GPU和数据集大小)

## 📚 更详细的文档

- 详细指南: `AFLFP_GUIDE.md`
- 实现总结: `AFLFP_IMPLEMENTATION_SUMMARY.md`

## ✨ 下一步

1. ✅ 准备好AFLFP数据
2. ✅ 创建CSV标注文件
3. ✅ 运行训练
4. ✅ 评估模型性能
5. ✅ 部署到应用

祝您使用愉快！🎉

---

**问题反馈**: 如遇到问题，请检查:
- [ ] CSV格式是否正确 (142列)
- [ ] 图像路径是否存在
- [ ] 配置文件中的数据路径是否正确
- [ ] GPU显存是否足够
