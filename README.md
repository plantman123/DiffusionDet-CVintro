# Task 1：基于扩散模型的肺炎病灶检测（DiffusionDet）

本项目为计算机视觉导论课程 Final Project Task 1，使用 **DiffusionDet**（生成式扩散模型检测器）和 **Faster R-CNN**（判别式基线）在 RSNA Pneumonia Detection 数据集上完成肺炎病灶检测，并进行多维度对比分析。

## 目录

- [项目结构](#项目结构)
- [环境配置](#环境配置)
- [数据集准备](#数据集准备)
- [模型训练](#模型训练)
- [模型评估](#模型评估)
- [可视化](#可视化)
- [预训练权重](#预训练权重)
- [主要结果](#主要结果)

---

## 项目结构

```
Assignment2_task1/
├── configs/                          # 模型配置文件
│   ├── Base-DiffusionDet.yaml        # DiffusionDet 基础配置
│   ├── diffdet.rsna.res50.yaml       # DiffusionDet ResNet-50 配置
│   ├── diffdet.rsna.res101.yaml      # DiffusionDet ResNet-101 配置
│   ├── faster_rcnn.rsna.res50.yaml   # Faster R-CNN ResNet-50 配置
│   └── faster_rcnn.rsna.res101.yaml  # Faster R-CNN ResNet-101 配置
├── diffusiondet/                     # DiffusionDet 模型核心代码
│   ├── detector.py                   # 检测器主体（前向传播 + DDIM 采样）
│   ├── head.py                       # 检测头
│   ├── loss.py                       # 损失函数
│   ├── config.py                     # DiffusionDet 自定义配置项
│   ├── dataset_mapper.py             # 数据增强 mapper
│   └── util/                         # 工具函数
├── visualize/                        # 可视化脚本
│   ├── plot_loss.py                  # 绘制训练 loss 曲线
│   └── experiments.py                # 实验配置列表
├── RSNA.ipynb                        # sRSNA 子集数据构建
├── LRSNA.ipynb                       # lRSNA 全集数据构建
├── train_net.py                      # DiffusionDet 训练入口
├── train_fasterrcnn.py               # Faster R-CNN 训练入口
├── predict.py                        # 推理 + 预测框可视化（支持两种模型）
├── vis_diffusion.py                  # DiffusionDet 渐进去噪过程可视化
├── demo.py                           # DiffusionDet 快速 demo
├── train.sh / trainl.sh              # DiffusionDet 训练脚本
├── train_fasterrcnn.sh               # Faster R-CNN 训练脚本
├── eval.sh                           # 模型评估脚本
├── predict.sh / predict_batch.sh     # 推理可视化脚本
├── REPORT.md                         # 实验报告
└── README.md                         # 本文件
```

### 数据集目录

```
dataset/
├── raw-RSNA/                         # 原始 RSNA 数据（实验一复用）
│   ├── stage_2_train_labels.csv      # 原始标注文件
│   └── pth_data/stage_2_train_images/  # Assignment 1 预处理的 .pth 文件
├── RSNA/                             # sRSNA 抽样子集（10,000 张）
│   ├── annotations/
│   │   ├── train_data.json
│   │   └── val_data.json
│   ├── train2017/                    # 7,000 张训练图像 (JPG)
│   └── val2017/                      # 3,000 张验证图像 (JPG)
└── LRSNA/                            # lRSNA 大规模集（18,512 张）
    ├── annotations/
    │   ├── train_data.json
    │   └── val_data.json
    ├── train2017/                    # 14,809 张训练图像 (JPG)
    └── val2017/                      # 3,703 张验证图像 (JPG)
```

---

## 环境配置

### 依赖安装

本项目基于 **PyTorch + Detectron2** 框架，需安装以下依赖：

```bash
# 1. 安装 PyTorch（根据 CUDA 版本选择，以下以 CUDA 11.8 为例）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 2. 安装 Detectron2
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# 3. 安装其他依赖
pip install pandas matplotlib tqdm pydicom nibabel opencv-python
```

便捷的conda环境安装见 Diffudet.yaml。

### 硬件要求

- **训练**：建议使用多块 GPU（实验使用 4×NVIDIA A40 进行分布式训练）
- **推理/评估**：单块 GPU 即可

---

## 数据集准备

### 前置条件

1. 从 [Kaggle RSNA Pneumonia Detection Challenge](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data) 下载原始数据集
2. 根据 Assignment 1 将 DICOM 图像预处理为 `.pth` 格式的 tensor 文件并将放置于 `dataset/raw-RSNA/` 目录，结构如下：

```
dataset/raw-RSNA/
├── stage_2_train_labels.csv
└── pth_data/stage_2_train_images/
    ├── <patientId>.pth
    └── ...
```

### Step 1：构建 sRSNA 抽样子集（10,000 张）

运行 `RSNA.ipynb`，该 notebook 将：

1. **均衡采样**：从全部患者中随机抽取正负各 5,000 名患者（共 10,000 张），保证正负样本 1:1 均衡
2. **按患者划分**：以患者为单位将数据按 **7:3** 比例划分为训练集（7,000 张）和验证集（3,000 张），避免同一患者的图像泄露到不同集合
3. **COCO 格式转换**：
   - 将 `.pth` tensor 转换为 JPG 图像，保存到 `train2017/` 和 `val2017/` 目录
   - 生成符合 COCO 格式的 `train_data.json` 和 `val_data.json` 标注文件
   - 类别统一为 `pneumonia`（category_id=1），边界框格式为 `[x, y, width, height]`

```bash
# 在 Jupyter 中逐 cell 执行 RSNA.ipynb
# 输出目录：dataset/RSNA/
```

### Step 2：构建 lRSNA 大规模集（18,512 张）

运行 `LRSNA.ipynb`，流程与 Step 1 类似，区别在于：

- 取全部可用患者（不限制数量），共 18,512 张图像
- 训练集占比 **80%**（14,809 张），验证集占比 20%（3,703 张）
- 正负样本比例约 1:2（未做均衡采样）

```bash
# 在 Jupyter 中逐 cell 执行 LRSNA.ipynb
# 输出目录：dataset/LRSNA/
```

### 数据集概览

| 数据集 | 总样本 | 训练集 | 验证集 | 正负比 | 训练集 bbox 数 |
|--------|--------|--------|--------|--------|---------------|
| sRSNA  | 10,000 | 7,000（正3521/负3479） | 3,000 | ≈1:1 | 5,596 |
| lRSNA  | 18,512 | 14,809（正4829/负9980） | 3,703 | ≈1:2 | 7,684 |

---

## 模型训练

### 预训练权重准备

训练前需下载 ImageNet 预训练的 ResNet 权重：

- **ResNet-50**：Detectron2 会自动下载 `detectron2://ImageNetPretrained/torchvision/R-50.pkl`
- **ResNet-101**：需手动下载并放置于 `models/torchvision-R-101.pkl`

```bash
mkdir -p models
# 下载 ResNet-101 预训练权重到 models/ 目录
```

### 训练 DiffusionDet

使用 `train_net.py` 进行训练，通过命令行参数或配置文件指定超参数：

```bash
# 示例：在 sRSNA 上训练 DiffusionDet-Res50
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_net.py \
    --num-gpus 4 \
    --config-file configs/diffdet.rsna.res50.yaml \
    --dataroot ./dataset/RSNA \
    SOLVER.IMS_PER_BATCH 56 \
    SOLVER.BASE_LR 8e-5 \
    SOLVER.MAX_ITER 8000 \
    OUTPUT_DIR ./output/optres50 \
    SOLVER.CHECKPOINT_PERIOD 500 \
    TEST.EVAL_PERIOD 500

# 示例：在 lRSNA 上训练 DiffusionDet-Res101
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_net.py \
    --num-gpus 4 \
    --config-file configs/diffdet.rsna.res101.yaml \
    --dist-url tcp://127.0.0.1:50160 \
    --dataroot ./dataset/LRSNA \
    SOLVER.IMS_PER_BATCH 32 \
    SOLVER.BASE_LR 8e-5 \
    SOLVER.MAX_ITER 3000 \
    OUTPUT_DIR ./output/lrsna_optres101 \
    SOLVER.CHECKPOINT_PERIOD 500 \
    TEST.EVAL_PERIOD 500
```

### 训练 Faster R-CNN

使用 `train_fasterrcnn.py` 进行训练：

```bash
# 示例：在 sRSNA 上训练 Faster R-CNN-Res101
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_fasterrcnn.py \
    --num-gpus 4 \
    --config-file ./configs/faster_rcnn.rsna.res101.yaml \
    --dataroot ./dataset/RSNA \
    SOLVER.MAX_ITER 5000 \
    OUTPUT_DIR ./output/fasterrcnn_res101 \
    SOLVER.CHECKPOINT_PERIOD 1000 \
    TEST.EVAL_PERIOD 1000

# 示例：在 lRSNA 上训练 Faster R-CNN-Res50
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_fasterrcnn.py \
    --num-gpus 4 \
    --config-file ./configs/faster_rcnn.rsna.res50.yaml \
    --dataroot ./dataset/LRSNA \
    SOLVER.MAX_ITER 6000 \
    OUTPUT_DIR ./output/lrsna_fasterrcnn_res50 \
    SOLVER.CHECKPOINT_PERIOD 1000 \
    TEST.EVAL_PERIOD 1000
```



### 从断点恢复训练

```bash
# 添加 --resume 参数即可从最后一个 checkpoint 恢复训练
python train_net.py \
    --num-gpus 4 \
    --config-file configs/diffdet.rsna.res50.yaml \
    --dataroot ./dataset/RSNA \
    --resume \
    OUTPUT_DIR ./output/optres50
```

> **注意**：使用多 GPU 训练时，若同一台机器上有多个训练任务，需通过 `--dist-url tcp://127.0.0.1:<PORT>` 指定不同端口避免冲突。

---

## 模型评估

### 在验证集上评估（COCO mAP）

使用 `--eval-only` 模式加载训练好的 checkpoint，在验证集上进行 COCO 标准评估：

```bash
# 评估 DiffusionDet（需注册数据集，通过 --dataroot 指定）
CUDA_VISIBLE_DEVICES=0 python train_net.py \
    --num-gpus 1 \
    --config-file configs/diffdet.rsna.res50.yaml \
    --dataroot ./dataset/RSNA \
    --eval-only \
    MODEL.WEIGHTS output/optres50/model_final.pth

# 评估 Faster R-CNN
CUDA_VISIBLE_DEVICES=0 python train_fasterrcnn.py \
    --num-gpus 1 \
    --config-file configs/faster_rcnn.rsna.res101.yaml \
    --dataroot ./dataset/RSNA \
    --eval-only \
    MODEL.WEIGHTS output/fasterrcnn_res101/model_final.pth
```

评估输出包括 COCO 标准指标：`AP`、`AP50`、`AP75`、`APs`、`APm`、`APl` 等。

### 评估不同去噪步数的影响

通过修改 `MODEL.DiffusionDet.SAMPLE_STEP` 可测试不同 DDIM 去噪步数对精度和速度的影响：

```bash
# 测试 step=1 的推理精度
python train_net.py \
    --num-gpus 1 \
    --config-file configs/diffdet.rsna.res50.yaml \
    --dataroot ./dataset/RSNA \
    --eval-only \
    MODEL.WEIGHTS output/optres50/model_final.pth \
    MODEL.DiffusionDet.SAMPLE_STEP 1

# 测试 step=3 的推理精度
python train_net.py \
    --num-gpus 1 \
    --config-file configs/diffdet.rsna.res50.yaml \
    --dataroot ./dataset/RSNA \
    --eval-only \
    MODEL.WEIGHTS output/optres50/model_final.pth \
    MODEL.DiffusionDet.SAMPLE_STEP 3
```

### 评估中间 checkpoint

训练过程中每 500 iteration 保存一次 checkpoint（`model_0000499.pth`、`model_0000999.pth` 等），可评估任意中间 checkpoint 来选择最佳模型：

```bash
python train_net.py \
    --num-gpus 1 \
    --config-file configs/diffdet.rsna.res50.yaml \
    --dataroot ./dataset/RSNA \
    --eval-only \
    MODEL.WEIGHTS output/optres50/model_0004999.pth
```

---

## 可视化

### 绘制训练 Loss 曲线

```bash
cd visualize
python plot_loss.py
# 输出: visualize/loss_all_experiments.png
```

该脚本从各实验输出目录的 `metrics.json` 中提取 loss 数据，绘制所有实验的训练 loss 变化曲线（含 EMA 平滑）。实验列表定义在 `visualize/experiments.py` 中。

### DiffusionDet 渐进去噪过程可视化

```bash
CUDA_VISIBLE_DEVICES=0 python vis_diffusion.py \
    --config-file configs/diffdet.rsna.res50.yaml \
    --input dataset/RSNA/val2017/*.jpg \
    --output eval/inferenceimgs \
    --opts MODEL.WEIGHTS output/optres50/model_final.pth \
    MODEL.DiffusionDet.SAMPLE_STEP 5
```

生成每张图像在各去噪步骤下的预测框和噪声框的对比大图，直观展示从噪声到检测结果的渐进生成过程。

### 预测结果可视化（预测框 vs GT 框）

```bash
# DiffusionDet 推理
python predict.py \
    --model-type diffdet \
    --config-file configs/diffdet.rsna.res50.yaml \
    --input dataset/RSNA/val2017/<patientId>.jpg \
    --output eval/predict \
    --score-thresh 0.3 \
    --opts MODEL.WEIGHTS output/optres50/model_final.pth

# Faster R-CNN 推理
python predict.py \
    --model-type fasterrcnn \
    --config-file configs/faster_rcnn.rsna.res50.yaml \
    --input dataset/RSNA/val2017/<patientId>.jpg \
    --output eval/predict \
    --score-thresh 0.3 \
    --dataroot dataset/RSNA \
    --opts MODEL.WEIGHTS output/fasterrcnn_res50/model_final.pth

# 批量预测（10 张阳性样本）
bash predict_batch.sh
```

输出图像中红色框为模型预测（带置信度分数），浅蓝色框为 Ground Truth。

---

## 预训练权重

训练完成后的模型权重保存在各实验的 `OUTPUT_DIR` 中：

| 实验 | 模型 | 权重路径 |
|------|------|---------|
| sRSNA DiffusionDet-Res50 最佳 | DiffusionDet | `output/optres50/model_final.pth` |
| sRSNA DiffusionDet-Res101 最佳 | DiffusionDet | `output/optres101/model_final.pth` |
| sRSNA Faster R-CNN-Res50 | Faster R-CNN | `output/fasterrcnn_res50/model_final.pth` |
| sRSNA Faster R-CNN-Res101 | Faster R-CNN | `output/fasterrcnn_res101/model_final.pth` |
| lRSNA DiffusionDet-Res50 | DiffusionDet | `output/lrsna_optres50/model_final.pth` |
| lRSNA DiffusionDet-Res101 | DiffusionDet | `output/lrsna_optres101/model_final.pth` |
| lRSNA Faster R-CNN-Res50 | Faster R-CNN | `output/lrsna_fasterrcnn_res50/model_final.pth` |
| lRSNA Faster R-CNN-Res101 | Faster R-CNN | `output/lrsna_fasterrcnn_res101/model_final.pth` |

---

## 主要结果

### sRSNA 数据集（10,000 张，正负 1:1）

| 模型 | Backbone | MAX_ITER | LR | Batch | mAP@0.5:0.95 | mAP@50 |
|------|----------|----------|-----|-------|-------------|--------|
| Faster R-CNN | Res101 | 5,000 | 0.005 | 16 | 16.05 | **48.20** |
| DiffusionDet | Res50 | 8,000 | 8e-5 | 56 | 14.58 | 46.92 |
| Faster R-CNN | Res50 | 5,000 | 0.005 | 16 | 14.97 | 46.15 |

### lRSNA 数据集（18,512 张，正负 1:2）

| 模型 | Backbone | MAX_ITER | LR | Batch | mAP@0.5:0.95 | mAP@50 |
|------|----------|----------|-----|-------|-------------|--------|
| DiffusionDet | Res101 | 3,000 | 8e-5 | 32 | **15.71** | **42.14** |
| Faster R-CNN | Res101 | 6,000 | 0.005 | 16 | 13.29 | 40.07 |

### 不同去噪步数（DiffusionDet-Res50，sRSNA）

| 去噪步数 | mAP@50 | 推理速度 (ms/img) | FPS |
|---------|--------|-------------------|-----|
| 1 | 37.82 | 35.5 | 28.2 |
| 3 | 43.58 | 78.6 | 12.7 |
| 5 | 44.92 | 120.8 | 8.3 |
| Faster R-CNN（单次前向） | 46.15 | 26.0 | 38.5 |

---
