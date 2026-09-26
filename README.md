# 雾天目标检测：图像增强预处理 + YOLOv8

> 一个极简的本科入门项目：验证「CLAHE / 暗通道去雾 预处理」能否提升雾天场景下
> YOLOv8 的目标检测精度。核心思想参考 **Image-Adaptive YOLO (IA-YOLO)** 的
> "图像增强改善恶劣天气检测"思路，但这里**不复现论文、不碰复杂理论**，
> 只用 Ultralytics YOLOv8 + OpenCV 做一次干净的对照实验。

---

## 1. 项目目标（一句话）

训练一个 YOLOv8 雾天检测模型，分别对**原始雾图 / CLAHE 增强图 / 去雾图**做推理评估，
用 mAP 对比说明"预处理能否帮助雾天检测"。

## 2. 实验设计

| 组 | 训练数据 | 测试数据 | 目的 |
|----|---------|---------|------|
| Baseline | 原始雾图 | 原始雾图 | 基线 |
| CLAHE | 原始雾图 | CLAHE 增强雾图 | 看增强是否提升检测 |
| Dehaze | 原始雾图 | 去雾后的雾图 | 看去雾是否提升检测 |

> 只训练**一次**模型（在原始雾图上），对同一模型做三组评估，控制变量、最快出结论。

## 3. 环境

- OS: Windows 11
- Python: 3.10（推荐，避免 3.14 兼容问题）
- GPU: NVIDIA GeForce RTX 5060（8GB，Blackwell）
- CUDA: 12.8
- PyTorch: ≥ 2.7（RTX 5060 必须 torch 2.7 + cu128 才能用 GPU）

### 安装步骤（PowerShell）

```powershell
# 1) 用 Python 3.10 建独立环境
py -3.10 -m venv C:\Users\HUA\yolo-env
C:\Users\HUA\yolo-env\Scripts\Activate.ps1

# 2) 装 CUDA 版 torch（RTX 5060 专用，约 2.5-3GB）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 3) 装 ultralytics + opencv
pip install -r requirements.txt

# 4) 自检，应显示 CUDA: True、GPU: RTX 5060
yolo checks
```

## 4. 数据

- 数据集：**DAWN**（Detection in Adverse Weather Nature），1000 张真实恶劣天气图，
  按 fog/rain/snow/sand 四类天气分卷，其中雾天约 300 张（Fog.zip），
  6 类：car / bus / truck / person / motorcycle / bicycle。
- 下载（任选其一，需免费注册账号）：
  - Mendeley：https://data.mendeley.com/datasets/766ygrbt8y/3 （DAWN.zip 约 132MB）
  - IEEE Dataport：https://ieee-dataport.org/open-access/dawn-vehicle-detection-adverse-weather-nature （DOI 10.21227/bw1x-yh39）
- 备用数据集：**RTTS**（RESIDE 的雾天检测集，4322 张，Roboflow 有现成 YOLO 格式）：
  https://universe.roboflow.com/yolo-26c5m/rtts-yfuou/dataset/4
- 许可：DAWN 仅限学习/科研用途，请勿商用。

解压后结构（fog 子目录内是 .jpg + .xml）：

```
DAWN/
├── fog/
├── rain/
├── snow/
└── sand/
```

## 5. 快速开始

### 5.1 转成 YOLO 格式

```powershell
python convert_voc_to_yolo.py --src "D:/DAWN/fog" --dst datasets/DAWN_yolo --val-ratio 0.2 --seed 42
```

得到：

```
datasets/DAWN_yolo/
├── fog.yaml
├── images/{train,val}/
└── labels/{train,val}/
```

### 5.2 训练（在原始雾图上，只训一次）

```powershell
yolo detect train model=yolov8n.pt data=datasets/DAWN_yolo/fog.yaml epochs=80 imgsz=640 batch=16 device=0
```

### 5.3 生成增强测试集（图片增强，标注不变）

```powershell
python preprocess.py --src datasets/DAWN_yolo/images/val `
      --dst datasets/DAWN_yolo/enhanced/clahe/images/val --method clahe --copy-labels

python preprocess.py --src datasets/DAWN_yolo/images/val `
      --dst datasets/DAWN_yolo/enhanced/dehaze/images/val --method dehaze --copy-labels
```

### 5.4 三组评估（用同一个 best.pt）

```powershell
# 基线：原始雾图
yolo detect val model=runs/detect/train/weights/best.pt data=datasets/DAWN_yolo/fog.yaml

# CLAHE 增强图（把 val 指向增强目录，见 5.5）
yolo detect val model=runs/detect/train/weights/best.pt data=datasets/DAWN_yolo/fog_clahe.yaml

# 去雾图
yolo detect val model=runs/detect/train/weights/best.pt data=datasets/DAWN_yolo/fog_dehaze.yaml
```

### 5.5 增强评估用的 yaml（手动建两个文件）

`datasets/DAWN_yolo/fog_clahe.yaml`：

```yaml
path: <你的绝对路径>/datasets/DAWN_yolo
train: images/train
val: enhanced/clahe/images/val
names:
  0: car
  1: bus
  2: truck
  3: person
  4: motorcycle
  5: bicycle
```

`datasets/DAWN_yolo/fog_dehaze.yaml` 同理，把 `val` 改成 `enhanced/dehaze/images/val`。

### 5.6 可视化预测（可选）

```powershell
yolo detect predict model=runs/detect/train/weights/best.pt `
      source=datasets/DAWN_yolo/images/val save=True project=runs/vis
```

## 6. 预处理方法说明

| 方法 | 原理（大白话） | 脚本 |
|------|--------------|------|
| CLAHE | 只对亮度(L)通道做局部直方图均衡，把雾里糊掉的对比度拉回来，不改变颜色 | `preprocess.py --method clahe` |
| Dehaze | 暗通道先验(DCP)：雾天图像"暗通道"偏亮，据此估计雾的浓度并减去 | `preprocess.py --method dehaze` |

参数可调：CLAHE 的 `--clip`（默认 2.0）、`--tile`（默认 8）；DCP 的 `win`、`omega`、`t0` 在 `dehaze()` 里。

## 7. 结果

模型：YOLOv8n（在 240 张原始雾图上训练 80 epoch，best.pt），在 59 张验证集（403 个目标）上评估。

| 测试数据 | mAP@50 | mAP@50-95 | Recall | 相对基线 |
|---------|--------|-----------|--------|---------|
| 原始雾图（基线） | **0.588** | 0.377 | 0.510 | — |
| CLAHE 增强 | 0.585 | 0.364 | 0.526 | -0.003 |
| 去雾（DCP） | 0.474 | 0.272 | 0.446 | **-0.114** |

分类别 mAP@50（样本数 = 验证集中该类别实例数）：

| 类别 | 样本数 | 原始 | CLAHE | Dehaze |
|------|-------|------|-------|--------|
| car | 307 | 0.855 | 0.824 | 0.769 |
| bus | 10 | 0.532 | 0.622 | 0.590 |
| truck | 47 | 0.529 | 0.468 | 0.201 |
| person | 30 | 0.573 | 0.511 | 0.456 |
| motorcycle | 3 | 0.995 | 0.913 | 0.764 |
| bicycle | 6 | 0.042 | 0.171 | 0.065 |

> 注：motorcycle(3)、bicycle(6) 样本极少，数值噪声大，不作为结论依据。

**对比图**（把增强前后 + 检测框截图放 `results/` 目录，可用 `yolo predict` 生成，见 5.6）：

- `results/baseline_vs_clahe.png`
- `results/baseline_vs_dehaze.png`

## 8. 结论

**一句话**：在"原始雾图训练 → 增强图测试"的设定下，离线图像增强预处理对雾天检测
**没有正面收益**——CLAHE 基本无影响（mAP@50 0.588 → 0.585，-0.003），去雾则显著下降
（0.588 → 0.474，-0.114）。

**关键洞察**：这恰恰印证了 IA-YOLO 的核心动机。图像增强不能当"即插即用"的离线预处理——
模型是在原始雾图分布上学习的，离线去雾把输入拉到了"像晴天"的分布（domain shift），
模型反而不认识目标了（truck 的 recall 从 0.32 崩到 0.09，person 从 0.43 掉到 0.31）。
所以增强必须**与检测联合训练**，让增强模块"为检测服务"，这正是 IA-YOLO 把可微图像处理
（DIP）模块嵌进 YOLO 一起端到端训练的原因。

**分项结论**：

- CLAHE（只拉亮度对比、不改变雾的分布）≈ 中性：recall 略升（0.510→0.526）但 precision
  略降（0.793→0.707），一正一负抵消，mAP 几乎不变。说明单纯提对比度不解决本质问题。
- Dehaze（真正去掉雾）反而最伤，因为彻底改变了输入分布。
- 唯一的小幅正收益出现在 `bus`（大目标、轮廓清晰），提示增强可能只对"已经比较清晰"的大目标有点用。

## 9. 失败分析（如实填写，科研诚信）

- [x] **去雾 mAP 显著下降（-0.114）** → 主因是 domain shift：模型在原始雾图上训练，
      去雾后输入分布改变。解决方向（下一步该做的）：把增强模块与检测**联合训练**（IA-YOLO 思路），
      或**同时增强训练集后重训**（train 和 test 都增强）再对比。
- [x] **去雾对小目标伤害最大**（truck recall 0.32→0.09，person 0.43→0.31）→ 本项目的 DCP 是简化版，
      用高斯模糊代替了论文里的引导滤波(guided filter)精化透射率，会产生光晕/伪影，小目标更易被干扰。
      解决：换成引导滤波，或调小 `win`、`omega` 参数。
- [x] **类别不均衡** → bicycle 仅 6 个实例，mAP 接近 0（recall=0）；motorcycle 3 个实例噪声大。
      这是 DAWN 数据集本身的特性，可说明"稀有类未学好"，但不影响整体结论（car 占 76% 目标）。
- [ ] 若想得到"增强有正收益"的结论 → 需要 train+test 同时增强重训，而非只增强测试集。
- [ ] 若训练不收敛 → 数据量小（240 张），可降低 epochs、加数据增强（mosaic/flip）。
- [ ] 若 CUDA OOM → 减小 `batch`（如 8）。

## 10. 目录结构

```
fog-yolo/
├── preprocess.py           # CLAHE / 去雾增强脚本
├── convert_voc_to_yolo.py  # DAWN VOC -> YOLO 格式
├── requirements.txt
├── fog.yaml                # 数据集配置（由转换脚本自动生成）
├── README.md
├── datasets/               # 数据集（gitignore，不上传）
├── runs/                   # 训练/评估结果（gitignore）
└── results/                # 对比图
```

## 11. 参考

- W. Liu, G. Ren, R. Yu, S. Guo, J. Zhu, L. Zhang, "Image-Adaptive YOLO for Object
  Detection in Adverse Weather Conditions", IEEE TPAMI, 2022.（课题组核心方向）
- Hassaballah, M., Kenk, M. A., "DAWN: Vehicle Detection in Adverse Weather Nature
  Dataset", IEEE T-ITS, 2020.
- He, K., Sun, J., Tang, X., "Single Image Haze Removal Using Dark Channel Prior",
  CVPR, 2009.
