# 雾天目标检测：图像增强 + YOLOv8 的对照实验

一个小实验：雾天图片检测不准，那先对图片做增强 / 去雾，再喂给 YOLO，检测会变好吗？

先把结论放前面：**不会，去雾反而更差。** 下面是我做了什么、看到了什么、以及我的理解。

## 想法从哪来

恶劣天气下的目标检测是 Image-Adaptive YOLO（IA-YOLO）的方向，它提出把"图像增强"和"检测"放在一起训练。我没能力复现那篇论文，只想先用最笨的方式验证一个前置问题：把增强当成"即插即用"的离线预处理，到底有没有用。

## 做了什么

1. 用 DAWN 数据集的雾天部分（约 300 张、6 类）训练了一个 YOLOv8n 基线
2. 对同一批测试图，分别用 CLAHE（提对比度）和暗通道先验去雾（DCP）做增强
3. 用同一个模型在"原图 / CLAHE / 去雾"三组测试集上评估，对比 mAP

全程只训练一次模型，剩下全靠换测试集，控制变量。

## 环境

- Windows 11 + Python 3.10
- GPU：NVIDIA RTX 5060（8GB），CUDA 12.8
- PyTorch 2.x（cu128），Ultralytics 8.x

## 数据

DAWN（Detection in Adverse Weather Nature）：1000 张真实恶劣天气图，按雾 / 雨 / 雪 / 沙尘分卷，雾天约 300 张，6 类：car / bus / truck / person / motorcycle / bicycle。

下载（需注册）：
- Mendeley：https://data.mendeley.com/datasets/766ygrbt8y/3
- IEEE Dataport：DOI 10.21227/bw1x-yh39

许可：仅学习 / 科研，请勿商用。

## 复现步骤

```powershell
# 1) VOC -> YOLO 格式
python convert_voc_to_yolo.py --src "D:/DAWN/Fog" --dst datasets/DAWN_yolo --val-ratio 0.2 --seed 42

# 2) 训练基线
yolo detect train model=yolov8n.pt data=datasets/DAWN_yolo/fog.yaml epochs=80 imgsz=640 batch=16 device=0

# 3) 生成两种增强测试集
python preprocess.py --src datasets/DAWN_yolo/images/val --dst datasets/DAWN_yolo/enhanced/clahe/images/val --method clahe --copy-labels
python preprocess.py --src datasets/DAWN_yolo/images/val --dst datasets/DAWN_yolo/enhanced/dehaze/images/val --method dehaze --copy-labels

# 4) 三组评估（同一个 best.pt）
yolo detect val model=runs/detect/train/weights/best.pt data=datasets/DAWN_yolo/fog.yaml
yolo detect val model=runs/detect/train/weights/best.pt data=datasets/DAWN_yolo/fog_clahe.yaml
yolo detect val model=runs/detect/train/weights/best.pt data=datasets/DAWN_yolo/fog_dehaze.yaml
```

`fog_clahe.yaml` / `fog_dehaze.yaml` 就是把 `val` 指向增强目录，标注不变。

## 两种预处理

- **CLAHE**：在 LAB 的 L（亮度）通道做对比度受限的自适应直方图均衡，只拉对比、不改颜色。
- **去雾 DCP**：暗通道先验——雾天图的"暗通道"偏亮，据此估计雾的浓度再减掉。

## 结果

YOLOv8n，240 张雾图训练 80 epoch，在 59 张验证集（403 个目标）上评估：

| 测试集 | mAP@50 | mAP@50-95 | Recall |
|---|---|---|---|
| 原图（基线） | **0.588** | 0.377 | 0.510 |
| CLAHE | 0.585 | 0.364 | 0.526 |
| 去雾 DCP | 0.474 | 0.272 | 0.446 |

分类别 mAP@50（括号里是验证集实例数）：

| 类别 | 原图 | CLAHE | 去雾 |
|---|---|---|---|
| car (307) | 0.855 | 0.824 | 0.769 |
| bus (10) | 0.532 | 0.622 | 0.590 |
| truck (47) | 0.529 | 0.468 | 0.201 |
| person (30) | 0.573 | 0.511 | 0.456 |
| motorcycle (3) | 0.995 | 0.913 | 0.764 |
| bicycle (6) | 0.042 | 0.171 | 0.065 |

motorcycle、bicycle 样本太少，数值参考意义不大。

## 结论

离线增强没有正面收益：CLAHE 基本没影响，去雾反而让 mAP 掉了 0.114。

我的理解：模型是在"原始雾图"上训练的，已经适应了雾的分布。去雾把图变成"像晴天"，模型反而不认识了——这就是 **domain shift（分布偏移）**。这也正好解释了 IA-YOLO 为什么要**把增强和检测一起训练**：让增强"为检测服务"，而不是离线硬套。

## 踩过的坑

- DAWN 的图片和标注不在同一个目录（图片在 `Fog/`，标注在 `Fog/Fog_PASCAL_VOC/`），转换脚本一开始找不到图，改成全目录索引才解决。
- 去雾对 truck、person 这类小目标伤害最大：简化版 DCP 用高斯模糊代替了导向滤波，会引入伪影。
- bicycle 只有 6 个样本，基本没学会（recall=0），是数据不均衡的问题。

## 参考

- W. Liu et al., "Image-Adaptive YOLO for Object Detection in Adverse Weather Conditions", AAAI TPAMI, 2022.
- Hassaballah & Kenk, "DAWN: Vehicle Detection in Adverse Weather Nature", IEEE T-ITS, 2020.
- He, Sun & Tang, "Single Image Haze Removal Using Dark Channel Prior", CVPR, 2009.
