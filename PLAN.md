# 5 天冲刺计划（每天 2-3 小时）

> 目标：只做一件事 —— 验证「图像增强预处理 + YOLOv8」在雾天检测的效果。
> 原则：不复现论文、不碰复杂理论，只用 Ultralytics + OpenCV。
> 前置：`yolo checks` 已通过，RTX 5060 + CUDA 12.8。

---

## Day 1 · 环境收尾 + 下载数据（约 2 小时）

- [ ] 确认 `yolo checks` 显示 `CUDA: True`、`GPU: RTX 5060`
- [ ] 建独立环境并装依赖（README 第 3 节命令）
- [ ] 注册 Mendeley / IEEE Dataport 账号，下载 DAWN.zip（132MB）并解压
- [ ] 打开 `fog/` 目录，随机看 10 张图 + 对应 .xml，理解"雾图 + 检测框"长什么样

**产出**：环境就绪，数据集在本地。

---

## Day 2 · 转格式 + 跑通预处理脚本（约 2.5 小时）

- [ ] 运行转换脚本（README 5.1），得到 `images/train|val` + `labels/train|val` + `fog.yaml`
- [ ] 用 `preprocess.py` 对 3-5 张图做 clahe / dehaze，肉眼对比效果
- [ ] 确认 `--copy-labels` 正常工作（增强后标注还在）

```powershell
python preprocess.py --src datasets/DAWN_yolo/images/val --dst tmp/clahe --method clahe
python preprocess.py --src datasets/DAWN_yolo/images/val --dst tmp/dehaze --method dehaze
```

**产出**：YOLO 格式数据集 + 能用的增强脚本。

---

## Day 3 · 训练 baseline（约 2.5 小时，含等待）

- [ ] 启动训练（README 5.2），`device=0` 用 GPU
- [ ] 观察 loss 下降、mAP 趋势；确认 best.pt 生成

```powershell
yolo detect train model=yolov8n.pt data=datasets/DAWN_yolo/fog.yaml epochs=80 imgsz=640 batch=16 device=0
```

**产出**：`runs/detect/train/weights/best.pt`

---

## Day 4 · 生成增强测试集 + 三组评估（约 2.5 小时）

- [ ] 增强整个 val 集：clahe 和 dehaze 各一份（README 5.3）
- [ ] 建 `fog_clahe.yaml`、`fog_dehaze.yaml`（README 5.5）
- [ ] 跑三组 `yolo val`，记录三个 mAP@50 / mAP@50-95

```powershell
yolo detect val model=runs/detect/train/weights/best.pt data=datasets/DAWN_yolo/fog.yaml
yolo detect val model=runs/detect/train/weights/best.pt data=datasets/DAWN_yolo/fog_clahe.yaml
yolo detect val model=runs/detect/train/weights/best.pt data=datasets/DAWN_yolo/fog_dehaze.yaml
```

**产出**：三组 mAP 数值 + 对比表。

---

## Day 5 · 报告 + README + 收尾（约 3 小时）

- [ ] 把结果填进 README 第 7、8、9 节
- [ ] 截增强前后 + 检测框对比图，放 `results/`
- [ ] 写结论 + 失败分析（如实写，加分项）
- [ ] `git init` + 推 GitHub，README 作为项目首页

```powershell
git init && git add . && git commit -m "fog detection: enhancement preprocessing + YOLOv8"
git remote add origin <你的仓库地址>
git push -u origin main
```

**产出**：完整 GitHub 项目 + 对比实验报告。

---

## 若还有富余时间（可选加分）

1. **训练集也增强后重训**：用 clahe 增强 train，重训一个模型，与 baseline 对比（更接近 IA-YOLO 的"增强提升训练"思路）。
2. 把 dehaze 的高斯模糊换成**引导滤波**，观察伪影是否减少。
3. 在 RTTS 上再验证一次结论的普适性。
