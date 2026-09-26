#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
雾天图像增强预处理脚本（Ultralytics YOLO 配套）

支持三种方法：
  - clahe  : 对 LAB 空间的 L 通道做 CLAHE（对比度受限自适应直方图均衡），增强雾天细节
  - dehaze : 简化版暗通道先验(DCP)去雾（He et al. 2009）
  - both   : 先去雾，再做 CLAHE

推荐目录结构（本项目约定）：
  datasets/DAWN_yolo/
  ├── fog.yaml
  ├── images/{train,val}/
  ├── labels/{train,val}/
  └── enhanced/
      ├── clahe/{images,labels}/val/
      └── dehaze/{images,labels}/val/

用法示例（Windows PowerShell）：
  python preprocess.py --src datasets/DAWN_yolo/images/val `
        --dst datasets/DAWN_yolo/enhanced/clahe/images/val --method clahe --copy-labels

  python preprocess.py --src datasets/DAWN_yolo/images/val `
        --dst datasets/DAWN_yolo/enhanced/dehaze/images/val --method dehaze --copy-labels

说明：--copy-labels 会把 src 同级 labels 里的同名 .txt 原样复制到 dst 同级 labels，
      因为增强只改图片、不改标注框。
"""

import argparse
import os
import shutil

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# 方法 1：CLAHE（对 LAB 的 L 通道做对比度受限自适应直方图均衡）
# --------------------------------------------------------------------------- #
def clahe(img, clip=2.0, tile=8):
    """img: BGR uint8 -> BGR uint8"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe_obj = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    l = clahe_obj.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


# --------------------------------------------------------------------------- #
# 方法 2：简化版暗通道先验去雾（He et al. 2009）
# --------------------------------------------------------------------------- #
def _dark_channel(img, win=15):
    """暗通道：先取 RGB 最小值，再做最小值滤波（腐蚀）。img: float32 [0,1]"""
    dc = img.min(axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (win, win))
    return cv2.erode(dc, kernel)


def _atmospheric_light(img, dark, top=0.001):
    """从暗通道最亮的 top 0.1% 像素里估计大气光 A（RGB 向量）"""
    h, w = dark.shape
    num = max(int(h * w * top), 1)
    flat_dark = dark.ravel()
    idx = np.argsort(flat_dark)[::-1][:num]   # 暗通道最亮的 num 个像素
    flat_img = img.reshape(-1, 3)
    brightest = idx[np.argmax(flat_img[idx].sum(axis=1))]
    return flat_img[brightest]                 # (3,)


def dehaze(img, omega=0.95, win=15, t0=0.1):
    """简化版 DCP 去雾。img: BGR uint8 -> BGR uint8"""
    im = img.astype(np.float32) / 255.0
    dark = _dark_channel(im, win)
    A = _atmospheric_light(im, dark)

    # 透射率 t(x) = 1 - omega * dark_channel(I/A)
    norm = im / np.maximum(A.reshape(1, 1, 3), 1e-6)
    t = 1.0 - omega * _dark_channel(norm, win)

    # 廉价精化：用高斯模糊代替导向滤波/软抠图，抑制光晕（可替换为引导滤波）
    t = cv2.GaussianBlur(t, (win, win), 0)
    t = np.clip(t, t0, 1.0)

    # 恢复 J(x) = (I - A) / max(t, t0) + A
    J = (im - A.reshape(1, 1, 3)) / t[..., None] + A.reshape(1, 1, 3)
    J = np.clip(J, 0, 1)
    return (J * 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# 统一入口
# --------------------------------------------------------------------------- #
def enhance(img, method, clip=2.0, tile=8):
    if method == "clahe":
        return clahe(img, clip=clip, tile=tile)
    if method == "dehaze":
        return dehaze(img)
    if method == "both":
        return clahe(dehaze(img), clip=clip, tile=tile)
    raise ValueError(f"未知方法: {method}")


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _labels_path(p):
    """把路径里的 'images' 替换成 'labels'（用于定位同级标注目录），兼容 / 和 \\ 两种分隔符"""
    for sep in (os.sep, "/"):
        p = p.replace(sep + "images" + sep, sep + "labels" + sep)
    return p


def main():
    ap = argparse.ArgumentParser(description="雾天图像增强预处理")
    ap.add_argument("--src", required=True, help="输入图片目录")
    ap.add_argument("--dst", required=True, help="输出目录")
    ap.add_argument("--method", required=True, choices=["clahe", "dehaze", "both"])
    ap.add_argument("--clip", type=float, default=2.0, help="CLAHE clipLimit（默认 2.0）")
    ap.add_argument("--tile", type=int, default=8, help="CLAHE tileGridSize（默认 8）")
    ap.add_argument("--copy-labels", action="store_true",
                    help="把 src 同级 labels 目录里的同名 .txt 复制到 dst 同级 labels")
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    files = sorted(
        f for f in os.listdir(args.src)
        if os.path.splitext(f)[1].lower() in IMG_EXTS
    )

    n = 0
    for f in files:
        p = os.path.join(args.src, f)
        img = cv2.imread(p)
        if img is None:
            print(f"[skip] 无法读取 {p}")
            continue
        out = enhance(img, args.method, clip=args.clip, tile=args.tile)
        cv2.imwrite(os.path.join(args.dst, f), out)
        n += 1
    print(f"[done] {args.method} 处理完成，共 {n} 张 -> {args.dst}")

    if args.copy_labels:
        src_labels = _labels_path(args.src)
        dst_labels = _labels_path(args.dst)
        if not os.path.isdir(src_labels):
            print(f"[warn] 未找到标注目录 {src_labels}，跳过复制标注")
            return
        os.makedirs(dst_labels, exist_ok=True)
        copied = 0
        for f in os.listdir(src_labels):
            if f.endswith(".txt"):
                shutil.copy2(os.path.join(src_labels, f),
                             os.path.join(dst_labels, f))
                copied += 1
        print(f"[done] 复制标注 {copied} 个 -> {dst_labels}")


if __name__ == "__main__":
    main()
