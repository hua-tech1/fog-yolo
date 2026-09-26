#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
把 DAWN（VOC XML 标注）转成 Ultralytics YOLO 格式。

DAWN 每个天气类别（fog/rain/snow/sand）一个目录，目录里是 .jpg + .xml。
本脚本读取一个天气目录（例如 fog），解析 VOC XML（兼容 LabelMe polygon），
按比例随机划分 train/val，输出：

  <dst>/
  ├── images/{train,val}/
  ├── labels/{train,val}/
  └── fog.yaml

用法：
  python convert_voc_to_yolo.py --src "D:/DAWN/fog" --dst datasets/DAWN_yolo --val-ratio 0.2 --seed 42
"""

import argparse
import os
import random
import shutil
import xml.etree.ElementTree as ET

# DAWN 的 6 个类别（顺序即 YOLO 的 class id）
CLASSES = ["car", "bus", "truck", "person", "motorcycle", "bicycle"]

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_image_index(src):
    """遍历 src 目录树，建立 {文件名小写(不含扩展名): 图片完整路径} 索引。
    兼容 DAWN 这种「图片在 Fog/、xml 在 Fog/Fog_PASCAL_VOC/」不在同一目录的情况。"""
    index = {}
    for root_dir, _, files in os.walk(src):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                index[os.path.splitext(f)[0].lower()] = os.path.join(root_dir, f)
    return index


def parse_voc_xml(xml_path):
    """解析 VOC XML -> [(class_id, cx, cy, w, h)]，返回 (img_w, img_h, boxes)"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    w = int(float(size.find("width").text))
    h = int(float(size.find("height").text))

    boxes = []
    for obj in root.findall("object"):
        name = (obj.find("name").text or "").strip().lower()
        if name not in CLASSES:
            continue
        cls = CLASSES.index(name)

        bndbox = obj.find("bndbox")
        if bndbox is not None:
            # 标准 VOC
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
        else:
            # LabelMe polygon 兜底：取多边形外接框
            poly = obj.find("polygon")
            pts = [(float(p.find("x").text), float(p.find("y").text))
                   for p in poly.findall("pt")]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            xmin, ymin, xmax, ymax = min(xs), min(ys), max(xs), max(ys)

        # 转 YOLO 归一化中心点格式
        cx = (xmin + xmax) / 2.0 / w
        cy = (ymin + ymax) / 2.0 / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h
        boxes.append((cls, cx, cy, bw, bh))

    return w, h, boxes


def write_yaml(dst, names):
    path = os.path.abspath(dst).replace("\\", "/")
    lines = [
        f"path: {path}",
        "train: images/train",
        "val: images/val",
        "",
        "names:",
    ]
    for i, n in enumerate(names):
        lines.append(f"  {i}: {n}")
    with open(os.path.join(dst, "fog.yaml"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description="DAWN VOC -> YOLO 格式转换")
    ap.add_argument("--src", required=True, help="DAWN 天气子目录（如 .../DAWN/fog）")
    ap.add_argument("--dst", required=True, help="输出目录（如 datasets/DAWN_yolo）")
    ap.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例（默认 0.2）")
    ap.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）")
    args = ap.parse_args()

    # 收集所有 xml
    xml_files = []
    for root_dir, _, files in os.walk(args.src):
        for f in files:
            if f.lower().endswith(".xml"):
                xml_files.append(os.path.join(root_dir, f))
    xml_files.sort()

    rng = random.Random(args.seed)
    rng.shuffle(xml_files)
    n_val = max(1, int(len(xml_files) * args.val_ratio))
    val_set = set(xml_files[:n_val])

    splits = {"train": [], "val": []}
    stats = {"total": 0, "skipped": 0}

    img_index = build_image_index(args.src)

    for xml_path in xml_files:
        split = "val" if xml_path in val_set else "train"
        stem = os.path.splitext(os.path.basename(xml_path))[0].lower()
        img_path = img_index.get(stem)
        if img_path is None:
            stats["skipped"] += 1
            print(f"[skip] 找不到图片: {xml_path}")
            continue

        _, _, boxes = parse_voc_xml(xml_path)
        if not boxes:
            stats["skipped"] += 1
            print(f"[skip] 无有效目标: {xml_path}")
            continue

        # 目标目录
        img_dst_dir = os.path.join(args.dst, "images", split)
        lbl_dst_dir = os.path.join(args.dst, "labels", split)
        os.makedirs(img_dst_dir, exist_ok=True)
        os.makedirs(lbl_dst_dir, exist_ok=True)

        name = os.path.splitext(os.path.basename(xml_path))[0]
        shutil.copy2(img_path, os.path.join(img_dst_dir, os.path.basename(img_path)))

        with open(os.path.join(lbl_dst_dir, name + ".txt"), "w") as f:
            for cls, cx, cy, bw, bh in boxes:
                f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        splits[split].append(name)
        stats["total"] += 1

    write_yaml(args.dst, CLASSES)

    print("=" * 50)
    print(f"转换完成：共 {stats['total']} 张，跳过 {stats['skipped']} 张")
    print(f"  train: {len(splits['train'])} 张")
    print(f"  val  : {len(splits['val'])} 张")
    print(f"  配置 : {os.path.join(args.dst, 'fog.yaml')}")
    print("=" * 50)


if __name__ == "__main__":
    main()
