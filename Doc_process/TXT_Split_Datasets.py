import os
import shutil

images_dir = "DataSets\\dogs-vs-cat\\train\\MyTrainning_data"
train_txt = "DataSets/dogs-vs-cat/train/labels/Train.txt"
labels_txt = "DataSets/dogs-vs-cat/train/labels/synsets.txt"
output_dir = "DataSets/dataset"

# ===== 读取 labels.txt（兼容多种格式）=====
id_to_label = {}

with open(labels_txt, "r", encoding="utf-8") as f:
    lines = [l.strip() for l in f if l.strip()]

for i, line in enumerate(lines):
    if " " in line:
        idx, name = line.split(maxsplit=1)
    elif "," in line:
        idx, name = line.split(",", 1)
    else:
        idx = str(i)
        name = line

    id_to_label[idx] = name

print("类别映射：", id_to_label)

# 创建类别文件夹
os.makedirs(output_dir, exist_ok=True)
for label in id_to_label.values():
    os.makedirs(os.path.join(output_dir, label), exist_ok=True)

# ===== 读取 train.txt 并拷贝图片 =====
with open(train_txt, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue

        img_path, label_id = line.strip().split()
        label_name = id_to_label[label_id]

        # 修正这里：使用 images_dir 而不是 "data"
        src = os.path.join(images_dir, os.path.basename(img_path))
        dst = os.path.join(output_dir, label_name, os.path.basename(img_path))

        shutil.copy(src, dst)
        print(f"拷贝: {os.path.basename(img_path)} -> {label_name}")

print("✅ 数据集整理完成")