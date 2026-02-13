# 随机划分数据集为训练集和测试集
import os
import random
from shutil import copy

def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

# 原始数据集路径
dir_path = 'datasets/fruits/raw'
classes = [c for c in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, c))]  # 仅保留目录

# 创建训练集和测试集目录
make_path(dir_path.replace('raw', 'train'))
make_path(dir_path.replace('raw', 'test'))
for c in classes:
    make_path(os.path.join(dir_path.replace('raw', 'train'), c))
    make_path(os.path.join(dir_path.replace('raw', 'test'), c))

# 划分比例：测试集占 10%
split_ratio = 0.1
random.seed(42)  # 设置随机种子，确保结果可复现

# 遍历每个类别
for c in classes:
    class_path = os.path.join(dir_path, c)
    images = [f for f in os.listdir(class_path) 
              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]  # 过滤非图像文件
    
    if not images:
        print(f"Warning: Class '{c}' has no images. Skipping.")
        continue
    
    random.shuffle(images)  # 随机打乱图像列表
    num_test = max(1, int(len(images) * split_ratio))  # 确保至少1张测试图
    test_images = set(images[:num_test])  # 转为set加速查找
    
    for image in images:
        src_path = os.path.join(class_path, image)
        if image in test_images:
            dst_path = os.path.join(dir_path.replace('raw', 'test'), c, image)
        else:
            dst_path = os.path.join(dir_path.replace('raw', 'train'), c, image)
        copy(src_path, dst_path)  # 复制文件
    
    print(f"Class '{c}': {len(test_images)} test images, {len(images) - len(test_images)} train images.")

print("Processing completed.")