# 计算训练集和测试集的均值和方差
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

folder_path = 'datasets/fruits/train'  # 替换为你的图像文件夹路径

total_pixels = 0
sum_normalized_pixel_values = np.zeros(3)  # 用于存储每个通道的像素值总和

for root, dirs, files in os.walk(folder_path):
    for filename in files:
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):  # 根据需要的图像格式进行筛选
            image_path = os.path.join(root, filename)
            image = Image.open(image_path)
            image_array = np.array(image)

            normalized_image_array = image_array / 255.0
            total_pixels += normalized_image_array.size
            sum_normalized_pixel_values += np.sum(normalized_image_array, axis=(0, 1))

mean = sum_normalized_pixel_values / total_pixels

sum_squared_diff = np.zeros(3)  # 用于存储每个通道的像素值与均值的平方差总和
for root, dirs, files in os.walk(folder_path):
    for filename in files:
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):  # 根据需要的图像格式进行筛选
            image_path = os.path.join(root, filename)
            image = Image.open(image_path)
            image_array = np.array(image)

            normalized_image_array = image_array / 255.0
            try:
                squared_diff = (normalized_image_array - mean) ** 2
                sum_squared_diff += np.sum(squared_diff, axis=(0, 1))
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")

variance = sum_squared_diff / total_pixels

print("Mean:", mean)
print("Variance:", variance)