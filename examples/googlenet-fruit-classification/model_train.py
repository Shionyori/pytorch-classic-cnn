import sys
import os

# 将项目根目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from torchvision import transforms
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
from models.googlenet import GoogleNet
import torch
from torch import nn
import copy
import time
import pandas as pd
from torchvision.datasets import ImageFolder

def train_val_data_process():
    # 数据集路径
    train_root = './datasets/fruits/train'

    # 数据预处理（替换为计算得到的均值和方差）
    normalize = transforms.Normalize(mean=[0.276, 0.247, 0.197], std=[0.119, 0.106, 0.099]) 

    train_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])

    # 加载数据集
    train_data = ImageFolder(root=train_root, transform=train_transform)

    train_data, val_data = Data.random_split(train_data, [round(0.8*len(train_data)), round(0.2*len(train_data))])
    
    train_dataloader = Data.DataLoader(dataset=train_data,
                                       batch_size=64,
                                       shuffle=True,
                                       num_workers=2)
    val_dataloader = Data.DataLoader(dataset=val_data,
                                     batch_size=64,
                                     shuffle=False,
                                     num_workers=2)
    
    return train_dataloader, val_dataloader

def train_model_process(model, train_dataloader, val_dataloader, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_dir = './checkpoints/googlenet-fruit-classification/'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # 使用Adam优化器，学习率设为0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 损失函数为交叉熵函数
    criterion = nn.CrossEntropyLoss()
    # 将模型放入训练设备
    model = model.to(device)
    # 保存当前模型的参数
    best_model_wts = copy.deepcopy(model.state_dict())
 
    # 初始化参数
    # 最高准确度
    best_acc = 0.0
    # 损失值
    train_loss_all = []
    val_loss_all = []
    # 准确度
    train_acc_all = []
    val_acc_all = []
    # 当前时间
    since = time.time()

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs-1))
        print("-"*10)

        # 初始化参数
        # 训练集
        train_loss = 0.0
        train_corrects = 0

        # 验证集
        val_loss = 0.0
        val_corrects = 0

        # 样本数量
        train_num = 0
        val_num = 0

        for step, (b_x, b_y) in enumerate(train_dataloader):
            # 将特征放入训练设备
            b_x = b_x.to(device)
            # 将标签放入训练设备
            b_y = b_y.to(device)
            # 设置模型为训练模式
            model.train()

            output = model(b_x)

            pre_lab = torch.argmax(output, dim=1)

            loss = criterion(output, b_y)

            # 梯度初始化为0
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()

            train_loss += loss.item() * b_x.size(0)
            train_corrects += torch.sum(pre_lab == b_y.data)

            train_num += b_x.size(0)

        for step, (b_x, b_y) in enumerate(val_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            model.eval()
            output = model(b_x)
            pre_lab = torch.argmax(output, dim=1)
            loss = criterion(output, b_y) 

            val_loss += loss.item() * b_x.size(0)
            val_corrects += torch.sum(pre_lab == b_y.data)

            val_num += b_x.size(0)
            
        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.double().item() / train_num)

        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_corrects.double().item() / val_num)

        print('{} Train Loss: {:.4f} Train Acc: {:.4f}'.format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print('{} Val Loss: {:.4f} Val Acc: {:.4f}'.format(epoch, val_loss_all[-1], val_acc_all[-1]))

        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]

            best_model_wts = copy.deepcopy(model.state_dict())

        time_use = time.time() - since
        print("训练和验证耗费的时间{:.0f}m{:.0f}s".format(time_use//60, time_use%60))

    torch.save(best_model_wts , checkpoint_dir + 'best_model.pth')

    train_process = pd.DataFrame(data={"epoch": range(num_epochs),
                                       "train_loss_all":train_loss_all,
                                       "val_loss_all":val_loss_all,
                                       "train_acc_all":train_acc_all,
                                       "val_acc_all":val_acc_all})
    
    return train_process

def matplot_acc_loss(train_process):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(train_process["epoch"], train_process.train_loss_all, 'ro-', label="train loss")
    plt.plot(train_process["epoch"], train_process.val_loss_all, 'bs-', label="val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")

    plt.subplot(1,2,2)
    plt.plot(train_process["epoch"], train_process.train_acc_all, 'ro-', label="train acc")
    plt.plot(train_process["epoch"], train_process.val_acc_all, 'bs-', label="val acc")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    GoogleNet = GoogleNet(3, 5)
    train_dataloader, val_dataloader = train_val_data_process()
    train_process = train_model_process(GoogleNet, train_dataloader, val_dataloader, 20)
    matplot_acc_loss(train_process)