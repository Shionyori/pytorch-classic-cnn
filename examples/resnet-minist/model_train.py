
import sys
import os

# 将项目根目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from torchvision import transforms
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
from models.resnet import ResNet18
import torch
from torch import nn
import copy
import time
import pandas as pd
from torchvision.datasets import ImageFolder
from torchvision.datasets import MNIST

def train_val_data_process(seed):
    # 数据预处理（替换为计算得到的均值和方差）
    normalize = transforms.Normalize(mean=[0.1307], std=[0.3081])

    train_transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), normalize])

    # 加载数据集
    train_data = MNIST(root='./datasets',
                          train=True,
                          transform=train_transform,
                          download=True)

    # 划分训练集和验证集
    seed_generator = torch.Generator().manual_seed(seed)  # 设置随机种子以确保可重复性

    train_data, val_data = Data.random_split(train_data, [round(0.8*len(train_data)), round(0.2*len(train_data))], generator=seed_generator)

    train_dataloader = Data.DataLoader(dataset=train_data,
                                       batch_size=128,
                                       shuffle=True,
                                       num_workers=0)
    val_dataloader = Data.DataLoader(dataset=val_data,
                                     batch_size=128,
                                     shuffle=False,
                                     num_workers=0)
    
    return train_dataloader, val_dataloader

def train_model_process(model, train_dataloader, val_dataloader, num_epochs, patience):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_dir = './checkpoints/resnet-mnist/'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # 使用Adam优化器，学习率设为0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 学习率衰减，每5个epoch衰减为原来的0.1
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

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

    no_improve_epochs = 0  # 记录验证集准确度没有提升的epoch数

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
            with torch.no_grad(): # 在验证阶段不需要计算梯度，节省显存+加速
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

        scheduler.step() # 更新学习率

        print('{} Train Loss: {:.4f} Train Acc: {:.4f}'.format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print('{} Val Loss: {:.4f} Val Acc: {:.4f}'.format(epoch, val_loss_all[-1], val_acc_all[-1]))

        # 保存当前模型的参数
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
            no_improve_epochs = 0  # 重置计数
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"验证集准确度连续{patience}个epoch没有提升，提前停止训练。")
                break

        time_use = time.time() - since
        print("训练和验证耗费的时间{:.0f}m{:.0f}s".format(time_use//60, time_use%60))

    # 当前只保存最后的（最佳）模型
    torch.save(best_model_wts , checkpoint_dir + 'best_model.pth')
    # 记录训练过程的数据
    actual_epochs = len(train_loss_all)  # 实际训练的epoch数（可能因为提前停止而少于num_epochs）
    train_process = pd.DataFrame(data={"epoch": range(actual_epochs),
                                       "train_loss_all":train_loss_all[:actual_epochs],
                                       "val_loss_all":val_loss_all[:actual_epochs],
                                       "train_acc_all":train_acc_all[:actual_epochs],
                                       "val_acc_all":val_acc_all[:actual_epochs]})
    
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
    model = ResNet18(1, 10) # 输入通道=1, 类别数=10
    train_dataloader, val_dataloader = train_val_data_process(seed=42)
    train_process = train_model_process(model, train_dataloader, val_dataloader, num_epochs=15, patience=5)
    matplot_acc_loss(train_process)