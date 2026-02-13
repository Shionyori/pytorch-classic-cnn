import sys
import os

# 将项目根目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch
import torch.utils.data as Data
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from models.alexnet import AlexNet

def test_data_process():
    test_data = FashionMNIST(root='./datasets',
                          train=False, # 加载测试集
                          transform=transforms.Compose([transforms.Resize(size=227), transforms.ToTensor()]),
                          download=True)
    
    test_dataloader = Data.DataLoader(dataset=test_data,
                                       batch_size=64,
                                       shuffle=False, # 测试集不打乱
                                       num_workers=0)

    return test_dataloader

def test_model_process(model, test_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    test_corrects = 0.0
    test_num = 0

    model.eval()
    # 只进行前向传播计算，不计算梯度，节省内存，加快运行速度
    with torch.no_grad():
        for test_data_x, test_data_y in test_dataloader:
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)

            output = model(test_data_x)
            pre_lab = torch.argmax(output, dim=1)
            
            test_corrects += torch.sum(pre_lab == test_data_y.data)

            test_num += test_data_x.size(0)

    test_acc = test_corrects.double().item() / test_num
    print("测试的准确率为：", test_acc)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = AlexNet(1, 10)
    model.load_state_dict(torch.load("./checkpoints/alexnet-fashionminist/best_model.pth", map_location=device, weights_only=True))

    test_dataloader = test_data_process()

    model = model.to(device)
    model.eval()

    # 打印测试集（只打印第一批次的前10个样本）
    with torch.no_grad():
        for b_x, b_y in test_dataloader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            output = model(b_x)
            pre_lab = torch.argmax(output, dim=1)
            for i in range(min(10, len(pre_lab))):
                result = pre_lab[i].item()
                label = b_y[i].item()
                print("预测值：", result, "------", "真实值：", label)
              
            break

    test_model_process(model, test_dataloader)