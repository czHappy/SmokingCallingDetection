import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from encoding.models.backbone.resnest import resnest101, resnest200
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import json

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--path', default='./output/net_023.pth', help="path to net (to continue training)")  # 恢复训练时的模型路径


args = parser.parse_args()
transform_test = transforms.Compose([
    transforms.Resize((250, 200)),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

testset = datasets.ImageFolder(root='/notebooks/', transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

# Cifar-10的标签
classes = ['calling', 'normal', 'smoking', 'smoking_calling']

# 模型定义-ResNet
net = resnest101()
fc_features = net.fc.in_features
net.fc = nn.Linear(fc_features, 4)
net = torch.nn.DataParallel(net, device_ids=[0])  # 多卡训练必须 否则无法导入模型 字段问题
modelState = torch.load(args.path)
net.load_state_dict(modelState)
print(net)

# 训练
if __name__ == "__main__":
    print("Start Testing, ResNest-101!")  # 定义遍历数据集的次数
    list = []
    with torch.no_grad():  # 没有求导
        correct = 0
        total = 0
        for i,data in enumerate(testloader):
            file_name=testloader.dataset.imgs[i][0].split('/')[-1]
            # print(file_name)
            net.eval()  # 运用net.eval()时，由于网络已经训练完毕，参数都是固定的，因此每个min-batch的均值和方差都是不变的，因此直接运用所有batch的均值和方差。
            images, _ = data
            images = images.to(device)
            outputs = net(images)

            p = outputs.data.exp()
            sum = torch.sum(p, 1)
            p = p / sum
            # print(p)

            # 取得分最高的那个类 (outputs.data的索引号)
            score, predicted = torch.max(p, 1)
            # print(torch.max(outputs.data, 1))
            # print(predicted, score)

            dict = {}
            dict['image_name'] = file_name
            dict['category'] = classes[predicted[0].item()]
            dict['score'] = score[0].item()

            list.append(dict)
            print(dict)

        with open('/notebooks/results/hualucup-18310588600.json', 'w') as f:
            json.dump(list, f)

    print("Testing Finished!")
