import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torchvision import transforms, datasets
from encoding.models.backbone.resnest import resnest101

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--output', default='./output/', help='folder to output images and model checkpoints')  # 输出结果保存路径
parser.add_argument('--models', default='./', help="path to net (to continue training)")  # 恢复训练时的模型路径
parser.add_argument('--epoch', default=200)
parser.add_argument('--batch_size', default=32)
parser.add_argument('--path_train', default='../senet.pytorch/data/train_yolov3')
parser.add_argument('--path_val', default='../senet.pytorch/data/val_yolov3')
args = parser.parse_args()

# 超参数设置
epoch = args.epoch  # 遍历数据集次数
pre_epoch = 0  # 定义已经遍历数据集的次数
batch_size = args.batch_size  # 批处理尺寸(batch_size)
lr = 0.0001  # 学习率

# 准备数据集并预处理
transform_train = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ColorJitter(brightness=0.5),
    # transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),  # 维度转化 由32x32x3  ->3x32x32
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # R,G,B每层的归一化用到的均值和方差     即参数为变换过程，而非最终结果。
])

transform_test = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = datasets.ImageFolder(root=args.path_train, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = datasets.ImageFolder(root=args.path_val, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)


# net = se_resnet50(num_classes=3, pretrained=False).to(device)
net = resnest101(pretrained=True, root='pretrained/resnest101-22405ba7.pth')
fc_features = net.fc.in_features
net.fc = nn.Linear(fc_features, 3)
print(net)
net = nn.DataParallel(net, device_ids=[0,1]).to(device)  # 多卡训练指定GPU


# 定义损失函数和优化方式
criterion = nn.CrossEntropyLoss()  # 损失函数为交叉熵，多用于多分类问题,此标准将LogSoftMax和NLLLoss集成到一个类中。
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9,
                      weight_decay=5e-4)  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
# optimizer = optim.SGD(net.parameters(), lr=lr)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=4, verbose=True)


# 训练
if __name__ == "__main__":
    best_acc = 85  # 2 初始化best test accuracy
    print("Start Training, ResNest-101!")  # 定义遍历数据集的次数
    with open(args.output+"val.txt", "a") as f:
        with open(args.output+"train.txt", "a")as f2:
            for epoch in range(pre_epoch, epoch):  # 从先前次数开始训练
                print('\nEpoch: %d' % (epoch + 1))  # 输出当前次数
                net.train()  # 这两个函数只要适用于Dropout与BatchNormalization的网络，会影响到训练过程中这两者的参数
                # 运用net.train()时，训练时每个min - batch时都会根据情况进行上述两个参数的相应调整，所有BatchNormalization的训练和测试时的操作不同。
                sum_loss = 0.0  # 损失数量
                correct = 0.0  # 准确数量
                total = 0.0  # 总共数量
                for i, data in enumerate(trainloader,
                                         0):  # 训练集合enumerate(sequence, [start=0])用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
                    # 准备数据  i是序号 data是遍历的数据元素
                    length = len(trainloader)  # 训练数量
                    inputs, labels = data  # data的结构是：[4x3x32x32的张量,长度4的张量]
                    # 假想： inputs是当前输入的图像，label是当前图像的标签，这个data中每一个sample对应一个label
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()  # 清空所有被优化过的Variable的梯度.

                    # forward + backward
                    outputs = net(inputs)  # 得到训练后的一个输出
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()  # 进行单次优化 (参数更新).

                    # 每训练1个batch打印一次loss和准确率
                    sum_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)  # 返回输入张量所有元素的最大值。 将dim维设定为1，其它与输入形状保持一致。
                    # 这里采用torch.max。torch.max()的第一个输入是tensor格式，所以用outputs.data而不是outputs作为输入；第二个参数1是代表dim的意思，也就是取每一行的最大值，其实就是我们常见的取概率最大的那个index；第三个参数loss也是torch.autograd.Variable格式。
                    total += labels.size(0)
                    correct += predicted.eq(labels.data).cpu().sum()
                    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('%03d  %05d | Loss: %.03f | Acc: %.3f%% '
                             % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('\n')
                    f2.flush()

                # 每训练完一个epoch测试一下准确率
                print("Waiting Test!")
                with torch.no_grad():  # 没有求导
                    correct = 0
                    total = 0
                    val_sum_loss = 0
                    for i, data in enumerate(testloader):
                        net.eval()  # 运用net.eval()时，由于网络已经训练完毕，参数都是固定的，因此每个min-batch的均值和方差都是不变的，因此直接运用所有batch的均值和方差。
                        images, labels = data
                        images, labels = images.to(device), labels.to(device)
                        outputs = net(images)
                        # 取得分最高的那个类 (outputs.data的索引号)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum()
                        loss = criterion(outputs, labels)
                        val_sum_loss += loss.item()
                    print('测试分类准确率为：%.3f%%' % (100 * correct / total))
                    print('测试分类损失为：%.3f' % (val_sum_loss / 1752))

                    acc = 100. * correct / total
                    # 将每次测试结果实时写入acc.txt文件中
                    print('Saving model......')
                    torch.save(net.state_dict(), '%s/net_%03d.pth' % (args.output, epoch + 1))
                    val_loss = val_sum_loss / 1752
                    f.write("epoch: %03d | Accuracy= %.3f%% | loss= %.3f" % (epoch + 1, acc, val_loss))
                    f.write('\n')
                    f.flush()
                    # 记录最佳测试分类准确率并写入best_acc.txt文件中
                    if acc > best_acc:
                        f3 = open(args.output+"best_acc.txt", "a")
                        f3.write("epoch=%d,best_acc= %.3f%%" % (epoch + 1, acc))
                        f3.close()
                        best_acc = acc
            print("Training Finished, TotalEPOCH=%d" % epoch)