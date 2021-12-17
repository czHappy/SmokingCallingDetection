from os.path import join as pjoin  # pylint: disable=g-importing-member
import time
import json
import sys
import os

import numpy as np
import torch
# from torchsummary import summary
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from encoding.models.backbone.resnest import resnest101, resnest200, resnest50

datadir = '/home/sk49/new_workspace/mengjunfeng/hualu/data/dataB_plus/test'
device = torch.device('cuda')


def compute_map(map):
    APs = []
    precisions = []
    recalls = []

    ap = 0
    tp = 0
    fp = 0
    fn = 0
    for i in range(len(map)):
        for index in range(len(map[i])):
            for k in range(0, index + 1):
                # print('i: {}, k: {}, gt_label: {}'.format(i, k, map[i][k]['gt_label']))
                if (map[i][k]['gt_label'] == i):
                    tp += 1
                else:
                    fp += 1
            for k in range(index + 1, len(map[i])):
                if (map[i][k]['gt_label'] == i):
                    fn += 1
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            # print("index: {}, precision: {}, recall: {}".format(index, precision, recall))
            if recall in recalls:
                ind = recalls.index(recall)
                if precisions[ind] < precision:
                    precisions[ind] = precision
            else:
                recalls.append(recall)
                precisions.append(precision)
        for k in range(len(recalls)):
            maxprec = max(precisions[k:])
            ap += maxprec
        if (len(recalls) != 0):
            ap = ap / len(recalls)
        else:
            ap = 0

        APs.append(ap)

        precisions = []
        recalls = []

        ap = 0.00000000000000000000001
        tp = 0.00000000000000000000001
        fp = 0.00000000000000000000001
        fn = 0.00000000000000000000001
    mAP = tuple(APs)
    print(mAP)
    return float(sum(mAP) / len(mAP))


classname = ['calling', 'normal', 'smoking', 'smoking_calling']
imgs = []

transform_test = transforms.Compose([
    transforms.Resize((250, 200)),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
	# transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

testset = datasets.ImageFolder(root=datadir, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

mAP = []
for i in range(4):
    mAP.append([])
if __name__ == '__main__':
    submission = []
    wrong_item = []
    best_map = 0
    model_path = 'output/net_023.pth'
    net = resnest101()
    fc_features = net.fc.in_features
    net.fc = nn.Linear(fc_features, 4)
    net = torch.nn.DataParallel(net, device_ids=[0])  # 多卡训练必须 否则无法导入模型 字段问题
    modelState = torch.load(model_path)
    net.load_state_dict(modelState)
    with torch.no_grad():
        correct = 0
        total = 0
        for i, data in enumerate(testloader):
            file_name = testloader.dataset.imgs[i][0].split('/')[-1]
            print(file_name)
            net.eval()
            value = {'image_name': None, 'category': None, 'score': None}
            wrong = {'image_name': None, 'label': None, 'predicted:': None}
            tmp = {'image_name': None, 'score': None, 'gt_label': None}

            images, label = data
            images, label = images.to(device), label.to(device)
            outputs = net(images)
            outputs = nn.functional.softmax(outputs, dim=1)
            # 取得分最高的那个类 (outputs.data的索引号)
            scores, predicted = torch.max(outputs.data, 1)
            scores, predicted = scores.cpu().numpy(), predicted.cpu().numpy()
            value['category'] = classname[predicted[0]]
            value['score'] = str(scores[0])
            value['image_name'] = file_name

            tmp['image_name'] = file_name
            tmp['score'] = float(scores[0])
            tmp['gt_label'] = int(label.cpu().numpy()[0])
            mAP[int(predicted[0])].append(tmp)

            submission.append(value)
            total += label.size(0)
            if predicted[0] == label.cpu().numpy()[0]:
                correct += 1
            else:
                wrong['image_name'] = file_name
                wrong['label'] = classname[label.cpu().numpy()[0]]
                wrong['predicted:'] = classname[predicted[0]]
                wrong_item.append(wrong)
        print('测试分类准确率为：%.3f%%' % (100 * correct / total))
        acc = 100. * correct / total

        for j in range(len(mAP)):
            mAP[j].sort(key=lambda x: (x.get('score', 0)), reverse=True)
        map = compute_map(mAP)
        if (map > best_map):
            best_map = map
            print('best_map: {}'.format(best_map))
        print('done!')
