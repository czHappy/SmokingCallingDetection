from os.path import join as pjoin  # pylint: disable=g-importing-member
import time
import argparse
import json
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import fewshot as fs
import lbtoolbox as lb
import models as models
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"  
device = torch.device('cuda')
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='the path of the model you want to use.', required=True)
parser.add_argument('--test_dir', help='the dir of the test set.', default='/notebooks')
args = parser.parse_args()
datadir = args.test_dir
model_path = args.model


transform_test = transforms.Compose([
    transforms.Resize((450, 450)),
    # transforms.RandomHorizontalFlip(1),
    # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),/
    # transforms.Resize((512, 512)),
    # transforms.RandomCrop((450, 450)),
    # transforms.RandomHorizontalFlip(1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                            (0.5, 0.5, 0.5)),
    #transforms.Normalize((0.40702813539470534, 0.4005526078656559, 0.39464261693601377,), (0.28119672440760013, 0.26599737107605065, 0.2687406301502049)),
])
# transforms = tta.Compose(
#     [
#         tta.Rotate90(angles=[0, 90,180,270]),
#         # tta.Scale(scales=[1, 2, 4]),
#         # tta.Multiply(factors=[0.9, 1, 1.1]),
#     ]
# )
#testset = torchvision.datasets.ImageFolder(os.path.join(datadir, 'testB/test/testA'), transform_test)
testset = torchvision.datasets.ImageFolder(datadir, transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=4) #debug时num_workers必须为0
classname = ['calling', 'normal', 'smoking', 'smoking_calling']
imgs = []
for i in range(len(testset.imgs)):
    imgs.append(testset.imgs[i][0].split('/')[-1])
    #print(testset.imgs[i][0].split('/')[-1])
print("测试集大小: ", len(testloader))

mAP = []
for i in range(4):
    mAP.append([])
if __name__ == '__main__':
    submission = []
    model = models.KNOWN_MODELS['BiT-M-R101x1'](head_size=4, zero_head=True)
    model = torch.nn.DataParallel(model)
    with torch.no_grad():
        model.load_state_dict(torch.load(model_path))
        # model.load_state_dict(torch.load('./net_513.pth'))
        # tta_model = tta.ClassificationTTAWrapper(model, transforms, merge_mode='mean')
        correct = 0
        total = 0
        for i, data in enumerate(testloader):
            value = {'image_name': None, 'category': None, 'score': None}

            model.eval()
            images, label = data
            #print("img_name = ", str(imgs[i]))
            #print("label = ", classname[int(label.cpu().numpy()[0])])
            #if classname[int(label.cpu().numpy()[0])] == 'smoking_calling':
            #    print("it is a s_c example!")
            images, label = images.to(device), label.to(device)

            # outputs = tta_model(images)
            outputs = model(images)
            outputs = nn.functional.softmax(outputs, dim=1)
            # 取得分最高的那个类 (outputs.data的索引号)
            score, predicted = torch.max(outputs.data, 1)
            score, predicted = score.cpu().numpy(), predicted.cpu().numpy()

            # 计算y_true y_pred


            value['category'] = classname[predicted[0]]
            value['score'] = str(score[0])
            value['image_name'] = str(imgs[i])

            submission.append(value)  # 存储预测结果 tojson

    with open('/notebooks/results/hualucup-13611377816.json', 'w') as f:
        json.dump(submission, f)
    print('result.json: done!')


'''PASCAL VOC 2010标准
From 2010 on, the method of computing AP by the PASCAL VOC challenge has changed. 
Currently, the interpolation performed by PASCAL VOC challenge uses all data points, 
rather than interpolating only 11 equally spaced points as stated in their paper. 
'''


