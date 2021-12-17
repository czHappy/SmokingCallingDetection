from os.path import join as pjoin  # pylint: disable=g-importing-member
import time
import json
import sys
import os
sys.path.append('/home/sk49/new_workspace/jym/big_transfer-master')

import numpy as np
import torch
# from torchsummary import summary
import torch.nn as nn
import torchvision
# import ttach as tta
import torchvision.transforms as transforms

import bit_pytorch.fewshot as fs
import bit_pytorch.lbtoolbox as lb
import bit_pytorch.models as models
#
# datadir = '/home/sk49/new_workspace/trj/big_transfer-master/bit_pytorch/data'
datadir = '/home/sk49/new_workspace/jym/big_transfer-master/'
device = torch.device('cuda')
def compute_map(map):
  APs = [] 
  precisions = [] # 计算presions 数组 纵坐标
  recalls= [] # 计算recalls数组 横坐标

  ap = 0 # ap
  tp = 0 # true positive
  fp = 0 # false positive
  fn = 0 # false negative
  # precision = tp / (tp + fp)
  # recall = tp / (tp + fn) 
  for i in range(len(map)):  # 取map[i], 求这个预测类的ap
    for index in range(len(map[i])): # 将这个类的预测信息划分为[0, index] [index+1, end) 前半部分是positive 后半部分是negative 计算该阈值下的tp fp fn 
      for k in range(0, index+1):# 前半部分是positive 用于计算tp fp
        # print('i: {}, k: {}, gt_label: {}'.format(i, k, map[i][k]['gt_label']))
        if(map[i][k]['gt_label']==i): # 如果GT值等于 预测值 tp+1
          tp += 1
        else: #否则fp+1
          fp +=1
      for k in range(index+1,len(map[i])): #后半部分是negative 用于计算fn
        if(map[i][k]['gt_label']==i): # 如果这个
          fn +=1
      precision = tp / (tp+fp) # 计算p r
      recall = tp / (tp+fn)
      # print("index: {}, precision: {}, recall: {}".format(index, precision, recall))
      if recall in recalls: #如果这个recall值之前已经有了，那么这个recall下的最大的precision作为其precision
        ind = recalls.index(recall)
        if precisions[ind] < precision:
          precisions[ind] = precision
      else:  # 如果这个recall值没有出现过则添加一个entry  事实上，recalls和precisions是两个平行数组，下标是对应的
        recalls.append(recall)
        precisions.append(precision)
		
	# 所有的二划分都计算完了，PR曲线已经得到
    for k in range(len(recalls)): # 枚举横坐标
      maxprec = max(precisions[k:]) #取该recall后面的最大precision  AP (Area under curve AUC，PASCAL VOC2010–2012评测指标)
      # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ MAKE CHANGES $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ #
      index_max = precisions.index(maxprec) # 取这个precision的下标，等价于对应recall在recalls中的下标 （由于precisions和recalls是平行数组，故二者共享下标）
      ap += maxprec * (recalls[index_max] - recalls[k]) # 计算面积
      ###
	 # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ MAKE CHANGES $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ #
    #if(len(recalls) != 0):  #这里不是11-points插值而是计算面积 不需要再进行除法平均
    #  ap = ap / len(recalls) 
    #else:
    #  ap = 0

    APs.append(ap)

    precisions = []
    recalls = []
	
    ap = 0
    tp = 0
    fp = 0
    fn = 0
  mAP = tuple(APs)
  print(mAP)
  return float(sum(mAP)/len(mAP))

transform_test = transforms.Compose([
  transforms.Resize((450, 450)),
  # transforms.RandomHorizontalFlip(1),
  # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),/
  # transforms.Resize((512, 512)),
  # transforms.RandomCrop((450, 450)),
  # transforms.RandomHorizontalFlip(1),
  transforms.ToTensor(),
  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
# transforms = tta.Compose(
#     [
#         tta.Rotate90(angles=[0, 90,180,270]),
#         # tta.Scale(scales=[1, 2, 4]),
#         # tta.Multiply(factors=[0.9, 1, 1.1]),
#     ]
# )
testset = torchvision.datasets.ImageFolder(os.path.join(datadir, 'test'), transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)
classname = ['calling', 'normal', 'smoking']
imgs = []
for i in range(len(testset.imgs)):
    imgs.append(testset.imgs[i][0].split('/')[-1])
print(len(testloader))

mAP = []
for i in range(3):
  mAP.append([])
if __name__ == '__main__':
  submission = []
  wrong_item = []
  best_map = 0
  model = models.KNOWN_MODELS['BiT-M-R101x1'](head_size=3, zero_head=True)
  model = torch.nn.DataParallel(model)
  with torch.no_grad():
    model.load_state_dict(torch.load('/home/sk49/new_workspace/jym/big_transfer-master/models/sc_10_10_R101x1_4/net_500.pth'))
    # model.load_state_dict(torch.load('./net_513.pth'))
    # tta_model = tta.ClassificationTTAWrapper(model, transforms, merge_mode='mean')
    correct = 0
    total = 0
    for i, data in enumerate(testloader):
      value = {'image_name': None, 'category': None, 'score': None}
      wrong = {'image_name': None, 'label': None, 'predicted:': None}
      tmp = {'image_name': None, 'score': None, 'gt_label': None}

      model.eval()
      images, label = data
      images, label = images.to(device), label.to(device)
      # outputs = tta_model(images)
      outputs = model(images)
      outputs = nn.functional.softmax(outputs, dim=1)
      # 取得分最高的那个类 (outputs.data的索引号)
      score, predicted = torch.max(outputs.data, 1)
      score, predicted = score.cpu().numpy(), predicted.cpu().numpy()

      value['category'] = classname[predicted[0]]
      value['score'] = str(score[0])
      value['image_name'] = str(imgs[i])

      tmp['image_name'] = str(imgs[i])
      tmp['score'] = float(score[0])
      tmp['gt_label'] = int(label.cpu().numpy()[0])
      mAP[int(predicted[0])].append(tmp) # 把每个类的预测信息存储起来

      submission.append(value) # 存储预测结果 tojson
      total += label.size(0) # 计数
      if predicted[0] == label.cpu().numpy()[0]: # 如果预测的标签和GT相同则correct+1
        correct += 1
      else:
        wrong['image_name'] = str(imgs[i]) # 存储哪些图片分类错了
        wrong['label'] = classname[label.cpu().numpy()[0]]
        wrong['predicted:'] = classname[predicted[0]]
        wrong_item.append(wrong)
    print('测试分类准确率为：%.3f%%' % (100 * correct / total)) # 计算accuracy
    acc = 100. * correct / total

    for j in range(len(mAP)):
      mAP[j].sort(key=lambda x: (x.get('score', 0)), reverse=True) # 每个类的预测信息按socre降序排列  这里get('score',0) 0是默认值 如果不存在该键就插入默认键值对
    map = compute_map(mAP) # 计算mAP
    if(map>best_map): # 只保留最好的mAP 
      best_map = map
      print('best_map: {}, model: {}'.format(best_map, 1))
    print('done!')

  with open('result.json', 'w') as f:
    json.dump(submission, f)
  print('result.json: done!')
  # with open('wrong.json', 'w') as f:
  #   for i in range(len(wrong_item)):
  #     json.dump(wrong_item[i], f)
  #     f.write('\n')
  
  
'''PASCAL VOC 2010标准
From 2010 on, the method of computing AP by the PASCAL VOC challenge has changed. 
Currently, the interpolation performed by PASCAL VOC challenge uses all data points, 
rather than interpolating only 11 equally spaced points as stated in their paper. 
'''


