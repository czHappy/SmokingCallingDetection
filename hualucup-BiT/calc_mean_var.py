import os
import cv2 as cv
import numpy as np
import sys
from tqdm import tqdm
from time import time

def get_all_files(img_dir):
    img_list = []
    for rt, ds, fs in os.walk(img_dir):
        for f in fs :
            img = os.path.join(rt, f)
            img_list.append(img)
    return img_list
imgs = get_all_files(r'../pure_data/train')
print(len(imgs))
# 测试集 mean =  (0.39464261693601377, 0.4005526078656559, 0.40702813539470534) BGR
# var =  (0.2687406301502049, 0.26599737107605065, 0.28119672440760013)

def calc_mean(img_list):
    print('==> calc mean......')
    #img_list = os.listdir(img_dir)
    mean_b = mean_g = mean_r = 0
    for img_name in tqdm(img_list):
        img = cv.imread(img_name) / 255.0
        mean_b += np.mean(img[:, :, 0])
        mean_g += np.mean(img[:, :, 1])
        mean_r += np.mean(img[:, :, 2])

    mean_b /= len(img_list)
    mean_g /= len(img_list)
    mean_r /= len(img_list)

    return mean_b, mean_g, mean_r

def calc_var(img_list, mean=None):
    print('==> calc var......')
    #img_list = os.listdir(img_dir)
    if mean is not None:
        mean_b, mean_g, mean_r = mean
    else:
        mean_b, mean_g, mean_r = calc_mean(img_list)
    var_b = var_g = var_r = 0
    total = 0
    for img_name in tqdm(img_list):
        img = cv.imread(img_name) / 255.0
        var_b += np.sum(np.power(img[:, :, 0] - mean_b, 2))
        var_g += np.sum(np.power(img[:, :, 1] - mean_g, 2))
        var_r += np.sum(np.power(img[:, :, 2] - mean_r, 2))
        total += np.prod(img[:, :, 0].shape)

    var_b = np.sqrt(var_b / total)
    var_g = np.sqrt(var_g / total)
    var_r = np.sqrt(var_r / total)

    return var_b, var_g, var_r

mean = calc_mean(imgs)
print("mean = ", mean) #多值返回 返回的是元组
var = calc_var(imgs, mean)
print("var = " , var)
import json
with open("mean_val_origin.json", 'w') as f:
    json.dump((mean, var), f)



