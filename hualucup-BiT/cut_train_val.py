import os
import random
import shutil
import shutil
import numpy
imgs_path = '/home/sk49/new_workspace/dataset/Smoking&TeleDetection/Repecharge/train'
imgs = []
calling_imgs = os.listdir(os.path.join(imgs_path, 'calling'))
for i in range(len(calling_imgs)):
    calling_imgs[i] = 'calling/' + calling_imgs[i]
normal_imgs = os.listdir(os.path.join(imgs_path, 'normal'))
for i in range(len(normal_imgs)):
    normal_imgs[i] = 'normal/' + normal_imgs[i]
smoking_imgs = os.listdir(os.path.join(imgs_path, 'smoking'))
for i in range(len(smoking_imgs)):
    smoking_imgs[i] = 'smoking/' + smoking_imgs[i]
smoking_calling_imgs = os.listdir(os.path.join(imgs_path, 'smoking_calling'))
for i in range(len(smoking_calling_imgs)):
    smoking_calling_imgs[i] = 'smoking_calling/' + smoking_calling_imgs[i]

print('call:{} normal:{} smoking:{} smoking_calling:{}'.format(len(calling_imgs), len(normal_imgs), len(smoking_imgs), len(smoking_calling_imgs)))

random.shuffle(calling_imgs)
random.shuffle(normal_imgs)
random.shuffle(smoking_imgs)
random.shuffle(smoking_calling_imgs)
pos1, pos2, pos3, pos4 = int(len(calling_imgs) * 0.9), int(len(normal_imgs) * 0.9), int(len(smoking_imgs) * 0.9), int(len(smoking_calling_imgs) * 0.9)
train_list = smoking_calling_imgs[:pos4]
val_list = smoking_calling_imgs[pos4:]
new_path = '/home/sk49/new_workspace/jym/sk49_2020/data'
for i in range(len(train_list)):
    shutil.copy2(os.path.join(imgs_path, train_list[i]), os.path.join(new_path, 'train/smoking_calling'))
for i in range(len(val_list)):
    shutil.copy2(os.path.join(imgs_path, val_list[i]), os.path.join(new_path, 'val/smoking_calling'))



# imgs.extend(calling_imgs)
# imgs.extend(normal_imgs)
# imgs.extend(smoking_imgs)
# imgs.extend(smoking_calling_imgs)
# random.shuffle(imgs)
# pos = int(len(imgs) * 0.9)
# train_list = imgs[:pos]
# val_iist = imgs[pos:]
# new_path = '/home/sk49/new_workspace/jym/sk49_2020/data'
# for i in range(len(train_list)):
#     shutil.copy2(os.path.join(imgs_path, train_list[i]), os.path.join(new_path, 'train', train_list[i]))
#     # if 'calling' in train_list[i]:
#     #     shutil.copy2(os.path.join(imgs_path, train_list[i]), os.path.join(new_path, 'train', train_list[i]))
#     # if 'normal' in train_list[i]:
#     #     shutil.copy2(os.path.join(imgs_path, train_list[i]), os.path.join(new_path, 'train', train_list[i]))
#     # if 'calling' in train_list[i]:
#     #     shutil.copy2(os.path.join(imgs_path, train_list[i]), os.path.join(new_path, 'train', train_list[i]))
# for i in range(len(val_iist)):
#     shutil.copy2(os.path.join(imgs_path, val_iist[i]), os.path.join(new_path, 'val', val_iist[i]))
