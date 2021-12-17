from PIL import Image
import os

dir_img = "/home/sk49/new_workspace/jym/sk49_2020/data/val/smoking/"
# 待处理的图片地址
dir_save = "/home/sk49/new_workspace/jym/sk49_2020/data/val/smoking/"
# 水平镜像翻转后保存的地址

list_img = os.listdir(dir_img)

for img_name in list_img:
    pri_image = Image.open(dir_img + img_name)
    tmppath = dir_save + "new_" + img_name
    pri_image.transpose(Image.FLIP_LEFT_RIGHT).convert('RGB').save(tmppath)
    print(str(img_name) + ": done")