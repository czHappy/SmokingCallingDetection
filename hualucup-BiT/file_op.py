import os
img_dir = r'../pure_data/train'
for root, dirs, files in os.walk(img_dir):
    for name in files:
        print(name)
        if name[0:3] == 'new':
            print(name)
            os.remove(os.path.join(root, name))
