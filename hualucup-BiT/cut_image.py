import os
import random

num = 0
for image_name in os.listdir("../new_balance_data/train/normal"):
    feed = random.randint(0, 10)
    if feed <= 3:
        os.remove(os.path.join("../new_balance_data/train/normal", image_name))
        # print(feed)
        # print(os.path.join("./data", image_name))
        num = num + 1
print(num)