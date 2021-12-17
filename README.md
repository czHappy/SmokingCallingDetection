# Smoking and Calling Detection
## Info
- Smoking and Calling Detection Challenge of HuaLu Cup 2020, Rank 9.
  - Solved by Classfication.
  - 4Labels: Normal, Calling, Smoking, Calling&Smoking.
- Explored models including ResNeSt and BiT, choose ResNeSt in the end.
  - Pretrained on ImageNet and tranfered to this dataset.
  - Data Augment such as resize, ColorJitter,RandomHorizontalFlip, RandomRotation and so on.
  - Dropout for Overfitting.
  - Adam Optimizer, MultiStepLR.
- mAP of Semifinals on A Rank list : 97.55
- mAP of Semifinals on B Rank list : 89.95
- Final Contest: Rank 9.
- Members: J.-F. Meng, Z. Cheng, C.-W. Deng, SK49B of 5G&Multimedia Lab.

## Usage (ResNeSt)
- Clone repo and cd into hualucup-ResNeSt
- Install Requirements.
- train
    ```py
    python train.py 
    ```
- test
    ```py
    python predict.py
    ```

- docker
  - download docker image with BaiduDisk, please [contact to me](https://github.com/czHappy) and I will send you link and code. 
  - Install nvidia-docker.
  - run docker image
    ```py
    docker run --shm-size 32G -it hualucup-18310588600:dcw_update /bin/bash
    cd /root/hualucup
    # predict
    python predict.py
    # train
    python train.py 
    # ...
    ```
