from src.model.SRGAN import SRGAN
import sys
import os

if __name__ == '__main__':

    # 读取根目录绝对路径
    cur_path = os.path.abspath(os.path.dirname(__file__))
    root_path = os.path.split(cur_path)[0]
    sys.path.append(root_path)

    srGAN = SRGAN()
    srGAN.train()
