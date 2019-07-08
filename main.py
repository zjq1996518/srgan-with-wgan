from src.model.SRGAN import SRGAN
from sys import argv

if __name__ == '__main__':

    srGAN = SRGAN(argv[1], argv[2], argv[3])
    srGAN.train()
