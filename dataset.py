import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class MNIST(Dataset):
    """ MNIST 데이터셋

    사용자 정의 데이터셋을 작성하려면 다음을 참조하세요:
    https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        data_dir: 이미지가 있는 디렉터리 경로

    주의:
        1) 각 이미지는 다음과 같이 전처리되어야 합니다:
            - 먼저 모든 값은 [0,1] 범위에 있어야 합니다.
            - 0.1307의 평균을 빼고 0.3081로 나눕니다.
            - 이러한 전처리는 torchvision.transforms를 사용하여 구현할 수 있습니다.
        2) 레이블은 파일 이름에서 얻을 수 있습니다: {숫자}_{레이블}.png
    """

    def __init__(self, data_dir):
        """
        데이터셋을 초기화합니다.

        Parameters:
            data_dir (str): 이미지 파일이 있는 디렉터리 경로
        """
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # 이미지를 텐서로 변환합니다.
            transforms.Normalize((0.1307,), (0.3081,))  # 평균 및 표준편차를 이용한 정규화를 수행합니다.
        ])

        self.images = []  # 이미지를 저장할 리스트
        self.labels = []  # 레이블을 저장할 리스트

        # 데이터 디렉터리의 각 이미지 파일을 반복하여 처리합니다.
        for filename in os.listdir(data_dir):
            if filename.endswith(".png"):
                # 파일 이름에서 레이블을 추출합니다.
                label = int(filename.split("_")[1].split(".")[0])
                self.labels.append(label)

                # 이미지를 불러와서 전처리합니다.
                image_path = os.path.join(data_dir, filename)
                image = Image.open(image_path)
                image = self.transform(image)

                self.images.append(image)

    def __len__(self):
        """
        데이터셋의 샘플 수를 반환합니다.
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        주어진 인덱스의 샘플을 반환합니다.

        Parameters:
            idx (int): 반환할 샘플의 인덱스

        Returns:
            tuple: 이미지 텐서와 해당하는 레이블
        """
        img = self.images[idx]
        label = self.labels[idx]

        return img, label

if __name__ == '__main__':
    # 구현 검증을 위한 테스트 코드를 작성합니다.
    data_dir = "C:/Users/yejun/OneDrive/문서/mnist-classification/data/train"
    dataset = MNIST(data_dir)

    # 데이터셋의 길이를 확인합니다.
    print(len(dataset))

    # 샘플 데이터를 확인합니다.
    if len(dataset) > 0:
        img, label = dataset[0]
        print(img.shape, label)
