import torch.nn as nn

class LeNet5(nn.Module):
    """ LeNet-5 (LeCun et al., 1998)

        - 자세한 아키텍처는 강의 노트를 참조하세요
        - 활성화 함수는 자유롭게 선택하세요
        - 하향 샘플링에 대해서는 kernel_size = (2,2)의 max pooling을 사용하세요
        - 출력은 로짓 벡터여야 합니다
    """

    def __init__(self):
        super(LeNet5, self).__init__()

        # Convolutional Layer
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)

        # Fully Connected Layers
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, img):
        # Convolutional Layer -> ReLU -> Max Pooling
        x = nn.functional.relu(self.conv1(img))
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        # Convolutional Layer -> ReLU -> Max Pooling
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        # Flatten
        x = x.view(-1, 16 * 4 * 4)
        # Fully Connected Layers -> ReLU
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        # Output Layer
        output = self.fc3(x)

        return output


class CustomMLP(nn.Module):
    """ 사용자 정의 MLP 모델

        - 모델 매개변수 수는 LeNet-5와 약간 동일하게 유지되어야 합니다
    """

    def __init__(self):
        super(CustomMLP, self).__init__()

        # Fully Connected Layers
        self.fc1 = nn.Linear(28 * 28, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, img):
        # Flatten
        x = img.view(-1, 28 * 28)
        # Fully Connected Layers -> ReLU
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        # Output Layer
        output = self.fc3(x)

        return output
