---
layout: post
title: "[모두를 위한 딥러닝 시즌2] Lab-10-5 Advance CNN(VGG)"
date: 2024-11-17 03:17:00+0900
categories: [Study, PyTorch]
tags: [Deep Learning Zero To All, 모두를 위한 딥러닝 시즌2, ML, DL, pytorch]
math: true
mermaid : true
---
## 이론적인 설명은 어디서?

**모두를 위한 딥러닝 시즌 1**

[https://www.youtube.com/watch?v=KbNbWTnlYXs&list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm&index=37&ab_channel=SungKim](https://www.youtube.com/watch?v=KbNbWTnlYXs&list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm&index=38)

**PyTorch Lecture 11: Advanced CNN**

[https://www.youtube.com/watch?v=hqYfqNAQIjE&ab_channel=SungKim](https://www.youtube.com/watch?v=hqYfqNAQIjE)

- 영상 요약
    
    1. 4차 산업혁명에서의 이미지 처리 응용 사례
    
       - 이미지 인식과 처리 기술이 4차 산업혁명에서 중요한 역할을 하며, 다양한 산업에서 이미지 처리 기술이 응용되고 있습니다.
       - ConvNet과 ResNet 등의 딥러닝 기술이 이미지 분석의 정확도를 크게 향상시켰습니다.
    
    2. ResNet의 구조와 혁신적 접근
    
       - **ResNet**은 2015년에 개발된 딥러닝 모델로, "잔차 연결(residual connection)"을 도입하여 깊은 신경망에서 발생하는 소실 기울기 문제를 해결합니다.
       - **잔차 학습**을 통해 더 깊은 네트워크에서도 학습 효율을 높일 수 있으며, 이를 통해 ImageNet 대회에서 우수한 성과를 거뒀습니다.
    
    3. 딥러닝의 발전과 ResNet의 성과 (2014-2015년)
    
       - 딥러닝의 빠른 발전과 함께 2014-2015년 사이에 VGGNet, Inception, ResNet 등 다양한 고성능 모델이 발표되었습니다.
       - 특히 ResNet은 깊은 네트워크 구조를 성공적으로 구현해냄으로써, 여러 이미지 인식 대회에서 우승을 차지하며 표준 모델로 자리 잡았습니다.
    
    4. 고급 CNN 구조와 Inception 구조
    
       - Inception 모델은 여러 크기의 필터(1x1, 3x3, 5x5)를 동시에 적용해 다양한 스케일의 특징을 추출하는 구조로, 연산 효율성을 높입니다.
       - **1x1 합성곱**을 활용하여 채널 수를 줄여 연산량을 감소시키는 전략을 채택해, 복잡한 모델에서도 계산 효율을 유지합니다.
    
    5. 1x1 합성곱의 중요성과 계산 효율성
      
      - 1x1 합성곱은 **채널을 조정**하고, 연산량을 크게 줄일 수 있어 Inception과 ResNet 같은 모델에서 자주 사용됩니다.
      - 5x5 필터 대신 1x1 필터를 활용하면 연산량을 최대 10배까지 줄일 수 있어, 큰 이미지를 효율적으로 처리할 수 있습니다.
      
    6. 소실 기울기 문제와 해결 방안
    
       - 딥 네트워크에서는 레이어가 깊어질수록 소실 기울기 문제(vanishing gradient)가 발생하여 학습이 어려워질 수 있습니다.
       - **잔차 연결**을 도입한 ResNet은 입력값을 다음 레이어 출력에 더하여 기울기 전파를 돕고, 이러한 문제를 효과적으로 완화합니다.
    
    7. 잔차 학습의 필요성과 구조
    
       - 잔차 학습을 통해 깊은 네트워크에서 학습 성능을 보장하며, ResNet은 **bypassing connections**를 활용해 기울기를 잘 전달하도록 합니다.
       - 1x1 합성곱을 사용해 연산량을 줄이며, 잔차 학습이 다양한 모델에 성공적으로 적용될 수 있음을 보여줍니다.
    
    8. ResNet과 다른 흥미로운 네트워크들
    
       - **ResNet**은 2015년 ImageNet 대회와 여러 주요 트랙에서 우수한 성과를 거두며 딥러닝의 표준으로 자리 잡았습니다.
       - DenseNet, Inception 등 ResNet 이후 다양한 네트워크들이 발표되었으며, 이들 모델은 ResNet의 잔차 학습 개념을 변형하여 적용하고 있습니다.
       - 다음 단계에서는 이러한 네트워크의 구현과 학습 과정을 통해 실전에서의 활용 가능성을 탐구합니다.
    

## VGG-net

- 영국 옥스포드 대학의 VGG(Visual Geometry Group) 연구팀에 의해 개발된 모델
- 2014년 이미지넷 이미지 인식 대회에서 준우승
- 의도 : 네트워크를 깊이 쌓으면 효과가 있는가? -> 단순하고 일관된 네트워크 구조로 통제하여 확인
- VGGNet 모델을 시작으로 네트워크의 깊이가 증가함
- 이후 ResNet, DenseNet 등 더 깊은 네트워크가 발전하는 계기
- **구조적 특징**
    - **고정된 필터 크기**: 모든 컨볼루션 레이어에서 3x3 크기의 필터를 사용. 작은 필터를 여러 층으로 쌓아 복잡한 특성을 효과적으로 학습
    - **풀링**: 각 컨볼루션 블록 뒤에는 Max Pooling 레이어가 있어, 공간 크기를 줄이면서도 주요 특성만 남김
    - **완전 연결(FC) 레이어**: 마지막에는 두 개의 4096 뉴런 FC 레이어와 소프트맥스 활성화 함수가 적용된 1000개의 클래스를 위한 최종 FC 레이어가 있음
    

![image.png](assets/img/posts/pytorch/10-5/image.png)

![image.png](assets/img/posts/pytorch/10-5/image%201.png)

### VGG 16

- 3x3 Conv : stride 1, padding 1

![image.png](assets/img/posts/pytorch/10-5/image%202.png)

### torchvision.models.vgg

- vgg11~vgg19 까지 만들 수 있음
- 3(RGB)x224x224 input
- 입력 이미지 크기가 달라지는 경우 : 컨볼루션 레이어를 거친 후의 텐서 크기도 변하게 되므로 FC Layer의 첫 번째 `nn.Linear` 레이어의 입력 크기를 이에 맞게 수정해야 함
- 수정 방법
    1. 입력 이미지 크기에 따라 컨볼루션과 풀링 과정을 거친 후의 최종 텐서 크기 계산
    2. 변경된 텐서 크기 반영: 예를 들어, 마지막 컨볼루션 층의 출력 크기가 `(512, H, W)`가 되었다면, `nn.Linear(512 * H * W, 4096)`으로 수정
        
        ```python
        # 입력 이미지가 224x224보다 작은 128x128이라면, 마지막 풀링 이후 텐서 크기가 (512, 4, 4)로 변할 수 있다. 이를 반영하여 다음과 같이 변경
        # 128x128 -> 64x64 -> 32x32 -> 16x16 -> 4x4
        nn.Linear(512 * 4 * 4, 096)
        ```
        

### **Advanced-CNN(VGG)**

```python
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]

# ImageNet 챌린지 데이터셋을 기반으로 사전 학습된 모델 URL들
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()

        # Convolution
        self.features = features 
        
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # FC Layer
        self.classifier = nn.Sequential(
            # 이미지 사이즈에 따라 수정해야 함
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x) # Convolution
        x = self.avgpool(x)  # avgpool
        x = x.view(x.size(0), -1) # 일렬로 펼침 (평탄화)
        x = self.classifier(x) # FC layer
        return x

    def _initialize_weights(self):
        # features의 값
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # activation function에 따라 초기화
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def make_layers(cfg, batch_norm=False):
    # 1. 빈 레이어 생성, input channel = 3
    layers = []
    in_channels = 3
    
    # 2. cfg 에서 v 반복
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            # 채널 수
            in_channels = v
                     
    return nn.Sequential(*layers)
    
# 'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
# 빈 레이어 생성, input channel = 3
# 1. v = 64
# conv2d = nn.Conv2d(3, 64, kernel_size=3, padding=1)
# layers += [conv2d, nn.ReLU(inplace=True)]
# in_channels = 64
# 2. v = 'M'
# layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
# 3. v = 128
# conv2d = nn.Conv2d(64, 128, kernel_size=3, padding=1)
# layers += [conv2d, nn.ReLU(inplace=True)]
# in_channels = 128
# 4. v = 'M'
# layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
# 5. v = 256
# conv2d = nn.Conv2d(128, 256, kernel_size=3, padding=1)
# layers += [conv2d, nn.ReLU(inplace=True)]
# in_channels = 256
# 6. v = 256
# conv2d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
# layers += [conv2d, nn.ReLU(inplace=True)]
# in_channels = 256
# ...

# VGG 모델 설정 (conv + fc)
cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], # vgg11: 8 + 3 = 11 계층
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], # vgg13: 10 + 3 = 13 계층
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'], # vgg16: 13 + 3 = 16 계층
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'], # vgg19: 16 + 3 = 19 계층
    'custom' : [64,64,64,'M',128,128,128,'M',256,256,256,'M'] # 사용자 정의 구성
}

# 사용자 정의 네트워크 구성으로 VGG 모델 생성
conv = make_layers(cfg['custom'], batch_norm=True)
CNN = VGG(make_layers(cfg['custom']), num_classes=10, init_weights=True)
print(CNN)
# VGG(
#   (features): Sequential(
#     (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (1): ReLU(inplace)
#     (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (3): ReLU(inplace)
#     (4): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (5): ReLU(inplace)
#     (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (7): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (8): ReLU(inplace)
#     (9): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (10): ReLU(inplace)
#     (11): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (12): ReLU(inplace)
#     (13): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (14): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (15): ReLU(inplace)
#     (16): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (17): ReLU(inplace)
#     (18): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (19): ReLU(inplace)
#     (20): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   )
#   (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
#   (classifier): Sequential(
#     (0): Linear(in_features=25088, out_features=4096, bias=True)
#     (1): ReLU(inplace)
#     (2): Dropout(p=0.5)
#     (3): Linear(in_features=4096, out_features=4096, bias=True)
#     (4): ReLU(inplace)
#     (5): Dropout(p=0.5)
#     (6): Linear(in_features=4096, out_features=10, bias=True)
#   )
# )
```

### **VGG for cifar10**

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import visdom

# Visdom 설정
vis = visdom.Visdom()
vis.close(env="main")

# 손실 추적 함수 정의
def loss_tracker(loss_plot, loss_value, num):
    '''손실값을 시각화'''
    vis.line(X=num,
             Y=loss_value,
             win=loss_plot,
             update='append'
             )

# 디바이스 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 시드 고정
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

# 데이터 전처리 정의
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# CIFAR-10 학습 데이터셋 로드
trainset = torchvision.datasets.CIFAR10(root='./cifar10', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=512,
                                          shuffle=True, num_workers=0)

# CIFAR-10 테스트 데이터셋 로드
testset = torchvision.datasets.CIFAR10(root='./cifar10', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=0)

# 클래스 라벨 정의
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 이미지 시각화를 위한 matplotlib 설정
import matplotlib.pyplot as plt
import numpy as np
# %matplotlib inline

# 이미지를 보여주는 함수 정의
def imshow(img):
    img = img / 2 + 0.5  # 정규화를 되돌림
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# 랜덤 학습 이미지 가져오기 및 시각화
dataiter = iter(trainloader)
images, labels = next(dataiter)
vis.images(assets/img/posts/pytorch/10-5/images / 2 + 0.5)  # 정규화된 이미지를 되돌림

# show images
#imshow(torchvision.utils.make_grid(assets/img/posts/pytorch/10-5/images))

# 이미지와 라벨 출력
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
# truck   dog horse truck

# VGG16 모델 구현
import torchvision.models.vgg as vgg
# import vgg

cfg = [32, 32, 'M', 64, 64, 128, 128, 128, 'M', 256, 256, 256, 512, 512, 512, 'M']  # VGG16 구조
# 32x32 -> 16 -> 8 -> 4

class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        #self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        # VGG16의 완전 연결층 정의
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 4096),  # CIFAR-10 이미지 크기에 맞춰서 조정
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)  # 합성곱 계층
        x = x.view(x.size(0), -1)  # 일렬로 펼침 (평탄화)
        x = self.classifier(x)  # 완전 연결층
        return x

    # 가중치 초기화
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# VGG16 모델 인스턴스 생성 및 장치로 이동
vgg16 = VGG(vgg.make_layers(cfg), 10, True).to(device)
a = torch.Tensor(1, 3, 32, 32).to(device)
out = vgg16(a)
print(out)
# tensor([[ 3.5377e+34,  6.0071e+34, -2.7727e+34,  2.0572e+35,  2.3735e+35,
#           2.2759e+35,  5.4568e+33, -1.1127e+35,  1.0189e+35,  3.9697e+34]],
#        grad_fn=<AddmmBackward>)

# 손실 함수 및 옵티마이저 정의
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(vgg16.parameters(), lr=0.005, momentum=0.9)

# 학습률 조정 스케줄러
lr_sche = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

# 손실 그래프 정의
loss_plt = vis.line(Y=torch.Tensor(1).zero_(), opts=dict(title='loss_tracker', legend=['loss'], showlegend=True))

# 학습 시작
epochs = 50
for epoch in range(epochs):  # 데이터셋을 여러 번 반복
    running_loss = 0.0
    lr_sche.step()
    for i, data in enumerate(trainloader, 0):
        # 입력 데이터 가져오기
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # 경사 초기화
        optimizer.zero_grad()

        # 순전파, 역전파, 최적화
        outputs = vgg16(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 손실 값 누적
        running_loss += loss.item()
        if i % 30 == 29:  # 30 미니 배치마다 출력
            loss_tracker(loss_plt, torch.Tensor([running_loss / 30]), torch.Tensor([i + epoch * len(trainloader)]))
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 30))
            running_loss = 0.0

print('Finished Training')
# [1,    30] loss: 2.302
# [1,    60] loss: 2.297
# [1,    90] loss: 2.288
# [2,    30] loss: 2.250
# [2,    60] loss: 2.290
# ...
# [49,    60] loss: 0.075
# [49,    90] loss: 0.082
# [50,    30] loss: 0.065
# [50,    60] loss: 0.064
# [50,    90] loss: 0.060

# 테스트 데이터셋에서 일부 이미지를 가져옴
dataiter = iter(testloader)
images, labels = next(dataiter)

# 테스트 이미지 출력
imshow(torchvision.utils.make_grid(assets/img/posts/pytorch/10-5/images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

# 예측 수행
outputs = vgg16(assets/img/posts/pytorch/10-5/images.to(device))
_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

# 전체 테스트셋에서 정확도 계산
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = vgg16(assets/img/posts/pytorch/10-5/images)
        
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

# GroundTruth:    cat  ship  ship plane
# Predicted:    cat  ship  ship plane
# Accuracy of the network on the 10000 test images: 72 %
```

![image.png](assets/img/posts/pytorch/10-5/image%203.png)

시간이 많이 걸린다
