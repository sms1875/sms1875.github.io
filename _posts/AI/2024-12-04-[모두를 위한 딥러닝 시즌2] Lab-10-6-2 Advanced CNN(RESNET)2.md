---
layout: post
title: "[모두를 위한 딥러닝 시즌2] Lab-10-6-2 Advanced CNN(RESNET)2"
date: 2024-12-04 17:02:00+0900
categories: [Study, AI]
tags: [Deep Learning Zero To All, 모두를 위한 딥러닝 시즌2, ML, DL, pytorch]
math: true
mermaid : true
---
## 특징

- ResNet50 기반의 커스텀 모델 구현
- CIFAR-10 데이터셋 사용
- 데이터 전처리 및 정규화 기법 적용
- Visdom을 통한 학습 과정 시각화
- 모델 성능 평가 및 체크포인트 저장

## 코드 설명

### 필요한 라이브러리 및 환경 설정

- PyTorch 및 torchvision을 사용해 데이터 로드와 모델 구성
- Visdom으로 학습 과정을 시각화

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import visdom
import torchvision.models.resnet as resnet
```

### Visdom 초기화

- Visdom 환경 초기화 및 시각화를 위한 준비

```python
vis = visdom.Visdom()
vis.close(env="main")

```

### Value Tracker 정의

- 손실(loss) 및 정확도(accuracy)를 Visdom에 실시간으로 시각화

```python
def value_tracker(value_plot, value, num):
    vis.line(X=num,
             Y=value,
             win=value_plot,
             update='append')

```

### 장치 설정 및 시드 초기화

- 훈련이 GPU에서 실행되도록 설정
- 재현성을 위한 랜덤 시드 고정

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

```

---

## 데이터 전처리

### Normalize 계산

- CIFAR-10 데이터의 각 채널 평균과 표준 편차를 계산하여 Normalize 값 설정

```python
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.CIFAR10(root='./cifar10', train=True, download=True, transform=transform)
train_data_mean = trainset.data.mean(axis=(0, 1, 2)) / 255
train_data_std = trainset.data.std(axis=(0, 1, 2)) / 255

```

### Random Crop 및 Normalize 적용

- `RandomCrop`으로 데이터 다양성 증가, `Normalize`로 데이터 정규화

```python
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(train_data_mean, train_data_std)
])

```

---

## 모델 구성

### ResNet 정의

- ResNet은 PyTorch ResNet 구조를 참고하여 CIFAR-10 크기에 맞춰 설계
- 첫 번째 레이어에서 채널 수를 16으로 설정하고, 3x3 컨볼루션을 사용

```python
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        ...
    def _make_layer(self, block, planes, blocks, stride=1):
        ...
    def forward(self, x):
        ...

```

### 모델 생성

- ResNet-50 생성 및 GPU에 업로드

```python
resnet50 = ResNet(resnet.Bottleneck, [3, 4, 6, 3], 10, True).to(device)

```

---

## 모델 학습

### 손실 및 옵티마이저 설정

- 크로스엔트로피 손실 함수, SGD 옵티마이저, 학습률 스케줄러 설정

```python
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(resnet50.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
lr_sche = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

```

### 학습 루프

- 배치 단위로 데이터를 모델에 입력, 손실 계산 및 역전파
- `lr_scheduler.step()`을 통해 학습률 조정
  
```python
for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = resnet50(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    lr_sche.step()
    acc = acc_check(resnet50, testloader, epoch, save=1)
    value_tracker(acc_plt, torch.Tensor([acc]), torch.Tensor([epoch]))

```

### 정확도 체크 및 모델 저장

- 모델 정확도 계산 및 `torch.save()`로 모델 가중치 저장

```python
def acc_check(net, test_set, epoch, save=1):
    correct, total = 0, 0
    with torch.no_grad():
        for data in test_set:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    torch.save(net.state_dict(), f"./model/model_epoch_{epoch}_acc_{int(acc)}.pth")
    return acc

```

## 학습 결과

- 30 epoch부터 변화가 미미하여 중단하였음

### Visdom

![image.png](assets/img/posts/AI/10-6-2/image.png)

![image.png](assets/img/posts/AI/10-6-2/image%201.png)

### Model Save

![image.png](assets/img/posts/AI/10-6-2/image%202.png)

## 전체코드

```python
# 필요한 라이브러리 임포트
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import visdom
import torchvision.models.resnet as resnet

# Visdom 초기화
vis = visdom.Visdom()
vis.close(env="main")

# Value Tracker 함수 정의
def value_tracker(value_plot, value, num):
    '''num, loss_value are Tensors'''
    vis.line(X=num,
             Y=value,
             win=value_plot,
             update='append')

# 장치 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print (f"device is {device}")

# 랜덤 시드 설정
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

# 데이터셋 전처리 및 정규화
transform = transforms.Compose([
    transforms.ToTensor()
])

# CIFAR10 데이터셋 로드 및 평균/표준편차 계산
trainset = torchvision.datasets.CIFAR10(root='./cifar10', train=True, download=True, transform=transform)
train_data_mean = trainset.data.mean(axis=(0, 1, 2)) / 255
train_data_std = trainset.data.std(axis=(0, 1, 2)) / 255

# 데이터셋 변환 설정
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(train_data_mean, train_data_std)
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(train_data_mean, train_data_std)
])

# 데이터셋 준비
trainset = torchvision.datasets.CIFAR10(root='./cifar10', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./cifar10', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# ResNet 모델 정의
conv1x1 = resnet.conv1x1
Bottleneck = resnet.Bottleneck
BasicBlock = resnet.BasicBlock

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(block, 16, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=1)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

# ResNet50 모델 생성
resnet50 = ResNet(Bottleneck, [3, 4, 6, 3], 10, True).to(device)

# 학습 준비
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(resnet50.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
lr_sche = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# Loss 및 Accuracy 플롯 초기화
loss_plt = vis.line(Y=torch.Tensor(1).zero_(), opts=dict(title='Loss Tracker', legend=['Loss'], showlegend=True))
acc_plt = vis.line(Y=torch.Tensor(1).zero_(), opts=dict(title='Accuracy', legend=['Accuracy'], showlegend=True))

# Accuracy 체크 함수 정의
def acc_check(net, test_set, epoch, save=1):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_set:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    acc = 100 * correct / total
    print(f'Accuracy of the network on the 10000 test images: {acc:.2f}%')
    if save:
        torch.save(net.state_dict(), f"./model/model_epoch_{epoch}_acc_{int(acc)}.pth")
    return acc

# 모델 학습
epochs = 150
for epoch in range(epochs):
    running_loss = 0.0
    # lr_sche.step()
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = resnet50(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 30 == 29:
            value_tracker(loss_plt, torch.Tensor([running_loss / 30]), torch.Tensor([i + epoch * len(trainloader)]))
            print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 30:.3f}')
            running_loss = 0.0

    lr_sche.step()
    acc = acc_check(resnet50, testloader, epoch, save=1)
    value_tracker(acc_plt, torch.Tensor([acc]), torch.Tensor([epoch]))

print('Finished Training')

# 최종 모델 평가
correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = resnet50(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')
# [1, 30] loss: 2.039
# [1, 60] loss: 1.885
# [1, 90] loss: 1.775
# [1, 120] loss: 1.753
# [1, 150] loss: 1.670
# [1, 180] loss: 1.609
# Accuracy of the network on the 10000 test images: 40.86%
# [2, 30] loss: 1.527
# [2, 60] loss: 1.472
# [2, 90] loss: 1.439
# [2, 120] loss: 1.368
# [2, 150] loss: 1.341
# [2, 180] loss: 1.268
# Accuracy of the network on the 10000 test images: 47.97%
# ...
# [28, 30] loss: 0.153
# [28, 60] loss: 0.154
# [28, 90] loss: 0.159
# [28, 120] loss: 0.160
# [28, 150] loss: 0.176
# [28, 180] loss: 0.178
# Accuracy of the network on the 10000 test images: 85.07%
# [29, 30] loss: 0.180
# [29, 60] loss: 0.149
# [29, 90] loss: 0.153
# [29, 120] loss: 0.154
# [29, 150] loss: 0.169
# [29, 180] loss: 0.167
# Accuracy of the network on the 10000 test images: 84.50%
# ...
# [39, 30] loss: 0.051
# [39, 60] loss: 0.056
# [39, 90] loss: 0.048
# [39, 120] loss: 0.057
# [39, 150] loss: 0.057
# [39, 180] loss: 0.064
# Accuracy of the network on the 10000 test images: 85.56%
# [40, 30] loss: 0.066
# [40, 60] loss: 0.058
# [40, 90] loss: 0.054
# [40, 120] loss: 0.065
# [40, 150] loss: 0.063
# [40, 180] loss: 0.064
# Accuracy of the network on the 10000 test images: 85.76%
```

## Warning

> **UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`. In PyTorch 1.1.0 and later, you should call them in the opposite order: `optimizer.step()` before `lr_scheduler.step()`.  Failure to do this will result in PyTorch skipping the first value of the learning rate schedule. See more details at [https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)**
**warnings.warn("Detected call of `lr_scheduler.step()` before `optimizer.step()`. "**
{: .prompt-warning}  



**해결법** 

- `lr_scheduler.step()`을 `optimizer.step()` 전에 호출했기 때문에 발생
- PyTorch 1.1.0 이후로는 `optimizer.step()`을 먼저 호출한 다음 `lr_scheduler.step()`을 호출해야 한다

```python
for epoch in range(epochs):
    running_loss = 0.0
    # lr_sche.step()
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = resnet50(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 30 == 29:
            value_tracker(loss_plt, torch.Tensor([running_loss / 30]), torch.Tensor([i + epoch * len(trainloader)]))
            print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 30:.3f}')
            running_loss = 0.0
            
    lr_sche.step()
    acc = acc_check(resnet50, testloader, epoch, save=1)
    value_tracker(acc_plt, torch.Tensor([acc]), torch.Tensor([epoch]))
```
