---
layout: post
title: "[모두를 위한 딥러닝 시즌2] Lab-10-4-1 ImageFolder1 + 10-4-2 ImageFolder2"
date: 2024-11-17 02:52:00+0900
categories: [Study, AI]
tags: [Deep Learning Zero To All, 모두를 위한 딥러닝 시즌2, ML, DL, pytorch]
math: true
mermaid : true
---
## DataSet 준비

강의에서 사용하는 의자 이미지 데이터셋 사용

![image.png](assets/img/posts/study/AI/10-4/image.png)

![image.png](assets/img/posts/study/AI/10-4/image%201.png)

깃허브 레포의 costom_data에 있다

![image.png](assets/img/posts/study/AI/10-4/image%202.png)

**origin data 크기 조절하기**

![image.png](assets/img/posts/study/AI/10-4/image%203.png)

```python
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from matplotlib.pyplot import imshow
import os

# 파일을 저장할 경로 생성
os.makedirs('custom_data/train_data/gray', exist_ok=True)
os.makedirs('custom_data/train_data/red', exist_ok=True)

# 이미지 전처리를 위한 변환(transform) 정의
trans = transforms.Compose([
    transforms.Resize((64, 128))  # 이미지를 (64, 128) 크기로 리사이즈
])

# ImageFolder를 이용해 데이터셋 로드
# 'root' 디렉토리의 하위 폴더를 클래스 레이블로 매핑
train_data = torchvision.datasets.ImageFolder(
    root='custom_data/origin_data',  # 원본 이미지 경로
    transform=trans  # 정의된 변환 적용
)

# 데이터셋 순회 및 저장
for num, value in enumerate(train_data):
    # 데이터와 레이블 분리
    data, label = value
    print(num, data, label)  # 현재 데이터의 순번, 이미지, 레이블 출력

    # 레이블에 따라 이미지를 다른 폴더에 저장
    if label == 0:
        # 레이블이 0인 이미지를 'gray' 폴더에 저장
        data.save('custom_data/train_data/gray/%d_%d.jpeg' % (num, label))
    else:
        # 레이블이 0이 아닌 이미지를 'red' 폴더에 저장
        data.save('custom_data/train_data/red/%d_%d.jpeg' % (num, label))

```

**결과**

128 x 64로 조절된 이미지가 custom_data/train_data 경로에 저장되었다

![image.png](assets/img/posts/study/AI/10-4/image%204.png)

## Neural Network 만들기

![image.png](assets/img/posts/study/AI/10-4/image%205.png)

- Layer 1
    - Convolution layer = (in_c = 3, out_c = 6, kernel_size = 5 stride = 1)
    - MaxPool layer = (kernel_size = 2, stride = 2)
- Layer 2
    - Convolution layer = (in_c = 6, out_c = 16, kernel_size = 5, stride = 1)
    - MaxPool layer = (kernel_size = 2, stride = 2)
- Layer 3
    - View  ⇒ (batch_size x [16,13,29] ⇒ batch_size x [6032])
    - Fully_Connect layer = (input = 6032, output = 120)
    - Fully_Connect layer = (input = 120, output = 2)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import os

# 파일을 저장할 경로 생성
os.makedirs('model', exist_ok=True)

# GPU 사용 여부 확인
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 랜덤 시드 설정 (재현성 보장)
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

# 데이터 전처리 (텐서로 변환)
trans = transforms.Compose([
    transforms.ToTensor()  # 이미지를 Tensor로 변환
])

# 훈련 데이터 로드
train_data = torchvision.datasets.ImageFolder(
    root='custom_data/train_data',  # 훈련 데이터 경로
    transform=trans  # 변환 적용
)
# DataLoader 생성 (배치 크기 8, 셔플 활성화)
data_loader = DataLoader(
    dataset=train_data,
    batch_size=8,
    shuffle=True,
    num_workers=2
)

# CNN 모델 정의
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 첫 번째 컨볼루션 레이어 (입력: 3채널, 출력: 6채널, 커널 크기: 5x5)
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # 두 번째 컨볼루션 레이어 (입력: 6채널, 출력: 16채널, 커널 크기: 5x5)
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # 완전 연결 레이어 (FC)
        # 입력 크기: 16x13x29 -> 출력 크기: 120 -> 2 (클래스 개수)
        self.layer3 = nn.Sequential(
            nn.Linear(16 * 13 * 29, 120),
            nn.ReLU(),
            nn.Linear(120, 2)
        )
        
    def forward(self, x):
        # 입력 데이터가 각 레이어를 통과하는 과정
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.shape[0], -1)  # Flatten (벡터화)
        out = self.layer3(out)
        return out

# 모델 인스턴스화 및 GPU로 이동
net = CNN().to(device)

# 테스트용 입력 데이터 (3개의 샘플, 3채널, 64x128 크기)
test_input = (torch.Tensor(3, 3, 64, 128)).to(device)
test_out = net(test_input)

# 옵티마이저 및 손실 함수 정의
optimizer = optim.Adam(net.parameters(), lr=0.00005)
loss_func = nn.CrossEntropyLoss().to(device)

# 훈련 데이터 배치 크기
total_batch = len(data_loader)

# 학습 진행
epochs = 7
for epoch in range(epochs):
    avg_cost = 0.0
    for num, data in enumerate(data_loader):
        imgs, labels = data  # 이미지와 레이블 분리
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()  # 기울기 초기화
        out = net(imgs)  # 모델 출력 계산
        loss = loss_func(out, labels)  # 손실 계산
        loss.backward()  # 역전파
        optimizer.step()  # 가중치 업데이트
        
        avg_cost += loss / total_batch  # 평균 손실 계산
        
    print('[Epoch:{}] cost = {}'.format(epoch + 1, avg_cost))
print('Learning Finished!')
# [Epoch:1] cost = 0.654132068157196
# [Epoch:2] cost = 0.4932752251625061
# [Epoch:3] cost = 0.2222609519958496
# [Epoch:4] cost = 0.06610226631164551
# [Epoch:5] cost = 0.02623838558793068
# [Epoch:6] cost = 0.01393833290785551
# [Epoch:7] cost = 0.008722379803657532

# 학습된 모델 저장
torch.save(net.state_dict(), "model/model.pth")

# 모델 불러오기 테스트
new_net = CNN().to(device)
new_net.load_state_dict(torch.load('model/model.pth'))

# 첫 번째 컨볼루션 레이어 확인
print(net.layer1[0])  # 기존 모델의 첫 번째 레이어
print(new_net.layer1[0])  # 불러온 모델의 첫 번째 레이어
# Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1))
# Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1))

# 첫 번째 레이어 가중치 확인
print(net.layer1[0].weight[0][0][0])  # 기존 모델
print(new_net.layer1[0].weight[0][0][0])  # 불러온 모델
# tensor([-0.0913,  0.0032, -0.0172, -0.0214,  0.0930], device='cuda:0',
#        grad_fn=<SelectBackward0>)
# tensor([-0.0913,  0.0032, -0.0172, -0.0214,  0.0930], device='cuda:0',
#        grad_fn=<SelectBackward0>)

# 모델의 첫 번째 레이어 가중치 비교
net.layer1[0].weight[0] == new_net.layer1[0].weight[0]

# 테스트 데이터셋 로드
trans = torchvision.transforms.Compose([
    transforms.Resize((64, 128)),  # 크기 조정
    transforms.ToTensor()
])
test_data = torchvision.datasets.ImageFolder(
    root='custom_data/test_data',  # 테스트 데이터 경로
    transform=trans
)
test_set = DataLoader(
    dataset=test_data,
    batch_size=len(test_data)  # 전체 데이터를 한 번에 로드
)

# 모델 평가
with torch.no_grad():
    for num, data in enumerate(test_set):
        imgs, label = data  # 이미지와 레이블 분리
        imgs = imgs.to(device)
        label = label.to(device)

        prediction = net(imgs)  # 모델 예측
        correct_prediction = torch.argmax(prediction, 1) == label  # 예측과 실제 값 비교
        accuracy = correct_prediction.float().mean()  # 정확도 계산
        print('Accuracy:', accuracy.item())
# Accuracy: 1.0
```

- test_data는 origin_data와 마찬가지로 레포에 있다
- 데이터셋이 좋기 때문에 epoch를 7번만 사용한다
- 학습시킨 모델을 save() 를 통해 저장할 수 있다
- 저장된 모델은 load()를 사용해 불러올 수 있다
