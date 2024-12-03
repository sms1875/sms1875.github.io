---
layout: post
title: "[모두를 위한 딥러닝 시즌2] Lab-04-2 Loading Data"
date: 2024-10-16 02:19:00+0900
categories: [Study, AI]
tags: [Deep Learning Zero To All, 모두를 위한 딥러닝 시즌2, ML, DL, pytorch]
math: true
---
{% include embed/youtube.html id='B3VG-TeO9Lk' %}  

## Multivariate Linear Regression 복습

- Simple Linear Regression : 하나의 정보 → 하나의 결론 예측
- Multivariate Linear Regression  : 복수의 정보 → 하나의 결론 예측
    
    3개의 쪽지시험 성적 데이터를 통해 기말고사 성적 예측
    
    경사하강법을 통해 모델을 학습 → 실제값에 수렴, 코스트가 점점 작아짐
    
    하지만, 우리가 사용한 데이터는 40명의 학생들에 대한 매우 적은 양의 데이터
    
    복잡한 머신러닝을 학습할 경우 엄청난 양의 데이터가 필요함
    

## “Minibatch” Gradient Descent 이론

### 배경

데이터가 많으면 모델이 학습하면서 견고하고 완성된 예측 가능함

경사하강법을 사용하려면 데이터마다 코스트를 부여해야함

데이터가 많을 경우 발생하는 문제점

1. 연산속도가 너무 느려짐
2. 데이터를 하드웨어에 저장할 수 없음

→ 데이터를 전부 사용하지말고 일부분만 사용해 학습하자! = Minibatch Gradient  Descent 

![image.png](assets/img/posts/AI/4-2/image.png)

## Minibatch Gradient  Descent란?

- 전체 데이터의 작은 부분을 나누어 각 미니 배치를 개별적으로 학습하는 방식
- 각 미니 배치에서만 코스트를 계산하여 업데이트하기 때문에 컴퓨터에 부담을 덜어주고, 업데이트 주기가 빠름
- 하지만 전체 데이터를 사용하지 않으면 데이터가 잘못된 방향으로 학습할 수 있어, 기존의 기존 경사하강법과 달리 코스트가 매끄럽게 줄어들지 않고 거칠게 감소할 수 있음

![image.png](assets/img/posts/AI/4-2/image%201.png)

## PyTorch Dataset and DataLoader 사용법

DataSet , DataLoader : 데이터 셋을 미니배치로 쪼개는데 사용하는 pyTorch 모듈

- DataLoader : 파이토치에서 데이터를 로드하는 모듈로, 상속하여 만든 새로운 클래스는 우리가 원하는 데이터셋을 지정할 수 있음
- 커스텀 데이터셋을 만들때 두가지 Magic Method를 만들어야 함
    - **__len__**() : 데이터셋의 총 개수 반환
    - **__getitem**__() : 어떤 인덱스 idx를 받을때 그에 상응하는 데이터 하나 반환
- DataLoader 모듈을 사용하려면 dataset, batch_size를 지정해줘야함
    - dataset
    - batch_size : 각 minibatch의 크기, 통상적으로 2의 제곱수로 설정
    - shuffle=True : Epoch마다 데이터셋을 섞어서 데이터가 학습되는 순서를 변경
- epoch for 문 안에 minibatch를 위한 for문이 하나 더 추가됨
    - enumerate(dataloader) : minibatch 인덱스와 데이터를 받음
    - 데이터를 x와 y로 나누어 경사하강법을 수행

```python
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x)

class CustomDataset(Dataset):
  def __init__(self):
    self.x_data=[
      [73,80,75],
      [93,88,93],
      [89,91,90],
      [96,98,100],
      [73,66,70]]
    self.y_data=[[152],[185],[180],[196],[142]]

  def __len__(self):
    return len(self.x_data)
  
  def __getitem__(self, index):
    x = torch.FloatTensor(self.x_data[index])
    y = torch.FloatTensor(self.y_data[index])

    return x,y

# 데이터 셋
dataset=CustomDataset() 

# 데이터 로더
dataloader=DataLoader(
  dataset,
  batch_size=2, # 2의 배수
  shuffle=True,
)

model = MultivariateLinearRegressionModel()

optimizer = optim.SGD(model.parameters(), lr=1e-5)

nb_epochs=20
for epoch in range(nb_epochs + 1):
    for batch_idx, samples in enumerate(dataloader):
      x_train, y_train = samples
      # H(x) 계산
      prediction = model(x_train)

      # cost 계산
      cost = F.mse_loss(prediction, y_train)

      # cost로 H(x) 개선
      optimizer.zero_grad()
      cost.backward()
      optimizer.step()

      # 로그 출력
      print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
         epoch, nb_epochs, batch_idx+1, len(dataloader), cost.item() 
      ))
      # Epoch    0/20 Batch 1/3 Cost: 35424.417969
      # Epoch    0/20 Batch 2/3 Cost: 7990.055664
      # Epoch    0/20 Batch 3/3 Cost: 2148.731934
      # Epoch    1/20 Batch 1/3 Cost: 986.442322
      # Epoch    1/20 Batch 2/3 Cost: 465.397614
      # Epoch    1/20 Batch 3/3 Cost: 88.113068
      # Epoch    2/20 Batch 1/3 Cost: 36.668221
      # Epoch    2/20 Batch 2/3 Cost: 9.448990
      # Epoch    2/20 Batch 3/3 Cost: 3.667141
      # Epoch    3/20 Batch 1/3 Cost: 0.975254
      # Epoch    3/20 Batch 2/3 Cost: 0.906193
      # Epoch    3/20 Batch 3/3 Cost: 0.006388
      # Epoch    4/20 Batch 1/3 Cost: 0.051697
      # Epoch    4/20 Batch 2/3 Cost: 0.054978
      # Epoch    4/20 Batch 3/3 Cost: 1.199195
      # Epoch    5/20 Batch 1/3 Cost: 0.401478
      # Epoch    5/20 Batch 2/3 Cost: 0.046481
      # Epoch    5/20 Batch 3/3 Cost: 0.388473
      # Epoch    6/20 Batch 1/3 Cost: 0.538637
      # Epoch    6/20 Batch 2/3 Cost: 0.145426
      # Epoch    6/20 Batch 3/3 Cost: 0.053031
      # Epoch    7/20 Batch 1/3 Cost: 0.601826
      # Epoch    7/20 Batch 2/3 Cost: 0.188046
      # Epoch    7/20 Batch 3/3 Cost: 0.048246
      # Epoch    8/20 Batch 1/3 Cost: 0.044538
      # Epoch    8/20 Batch 2/3 Cost: 0.509270
      # Epoch    8/20 Batch 3/3 Cost: 0.279848
      # Epoch    9/20 Batch 1/3 Cost: 0.051684
      # Epoch    9/20 Batch 2/3 Cost: 0.079211
      # Epoch    9/20 Batch 3/3 Cost: 1.242455
      # Epoch   10/20 Batch 1/3 Cost: 0.039547
      # Epoch   10/20 Batch 2/3 Cost: 0.284169
      # Epoch   10/20 Batch 3/3 Cost: 1.030069
      # Epoch   11/20 Batch 1/3 Cost: 0.402872
      # Epoch   11/20 Batch 2/3 Cost: 0.285181
      # Epoch   11/20 Batch 3/3 Cost: 0.017461
      # Epoch   12/20 Batch 1/3 Cost: 0.045682
      # Epoch   12/20 Batch 2/3 Cost: 0.093299
      # Epoch   12/20 Batch 3/3 Cost: 1.044303
      # Epoch   13/20 Batch 1/3 Cost: 0.401862
      # Epoch   13/20 Batch 2/3 Cost: 0.055938
      # Epoch   13/20 Batch 3/3 Cost: 0.412103
      # Epoch   14/20 Batch 1/3 Cost: 0.537061
      # Epoch   14/20 Batch 2/3 Cost: 0.010415
      # Epoch   14/20 Batch 3/3 Cost: 0.253107
      # Epoch   15/20 Batch 1/3 Cost: 0.625113
      # Epoch   15/20 Batch 2/3 Cost: 0.015669
      # Epoch   15/20 Batch 3/3 Cost: 0.070435
      # Epoch   16/20 Batch 1/3 Cost: 0.551100
      # Epoch   16/20 Batch 2/3 Cost: 0.126779
      # Epoch   16/20 Batch 3/3 Cost: 0.001243
      # Epoch   17/20 Batch 1/3 Cost: 0.074117
      # Epoch   17/20 Batch 2/3 Cost: 0.049969
      # Epoch   17/20 Batch 3/3 Cost: 1.040596
      # Epoch   18/20 Batch 1/3 Cost: 0.064074
      # Epoch   18/20 Batch 2/3 Cost: 0.334032
      # Epoch   18/20 Batch 3/3 Cost: 0.990896
      # Epoch   19/20 Batch 1/3 Cost: 0.354361
      # Epoch   19/20 Batch 2/3 Cost: 0.372612
      # Epoch   19/20 Batch 3/3 Cost: 0.308847
      # Epoch   20/20 Batch 1/3 Cost: 0.041474
      # Epoch   20/20 Batch 2/3 Cost: 0.466467
      # Epoch   20/20 Batch 3/3 Cost: 0.352104
```
