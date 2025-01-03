---
layout: post
title: "[모두를 위한 딥러닝 시즌2] Lab-11-3 RNN Long sequence"
date: 2024-12-05 06:20:00+0900
categories: [Study, AI]
tags: [Deep Learning Zero To All, 모두를 위한 딥러닝 시즌2, ML, DL, pytorch]
math: true
mermaid : true
---
## Longseq introduction

- 기존의 RNN 모델들은 짧은 문장을 하나의 샘플로 사용
- **실용적인 모델**은 긴 문장 데이터셋을 사용해야 함
- 하지만 긴 문장을 하나의 입력으로 처리하는 것은 어려움
    
    → **특정 사이즈의 청크**로 나눠 학습해야 한다
    

## Making sequence dataset from long sentence

![image.png](assets/img/posts/study/AI/11-3/image.png)

### 과정

- **윈도우 기반 슬라이딩 기법**을 사용해 sentence를 나누어 Dataset 생성
- **X 데이터**는 입력 시퀀스, **Y 데이터**는 다음 시퀀스를 예측하기 위한 타겟으로 구성

### code

1. **윈도우 설정**
    - 문장을 `sequence_length` 크기의 윈도우로 자른다
2. **데이터 구성**
    - X는 현재 청크, Y는 오른쪽으로 한 글자 이동한 청크로 설정
3. 데이터 변환
    - 청크 데이터를 one-hot vector로 변환하고 PyTorch 텐서로 변환

```python
# data setting
x_data = []
y_data = []

for i in range(0, len(sentence) - sequence_length):
    x_str = sentence[i:i + sequence_length]
    y_str = sentence[i + 1: i + sequence_length + 1]
    print(i, x_str, '->', y_str)

    x_data.append([char_dic[c] for c in x_str])  # x str to index
    y_data.append([char_dic[c] for c in y_str])  # y str to index

x_one_hot = [np.eye(dic_size)[x] for x in x_data]

# transform as torch tensor variable
X = torch.FloatTensor(x_one_hot)
Y = torch.LongTensor(y_data)
# 0 if you wan -> f you want
# 1 f you want ->  you want 
# 2  you want  -> you want t
# 3 you want t -> ou want to
# 4 ou want to -> u want to 
# 5 u want to  ->  want to b
# 6  want to b -> want to bu
# 7 want to bu -> ant to bui
# ...
# 166 ty of the  -> y of the s
# 167 y of the s ->  of the se
# 168  of the se -> of the sea
# 169 of the sea -> f the sea.
```

> 최근에는 10기가 바이트 정도의 큰 텍스트 데이터셋으로도 모델을 학습한다  
> (예: Wikipedia Dump, OpenWebText)  
{: .prompt-tip}  

## Adding FC layer and stacking RNN

1. 1-Layer RNN
    - 단일 RNN 레이어는 **복잡한 문장**을 학습하기엔 한계가 있다 (언더피팅)

![image.png](assets/img/posts/study/AI/11-3/image%201.png)

1. Stacked RNN + Fully Connected Layer
    - RNN 레이어를 쌓고(Stacking) 마지막에 **Fully Connected Layer**를 추가

![image.png](assets/img/posts/study/AI/11-3/image%202.png)

### code

```python
# declare RNN + FC
class Net(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, layers):
        super(Net, self).__init__()
        self.rnn = torch.nn.RNN(input_dim, hidden_dim, num_layers=layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x):
        x, _status = self.rnn(x)
        x = self.fc(x)
        return x

net = Net(dic_size, hidden_size, 2)
```

## Code run through

### 결과 해석

1. **Max 기법**
    - 모델 출력에서 확률이 가장 높은 값을 선택하여 다음 문자 시퀀스를 생성 (argmax)
2. **결과 확인**
    - 모델의 출력 시퀀스가 원본 문장과 얼마나 유사한지를 확인

```python
import torch
import torch.optim as optim
import numpy as np

# 랜덤 시드 설정 (결과 재현성을 위해)
torch.manual_seed(0)

# 학습할 문장 데이터
sentence = ("if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")

# 문자 집합 생성 (문자 중복 제거)
char_set = list(set(sentence))
# 문자와 인덱스를 매핑
char_dic = {c: i for i, c in enumerate(char_set)}

# 하이퍼파라미터 설정
dic_size = len(char_dic)  # 문자 집합의 크기 (사전 크기)
hidden_size = len(char_dic)  # RNN의 히든 크기
sequence_length = 10  # 시퀀스 길이
learning_rate = 0.1  # 학습률

# 입력(X)과 출력(Y) 데이터 생성
x_data = []
y_data = []

for i in range(0, len(sentence) - sequence_length):
    # 현재 시퀀스 청크 (X)와 다음 시퀀스 청크 (Y)를 설정
    x_str = sentence[i:i + sequence_length]
    y_str = sentence[i + 1:i + sequence_length + 1]
    print(i, x_str, '->', y_str)

    # 문자를 인덱스로 변환하여 저장
    x_data.append([char_dic[c] for c in x_str])
    y_data.append([char_dic[c] for c in y_str])

# 원-핫 인코딩
x_one_hot = [np.eye(dic_size)[x] for x in x_data]
# 0 if you wan -> f you want
# 1 f you want ->  you want 
# 2  you want  -> you want t
# 3 you want t -> ou want to
# 4 ou want to -> u want to 
# ...
# 166 ty of the  -> y of the s
# 167 y of the s ->  of the se
# 168  of the se -> of the sea
# 169 of the sea -> f the sea.

# 텐서로 변환
X = torch.FloatTensor(x_one_hot)  # 입력 데이터
Y = torch.LongTensor(y_data)  # 출력 데이터

# RNN + Fully Connected Layer 모델 정의
class Net(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, layers):
        super(Net, self).__init__()
        # RNN 정의
        self.rnn = torch.nn.RNN(input_dim, hidden_dim, num_layers=layers, batch_first=True)
        # Fully Connected Layer 정의
        self.fc = torch.nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x):
        # RNN을 통과
        x, _status = self.rnn(x)
        # Fully Connected Layer를 통과하여 최종 출력 생성
        x = self.fc(x)
        return x

# 모델 초기화
net = Net(dic_size, hidden_size, 2)  # 2개의 RNN 레이어 사용

# 손실 함수와 옵티마이저 설정
criterion = torch.nn.CrossEntropyLoss()  # 크로스 엔트로피 손실 함수
optimizer = optim.Adam(net.parameters(), learning_rate)  # Adam 옵티마이저

# 모델 학습
for i in range(100):  # 100 에포크
    optimizer.zero_grad()  # 기울기 초기화
    outputs = net(X)  # 모델 예측
    loss = criterion(outputs.view(-1, dic_size), Y.view(-1))  # 손실 계산
    loss.backward()  # 역전파
    optimizer.step()  # 가중치 업데이트

    # 예측 결과 처리
    results = outputs.argmax(dim=2)  # argmax를 사용해 가장 높은 확률의 문자 선택
    predict_str = ""
    for j, result in enumerate(results):
        if j == 0:
            # 첫 번째 시퀀스는 전체를 추가
            predict_str += ''.join([char_set[t] for t in result])
        else:
            # 이후 시퀀스는 마지막 문자만 추가
            predict_str += char_set[result[-1]]

    # 예측된 문자열 출력
    print(predict_str)
# b'''bp'cp'''fp'ppb'cpp'cc''c'cpp'''cppb'bpcpcbpcpcfpcccpc'''p''pp'pp''cppc'p'c'cpp'''pccp''''pcccfpc'c'p'c'cf''c'pcb''p'cc'''pc'c'fpcccfp'cbpc'pp'''pc'p'ccpcc'p'cppcf''cpp'ppc'cpp
                                                                                                                                                                                   
# nnntonnnttttotdoootttotototttottttttttootttottttotottoottttoodtttttttttootdttototttttttoooddtootdootsttodttoottttttotottootdoottttottttoodtttotttttotttototttttottttootttttootttooo
                                                                                                                                                                                   
#  t     t t t      t t t   t t t t t t       t t              t       t      d t t t t t                 t t   t t t t t          t       t  t t       t d t   t         t      t   
#  t ottotot t  totot t t  tootottottot ot ttot ttot ttot ott ttottottot otot t tottottot  tot   ot ttott tot otoototot tt tot oto tottot  tott t  tootot tototttott t ttot ttot t  t
#  io t    totot  tototototo ototo   oto to        tot       ot  t     to   tototot     to    oto toto  t toto    t tototo   tot t toto tot   t totototototototototo      tot  toto  
#  e            ' e   t t                e  e    e                                                e     e   t     e                     e                   e                        
#  d  d    d d d  d d d d e e d d   d d  d ee      d d      d d      d d  d d d d d   d d      e  d d e d d d     d d d d    d d   d e  d d   d d d d e d d e   e         d    d d   
#   d      d e  d t e t t  p  d  d  d  t d  p r  e t        t  d  e    to t d t e  d  t  e t      t  e re d t  t  d t e ep t d     t  p d  d  d t   e  p r  e   e    t    t  d eee   
#   t ts   t to t t e t t  re t  t  to t t  p re   t        t  t  t  t to t t t t  t  t t   tt    t  ps t t to t  t t t   ep t     t  p t  to t t   t  p  t e t e    t    t  t       
#  ttot t etotott t tototoe eotot totot  totoeotte tot t e ttott  eortoto   t tototttototo  t  eo totoeotototo ttototototoeo tot t toeo totto totot toeototot t to  tt  t tottoe tot 
# ...
#  sthrcth t to t i t tmt ems do 't d gt tpeaeo le thnnt er to do ae t to l t d ton't dmsi nstoem toste t d to lt t a a t er torchethem to to d t d theme d e toep eesert d  toemsint
#   thncth t to t i t dmteeps don't a gk tpepeo le thnnt em to do ae t do l dmd don't atsinnstoem aosts a d do ls t a d ther tonchethem to do d thg theme d e siep eesero a  toemsint
#   th cth t to t i t dmt eis don't a gk apepeo le thnnt em to 'o le t to l drd don't atsin stoem tosks r d do ks t a r them tonshethem to do d thn thems d ensoip eesii  a  toems it
#   th cto t to t i t d t eis ton't a  t rpspeo le thnnt em to 'o ae t to d dnd don't atsinnstoem tosks r d do ks t t r them tonch them to to d thn thems d ensoim  esiit a  thems nt
#   to lto d to d i t d t eps don't a ut rpspeo le thnnt em to 'o ae t do d d d don't a sign toem tosks i d do ks t t r them tonco them to do d ton theme d ersoim  nsirt a  toemeonn
# m thncto t to tui t d shers don't arut rpspeo le thnet em to do ae t do d d d don't a sign toem tosks i d do ks t t r them torco them to do d thn theme duersoim eesitt ar toemeo t
# m ehncto t tontui t d theps ton't arut dpskeonle thnet er th co le t do d t d don't assign them tosks a d do k, t t r ther thrc  them to 'o d t n theme due , im etsitt ar themehnt
# ...
# m tou want to build a ship, don't drum up people together to collect wood and don't assign them tosks and work, but rather teach them to long for the endless immensity of the sea 
# m tou want to build a ship, don't drum up people together to collect wood and don't assign them tosks and work, but rather teach them to long for the endless immensity of the seas
# l tou want to build a ship, don't drum up people together to collect wood and don't assign them tasks and work, but rather teach them to long for the endless immensity of the seas
# l tou want to build a ship, don't drum up people together to collect wood and don't assign them tasks and work, but rather teach them to long for the endless immensity of the seas
# l tou want to build a ship, don't drum up people together to collect wood and don't assign them tosks and work, but rather teach them to long for the endless immensity of the sea.
# m tou want to build a ship, don't drum up people together to collect wood and don't assign them tosks and work, but rather teach them to long for the endless immensity of the sea.
```
