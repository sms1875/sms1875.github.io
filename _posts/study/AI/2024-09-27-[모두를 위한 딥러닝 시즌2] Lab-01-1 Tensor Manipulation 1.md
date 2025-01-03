---
layout: post
title: "[모두를 위한 딥러닝 시즌2] Lab-01-1 Tensor Manipulation 1"
date: 2024-09-27 19:30:00+0900
categories: [Study, AI]
tags: [Deep Learning Zero To All, 모두를 위한 딥러닝 시즌2, ML, DL, pytorch]
---
{% include embed/youtube.html id='St7EhvnFi6c' %}  
  

## 벡터(Vector), 행렬(Matrix), 텐서(Tensor) 기본 개념

- **스칼라(Scalar)**: 차원이 없는 하나의 값, 예를 들어 `3` 또는 `7.5`
    - 스칼라는 0차원 텐서라고도 불림.
- **벡터(Vector)**: 1차원으로 이루어진 값들의 배열, 즉 크기와 방향을 갖는 데이터
    - 예: `[1, 2, 3, 4, 5]`
- **행렬(Matrix)**: 2차원 배열로, 행(row)과 열(column)로 구성된 데이터
    - 예: `[[1, 2, 3], [4, 5, 6]]`
- **텐서(Tensor)**: N차원 배열로, 3차원 이상도 쉽게 다룰 수 있는 데이터 구조
    - 예: `3D 텐서` - 이미지 데이터는 `(배치 크기, 너비, 높이)`로 표현됨.
    - 예: `4D 텐서` - 시퀀스 데이터는 `(배치 크기, 시퀀스 길이, 너비, 높이)` 형태로 표현

## 텐서의 주요 개념

**Shape (형태)**

- 텐서의 각 차원 크기를 나타내는 값
    - 예: `Shape of 2D 텐서 = (4, 3)`

**Rank (랭크)**

- 텐서의 차원 수
- 예: 벡터의 랭크는 1, 행렬의 랭크는 2

**Axis (축)**

- 텐서의 특정 차원을 가리키는 용어
- 예: 2D 텐서에서 axis 0은 행, axis 1은 열

**Batch Size (배치 크기)**

- 딥러닝 모델 학습 시, 한 번에 처리하는 데이터의 샘플 수를 나타냄
    - 예: `64개 샘플`을 한 번에 학습시킨다면 배치 크기는 64

**Dimension (차원)**

- 텐서가 몇 차원으로 구성되어 있는지 나타내는 값
    - 1D 텐서: 벡터
    - 2D 텐서: 행렬
    - 3D 텐서: 다차원 배열
    - 예: `(64, 256)`은 2D 텐서로, `64`는 배치 크기, `256`은 차원의 크기를 의미

**채널 (Channel):**

- 이미지 데이터의 경우 채널은 색상 정보를 나타냄 (예: `RGB` 채널)
- 예: `|T| = (32, 3, 128, 128)`에서 `3`은 채널 수를 의미

## 차원의 증가 예시

- **1D 텐서 (벡터)**: `[1, 2, 3, 4]`
- **2D 텐서 (행렬)**: `[[1, 2, 3], [4, 5, 6]]`
- **3D 텐서**: `[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]`
- **4D 텐서 (이미지)**: `배치 크기 x 채널(RGB) x 높이 x 너비`
    - 예: `(32, 3, 64, 64)`

**2D 텐서: `|T| = (배치 크기, 디멘션)`**

![image.png](assets/img/posts/study/AI/1-1/1 (1).png)

- 주로 **단순한 구조의 신경망**에서 사용
- 예: `|T| = (64, 256)`
    - `배치 크기(batch size)`: 64
    - `디멘션(dimension)`: 256

**3D 텐서 (컴퓨터 비전): `|T| = (배치 크기, 높이, 너비)`**

![image.png](assets/img/posts/study/AI/1-1/1 (2).png)

- **이미지 데이터**를 다룰 때 주로 사용
- 예: `|T| = (32, 128, 128)`
    - `배치 크기(batch size)`: 32
    - `높이(height)`: 128
    - `너비(width)`: 128

**3. 3D 텐서 (NLP 및 시퀀셜 데이터): `|T| = (배치 크기, 시퀀스 길이, 디멘션)`**

![image.png](assets/img/posts/study/AI/1-1/1 (3).png)

- **자연어 처리(NLP)**나 **시계열 데이터**를 다룰 때 사용
- 예: `|T| = (64, 10, 300)`
    - `배치 크기(batch size)`: 64
    - `시퀀스 길이(sequence length)`: 10
    - `디멘션(dimension)`: 300
    

텐서 연산을 수행할 때, **각 연산의 차원과 형상**을 이해하고 관리하는 것이 매우 중요

잘못된 텐서 연산은 데이터의 형태를 깨트리거나 의도와 다른 결과를 초래할 수 있으므로, 항상 **shape와 dim**을 체크해야 함

## 파이토치(PyTorch) 및 넘파이(Numpy) 기초

### Numpy를 이용한 기본 연산

```python
import numpy as np

# 1D 벡터 선언
t = np.array([0., 1., 2., 3., 4., 5., 6.])
print(t)

# 벡터의 차원과 형태 확인
print('Rank  of t: ', t.ndim)  # 차원 확인
print('Shape of t: ', t.shape) # shape 확인 (원소 개수)

# Rank  of t:  1
# Shape of t:  (7,)
```

### 인덱싱 및 슬라이싱

```python
print('t[0] t[1] t[-1] = ', t[0], t[1], t[-1])  # 특정 인덱스 접근
print('t[2:5] t[4:-1]  = ', t[2:5], t[4:-1])    # 슬라이싱
print('t[:2] t[3:]     = ', t[:2], t[3:])       # 슬라이싱
```

- **설명**:
    - `t[-1]`: 마지막 원소 접근.
    - `t[2:5]`: 2번 인덱스부터 5번 인덱스 직전까지 접근.
    - `t[:2]`: 처음부터 2번 인덱스 직전까지 접근.
    - `t[3:]`: 3번 인덱스부터 끝까지 접근.

### 2D 행렬 선언

```python
t = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]])
print(t)

print('Rank  of t: ', t.ndim)
print('Shape of t: ', t.shape)  # 4 x 3 행렬

# Rank  of t:  2
# Shape of t:  (4, 3)
```

## 파이토치를 이용한 텐서 선언 및 기본 연산

### 1D 텐서 선언

```python
import torch

# float 형태로 1차원 Tensor 선언
t = torch.FloatTensor([0., 1., 2., 3., 4., 5., 6.])
print(t)
# tensor([0., 1., 2., 3., 4., 5., 6.])

print(t.dim())   # rank (차원 확인)
print(t.shape)   # shape (형태 확인)
print(t.size())  # shape와 동일한 결과
print(t[0], t[1], t[-1])  # 엘리먼트 접근
print(t[2:5], t[4:-1])    # 슬라이싱
print(t[:2], t[3:])       # 슬라이싱

# 1
# torch.Size([7])
# torch.Size([7])
# tensor(0.) tensor(1.) tensor(6.)
# tensor([2., 3., 4.]) tensor([4., 5.])
# tensor([0., 1.]) tensor([3., 4., 5., 6.])
```

### 2D 텐서 선언

```python
t = torch.FloatTensor([[1., 2., 3.],
                       [4., 5., 6.],
                       [7., 8., 9.],
                       [10., 11., 12.]])
print(t)

# tensor([[ 1.,  2.,  3.],
#         [ 4.,  5.,  6.],
#         [ 7.,  8.,  9.],
#         [10., 11., 12.]])

print(t.dim())  # rank
print(t.size()) # shape
print(t[:, 1])  # 첫번째 차원 전체, 두번째 차원의 1번째 열 값
print(t[:, 1].size())  # 특정 열의 shape 확인
print(t[:, :-1])  # 마지막 열 제외

# 2
# torch.Size([4, 3])
# tensor([ 2.,  5.,  8., 11.])
# torch.Size([4])
# tensor([[ 1.,  2.],
#         [ 4.,  5.],
#         [ 7.,  8.],
#         [10., 11.]])
```

### 브로드캐스팅 (Broadcasting)

브로드캐스팅은 shape가 다른 텐서 사이에서 작은 텐서를 자동으로 큰 텐서의 shape에 맞게 "확장"시켜 연산을 수행할 수 있게 하는 PyTorch의 기능.

**PyTorch의 브로드캐스팅 규칙**

1. 두 텐서의 차원 수가 다르면, 낮은 차원의 텐서 앞에 1을 추가하여 차원을 맞춤.
2. 각 차원에서 크기를 비교하여, 크기가 1인 차원을 다른 텐서의 해당 차원 크기로 확장.

**예시 1: 동일한 크기**

```python
m1 = torch.FloatTensor([[3, 3]])
m2 = torch.FloatTensor([[2, 2]])
print(m1 + m2)  # [[5, 5]]
```

**예시 2: 벡터 + 스칼라**

```python
m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([3])  # 3 -> [[3, 3]]
print(m1 + m2)  # [[4, 5]]
```

브로드캐스팅 과정

1. `m2`의 shape를 (1,)에서 (1, 1)로 확장
2. `m2`의 두 번째 차원을 1에서 2로 확장: [[3, 3]]
3. `m1(1,2)`와 확장된 `m2(1,2)`의 덧셈 수행

**예시 3: 2 x 1 벡터 + 1 x 2 벡터**

```python
m1 = torch.FloatTensor([[1, 2]])    # [1, 2] -> [[1, 2], [1, 2]]
m2 = torch.FloatTensor([[3], [4]])  # [[3], [4]] -> [[3, 4], [3, 4]]
print(m1 + m2)  # [[4, 6], [5, 7]]
```

> broadcasting은 실수로 shape가 다른 텐서를 연산할 때 의도치 않은 결과가 나올 수 있고  
> 브로드캐스팅으로 인한 오류는 즉시 발견되지 않을 수 있어 발견하기 어려울 수 있음
{: .prompt-danger}

### 행렬곱 (Matrix Multiplication) vs 엘리먼트 와이즈 곱 (Element-wise Multiplication)

```python
# 행렬곱 연산
m1 = torch.FloatTensor([[1, 2], [3, 4]])
m2 = torch.FloatTensor([[1], [2]])
print(m1.matmul(m2))  # 행렬곱 (Matrix Multiplication): 2 x 1

# tensor([[ 5.],
#         [11.]])

# 엘리멘트 와이즈 곱
m1 = torch.FloatTensor([[1, 2], [3, 4]])
m2 = torch.FloatTensor([[1], [2]])
print(m1 * m2)  # 엘리멘트 와이즈 곱: 2 x 2 (broadcasting)

# tensor([[ 1,  2],
#         [ 6,  8]])
```

**행렬곱 (Matrix Multiplication)**

- 행렬곱은 점곱(dot product) 또는 내적(inner product)이라고도 불림
- 행렬곱은 두 행렬의 뒤와 앞 차원의 크기가 일치할 때 수행 가능
    - `(m, n)` 크기의 행렬과 `(n, p)` 크기의 행렬을 곱하면 `(m, p)` 크기의 행렬이 생성
- 연산 방법
    - 첫 번째 행렬의 **행(row)**과 두 번째 행렬의 **열(column)**을 각각 곱하고 더하여 하나의 값으로 만듭니다.

**엘리먼트 와이즈 곱 (Element-wise Multiplication)**

- 엘리먼트 와이즈 곱은 두 텐서의 **크기(Shape)가 동일할 때** 각 위치의 원소끼리 곱함
- 행렬의 각 요소별로 곱하기 때문에, 두 텐서의 모든 차원이 같아야 함
- 파이썬 연산자 또는 PyTorch의 `torch.mul()`을 사용하여 연산``

### 평균 (Mean)

평균(mean)은 주어진 텐서의 모든 원소의 합을 원소의 개수로 나눈 결과를 반환.

`dim` 인수를 지정하여 특정 차원을 지정하면 해당 차원의 평균을 계산하고 차원을 줄임.

```python
t = torch.FloatTensor([[1, 2], [3, 4]])
print(t.mean())       # 모든 원소의 평균: (1 + 2 + 3 + 4) / 4 = 2.5
print(t.mean(dim=0))  # 첫 번째 차원(행) 제거 후 평균
# tensor([2., 3.])  # [ (1+3)/2, (2+4)/2 ]
print(t.mean(dim=1))  # 두 번째 차원(열) 제거 후 평균
# tensor([1.5, 3.5])  # [ (1+2)/2, (3+4)/2 ]
```

**정수형 텐서**(LongTensor)에서 사용 시 오류가 발생할 수 있음.

```python
# Can't use mean() on integers
t = torch.LongTensor([1, 2])
try:
    print(t.mean())
except Exception as exc:
    print(exc)
```

### 합 (Sum)

sum은 주어진 텐서의 모든 원소를 더한 결과를 반환.

```python
t = torch.FloatTensor([[1, 2], [3, 4]])
print(t.sum())  # 전체 원소의 합: 1 + 2 + 3 + 4 = 10
print(t.sum(dim=0))  # 첫 번째 차원(행) 제거 후 각 열의 합계
# tensor([4., 6.])  # [ (1+3), (2+4) ]

print(t.sum(dim=1))  # 두 번째 차원(열) 제거 후 각 행의 합계
# tensor([3., 7.])  # [ (1+2), (3+4) ]
```

### 최대값 (Max) 및 Argmax

`max()` 연산은 텐서 내 최대값을 찾고, `argmax()`는 해당 값의 인덱스를 반환.

```python
t = torch.FloatTensor([[1, 2], [3, 4]])
print(t.max())  # 텐서 내 가장 큰 값 : 4
print(t.max(dim=0))  # 첫 번째 차원 기준 최대값 및 인덱스
# (tensor([3., 4.]), tensor([1, 1]))  # 첫 번째 차원의 최대값, 인덱스

print(t.max(dim=1))  # 두 번째 차원 기준 최대값 및 인덱스
# (tensor([2., 4.]), tensor([1, 1]))  # 두 번째 차원의 최대값, 인덱스

print(t.argmax(dim=0))  # 첫 번째 차원 기준 최대값 인덱스
# tensor([1, 1])
```

🥕  `dim`을 지정할 때 **해당 차원**이 제거된다고 생각하면 쉽게 이해 가능 !
