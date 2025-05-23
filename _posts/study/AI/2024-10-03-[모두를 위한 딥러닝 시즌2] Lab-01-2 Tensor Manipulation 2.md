---
layout: post
title: "[모두를 위한 딥러닝 시즌2] Lab-01-2 Tensor Manipulation 2"
date: 2024-10-03 16:35:00+0900
categories: [Study, AI]
tags: [Deep Learning Zero To All, 모두를 위한 딥러닝 시즌2, ML, DL, pytorch]
---
{% include embed/youtube.html id='XkqdNaNQGx8' %}  

> **Reshape**는  **원소 값 순서**는 그대로 유지하면서, **모양(shape)만 변경**  
> 데이터의 실제 값은 바꾸지 않고 **형태만 재구성**  
> 기존 차원의 곱과 변경할 차원의 곱이 일치해야 함   
{: .prompt-info}  

### View (Reshape)

텐서의 모양을 원하는 대로 변경하는 기능  
**-1**: 차원을 지정하지 않음, 보통 가장 변동이 심한 배치 사이즈에 사용

```python
t = np.array([[[0, 1, 2],
               [3, 4, 5]],

              [[6, 7, 8],
               [9, 10, 11]]])
ft = torch.FloatTensor(t)
print(ft.shape)
# torch.Size([2, 2, 3])

# View로 텐서의 차원을 바꾸기
print(ft.view([-1, 3]))
print(ft.view([-1, 3]).shape)
# tensor([[ 0.,  1.,  2.],
#         [ 3.,  4.,  5.],
#         [ 6.,  7.,  8.],
#         [ 9., 10., 11.]])
# torch.Size([4, 3])

print(ft.view([-1, 1, 3]))
print(ft.view([-1, 1, 3]).shape)
# tensor([[[ 0.,  1.,  2.]],
#         [[ 3.,  4.,  5.]],
#         [[ 6.,  7.,  8.]],
#         [[ 9., 10., 11.]]])
# torch.Size([4, 1, 3])
```
![alt text](assets/img/posts/study/AI/1-2/image.png)
---

### Squeeze

디멘션에서 엘리먼트의 값이 1인 경우 그 차원을 제거하는 함수  
**dim**을 명시하지 않으면 1인 차원 모두를 제거  

```python
ft = torch.FloatTensor([[0], [1], [2]])
print(ft)
print(ft.shape)
# tensor([[0.],
#         [1.],
#         [2.]])
# torch.Size([3, 1])

print(ft.squeeze())
print(ft.squeeze().shape)
# tensor([0., 1., 2.])
# torch.Size([3])
```

---

### Unsqueeze

원하는 차원에 1을 추가하여 새로운 차원을 만들 때 사용  
**dim**을 명시하여 해당 차원에 1을 추가  

```python
ft = torch.FloatTensor([0, 1, 2])
print(ft.shape)
# torch.Size([3])

print(ft.unsqueeze(0))
print(ft.unsqueeze(0).shape)
# tensor([[0., 1., 2.]])
# torch.Size([1, 3])

print(ft.unsqueeze(1))
print(ft.unsqueeze(1).shape)
# tensor([[0.],
#         [1.],
#         [2.]])
# torch.Size([3, 1])
```
![alt text](assets/img/posts/study/AI/1-2/image-1.png)
---
### Scatter (for One-Hot Encoding)


텐서의 **지정된 위치**에 **특정 값을 할당**  
주로 **One-hot Encoding**을 구현하거나, **복잡한 인덱싱 및 할당 연산**을 수행할 때 사용  
`scatter(dim, index, src)` 형식으로 사용되며, `dim`은 값을 채울 **차원**, `index`는 **인덱스 위치**, `src`는 해당 위치에 **할당할 값**

```python
import torch

# LongTensor로 인덱스를 정의
lt = torch.LongTensor([[0], [1], [2], [0]])
print("Index Tensor:")
print(lt)
# tensor([[0],
#         [1],
#         [2],
#         [0]])

# zeros_like로 크기가 동일한 4x3의 텐서 생성
one_hot = torch.zeros(4, 3)
print("Initial Zero Tensor:")
print(one_hot)
# tensor([[1., 0., 0.],
#         [0., 1., 0.],
#         [0., 0., 1.],
#         [1., 0., 0.]])
```

> One-hot Encoding이란?  
> **범주형 데이터(categorical data)**를 벡터화하여 표현하는 방법  
> 각 클래스의 인덱스에 해당하는 위치에 1을 채우고, 나머지 모든 값은 0으로 채워 범주형 변수를 수치형으로 변환 가능  
> 예: [고양이, 개, 토끼]를 One-hot Encoding으로 변환  
> ```
> 고양이: [1, 0, 0]  
> 개: [0, 1, 0]  
> 토끼: [0, 0, 1]   
> ```  
{: .prompt-info}

---

### Type Casting

텐서의 타입을 변경 (예: long → float)  

```python
lt = torch.LongTensor([1, 2, 3, 4])
print(lt)
# tensor([1, 2, 3, 4])

print(lt.float())
# tensor([1., 2., 3., 4.])

bt = torch.ByteTensor([True, False, False, True])
print(bt)
# tensor([1, 0, 0, 1], dtype=torch.uint8)

print(bt.long())
print(bt.float())
# tensor([1, 0, 0, 1])
# tensor([1., 0., 0., 1.])
```

> **byte type**은 조건에 따라 True, False를 자동으로 생성이 가능
> ```python
> # LongTensor 생성
> lt = torch.LongTensor([1, 2, 3, 4, 5, 6])
> 
> # 조건에 따라 ByteTensor 생성: lt의 값이 3보다 큰 경우 True, 나머지는 False
> bt = lt > 3
> print(bt)
> # tensor([False, False, False,  True,  True,  True])
> ```
> 마스킹 등에 활용  
{: .prompt-info}  

---

### Concatenation

텐서들을 지정한 **dim**을 기준으로 이어붙임  

```python
x = torch.FloatTensor([[1, 2], [3, 4]])
y = torch.FloatTensor([[5, 6], [7, 8]])

print(torch.cat([x, y], dim=0))
# tensor([[1., 2.],
#         [3., 4.],
#         [5., 6.],
#         [7., 8.]])

print(torch.cat([x, y], dim=1))
# tensor([[1., 2., 5., 6.],
#         [3., 4., 7., 8.]])
```
![alt text](assets/img/posts/study/AI/1-2/image-2.png)
---

### Stacking

텐서를 쌓아 새로운 차원을 생성하는 방식   
**unsqueeze**와 **cat**을 이용한 방식과 동일하게 동작  

```python
x = torch.FloatTensor([1, 4])
y = torch.FloatTensor([2, 5])
z = torch.FloatTensor([3, 6])

print(torch.stack([x, y, z])) # torch.cat([x.unsqueeze(0), y.unsqueeze(0), z.unsqueeze(0)], dim=0) 와 같음
# tensor([[1., 4.],
#         [2., 5.],
#         [3., 6.]])

print(torch.stack([x, y, z], dim=1)) # torch.cat([x.unsqueeze(1),y.unsqueeze(1),z.unsqueeze(1)], dim=1) 와 같음
# tensor([[1., 2., 3.],
#         [4., 5., 6.]])
```
![alt text](assets/img/posts/study/AI/1-2/image-3.png)
---

## Ones and Zeros Like

주어진 텐서와 동일한 크기(shape)로 **1** 또는 **0**을 채운 새로운 텐서를 생성  

```python
x = torch.FloatTensor([[0, 1, 2], [2, 1, 0]])
print(x)
# tensor([[0., 1., 2.],
#         [2., 1., 0.]])

print(torch.ones_like(x))
# tensor([[1., 1., 1.],
#         [1., 1., 1.]])

print(torch.zeros_like(x))
# tensor([[0., 0., 0.],
#         [0., 0., 0.]])
```

---

## In-place Operation

메모리를 새로 할당하지 않고 기존 텐서를 수정하는 연산  
연산 함수에 **언더바(_)**를 붙여서 사용  

```python
x = torch.FloatTensor([[1, 2], [3, 4]])

print(x.mul(2.))
# tensor([[2., 4.],
#         [6., 8.]])
print(x)
# tensor([[1., 2.],
#         [3., 4.]])

print(x.mul_(2.))
print(x)
# tensor([[2., 4.],
#         [6., 8.]])
```

> 심화 내용 : device
{: .prompt-tip}  
