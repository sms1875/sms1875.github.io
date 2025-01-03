---
layout: post
title: "[모두를 위한 딥러닝 시즌2] Lab-10-7 Next step of CNN"
date: 2024-12-04 18:29:00+0900
categories: [Study, AI]
tags: [Deep Learning Zero To All, 모두를 위한 딥러닝 시즌2, ML, DL, pytorch]
math: true
mermaid : true
---
## 앞으로 뭘 하면 좋을까?

### Classification

![image.png](assets/img/posts/study/AI/10-7/image.png)

- 이미지의 내용을 분석하여 어떤 객체인지 식별  
  예) 강아지 사진 → "강아지"
- 주요 모델
    - **DenseNet:** 레이어 간 연결성을 극대화하여 학습 효율을 높인 신경망 구조
    - **SENet:** 채널 간의 중요도를 학습하여 성능을 높이는 네트워크
    - **MobileNet:** 모바일 환경을 위한 경량화된 딥러닝 모델
    - **SqueezeNet:** 모델 크기를 줄여도 높은 성능을 유지하는 효율적인 네트워크
    - **AutoML (NAS, NASNet):** **강화학습** 기반으로 최적화된 딥러닝 아키텍처를 자동으로 생성 (RNN 이해 필요)

### Detection

![image.png](assets/img/posts/study/AI/10-7/image%201.png)

- 이미지 내 객체의 위치와 종류를 판별  
  예) 사진 내 강아지의 위치와 종류 식별
- 검색 키워드 : Latest Object Detection
- 참고 사이트
    - 객체 탐지 관련 논문과 모델 리스트를 정리한 리소스
        [GitHub - hoya012/deep_learning_object_detection](https://github.com/hoya012/deep_learning_object_detection)
        

### Tracking

![image.png](assets/img/posts/study/AI/10-7/image%202.png)

- 영상 내에서 연속적인 프레임 간 객체의 위치를 추적
- **주요 모델**
    1. **MDNet:** CNN 기반의 다중 도메인 학습을 활용한 추적 네트워크
    2. **GOTURN:** 영상 내 객체를 단일 패스(single pass)로 빠르게 추적하는 모델
    3. **CFNet:** 크로스-코릴레이션을 활용한 효율적인 추적 모델
    4. **ROLO:** LSTM을 활용해 객체의 위치 정보를 예측하는 모델
- 검색 키워드 : Tracking the Untrackable

### Segmentation

![image.png](assets/img/posts/study/AI/10-7/image%203.png)

- 객체와 배경을 분리하여 특정 객체를 시각적으로 구분  
  예) 강아지 부분에 색칠하여 특정 객체 표시
- **주요 모델**
    1. **FCN (Fully Convolutional Network):** 모든 계층을 컨볼루션으로 구성한 세그멘테이션 모델
    2. **U-Net:** 의료 이미지 세그멘테이션에 특화된 구조로, 업샘플링을 통해 정확도 향상
    3. **Mask RCNN:** 객체 탐지와 세그멘테이션을 동시에 수행하는 모델
- 검색 키워드: Image segmentation deep learning

### **커스텀 데이터셋 제작**

- 커스텀 데이터셋 만드는 방법을 익히는 것은 원하는 아키텍처와 데이터셋에 딥러닝을 적용하기 위해 필수적이다
- 토치 비전을 통해 정의된 데이터셋의 규칙에 맞춰 클래스를 구성하면 파이토치에서 사용 가능한 형태의 커스텀 데이터셋을 쉽게 만들 수 있다
- **추천 주제**
    1. **이미지 캡셔닝:** 이미지 설명 생성
    2. **슈퍼 레졸루션:** 저해상도 이미지를 고해상도로 변환
    3. **제너레이티브 모델:** 생성적 적대 신경망(GAN)을 활용한 데이터 생성
    4. **오픈 포즈:** 사람의 신체 포즈를 추출하는 기술

- 참고 사이트
    - Custom DataSet 만드는 방법 익히기  
    [Sign in to Roboflow](https://app.roboflow.com/)
    - Pytorch가 제공하지 않는 데이터셋 다운받아 학습해보기
        1. [Tiny ImageNet Challenge](https://www.kaggle.com/c/thu-deep-learning/overview)
        2. [Kaggle Competitions](https://www.kaggle.com/competitions?hostSegmentIdFilter=5)
          
