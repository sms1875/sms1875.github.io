---
layout: post
title: "최장 증가 수열 (LIS, Longest Increasing Subsequence)"
date: 2024-09-05 10:21:00
categories: [Study, Algorithm]
tags: [LIS, Longest Increasing Subsequence]
---
# 최장 증가 수열(LIS)이란?
주어진 수열에서 순서대로 정렬된 가장 긴 증가하는 부분 수열  

**예시**  

수열: [3, 10, 2, 1, 20]  
LIS: [3, 10, 20]  
길이: 3 

수열: [50, 3, 10, 7, 40, 80]  
LIS: [3, 7, 40, 80]  
길이: 4  

LIS 문제는 **동적 계획법(Dynamic Programming)** 과 **이분 탐색(Binary Search)** 을 활용하여 해결 가능


## LIS 알고리즘의 장점과 단점

### 장점  

다양한 알고리즘(동적 계획법, 이분 탐색)을 사용하여 문제를 해결할 수 있어, 시간 복잡도를 최적화 가능  
LIS 알고리즘을 활용하여 다양한 응용 문제를 해결  

### 단점  

동적 계획법을 사용할 경우, O(N²)의 시간 복잡도를 가지므로 큰 입력에 대해 비효율적일 수 있음  
실제 LIS를 추적하여 구할 때는 추적 배열을 추가해야 하므로, 메모리 사용이 늘어날 수 있음  

# LIS 알고리즘 종류

## 동적 계획법 (Dynamic Programming) - 시간 복잡도 O(N²)

dp[i]는 i번째 원소를 마지막으로 포함하는 LIS의 길이를 의미  
이전 모든 원소와 비교하여, 증가하는 순서가 유지되면 dp 값을 갱신  

```cpp
int LIS_DP(int arr[], int n) {
    int dp[n];  // dp 배열 생성
    fill(dp, dp + n, 1);  // 모든 값 초기화 (각 원소는 최소 길이 1)

    // 각 원소에 대해 이전 원소들과 비교하여 최장 증가 수열 계산
    for (int i = 1; i < n; i++) {
        for (int j = 0; j < i; j++) {
            if (arr[i] > arr[j]) {  // 증가하는 관계일 경우
                dp[i] = max(dp[i], dp[j] + 1);
            }
        }
    }

    // dp 배열의 최댓값이 LIS의 길이
    return *max_element(dp, dp + n);
}
```

## 이분 탐색을 이용한 그리디 방법 - 시간 복잡도 O(N log N)

그리디 알고리즘과 **이분 탐색(Binary Search)** 을 결합  
LIS를 구성할 수 있는 가장 작은 원소들을 유지하면서 길이를 확장  
이를 위해 임시 배열을 사용하고, 각 원소를 이분 탐색으로 삽입하여 LIS의 길이를 구함  

```cpp

#include <bits/stdc++.h>
using namespace std;

int LIS_BinarySearch(int arr[], int n) {
    vector<int> lis;  // LIS를 저장할 임시 배열

    for (int i = 0; i < n; i++) {
        auto it = lower_bound(lis.begin(), lis.end(), arr[i]);  // 이분 탐색을 이용한 위치 찾기
        if (it == lis.end()) {  // 현재 원소가 LIS 배열의 마지막 값보다 크면 추가
            lis.push_back(arr[i]);
        } else {  // 아니면 기존 값 교체
            *it = arr[i];
        }
    }

    return lis.size();  // LIS 배열의 크기가 LIS의 길이
}
```

## 동적 계획법과 이분 탐색을 결합한 방법 - 시간 복잡도 O(N log N)

DP 배열을 이분 탐색을 통해 갱신하여 LIS의 길이를 계산  
DP 배열을 사용하여 각 길이에 해당하는 최소값을 기록하고, 이를 이분 탐색으로 갱신  

```cpp
int LIS_Optimized(int arr[], int n) {
    vector<int> dp;  // dp 배열: 각 길이에 해당하는 최소값을 저장

    for (int i = 0; i < n; i++) {
        auto it = lower_bound(dp.begin(), dp.end(), arr[i]);  // 이분 탐색으로 삽입 위치 찾기
        if (it == dp.end()) dp.push_back(arr[i]);  // 배열의 마지막 위치에 추가
        else *it = arr[i];  // 해당 위치에 원소 갱신
    }

    return dp.size();  // dp 배열의 크기가 LIS의 길이
}
```
# LIS의 응용

## 최장 감소 수열 (LDS, Longest Decreasing Subsequence)
LIS와 반대로 가장 긴 감소 수열  
arr 배열의 각 원소를 반대로 정렬한 후 LIS 알고리즘을 적용하면 해결 가능

## 2차원 평면에서 LIS (Longest Increasing Subsequence in 2D)  
2차원 평면에 있는 점들의 좌표가 주어졌을 때, 가장 긴 증가 수열  
먼저 x 좌표를 기준으로 정렬하고, 이후 y 좌표에 대해서 LIS를 찾음  

## LIS의 길이뿐만 아니라 실제 수열 찾기  
LIS 알고리즘을 사용하여 단순히 길이만 구하는 것이 아니라, 실제 LIS를 추적하여 부분 수열을 반환 가능  
DP 배열의 값을 이용하거나, 그리디 방법에서 백트래킹을 사용하여 수열을 구함

```cpp
vector<int> findLIS(int arr[], int n) {
    vector<int> lis;
    vector<int> parent(n, -1);  // 추적을 위한 부모 배열
    vector<int> lis_idx(n, 0);  // LIS 값의 인덱스 추적
    int lis_length = 0, lis_end = 0;

    for (int i = 0; i < n; i++) {
        int pos = lower_bound(lis.begin(), lis.end(), arr[i]) - lis.begin();
        if (pos >= lis.size()) lis.push_back(arr[i]);
        else lis[pos] = arr[i];

        lis_idx[pos] = i;  // LIS의 각 값의 위치 저장
        if (pos > 0) parent[i] = lis_idx[pos - 1];  // 부모 노드 설정

        if (pos + 1 > lis_length) {
            lis_length = pos + 1;
            lis_end = i;  // 최종 LIS 끝 위치 저장
        }
    }

    vector<int> result(lis_length);
    for (int i = lis_length - 1; i >= 0; i--) {
        result[i] = arr[lis_end];
        lis_end = parent[lis_end];
    }
    return result;
}
```

<details>
    <summary>동작과정</summary>
  
**입력**

```
n = 8
arr[] = [6 2 5 1 7 4 8 3]
```

**초기 변수 상태**

```
lis: 빈 배열 ([])
parent: [-1, -1, -1, -1, -1, -1, -1, -1]
lis_idx: [0, 0, 0, 0, 0, 0, 0, 0]
lis_length: 0
lis_end: 0
```

**각 반복 단계별 상태 변화**

***i = 0, arr[i] = 6***  
pos = lower_bound(lis.begin(), lis.end(), 6) - lis.begin() = 0  
lis가 비어있으므로, lis.push_back(6)  
lis: [6]  
parent: [-1, -1, -1, -1, -1, -1, -1, -1]  
lis_idx: [0, 0, 0, 0, 0, 0, 0, 0]  
lis_length: 1  
lis_end: 0  

***i = 1, arr[i] = 2***  
pos = lower_bound(lis.begin(), lis.end(), 2) - lis.begin() = 0  
lis의 첫 번째 원소(6)를 2로 교체 → lis[0] = 2  
lis: [2]  
parent: [-1, -1, -1, -1, -1, -1, -1, -1]  
lis_idx: [1, 0, 0, 0, 0, 0, 0, 0]  
lis_length: 1  
lis_end: 1  

***i = 2, arr[i] = 5***  
pos = lower_bound(lis.begin(), lis.end(), 5) - lis.begin() = 1  
pos == lis.size(), lis.push_back(5)  
parent[2] = lis_idx[0] = 1  
lis: [2, 5]  
parent: [-1, -1, 1, -1, -1, -1, -1, -1]  
lis_idx: [1, 2, 0, 0, 0, 0, 0, 0]  
lis_length: 2  
lis_end: 2  

***i = 3, arr[i] = 1***  
pos = lower_bound(lis.begin(), lis.end(), 1) - lis.begin() = 0  
lis의 첫 번째 원소(2)를 1로 교체 → lis[0] = 1  
lis: [1, 5]  
parent: [-1, -1, 1, -1, -1, -1, -1, -1]  
lis_idx: [3, 2, 0, 0, 0, 0, 0, 0]  
lis_length: 2  
lis_end: 2  

***i = 4, arr[i] = 7***  
pos = lower_bound(lis.begin(), lis.end(), 7) - lis.begin() = 2  
pos == lis.size(), lis.push_back(7)  
parent[4] = lis_idx[1] = 2  
lis: [1, 5, 7]  
parent: [-1, -1, 1, -1, 2, -1, -1, -1]  
lis_idx: [3, 2, 4, 0, 0, 0, 0, 0]  
lis_length: 3  
lis_end: 4  

***i = 5, arr[i] = 4***  
pos = lower_bound(lis.begin(), lis.end(), 4) - lis.begin() = 1  
lis의 두 번째 원소(5)를 4로 교체 → lis[1] = 4  
parent[5] = lis_idx[0] = 3  
lis: [1, 4, 7]  
parent: [-1, -1, 1, -1, 2, 3, -1, -1]   
lis_idx: [3, 5, 4, 0, 0, 0, 0, 0]  
lis_length: 3  
lis_end: 4  

***i = 6, arr[i] = 8***  
pos = lower_bound(lis.begin(), lis.end(), 8) - lis.begin() = 3  
pos == lis.size(), lis.push_back(8)  
parent[6] = lis_idx[2] = 4  
lis: [1, 4, 7, 8]  
parent: [-1, -1, 1, -1, 2, 3, 4, -1]  
lis_idx: [3, 5, 4, 6, 0, 0, 0, 0]  
lis_length: 4  
lis_end: 6  

***i = 7, arr[i] = 3***    
pos = lower_bound(lis.begin(), lis.end(), 3) - lis.begin() = 1  
lis의 두 번째 원소(4)를 3으로 교체 → lis[1] = 3  
parent[7] = lis_idx[0] = 3  
lis: [1, 3, 7, 8]  
parent: [-1, -1, 1, -1, 2, 3, 4, 3]  
lis_idx: [3, 7, 4, 6, 0, 0, 0, 0]  
lis_length: 4  
lis_end: 6  

**LIS의 추적 과정**   

parent 배열을 통해 LIS 수열을 역추적함  
lis_length = 4이고, lis_end = 6이므로 arr[6] = 8에서부터 추적 시작  

result 배열 생성 ([0, 0, 0, 0])  

***i = 3***  
result[3] = arr[lis_end] = arr[6] = 8  
lis_end = parent[6] = 4  

***i = 2***  
result[2] = arr[4] = 7  
lis_end = parent[4] = 2  

***i = 1***  
result[1] = arr[2] = 5  
lis_end = parent[2] = 1  

***i = 0***  
result[0] = arr[1] = 2  
lis_end = parent[1] = -1 (종료)  

**최종 결과**  

```
LIS: [2, 5, 7, 8]  
길이: 4
```

</details>
