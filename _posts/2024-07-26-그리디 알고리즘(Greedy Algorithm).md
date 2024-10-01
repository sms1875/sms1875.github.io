---
layout: post
title: "그리디 알고리즘(Greedy Algorithm)"
date: 2024-07-26 09:05:00
categories: [Study, Algorithm]
tags: [Greedy]
---
## 그리디 알고리즘(Greedy Algorithm)이란?

각 단계에서 최선의 선택을 반복적으로 수행하여 전체 문제를 해결하는 알고리즘  
문제의 최적해를 보장하지 않지만, 특정 조건을 만족하는 경우 최적해를 찾을 수 있음


### 특징

* 현재의 최선 선택: 각 단계에서 가장 좋은 선택을 하여 문제를 해결
* 지역 최적해를 반복하여 전역 최적해를 구하려고 시도함
* 백트래킹이나 다이나믹 프로그래밍과는 다르게 상황을 되돌아보지 않음


### 그리디 알고리즘이 최적해를 보장할 수 있는 조건

* 탐욕적 선택 속성(Greedy Choice Property): 문제의 부분해가 항상 전체 문제의 최적해로 이어질 수 있어야 함
* 최적 부분 구조(Optimal Substructure): 부분 문제의 최적해가 전체 문제의 최적해로 확장될 수 있어야 함
  
### 장점

1. 구현이 간단하고, 직관적  
2. 특정 문제에 대해 최적해를 빠르게 구할 수 있음
3. 탐색 공간을 줄여서 성능을 개선할 수 있음
   
### 단점

1. 모든 문제에 대해 최적해를 보장하지는 않습니다.
2. 문제의 특성을 이해하고, 그리디 알고리즘이 적합한지 확인해야 함
3. 경우에 따라 다이나믹 프로그래밍 또는 백트래킹 알고리즘이 더 나은 성능을 보일 수 있음

## 활용 문제 예시

### 최소 신장 트리 (Minimum Spanning Tree, MST)
그래프에서 모든 정점을 포함하는 최소 비용의 신장 트리를 구하는 문제  
대표적인 알고리즘으로 **크루스칼 알고리즘(Kruskal's Algorithm)** 과 **프림 알고리즘(Prim's Algorithm)** 이 있음

### 허프만 코딩 (Huffman Code)
문자열의 각 문자의 빈도수를 고려하여 최단 길이의 이진 코드로 압축하는 방법  
각 문자의 빈도수를 기준으로 허프만 트리를 생성하고, 이를 통해 최적의 압축 코드를 만듬

### 동전 거스름돈 문제

손님에게 거스름돈을 줄 때, 가장 적은 수의 동전을 사용하는 방법을 찾는 문제  
일반적으로 큰 단위의 동전부터 거슬러주는 것이 최선의 해   
단, 동전의 단위가 1, 3, 4와 같이 비표준적인 경우에는 그리디 알고리즘이 최적해를 보장하지 못할 수 있음

``` cpp
void minCoinChange(int coins[], int m, int amount) {
    int count = 0;
    for (int i = m - 1; i >= 0; i--) {  // 큰 단위의 동전부터 탐색
        if (amount >= coins[i]) {
            count += amount / coins[i];  // 해당 동전으로 거슬러 줄 수 있는 최대 개수
            amount %= coins[i];  // 남은 금액
        }
    }
    cout << "최소 동전 개수: " << count << endl;
}
```

### 배낭 문제 (Fractional Knapsack Problem)
무게 제한이 있는 배낭에 물건을 넣을 때, 물건의 가치 대비 무게 비율이 높은 순서대로 선택하여 배낭의 총 가치를 최대화하는 문제  
물건을 쪼갤 수 있는 경우에 그리디 알고리즘이 최적해를 보장

```cpp
struct Item {
    int weight, value;
    Item(int w, int v) : weight(w), value(v) {}
};

// 무게 대비 가치 비율을 기준으로 정렬
bool compare(Item a, Item b) {
    double r1 = (double)a.value / a.weight;
    double r2 = (double)b.value / b.weight;
    return r1 > r2;
}

double fractionalKnapsack(Item arr[], int n, int W) {
    sort(arr, arr + n, compare);  // 가치 비율이 높은 순서대로 정렬
    double totalValue = 0.0;

    for (int i = 0; i < n; i++) {
        if (arr[i].weight <= W) {  // 배낭에 전부 넣을 수 있는 경우
            W -= arr[i].weight;
            totalValue += arr[i].value;
        } else {  // 배낭에 일부만 넣을 수 있는 경우
            totalValue += arr[i].value * ((double)W / arr[i].weight);
            break;
        }
    }
    return totalValue;
}
```
### 활동 선택 문제 (Activity Selection Problem)

여러 개의 활동이 주어졌을 때, 각 활동이 겹치지 않도록 최대한 많은 활동을 선택하는 문제  
활동의 종료 시간이 빠른 순서로 정렬하여 선택하면 최적해를 얻을 수 있음

```cpp
struct Activity {
    int start, end;
};

// 종료 시간을 기준으로 정렬
bool compare(Activity a, Activity b) {
    return a.end < b.end;
}

void activitySelection(Activity arr[], int n) {
    sort(arr, arr + n, compare);  // 종료 시간 기준으로 정렬

    int count = 1;  // 첫 번째 활동 선택
    int prev_end = arr[0].end;

    for (int i = 1; i < n; i++) {
        if (arr[i].start >= prev_end) {  // 현재 활동의 시작 시간이 이전 활동의 종료 시간 이후일 경우
            count++;
            prev_end = arr[i].end;  // 종료 시간 업데이트
        }
    }
    cout << "최대 선택 가능한 활동 개수: " << count << endl;
}
```
