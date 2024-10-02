---
layout: post
title: "우선순위 큐 (Priority Queue)"
date: 2024-10-02 13:44:00+0900
categories: [Study, Data Structure]
tags: [Priority Queue, Queue, Heap]
---
## 우선순위 큐(Priority Queue)란?

우선순위가 높은 요소가 가장 먼저 처리되는 선형 자료구조   
일반적인 큐는 **선입선출(FIFO)**의 원칙을 따르지만, 우선순위 큐는 요소의 우선순위에 따라 추출되는 순서가 결정됨  
즉, 우선순위가 높은 요소가 큐의 맨 앞에 위치하며, 우선순위가 같은 경우 선입선출의 규칙을 따름  

### 특징
1. 우선순위에 따라 추출: 우선순위가 높은 요소가 먼저 처리
2. 동적 정렬: 삽입과 삭제 시마다 요소가 우선순위에 따라 자동으로 정렬
3. 삽입/삭제의 시간 복잡도: 일반적으로 우선순위 큐의 삽입 및 삭제는 O(log n)의 시간 복잡도

### 장점
1. 효율적 우선순위 처리: 여러 작업의 우선순위를 고려하여 중요한 작업을 먼저 처리
2. 다익스트라(Dijkstra) 알고리즘, 작업 스케줄링, 네트워크 패킷 처리 등에서 사용

### 단점
1. 구현의 복잡성: 일반 큐와 달리 우선순위를 고려해야 하므로, 구현이 복잡해질 수 있음
2. 삽입/삭제 비용: 일반 큐에 비해 삽입과 삭제 연산의 비용이 높음

### 사용 예시
* 최단 경로 탐색: 그래프의 다익스트라 알고리즘에서, 최단 경로를 찾기 위해 가장 비용이 낮은 정점을 우선적으로 탐색
* 작업 스케줄링: CPU 작업 스케줄링에서 작업의 우선순위에 따라 작업을 처리
* 이벤트 처리 시스템: 이벤트의 우선순위에 따라 사용자 입력을 처리하거나, 게임 AI에서 우선순위가 높은 행동을 먼저 수행
* 네트워크 패킷 처리: 네트워크 라우터가 패킷을 처리할 때, 중요한 패킷을 먼저 처리하여 QoS(Quality of Service)를 보장
* 긴급 상황 관리: 병원에서 응급 환자나 심각한 상태의 환자를 우선적으로 진료할 때


### 구현 방식
 

**우선순위 큐를 구현할 수 있는 주요 자료구조**

1. 배열 (Array)
    정렬되지 않은 배열을 사용하면, 삽입은 O(1)이지만, 최대/최소값을 추출하는 비용이 O(n)

2. 연결 리스트 (Linked List)
    삽입 시 올바른 위치를 찾아서 O(n)의 시간이 필요하지만, 최대/최소값 추출은 O(1)로 매우 효율적  

3. 힙 (Heap)
    최대 힙과 최소 힙을 사용하여 우선순위 큐를 구현하면, 삽입과 삭제 연산이 O(log n)의 시간 복잡도를 가집니다.

> 힙은 우선순위 큐의 구현에서 가장 많이 사용되는 자료구조로, 최대값 또는 최소값을 빠르게 추출할 수 있으며, 삽입과 삭제의 시간 복잡도가 O(log n)으로 매우 효율적  
{: .prompt-tip}  

## 구현 예시
### 배열 기반
```cpp
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

class PriorityQueue {
    vector<int> data;

public:
    // 요소 삽입 (우선순위에 따라 정렬)
    void enqueue(int value) {
        data.push_back(value);
        sort(data.begin(), data.end(), greater<int>());  // 내림차순 정렬 (최대값이 맨 앞)
    }

    // 요소 제거 (최대값 제거)
    void dequeue() {
        if (data.empty()) {
            cout << "Queue is Empty\n";
            return;
        }
        cout << "Removed: " << data.front() << endl;
        data.erase(data.begin());  // 가장 앞의 요소 제거
    }

    // 큐의 맨 앞 요소 반환
    int peek() {
        if (data.empty()) {
            cout << "Queue is Empty\n";
            return -1;
        }
        return data.front();  // 가장 우선순위가 높은 요소 반환
    }

    // 큐의 상태 출력
    void printQueue() {
        cout << "Priority Queue: ";
        for (int i = 0; i < data.size(); i++) {
            cout << data[i] << " ";
        }
        cout << endl;
    }
};

int main() {
    PriorityQueue pq;
    pq.enqueue(30);
    pq.enqueue(20);
    pq.enqueue(50);
    pq.enqueue(10);

    pq.printQueue();  // Priority Queue: 50 30 20 10
    pq.dequeue();     // Removed: 50
    pq.printQueue();  // Priority Queue: 30 20 10

    return 0;
}
```

### 라이브러리

<queue> 라이브러리의 priority_queue 클래스를 사용  

```cpp
#include <iostream>
#include <queue>
using namespace std;

int main() {
    priority_queue<int> pq;  // 기본적으로 최대 힙으로 구성된 우선순위 큐

    // 요소 삽입
    pq.push(30);
    pq.push(20);
    pq.push(50);
    pq.push(10);

    // 우선순위 큐의 맨 앞 요소 출력
    cout << "Top element is: " << pq.top() << endl;  // 50 (최대값)

    // 요소 제거
    pq.pop();
    cout << "Top element after removal: " << pq.top() << endl;  // 30

    return 0;
}
``` 
### Python의 heapq 라이브러리를 사용한 우선순위 큐

```python
import heapq

# 우선순위 큐 초기화
pq = []

# 요소 삽입
heapq.heappush(pq, 30)
heapq.heappush(pq, 20)
heapq.heappush(pq, 50)
heapq.heappush(pq, 10)

# 우선순위 큐의 맨 앞 요소 출력
print("Top element is:", pq[0])  # 10 (최소값)

# 요소 제거
heapq.heappop(pq)
print("Top element after removal:", pq[0])  # 20
```

> priority_queue는 C++에서 기본적으로 최대 힙으로 동작하지만, Python의 heapq는 최소 힙이므로 주의해야 함.  
> heapq에서 최대 힙을 구현하려면 요소를 삽입할 때 -value로 반전하여 처리
{: .prompt-info}