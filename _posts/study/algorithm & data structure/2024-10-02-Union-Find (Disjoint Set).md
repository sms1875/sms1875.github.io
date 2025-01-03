---
layout: post
title: "Union-Find (Disjoint Set)"
date: 2024-10-02 01:20:00+0900
categories: [Study, Algorithm & Data Structure]
tags: [Algorithm & Data Structure, Union-Find, Disjoint Set, Graph]
---
## Union-Find(Disjoint Set)란?  
서로소 집합 자료구조  
여러 개의 서로 중복되지 않는 집합을 관리하고, 집합을 효율적으로 합치거나(Union), 특정 원소가 속한 집합을 찾는(Find) 연산을 수행할 수 있는 자료구조  
주로 그래프 사이클 판별, 최소 신장 트리(MST) 구성과 같은 문제를 해결할 때 사용  

### 시간 복잡도
경로 압축과 랭크 결합을 적용한 경우, 각 연산이 **거의 상수 시간 O(α(N))** 에 수행됨  
여기서 α(N)은 매우 느리게 증가하는 **역 아커만 함수(inverse Ackermann function)** 로, 대부분의 경우 4 이하로 작아져, 사실상 O(1)과 같음  

### 주요 연산

* Find 연산: 특정 원소가 속한 집합의 대표 노드(루트 노드)를 찾습니다.
* Union 연산: 두 개의 집합을 하나의 집합으로 합칩니다.
* Connected 연산: 두 원소가 같은 집합에 속하는지 확인합니다. (주로 Find를 통해 확인)

### 장점

1. 효율적인 연산: Find와 Union 연산이 거의 상수 시간에 수행되므로 대규모 데이터 처리에 적합
2. 간단한 구현: 자료구조가 간단하여 쉽게 구현
3. 다양한 활용: 그래프 문제뿐만 아니라 네트워크 연결, 클러스터링 등 여러 문제에서 사용
   
### 단점

1. 메모리 사용: 각 원소마다 부모와 랭크를 저장하므로, 메모리 사용량이 많을 수 있음
2. 초기화 필요: 초기화 단계에서 모든 원소를 초기화해야 하므로, 작은 입력에서는 단순 구현이 더 나을 수 있음
  
### 사용 예시

1. 그래프 사이클 판별: 그래프에서 사이클이 존재하는지 확인할 때  
2. 최소 신장 트리(MST) 알고리즘: **크루스칼 알고리즘(Kruskal's Algorithm)** 에서 사용  
3. 네트워크 연결 여부: 네트워크 상의 두 노드가 같은 연결된 컴포넌트에 있는지 확인  
  
## Union-Find의 기본 개념
부모 배열을 사용하여 각 원소의 루트를 추적하고, 경로 압축(Path Compression) 및 랭크를 통한 결합(Union by Rank) 기법을 활용하여 연산을 최적화

### 자료구조의 기본 형태
* 부모 배열 (parent): 각 원소가 속한 집합의 루트 노드를 저장  
* 랭크 배열 (rank): 각 집합의 트리 높이를 저장하여, 트리의 균형을 유지하고 성능을 최적화  
  
### Find 연산

원소 x가 속한 집합의 **대표 노드(루트 노드)** 를 반환  
일반적인 방법은 재귀적으로 부모 노드를 찾아가면서 루트 노드를 찾음 
경로 압축(Path Compression) 기법을 사용하여, 모든 노드가 루트 노드를 직접 가리키도록 하여 Find 연산의 효율성을 높임  

```cpp
int find(int parent[], int x) {
    if (parent[x] == x) return x;  // 자기 자신이 부모인 경우 (루트 노드)
    return parent[x] = find(parent, parent[x]);  // 경로 압축: 루트 노드로 부모 설정
}
```

### Union 연산
원소 x가 속한 집합과 y가 속한 집합을 합치는 연산  
두 집합의 루트 노드를 찾고, **랭크(rank)** 를 이용하여 트리의 높이가 낮은 집합을 높은 집합에 연결하여 트리의 균형을 유지하여 최적화


```cpp
void unionSets(int parent[], int rank[], int x, int y) {
    int rootX = find(parent, x);  // x의 루트 노드
    int rootY = find(parent, y);  // y의 루트 노드

    if (rootX != rootY) {  // 두 원소의 루트가 다르면 집합을 합침
        if (rank[rootX] > rank[rootY]) {  // 랭크가 높은 쪽을 부모로 설정
            parent[rootY] = rootX;
        } else if (rank[rootX] < rank[rootY]) {
            parent[rootX] = rootY;
        } else {  // 두 랭크가 같은 경우
            parent[rootY] = rootX;
            rank[rootX]++;  // 랭크 증가
        }
    }
}
```

### Connected 연산
원소 x와 y가 같은 집합에 속해 있는지 확인함  
Find 연산을 사용하여 두 원소의 루트가 같은지 확인

```cpp
bool connected(int parent[], int x, int y) {
    return find(parent, x) == find(parent, y);  // 두 원소의 루트가 같은지 확인
}
```

## Union-Find 자료구조의 최적화 기법

### 경로 압축 (Path Compression)
Find 연산을 수행할 때, 각 노드를 루트 노드에 직접 연결하여 트리의 높이를 줄이는 기법  
경로 압축을 통해 Find 연산의 시간 복잡도는 거의 상수 시간에 수렴함  

```cpp
int find(int parent[], int x) {
    if (parent[x] != x) {  // 루트 노드가 아니면
        parent[x] = find(parent, parent[x]);  // 경로 압축
    }
    return parent[x];
}
```

### 랭크를 통한 결합 (Union by Rank)
Union 연산을 수행할 때, 트리의 **높이(랭크)** 를 유지하여 작은 트리를 큰 트리에 합쳐 트리의 균형을 유지하고, 높이를 줄여 Find 연산의 효율성을 높임  

```cpp
void unionSets(int parent[], int rank[], int x, int y) {
    int rootX = find(parent, x);
    int rootY = find(parent, y);

    if (rank[rootX] > rank[rootY]) {
        parent[rootY] = rootX;
    } else if (rank[rootX] < rank[rootY]) {
        parent[rootX] = rootY;
    } else {
        parent[rootY] = rootX;
        rank[rootX]++;
    }
}
```

## Union-Find의 활용 예제

### 그래프의 사이클 판별

각 간선을 처리하면서 두 정점을 Union 연산으로 합치고, 두 정점이 이미 같은 집합에 속해 있다면 사이클이 존재한다고 판단

```cpp
bool hasCycle(int parent[], int rank[], vector<pair<int, int>>& edges) {
    for (auto& edge : edges) {
        int u = edge.first;
        int v = edge.second;

        if (connected(parent, u, v)) return true;  // 이미 같은 집합에 속해 있으면 사이클 발생
        unionSets(parent, rank, u, v);  // 두 정점을 같은 집합으로 합침
    }
    return false;
}
```

### 크루스칼 알고리즘 (Kruskal's Algorithm)

**최소 신장 트리(MST)** 를 구성하는 알고리즘   
간선의 가중치를 기준으로 오름차순 정렬한 후, Union-Find를 사용하여 사이클을 방지하면서 최소 신장 트리를 구성   

```cpp

int kruskalMST(int V, vector<pair<int, pair<int, int>>>& edges) {
    sort(edges.begin(), edges.end());  // 간선의 가중치를 기준으로 정렬
    int parent[V], rank[V];
    for (int i = 0; i < V; i++) {
        parent[i] = i;
        rank[i] = 0;
    }

    int mst_weight = 0;  // MST 가중치
    for (auto& edge : edges) {
        int weight = edge.first;
        int u = edge.second.first;
        int v = edge.second.second;

        if (!connected(parent, u, v)) {  // 사이클이 발생하지 않으면
            mst_weight += weight;  // 간선을 추가하고
            unionSets(parent, rank, u, v);  // 두 정점을 합침
        }
    }
    return mst_weight;
}
```
