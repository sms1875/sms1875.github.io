---
layout: post
title: "재귀 알고리즘(Recursion Algorithm)"
date: 2024-07-26 10:34:00
categories: [Study, Algorithm & Data Structure]
tags: [Algorithm & Data Structure, Recursion]
---
## 재귀(Recursion) 알고리즘이란?
함수가 자기 자신을 호출하여 문제를 해결하는 방법  
큰 문제를 동일한 형태의 더 작은 문제로 분할하고, 더 이상 분할할 수 없을 때(기저 조건에 도달) 결과를 도출하여 문제를 해결  
재귀는 분할 정복(Divide and Conquer) 또는 **백트래킹(Backtracking)** 과 같은 기법을 사용할 때 자주 사용

### 재귀의 구조

* 기저 조건(Base Case): 더 이상 분할할 수 없을 때, 재귀 호출을 종료하고 결과를 반환하는 조건.
* 재귀 호출(Recursive Call): 함수를 자기 자신으로 호출하여 문제를 더 작은 단위로 나누는 부분.

기본적인 재귀 구조

```cpp
void recursiveFunction(int n) {
    if (n <= 0) {  // 기저 조건: n이 0 이하일 때 함수 종료
        return;
    }
    cout << n << " ";
    recursiveFunction(n - 1);  // 재귀 호출: n을 줄여서 자신을 다시 호출
}
```

## 재귀의 활용 예시
### 팩토리얼 (Factorial)

```math  
n! = n * (n-1) * (n-2) * ... * 1
```  

이 식을 재귀적으로 구현하면 (n! = n * (n-1)!)로 표현  

```cpp
int factorial(int n) {
    if (n == 0 || n == 1) return 1;  // 기저 조건: n이 0 또는 1일 때
    return n * factorial(n - 1);  // 재귀 호출: n * (n-1)!
}
```

### 피보나치 수열 (Fibonacci Sequence)

```math
F(n) = F(n-1) + F(n-2), F(0) = 0, F(1) = 1
```

```cpp
int fibonacci(int n) {
    if (n == 0) return 0;  // 기저 조건: F(0) = 0
    if (n == 1) return 1;  // 기저 조건: F(1) = 1
    return fibonacci(n - 1) + fibonacci(n - 2);  // 재귀 호출: F(n-1) + F(n-2)
}
```

### 하노이의 탑 (Tower of Hanoi)
하노이의 탑 문제는 크기가 다른 원판을 한 기둥에서 다른 기둥으로 옮기는 문제임  
한 번에 한 개의 원판만 옮길 수 있으며, 더 큰 원판이 더 작은 원판 위에 올려져서는 안 됨  
재귀를 이용하여 n개의 원판을 옮기기 위해서는 다음과 같은 과정을 수행함 

1. n-1개의 원판을 첫 번째 기둥에서 두 번째 기둥으로 이동
2. 가장 큰 원판을 첫 번째 기둥에서 세 번째 기둥으로 이동
3. n-1개의 원판을 두 번째 기둥에서 세 번째 기둥으로 이동

```cpp
void hanoi(int n, char from, char to, char aux) {
    if (n == 1) {
        cout << "Move disk 1 from " << from << " to " << to << endl;
        return;
    }
    hanoi(n - 1, from, aux, to);  // Step 1: n-1개의 원판을 보조 기둥으로 이동
    cout << "Move disk " << n << " from " << from << " to " << to << endl;  // Step 2: 가장 큰 원판 이동
    hanoi(n - 1, aux, to, from);  // Step 3: n-1개의 원판을 최종 기둥으로 이동
}
```

## 재귀와 반복의 차이점

### 반복문과 재귀의 비교

재귀는 함수 호출을 통해 스택 메모리를 사용하여 값을 저장  
반복문은 고정된 메모리를 사용하며, 메모리 사용 측면에서 더 효율적일 수 있음  

### 반복으로 변환 가능한 재귀

모든 재귀는 반복문으로 변환 가능  
하지만 재귀를 사용하면 코드가 간결해지고 이해하기 쉬워지는 경우가 많음

### 재귀 알고리즘의 장점과 단점

### 장점

* 간결한 코드: 복잡한 문제를 간단하게 표현할 수 있음
* 논리적 표현: 문제의 논리적 구조를 그대로 반영하여 이해하기 쉬움

### 단점

* 스택 오버플로우: 너무 많은 재귀 호출은 스택 메모리를 초과할 수 있음
* 비효율성: 중복된 계산이 발생할 수 있어, 메모이제이션(Memoization) 또는 동적 계획법(DP)과 결합해야 할 때가 많음

### 언제 재귀를 사용할까?

문제가 반복적인 구조를 가지고 있거나, 분할 정복 방식으로 해결할 수 있을 때 사용  
예를 들어, 트리 탐색, 백트래킹, DFS와 같은 알고리즘에 적합  

## 재귀를 사용한 주요 알고리즘

### 이진 탐색 (Binary Search)

재귀를 사용하여 정렬된 배열에서 특정 값을 찾는 방법  
배열을 반으로 나누어, 중간값을 기준으로 탐색을 줄여 나감

```cpp
int binarySearch(int arr[], int l, int r, int x) {
    if (r >= l) {
        int mid = l + (r - l) / 2;

        if (arr[mid] == x) return mid;  // 값이 중간값일 경우
        if (arr[mid] > x) return binarySearch(arr, l, mid - 1, x);  // 왼쪽 탐색
        return binarySearch(arr, mid + 1, r, x);  // 오른쪽 탐색
    }
    return -1;  // 값이 없는 경우
}
```

### 그래프 탐색 (DFS)

깊이 우선 탐색(Depth First Search)은 그래프에서 재귀를 이용하여 깊이 탐색을 수행하는 알고리즘  
노드를 방문하고, 그와 연결된 모든 노드를 방문하는 방식으로 이루어짐 

```cpp
void DFS(int v, vector<int> adj[], bool visited[]) {
    visited[v] = true;  // 현재 노드 방문 표시
    cout << v << " ";  // 방문한 노드 출력

    for (int u : adj[v]) {
        if (!visited[u]) {
            DFS(u, adj, visited);  // 재귀 호출로 다음 노드 방문
        }
    }
}
```

### 백트래킹 (Backtracking)

백트래킹은 모든 가능한 조합을 탐색하여 문제의 해를 구하는 방법  
특정 조건을 만족하지 않으면 탐색을 중지하고, 되돌아가는 방식으로 최적해를 찾음  

**예시 문제: N-Queen**  
N-Queen 문제는 N x N 크기의 체스판에 N개의 퀸을 서로 공격하지 않도록 배치하는 문제

```cpp
bool isSafe(int board[], int row, int col, int n) {
    for (int i = 0; i < row; i++) {
        if (board[i] == col || abs(board[i] - col) == abs(i - row)) {
            return false;  // 같은 열이거나, 대각선에 위치한 경우
        }
    }
    return true;
}

void nQueen(int board[], int row, int n) {
    if (row == n) {
        for (int i = 0; i < n; i++) cout << board[i] << " ";
        cout << endl;
        return;
    }

    for (int col = 0; col < n; col++) {
        if (isSafe(board, row, col, n)) {
            board[row] = col;  // 퀸 배치
            nQueen(board, row + 1, n);  // 다음 행으로 이동
        }
    }
}
```
