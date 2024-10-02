---
layout: post
title: "정렬 알고리즘(Sort Algorithm)"
date: 2024-07-25 15:20:00
categories: [Study, Algorithm]
tags: [Sort]
---
## 정렬 알고리즘이란?

데이터를 특정 기준에 따라 정렬하는 방법  
크게 비교 기반 정렬과 비교하지 않는 정렬로 나뉘며, 각 알고리즘은 데이터의 크기, 데이터의 특성, 메모리 제약 등에 따라 다른 성능을 보임

* 비교 기반 정렬 (Comparison-based Sort): 각 원소를 비교하여 정렬 순서를 결정하는 방법
* 비교하지 않는 정렬 (Non-Comparison-based Sort): 데이터를 직접 비교하지 않고 정렬하는 방법으로, 특정 상황에서 더 효율적인 경우가 있음

### 정렬 알고리즘의 시간 복잡도

* $$O(N²)$: 작은 데이터에 적합. 비효율적이지만 구현이 간단.
  * 버블 정렬, 선택 정렬, 삽입 정렬
* $$O(N log N)$$: 대부분의 경우 효율적. 대규모 데이터에서 많이 사용.
  * 병합 정렬, 퀵 정렬, 힙 정렬
* $$O(N)$$: 특정 조건이나 특수한 경우에만 사용.
  * 계수 정렬, 기수 정렬, 버킷 정렬

## O(N²) 정렬 알고리즘

### 버블 정렬 (Bubble Sort)

인접한 두 원소를 반복적으로 비교하여 정렬 

```cpp
void bubbleSort(int arr[], int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - 1 - i; j++) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);  // 두 원소를 교환
            }
        }
    }
}
```

### 선택 정렬 (Selection Sort)
매번 최소값을 찾아 첫 번째 원소와 교환하는 방식으로 정렬

```cpp
void selectionSort(int arr[], int n) {
    for (int i = 0; i < n - 1; i++) {
        int min_idx = i;
        for (int j = i + 1; j < n; j++) {
            if (arr[j] < arr[min_idx]) {
                min_idx = j;  // 최소값의 인덱스 업데이트
            }
        }
        swap(arr[min_idx], arr[i]);  // 최소값과 현재 위치의 원소 교환
    }
}
```

### 삽입 정렬 (Insertion Sort)
정렬된 배열의 끝에서부터 새로운 값을 삽입할 위치를 찾아 정렬  
작은 데이터에 대해서는 효율적이며, 거의 정렬된 데이터에는 O(N) 성능을 보이기도 함

```cpp
void insertionSort(int arr[], int n) {
    for (int i = 1; i < n; i++) {
        int key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];  // 값을 오른쪽으로 이동
            j--;
        }
        arr[j + 1] = key;  // 삽입할 위치에 key 값 넣기
    }
}
```

## O(N log N) 정렬 알고리즘

### 퀵 정렬 (Quick Sort)

분할 정복 알고리즘으로, 기준값(pivot)을 설정하고 좌우로 분할하여 정렬  
평균 시간 복잡도는 O(N log N)이며, 최악의 경우 O(N²)

```cpp
int partition(int arr[], int low, int high) {
    int pivot = arr[high];  // 기준값 설정
    int i = (low - 1);  // 작은 원소의 인덱스

    for (int j = low; j <= high - 1; j++) {
        if (arr[j] < pivot) {
            i++;
            swap(arr[i], arr[j]);  // 기준값보다 작은 값을 앞으로 이동
        }
    }
    swap(arr[i + 1], arr[high]);  // 기준값을 중앙으로 이동
    return (i + 1);
}

void quickSort(int arr[], int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quickSort(arr, low, pi - 1);  // 왼쪽 부분 배열 정렬
        quickSort(arr, pi + 1, high);  // 오른쪽 부분 배열 정렬
    }
}
```

### 병합 정렬 (Merge Sort)

배열을 반으로 나누고, 정렬된 두 배열을 병합하여 하나의 정렬된 배열로 만드는 방식  
안정적인 정렬 방법으로, 최악의 경우에도 O(N log N)을 유지함

```cpp
void merge(int arr[], int l, int m, int r) {
    int n1 = m - l + 1;
    int n2 = r - m;

    int L[n1], R[n2];  // 두 개의 임시 배열 생성

    for (int i = 0; i < n1; i++) L[i] = arr[l + i];
    for (int j = 0; j < n2; j++) R[j] = arr[m + 1 + j];

    int i = 0, j = 0, k = l;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) arr[k++] = L[i++];
        else arr[k++] = R[j++];
    }

    while (i < n1) arr[k++] = L[i++];  // 남은 값 복사
    while (j < n2) arr[k++] = R[j++];  // 남은 값 복사
}

void mergeSort(int arr[], int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;
        mergeSort(arr, l, m);  // 왼쪽 부분 배열 정렬
        mergeSort(arr, m + 1, r);  // 오른쪽 부분 배열 정렬
        merge(arr, l, m, r);  // 두 부분 배열을 병합
    }
}
```

### 힙 정렬 (Heap Sort)
힙 정렬은 최대/최소 힙을 사용하여 정렬  
우선순위 큐를 사용하여 O(N log N) 시간에 정렬이 가능  

```cpp
void heapify(int arr[], int n, int i) {
    int largest = i;  // 루트
    int left = 2 * i + 1;
    int right = 2 * i + 2;

    if (left < n && arr[left] > arr[largest]) largest = left;
    if (right < n && arr[right] > arr[largest]) largest = right;

    if (largest != i) {
        swap(arr[i], arr[largest]);  // 루트와 교환
        heapify(arr, n, largest);  // 하위 서브트리 재귀적으로 힙 정렬
    }
}

void heapSort(int arr[], int n) {
    for (int i = n / 2 - 1; i >= 0; i--) heapify(arr, n, i);  // 힙 구성
    for (int i = n - 1; i > 0; i--) {
        swap(arr[0], arr[i]);  // 루트와 마지막 원소 교환
        heapify(arr, i, 0);  // 남은 원소 재정렬
    }
}
```

## $$O(N)$$ 정렬 알고리즘

### 계수 정렬 (Counting Sort)

정수 배열의 값들을 기준으로 빈도수를 계산하여 정렬  
데이터가 정수이고, 값의 범위가 제한된 경우 매우 효율적  

```cpp
void countingSort(int arr[], int n, int max_val) {
    int* count = new int[max_val + 1]{0};  // 빈도수 배열 초기화
    int* output = new int[n];

    // 1. 각 값의 빈도수 카운트
    for (int i = 0; i < n; i++) {
        count[arr[i]]++;
    }

    // 2. 빈도수 누적합 계산
    for (int i = 1; i <= max_val; i++) {
        count[i] += count[i - 1];
    }

    // 3. 빈도수를 기반으로 각 값을 올바른 위치에 삽입
    for (int i = n - 1; i >= 0; i--) {
        output[--count[arr[i]]] = arr[i];
    }

    // 4. 정렬된 배열을 원본 배열에 복사
    for (int i = 0; i < n; i++) {
        arr[i] = output[i];
    }

    delete[] count;
    delete[] output;
}
```

### 기수 정렬 (Radix Sort)

가장 낮은 자릿수부터 높은 자릿수까지 정렬하여 전체를 정렬  
각 자릿수의 정렬에는 계수 정렬을 사용하므로 시간 복잡도는 O(d * N)

```cpp
void countingSortForRadix(int arr[], int n, int exp) {
    int output[n];
    int count[10] = {0};  // 자릿수는 0~9까지이므로 크기 10

    // 1. 현재 자릿수를 기준으로 각 값의 빈도수 계산
    for (int i = 0; i < n; i++) {
        count[(arr[i] / exp) % 10]++;
    }

    // 2. 빈도수 누적합 계산
    for (int i = 1; i < 10; i++) {
        count[i] += count[i - 1];
    }

    // 3. 현재 자릿수를 기준으로 정렬된 배열 생성
    for (int i = n - 1; i >= 0; i--) {
        output[--count[(arr[i] / exp) % 10]] = arr[i];
    }

    // 4. 정렬된 배열을 원본 배열에 복사
    for (int i = 0; i < n; i++) {
        arr[i] = output[i];
    }
}

void radixSort(int arr[], int n) {
    int max_val = *max_element(arr, arr + n);  // 최대값 탐색

    // 각 자릿수를 기준으로 계수 정렬 수행
    for (int exp = 1; max_val / exp > 0; exp *= 10) {
        countingSortForRadix(arr, n, exp);
    }
}
```

### 버킷 정렬 (Bucket Sort)

데이터를 여러 개의 버킷으로 나누고, 각 버킷을 개별적으로 정렬하여 최종적으로 합치는 방식  
정렬할 데이터가 균등하게 분포된 경우 효율적으로 동작
 
```cpp
void bucketSort(float arr[], int n) {
    vector<float> bucket[n];  // n개의 버킷 생성

    // 1. 각 원소를 버킷에 분배
    for (int i = 0; i < n; i++) {
        int idx = n * arr[i];  // 값에 따라 버킷의 인덱스 결정
        bucket[idx].push_back(arr[i]);
    }

    // 2. 각 버킷을 개별적으로 정렬
    for (int i = 0; i < n; i++) {
        sort(bucket[i].begin(), bucket[i].end());
    }

    // 3. 정렬된 버킷을 합침
    int index = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < bucket[i].size(); j++) {
            arr[index++] = bucket[i][j];
        }
    }
}

```

## 특수한 정렬 알고리즘

### 트리 정렬 (Tree Sort)

이진 탐색 트리를 이용하여 정렬하는 방법  
트리에 데이터를 삽입하고, 중위 순회를 통해 정렬  
시간 복잡도는 O(N log N)이며, 트리의 높이에 따라 성능이 달라진다

```cpp
struct Node {
    int data;
    Node* left, * right;
    Node(int val) : data(val), left(NULL), right(NULL) {}
};

Node* insert(Node* root, int data) {
    if (!root) return new Node(data);
    if (data < root->data) root->left = insert(root->left, data);
    else root->right = insert(root->right, data);
    return root;
}

void inorder(Node* root, vector<int>& sorted) {
    if (root) {
        inorder(root->left, sorted);
        sorted.push_back(root->data);
        inorder(root->right, sorted);
    }
}

void treeSort(int arr[], int n) {
    Node* root = NULL;
    for (int i = 0; i < n; i++) {
        root = insert(root, arr[i]);  // 트리에 삽입
    }

    vector<int> sorted;
    inorder(root, sorted);  // 중위 순회로 정렬된 데이터 추출

    for (int i = 0; i < n; i++) {
        arr[i] = sorted[i];  // 정렬된 데이터를 배열에 복사
    }
}
```

### 위상 정렬 (Topological Sort)

위상 정렬은 DAG(Directed Acyclic Graph)에서 각 정점의 순서를 정렬하는 방법  
보통 그래프 알고리즘에서 작업 순서 결정 등에 사용  
진입 차수가 0인 정점을 찾아 순서대로 처리하며, 시간 복잡도는 O(V + E)

``` cpp
void topologicalSort(vector<int> adj[], int V) {
    vector<int> indegree(V, 0);  // 모든 정점의 진입 차수 저장

    // 각 정점의 진입 차수 계산
    for (int i = 0; i < V; i++) {
        for (int u : adj[i]) {
            indegree[u]++;
        }
    }

    queue<int> q;
    for (int i = 0; i < V; i++) {
        if (indegree[i] == 0) q.push(i);  // 진입 차수가 0인 정점 삽입
    }

    // 위상 정렬 수행
    while (!q.empty()) {
        int node = q.front();
        q.pop();
        cout << node << " ";  // 출력

        // 인접한 정점의 진입 차수 감소
        for (int u : adj[node]) {
            if (--indegree[u] == 0) {
                q.push(u);  // 진입 차수가 0이 된 정점 삽입
            }
        }
    }
}
```
