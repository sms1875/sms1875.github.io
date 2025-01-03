---
layout: post
title: "Flood Fill Algorithm"
date: 2024-10-02 05:28:00+0900
categories: [Study, Algorithm & Data Structure]
tags: [Algorithm & Data Structure, Flood Fill, BFS, DFS, Graph]
---
## Flood Fill Algorithm이란?
그래프 탐색의 한 종류로, 특정한 영역을 채우는 알고리즘   
시작 지점에서부터 연결된 모든 셀을 특정한 색상 또는 값으로 변경시킴  
주로 **그래프 탐색(DFS, BFS)**을 이용하여 인접한 모든 셀을 탐색하는 방식으로 구현  
 
 
### 장점

1. DFS 또는 BFS를 이용하여 간단하게 구현할 수 있음
2. 그래프 탐색, 이미지 영역 채우기, 경로 찾기 등 다양한 문제에 적용 가능

### 단점

1. DFS를 사용할 경우, 매우 큰 입력에 대해서는 스택 오버플로우가 발생할 수 있음
2. BFS를 사용할 경우, 큐의 크기가 매우 커질 수 있음.

### 사용 예시
1. 그래픽 프로그램의 영역 채우기: 특정 영역을 선택하고 색을 채울 때 사용
2. 맵 탐색: 미로 찾기, 영역 탐색 등에서 특정 영역을 색칠하거나 경로를 찾을 때
3. 게임 개발: 지형 탐색, 영역 분할, 경로 찾기 등.

## Flood Fill 알고리즘의 구현

### 깊이 우선 탐색 (DFS, Depth First Search)

DFS를 사용한 Flood Fill은 재귀 호출 또는 스택을 사용하여, 시작 지점에서 연결된 모든 셀을 깊이 우선으로 탐색

```cpp

void dfs(vector<vector<int>>& image, int x, int y, int newColor, int originalColor) {
    if (x < 0 || y < 0 || x >= image.size() || y >= image[0].size()) return;  // 범위 벗어남
    if (image[x][y] != originalColor) return;  // 채울 셀이 아니면 종료

    image[x][y] = newColor;  // 현재 셀을 새로운 색으로 변경

    // 상하좌우 탐색
    dfs(image, x - 1, y, newColor, originalColor);  // 위쪽
    dfs(image, x + 1, y, newColor, originalColor);  // 아래쪽
    dfs(image, x, y - 1, newColor, originalColor);  // 왼쪽
    dfs(image, x, y + 1, newColor, originalColor);  // 오른쪽
}

void floodFillDFS(vector<vector<int>>& image, int sr, int sc, int newColor) {
    int originalColor = image[sr][sc];
    if (originalColor != newColor) {
        dfs(image, sr, sc, newColor, originalColor);  // DFS 시작
    }
}

``` 

### 너비 우선 탐색 (BFS, Breadth First Search)

BFS를 사용한 Flood Fill은 **큐(Queue)**를 사용하여 시작점에서 가까운 셀부터 탐색  
BFS는 방문 순서가 균등하여, 특정 조건에서 더 안정적인 탐색 제공  

```cpp
void floodFillBFS(vector<vector<int>>& image, int sr, int sc, int newColor) {
    int originalColor = image[sr][sc];
    if (originalColor == newColor) return;  // 이미 같은 색상으로 채워져 있음

    int n = image.size(), m = image[0].size();
    queue<pair<int, int>> q;
    q.push({sr, sc});  // 시작 지점 큐에 삽입
    image[sr][sc] = newColor;  // 시작 지점 색상 변경

    // 상하좌우 방향 배열
    vector<int> dx = {-1, 1, 0, 0};
    vector<int> dy = {0, 0, -1, 1};

    while (!q.empty()) {
        auto [x, y] = q.front();
        q.pop();

        for (int i = 0; i < 4; i++) {  // 상하좌우 탐색
            int nx = x + dx[i];
            int ny = y + dy[i];

            if (nx >= 0 && ny >= 0 && nx < n && ny < m && image[nx][ny] == originalColor) {
                q.push({nx, ny});  // 다음 지점을 큐에 추가
                image[nx][ny] = newColor;  // 색상 변경
            }
        }
    }
}

```

### 선택 기준  

* DFS: 재귀적인 구현이 간단하며, 깊이 탐색이 필요한 경우 유리함. 그러나 스택 오버플로우가 발생할 수 있으므로 매우 큰 데이터에서는 주의 필요  
* BFS: 너비 우선 탐색을 통해 방문 순서가 균등하게 퍼지며, 메모리 사용량이 더 안정적일 수 있음  

> DFS와 BFS의 차이를 이해하고, 상황에 맞는 탐색 방식을 선택하는 것이 중요  
{: .prompt-tip}

## 응용 문제

### 섬의 개수 찾기 (Number of Islands)
1은 육지, 0은 바다로 표시된 2D 배열이 있을 때, 연결된 모든 육지를 하나의 섬으로 간주하여 총 섬의 개수를 찾음

```cpp
int numIslands(vector<vector<char>>& grid) {
    int n = grid.size(), m = grid[0].size();
    int count = 0;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            if (grid[i][j] == '1') {  // 새로운 섬 발견
                floodFillDFS(grid, i, j, '0');  // 섬을 '0'으로 채워서 방문 표시
                count++;
            }
        }
    }

    return count;
}

void floodFillDFS(vector<vector<char>>& grid, int x, int y, char newChar) {
    if (x < 0 || y < 0 || x >= grid.size() || y >= grid[0].size()) return;  // 범위 벗어남
    if (grid[x][y] != '1') return;  // 육지가 아니면 종료

    grid[x][y] = newChar;  // 현재 육지를 새로운 값으로 채움
    // 상하좌우로 연결된 모든 육지 탐색
    floodFillDFS(grid, x - 1, y, newChar);  // 위
    floodFillDFS(grid, x + 1, y, newChar);  // 아래
    floodFillDFS(grid, x, y - 1, newChar);  // 왼쪽
    floodFillDFS(grid, x, y + 1, newChar);  // 오른쪽
}
``` 

### 영역의 크기 구하기 (Region Size Calculation)

주어진 2D 배열에서 특정 셀을 선택하고, 그 셀과 연결된 모든 셀의 개수를 반환  


```cpp
int getRegionSize(vector<vector<int>>& grid, int sr, int sc) {
    int originalColor = grid[sr][sc];
    return floodFillCount(grid, sr, sc, originalColor, -1);  // 채운 셀은 -1로 표시
}

int floodFillCount(vector<vector<int>>& grid, int x, int y, int originalColor, int newColor) {
    if (x < 0 || y < 0 || x >= grid.size() || y >= grid[0].size()) return 0;
    if (grid[x][y] != originalColor) return 0;

    grid[x][y] = newColor;  // 셀 색상 변경
    int size = 1;  // 현재 셀 포함

    // 상하좌우 방향 탐색으로 크기 계산
    size += floodFillCount(grid, x - 1, y, originalColor, newColor);
    size += floodFillCount(grid, x + 1, y, originalColor, newColor);
    size += floodFillCount(grid, x, y - 1, originalColor, newColor);
    size += floodFillCount(grid, x, y + 1, originalColor, newColor);

    return size;  // 영역의 총 크기 반환
}

```
