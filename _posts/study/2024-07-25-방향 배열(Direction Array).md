---
layout: post
title: "방향 배열(Direction Array)"
date: 2024-07-25 14:45:00
categories: [Study, Algorithm]
tags: [Direction Array]
---
## 방향 배열(Direction Array)이란?  
2차원 평면에서 상하좌우 및 대각선 이동을 쉽게 구현하기 위한 배열  
이를 통해 특정 좌표의 상하좌우 또는 대각선 방향으로의 이동을 간편하게 처리 가능  
게임 개발, 그래프 탐색, 경로 찾기 알고리즘 등에서 자주 사용  
  
```cpp
#include <iostream>
using namespace std;

int dy[4] = {-1, 1, 0, 0};  // 상, 하, 좌, 우 y 변화량
int dx[4] = {0, 0, -1, 1};  // 상, 하, 좌, 우 x 변화량

struct Point {
    int y;
    int x;
};

int map[4][4] = {
    {1, 3, 7, 2},
    {2, 2, 6, 1},
    {1, 4, 5, 1},
    {1, 1, 2, 1}
};

int main() {
    Point sp = {1, 1};  // 시작 좌표 (1, 1)
    int sum = 0;
    
    // 상하좌우 탐색
    for (int i = 0; i < 4; i++) {
        int ny = sp.y + dy[i];
        int nx = sp.x + dx[i];

        // 맵을 벗어나는 경우를 체크
        if (ny < 0 || nx < 0 || ny >= 4 || nx >= 4) continue;  
        sum += map[ny][nx];  // 상하좌우의 값 합산
    }

    cout << "상하좌우 합: " << sum << "\n";
    return 0;
}
```

### 대각선 이동  

대각선 방향 배열도 상하좌우와 비슷하게 정의 가능  

```cpp
#include <iostream>
using namespace std;

int dy[4] = {-1, -1, 1, 1};  // 좌상, 우상, 좌하, 우하 y 변화량
int dx[4] = {-1, 1, -1, 1};  // 좌상, 우상, 좌하, 우하 x 변화량

struct Point {
    int y;
    int x;
};

int map[4][4] = {
    {1, 3, 7, 2},
    {2, 2, 6, 1},
    {1, 4, 5, 1},
    {1, 1, 2, 1}
};

int main() {
    Point sp = {1, 1};  // 시작 좌표 (1, 1)
    int sum = 0;
    
    // 대각선 탐색
    for (int i = 0; i < 4; i++) {
        int ny = sp.y + dy[i];
        int nx = sp.x + dx[i];

        // 맵을 벗어나는 경우를 체크
        if (ny < 0 || nx < 0 || ny >= 4 || nx >= 4) continue;
        sum += map[ny][nx];  // 대각선 방향의 값 합산
    }

    cout << "대각선 합: " << sum << "\n";
    return 0;
}
```

이를 응용해 2칸씩 이동, L자 이동, 나이트 이동 등 구현이 가능
