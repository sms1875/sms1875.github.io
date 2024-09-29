---
title: "# Silver II DFS와 BFS: 1260 "
date: 2024-09-29
categories: [ "coding test" ]
tags: [ "#", "coding test" ]
---

# [Silver II] DFS와 BFS - 1260 

[문제 링크](https://www.acmicpc.net/problem/1260) 

### 성능 요약

메모리: 2296 KB, 시간: 8 ms

### 분류

그래프 이론, 그래프 탐색, 너비 우선 탐색, 깊이 우선 탐색

### 제출 일자

2024년 7월 24일 20:22:32

### 문제 설명

<p>그래프를 DFS로 탐색한 결과와 BFS로 탐색한 결과를 출력하는 프로그램을 작성하시오. 단, 방문할 수 있는 정점이 여러 개인 경우에는 정점 번호가 작은 것을 먼저 방문하고, 더 이상 방문할 수 있는 점이 없는 경우 종료한다. 정점 번호는 1번부터 N번까지이다.</p>

### 입력 

 <p>첫째 줄에 정점의 개수 N(1 ≤ N ≤ 1,000), 간선의 개수 M(1 ≤ M ≤ 10,000), 탐색을 시작할 정점의 번호 V가 주어진다. 다음 M개의 줄에는 간선이 연결하는 두 정점의 번호가 주어진다. 어떤 두 정점 사이에 여러 개의 간선이 있을 수 있다. 입력으로 주어지는 간선은 양방향이다.</p>

### 출력 

 <p>첫째 줄에 DFS를 수행한 결과를, 그 다음 줄에는 BFS를 수행한 결과를 출력한다. V부터 방문된 점을 순서대로 출력하면 된다.</p>


### 소스 코드 (cc)
```cc
#include <iostream>
#include <stack>
#include <queue>
#include <unordered_map>
#include <algorithm>

using namespace std;

unordered_map<int, vector<int>> map;
int N, M, V;

void bfs()
{
  unordered_map<int, bool> visited;
  queue<int> q;

  q.push(V);
  visited[V] = true;

  while (!q.empty())
  {
    int current = q.front();
    q.pop();
    for (int next : map[current])
    {
      if (!visited[next])
      {
        q.push(next);
        visited[next] = true;
      }
    }
    cout << current << " ";
  }
}

void dfs()
{
  unordered_map<int, bool> visited;
  stack<int> s;

  s.push(V);

  while (!s.empty())
  {
    int current = s.top();
    s.pop();

    if (!visited[current])
    {
      visited[current] = true;

      // 정점 번호가 작은 것을 먼저 방문하기 위해 스택에 큰 순서부터 넣음
      for (auto it = map[current].rbegin(); it != map[current].rend(); ++it)
      {
        if (!visited[*it])
        {
          s.push(*it);
        }
      }

      cout << current << " ";
    }
  }
}

void input()
{
  cin >> N >> M >> V;
  for (int i = 0; i < M; i++)
  {
    int a, b;
    cin >> a >> b;
    map[a].push_back(b);
    map[b].push_back(a);
  }

  for (auto &m : map)
  {
    sort(m.second.begin(), m.second.end());
  }
}

int main()
{

  input();
  dfs();
  cout << "
";
  bfs();

  return 0;
}
```
