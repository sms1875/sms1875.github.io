---
layout: post
title: "[Gold V] 선수과목 (Prerequisite) - 14567"
date: 2024-09-10 14:57:14
categories: [Coding Test, Baekjoon]
tags: [방향 비순환 그래프, 다이나믹 프로그래밍, 그래프 이론, 위상 정렬,cpp]
---

### 문제 링크

[문제 링크](https://www.acmicpc.net/problem/14567)

### 성능 요약

메모리: 4936 KB, 시간: 68 ms

### 문제 설명

<p>올해 Z대학 컴퓨터공학부에 새로 입학한 민욱이는 학부에 개설된 모든 전공과목을 듣고 졸업하려는 원대한 목표를 세웠다. 어떤 과목들은 선수과목이 있어 해당되는 모든 과목을 먼저 이수해야만 해당 과목을 이수할 수 있게 되어 있다. 공학인증을 포기할 수 없는 불쌍한 민욱이는 선수과목 조건을 반드시 지켜야만 한다. 민욱이는 선수과목 조건을 지킬 경우 각각의 전공과목을 언제 이수할 수 있는지 궁금해졌다. 계산을 편리하게 하기 위해 아래와 같이 조건을 간소화하여 계산하기로 하였다.</p>

<ol>
	<li>한 학기에 들을 수 있는 과목 수에는 제한이 없다.</li>
	<li>모든 과목은 매 학기 항상 개설된다.</li>
</ol>

<p>모든 과목에 대해 각 과목을 이수하려면 최소 몇 학기가 걸리는지 계산하는 프로그램을 작성하여라.</p>

### 입력

 <p>첫 번째 줄에 과목의 수 N(1 ≤ N ≤ 1000)과 선수 조건의 수 M(0 ≤ M ≤ 500000)이 주어진다. 선수과목 조건은 M개의 줄에 걸쳐 한 줄에 정수 A B 형태로 주어진다. A번 과목이 B번 과목의 선수과목이다. A < B인 입력만 주어진다. (1 ≤ A < B ≤ N)</p>

### 출력

 <p>1번 과목부터 N번 과목까지 차례대로 최소 몇 학기에 이수할 수 있는지를 한 줄에 공백으로 구분하여 출력한다.</p>

### 코드

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

void solve(const int &cur, vector<vector<int>> &prerequisites, vector<int> &dp) {
	if (prerequisites[cur].size() == 0) {
		dp[cur] = 1;
	}
	else {
		int maxV = 0;
		for (int i = 1; i <= prerequisites[cur].size(); i++)
		{
			int prerequisite = prerequisites[cur][i - 1];
			if (dp[prerequisite] == -1) solve(prerequisites[cur][i], prerequisites, dp);
			maxV = max(maxV, dp[prerequisite] + 1);
		}
		dp[cur] = maxV;
	}
}

int main() {
    std::ios::sync_with_stdio(0);
	std::cin.tie(0); std::cout.tie(0);
    
	int N, M;
	cin >> N >> M;

	vector<vector<int>> prerequisites(N+1);
	vector<int> dp(N+1, -1);

	for (int i = 1; i <= M; i++)
	{
		int A, B;
		cin >> A >> B;
		prerequisites[B].push_back(A);
	}

	for (int i = 1; i <= N ; i++)
	{
		solve(i, prerequisites, dp);
	}

	for (int i = 1; i <= N; i++)
	{
		cout << dp[i] << " ";
	}

	return 0;
}

```
