---
title: "#_CodingTestStudy"
date: 2024-09-29
categories: [ "coding test" ]
tags: [ "#", "coding test" ]
---

# CodingTestStudy
### 코딩테스트 문제 풀이기록
BaekjoonHub auto cummit <br>
[BaekjoonHub](https://github.com/BaekjoonHub/BaekjoonHub)

# Site Link
### [https://solved.ac](https://solved.ac) <br>
### [https://swexpertacademy.com](https://swexpertacademy.com) <br>
### [https://programmers.co.kr/](https://programmers.co.kr/) <br>


### 소스 코드 (cpp)
```cpp
#include<iostream>

using namespace std;

int main(int argc, char** argv)
{
	int test_case;
	int T;
	cin>>T;
	for(test_case = 1; test_case <= T; ++test_case)
	{
        int sum=0;
        int tmp=0;
        for(int i=0;i<10;i++){
            cin >> tmp;
            if(tmp%2==1) sum += tmp;
        }
        cout << "#"<<test_case<<" "<<sum<<"
";
	}
	return 0;//정상종료시 반드시 0을 리턴해야합니다.
}
```

### 소스 코드 (cpp)
```cpp
#include<iostream>
#include<cstring>
#include <algorithm>
#include<vector>
using namespace std;

int arr[21][21];
int N;
int res;
int st_x, st_y;
bool visited[101];

int dx[4] = { 1 , -1, -1, 1 };
int dy[4] = { 1, 1, -1, -1 };


void dfs(int y, int x, int dir) {

	if (dir == 4) { 
		//도착 확인
		if (st_x == x && st_y == y) {
			int sum = 0;
			for (int i = 0; i < 101; i++)
			{
				if (visited[i]) {
					sum++;
				}
			}
			res = max(sum, res);
		}
		return; 
	}

	for (int i = 1; i < N; i++)
	{
		int nx = dx[dir]*i + x;
		int ny = dy[dir]*i + y;

		if (nx < 0 || ny < 0 || nx >= N || ny >= N) return;
	
		// 사이 경로 방문 처리
		for (int j = 1; j <= i; j++)
		{
			int nx2 = dx[dir] * j + x;
			int ny2 = dy[dir] * j + y;
			// 경로 사이에 중복되는 디저트 종류 존재할 경우
			if (visited[arr[ny2][nx2]]) {
				for (int k = 0; k < j; k++)
				{
					int nx3 = dx[dir] * k + x;
					int ny3 = dy[dir] * k + y;
					visited[arr[ny3][nx3]] = false;
				}
				return;
			}
			visited[arr[ny2][nx2]] = true;
		}

		dfs(ny, nx, dir + 1);

		for (int j = 1; j <= i; j++)
		{
			int nx2 = dx[dir] * j + x;
			int ny2 = dy[dir] * j + y;
			visited[arr[ny2][nx2]] = false;
		}
	}
}

int main(int argc, char** argv)
{
	int test_case;
	int T;

	cin >> T;
	for (test_case = 1; test_case <= T; ++test_case)
	{
		res = -1;
		memset(arr, 0, sizeof(arr));

		cin >> N;
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				cin >> arr[i][j];
			}
		}

		// 우하 -> 좌하 -> 좌상 -> 우상
		for (int i = 0; i < N-2; i++)
		{
			for (int j = 1; j < N-1; j++)
			{
				memset(visited, 0, sizeof(visited));
				st_y = i; st_x = j; 
				dfs(st_y, st_x, 0);
			}
		}

		cout << "#" << test_case << " "<< res << "
";
	}
	return 0;//정상종료시 반드시 0을 리턴해야합니다.
}
```

### 소스 코드 (cpp)
```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace std;

struct BC {
	int x, y, c, p; // 위치 (x, y), 충전 범위 c, 성능 p
};

vector<int> movements_A, movements_B;
vector<BC> batteryChargers;
vector<vector<int>> map;
int M, A;

//(0: 이동 안함, 1: 상, 2: 우, 3: 하, 4: 좌)
int dx[] = { 0, 0, 1, 0, -1 };
int dy[] = { 0, -1, 0, 1, 0 };

void initialize_map() {
	// 맵 초기화
	map.clear();
	map.resize(11, vector<int>(11, 0));

	// 각 BC의 범위를 비트마스크로 맵에 저장
	for (int i = 0; i < A; ++i) {
		int x = batteryChargers[i].x;
		int y = batteryChargers[i].y;
		int c = batteryChargers[i].c;
		int bitmask = (1 << i); // BC i의 비트마스크

		for (int dx = -c; dx <= c; ++dx) {
			for (int dy = -c; dy <= c; ++dy) {
				int nx = x + dx;
				int ny = y + dy;
				if (nx >= 1 && nx <= 10 && ny >= 1 && ny <= 10) {
					if (abs(x - nx) + abs(y - ny) <= c) {
						map[nx][ny] |= bitmask; // 비트마스크 추가
					}
				}
			}
		}
	}
}

vector<int> get_charging_options(int x, int y) {
	vector<int> options;
	int bitmask = map[x][y];

	for (int i = 0; i < batteryChargers.size(); ++i) {
		if (bitmask & (1 << i)) {
			options.push_back(i);
		}
	}

	return options;
}

int simulate() {
	// 사용자 초기 위치
	pair<int, int> pos_A = { 1, 1 };
	pair<int, int> pos_B = { 10, 10 };

	int total_charge_A = 0;
	int total_charge_B = 0;

	// 매초마다 위치 갱신 및 충전 계산
	for (int t = 0; t <= M; t++) {  // 0초부터 M초까지
		// 현재 위치에서 가능한 충전기 리스트 구하기
		vector<int> options_A = get_charging_options(pos_A.first, pos_A.second);
		vector<int> options_B = get_charging_options(pos_B.first, pos_B.second);

		int max_charge = 0;

		if (!options_A.empty() && !options_B.empty()) {  // 두 사용자 모두 충전기가 있을 때
			for (int i : options_A) {
				for (int j : options_B) {
					int charge;
					if (i == j) {  // 같은 충전기일 경우
						charge = batteryChargers[i].p;
					}
					else {  // 다른 충전기를 선택한 경우 각각 충전 가능
						charge = batteryChargers[i].p + batteryChargers[j].p;
					}
					max_charge = max(max_charge, charge);  // 최댓값 갱신
				}
			}
		}
		else if (!options_A.empty()) {  // 사용자 A만 충전 가능한 경우
			int best_A = 0;
			for (int i : options_A) {
				best_A = max(best_A, batteryChargers[i].p);
			}
			max_charge = best_A;
		}
		else if (!options_B.empty()) {  // 사용자 B만 충전 가능한 경우
			int best_B = 0;
			for (int i : options_B) {
				best_B = max(best_B, batteryChargers[i].p);
			}
			max_charge = best_B;
		}

		total_charge_A += max_charge;

		// 다음 초로 이동 (0초는 이동하지 않음)
		if (t < M) {
			pos_A.first += dx[movements_A[t]];
			pos_A.second += dy[movements_A[t]];
			pos_B.first += dx[movements_B[t]];
			pos_B.second += dy[movements_B[t]];
		}
	}

	return total_charge_A;
}

int main() {
	int T;  // 테스트 케이스 개수
	cin >> T;

	for (int t = 1; t <= T; t++) {
		cin >> M >> A;

		movements_A.resize(M);
		movements_B.resize(M);
		for (int i = 0; i < M; i++) cin >> movements_A[i];
		for (int i = 0; i < M; i++) cin >> movements_B[i];

		batteryChargers.resize(A);
		for (int i = 0; i < A; i++) {
			cin >> batteryChargers[i].x >> batteryChargers[i].y >> batteryChargers[i].c >> batteryChargers[i].p;
		}

		// 맵 초기화 및 비트마스크 설정
		initialize_map();

		// 시뮬레이션 수행
		int result = simulate();

		// 결과 출력
		cout << "#" << t << " " << result << "
";
	}

	return 0;
}
```

### 소스 코드 (cpp)
```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <queue>
using namespace std;


struct Point {
	int y, x;
	int day;
};

int N, M;
int R, C;
int L;
vector<vector<int>> arr;

int cnt = 0;

vector<pair<int, int>> pipeDir[8] = {
	{},
	{{0,-1},{-1,0},{0,1},{1,0}},
	{{-1,0},{1,0}},
	{{0,-1},{0,1}},
	{{-1,0},{0,1}},
	{{0,1},{1,0}},
	{{0,-1},{1,0}},
	{{0,-1},{-1,0}},
};

bool canMove(int curPipeType, int nextDir , int nextPipeType) {
	if (nextPipeType == 1)
		return true;
	else if (curPipeType == 1) {
		if (nextDir == 0)
			return (nextPipeType == 3 || nextPipeType == 4 || nextPipeType == 5) ? true : false;
		else if (nextDir == 1)
			return (nextPipeType == 2 || nextPipeType == 5 || nextPipeType == 6) ? true : false;
		else if (nextDir == 2)
			return (nextPipeType == 3 || nextPipeType == 6 || nextPipeType == 7) ? true : false;
		else
			return (nextPipeType == 2 || nextPipeType == 4 || nextPipeType == 7) ? true : false;
	}
	else if (curPipeType == 2) {
		if (nextDir == 0)
			return (nextPipeType == 2 || nextPipeType == 5 || nextPipeType == 6) ? true : false;
		else
			return (nextPipeType == 2 || nextPipeType == 4 || nextPipeType == 7) ? true : false;
	}
	else if (curPipeType == 3) {
		if (nextDir == 0)
			return (nextPipeType == 3 || nextPipeType == 4 || nextPipeType == 5) ? true : false;
		else
			return (nextPipeType == 3 || nextPipeType == 6 || nextPipeType == 7) ? true : false;
	}
	else if (curPipeType == 4) {
		if (nextDir == 0)
			return (nextPipeType == 2 || nextPipeType == 5 || nextPipeType == 6) ? true : false;
		else
			return (nextPipeType == 3 || nextPipeType == 6 || nextPipeType == 7) ? true : false;
	}
	else if (curPipeType == 5) {
		if (nextDir == 0)
			return (nextPipeType == 3 || nextPipeType == 6 || nextPipeType == 7) ? true : false;
		else
			return (nextPipeType == 2 || nextPipeType == 4 || nextPipeType == 7) ? true : false;
	}
	else if (curPipeType == 6) {
		if (nextDir == 0)
			return (nextPipeType == 3 || nextPipeType == 4 || nextPipeType == 5) ? true : false;
		else
			return (nextPipeType == 2 || nextPipeType == 4 || nextPipeType == 7) ? true : false;
	}
	else if (curPipeType == 7) {
		if (nextDir == 0)
			return (nextPipeType == 3 || nextPipeType == 4 || nextPipeType == 5) ? true : false;
		else
			return (nextPipeType == 2 || nextPipeType == 5 || nextPipeType == 6) ? true : false;
	}
}

void bfs() {
	vector<vector<bool>> visited(N, vector<bool>(M, false));
	queue<Point> q;
	q.push({ R,C,1 });
	while (!q.empty()) {

		int cur_y = q.front().y;
		int cur_x = q.front().x;
		int cur_day = q.front().day;
		q.pop();

		if (cur_day > L) continue; //날짜가 넘으면 통과

		int curPipeType = arr[cur_y][cur_x];
		visited[cur_y][cur_x] = true; // 방문한 곳 기록

		for (int nextDir = 0; nextDir < pipeDir[curPipeType].size(); nextDir++)
		{
			int ny = pipeDir[curPipeType][nextDir].first + cur_y;
			int nx = pipeDir[curPipeType][nextDir].second + cur_x;

			if (ny < 0 || nx < 0 || ny >= N || nx >= M) continue; // 범위 체크
			if (visited[ny][nx]) continue; // 방문했으면 통과
			if (arr[ny][nx] == 0) continue; // 파이프 없으면 통과 
			if (!canMove(curPipeType, nextDir, arr[ny][nx])) continue; // 다음 파이프에 갈 수 없으면 통과

			q.push({ ny,nx,cur_day + 1 });
		}
	}

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < M; j++)
		{
			if (visited[i][j]) cnt++; // 방문한 곳 카운트
		}
	}
}

int main() {
	int T;
	cin >> T;
	for (int tc = 1; tc <= T; tc++)
	{
		cnt = 0;
		cin >> N >> M;
		cin >> R >> C >> L;

		arr.clear(); arr.resize(N, vector<int>(M));

		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < M; j++)
			{
				cin >> arr[i][j];
			}
		}

		bfs();

		cout << "#" << tc << " " << cnt << "
";
	}
	return 0;
}
```

### 소스 코드 (cpp)
```cpp
#include <iostream>
#include <vector>
#include <cstring>

using namespace std;

int T, tc;
int N;
int diff;
int myOperator[4];
int nums[13];

int usingOp;
const int ADD = 0;
const int SUB = 1;
const int MUL = 2;
const int DIV = 3;

int maxVal = -21e8;
int minVal = 21e8;

void init()
{
	memset(myOperator, -1, sizeof(myOperator));
	memset(nums, 0, sizeof(nums));
}

void input()
{
	cin >> N;
	// 연산자 input
	for (int i = 0; i < 4; i++)
	{
		cin >> myOperator[i];
	}
	// 피 연산자 input
	for (int i = 0; i < N; i++)
	{
		cin >> nums[i];
	}
}

void DFS(int lev, int val)
{
	if (lev == N)
	{
		if (maxVal < val)
		{
			maxVal = val;
		}
		if (minVal > val)
		{
			minVal = val;
		}
		diff = maxVal - minVal;
		return;
	}

	for (int i = 0; i < 4; i++)
	{
		if (myOperator[i] == 0) continue;
		myOperator[i]--;
		usingOp = i;
		switch (usingOp)
		{
		case ADD:
			DFS(lev + 1, val + nums[lev]);
			break;
		case SUB:
			DFS(lev + 1, val - nums[lev]);
			break;
		case MUL:
			DFS(lev + 1, val * nums[lev]);
			break;
		case DIV:
			DFS(lev + 1, val / nums[lev]);
			break;
		}
		myOperator[i]++;
	}
}

void solve()
{
	maxVal = -21e8;
	minVal = 21e8;
	DFS(1, nums[0]);
}

void output()
{
	cout << "#" << tc << " " << diff << "
";
}
int main()
{
	cin >> T;
	for (tc = 1; tc <= T; tc++)
	{
		init();
		input();
		solve();
		output();
	}

	return 0;
}
```

### 소스 코드 (cpp)
```cpp
#include <iostream>
#include <algorithm>

using namespace std;

int main(int argc, char** argv) {
	int test_case;
	int T;
	cin >> T;
	for (test_case = 1; test_case <= T; ++test_case) {
		int oneDay, oneMonth, threeMonth, oneYear;
		int month[12];

		cin >> oneDay >> oneMonth >> threeMonth >> oneYear;
		for (int i = 0; i < 12; i++) {
			cin >> month[i];
		}

		int dp[13] = { 0 }; // dp[i]는 i번째 달까지의 최소 비용

		for (int i = 0; i < 12; i++) {
			// 1일 이용권 사용
			dp[i + 1] = dp[i] + month[i] * oneDay;
			// 1달 이용권 사용
			dp[i + 1] = min(dp[i + 1], dp[i] + oneMonth);
			// 3달 이용권 사용
			if (i >= 2) {
				dp[i + 1] = min(dp[i + 1], dp[i - 2] + threeMonth);
			}
			else {
				dp[i + 1] = min(dp[i + 1], threeMonth);
			}
		}

		// 1년 이용권 사용
		int result = min(dp[12], oneYear);

		cout << "#" << test_case << " " << result << endl;
	}
	return 0;
}
```

### 소스 코드 (cpp)
```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <queue>
#include <limits.h>

using namespace std;

struct Stair {
	int x;   
	int y; 
	int len; 
};


struct People {
	int x;     
	int y;     
	int s1_len;  
	int s2_len; 
};

int n;              
vector<Stair> stairs;  
vector<People> people;
int minTime; 

void init() {
	minTime = INT_MAX;
	stairs.clear();
	people.clear();
}

void input() {
	// 방의 정보를 입력받고, 사람과 계단의 위치를 저장
	cin >> n;
	for (int y = 0; y < n; y++) {
		for (int x = 0; x < n; x++) {
			int num;
			cin >> num;

			if (num > 1) {
				stairs.push_back(Stair{ x, y, num });
			}
			else if (num == 1) {
				people.push_back(People{ x, y });
			}
		}
	}

	// 사람과 계단 간의 거리 계산
	for (int i = 0; i < people.size(); i++) {
		people[i].s1_len = abs(people[i].x - stairs[0].x) + abs(people[i].y - stairs[0].y);
		people[i].s2_len = abs(people[i].x - stairs[1].x) + abs(people[i].y - stairs[1].y);
	}
}

int calculateTime(const vector<int>& stair_choice) {
	// 사람들이 계단에 도착하는 시간 추가
	vector<int> times[2]; 
	for (int i = 0; i < people.size(); ++i) {
		if (stair_choice[i] == 0) {
			times[0].push_back(people[i].s1_len);
		}
		else {
			times[1].push_back(people[i].s2_len);
		}
	}

	int max_time = 0;
	for (int i = 0; i < 2; ++i) {
		if (times[i].empty()) continue; // 계단에 가는 사람이 없으면 패스
		sort(times[i].begin(), times[i].end()); // 계단에 도착하는 시간이 짧은 순으로 정렬
		
		// 대기 인원이 계단에서 내려가는 시간을 처리
		queue<int> q;
		for (int t : times[i]) {
			while (!q.empty() && q.front() <= t) q.pop(); // 시간 지나간 인원 제거
			if (q.size() < 3) {
				q.push(t + stairs[i].len + 1); // 계단에 대기
			}
			else {
				int wait_time = q.front(); // 대기 중인 사람의 시간
				q.pop();
				q.push(wait_time + stairs[i].len); // 계단을 내려가고 나서 다음 사람 대기
			}
		}
		// 큐에 남아 있는 사람들의 완료 시간을 계산
		while (!q.empty()) {
			max_time = max(max_time, q.front());
			q.pop();
		}
	}
	return max_time;
}

int solve() {
	int num_people = people.size(); 
	int total_combinations = (1 << num_people); // 가능한 모든 계단 선택 조합의 수

	for (int mask = 0; mask < total_combinations; ++mask) {
		// 비트마스킹을 통해 계단 선택 조합 생성
		vector<int> stair_choice(num_people);
		for (int i = 0; i < num_people; ++i) {
			stair_choice[i] = (mask & (1 << i)) ? 1 : 0;
		}

		minTime = min(minTime, calculateTime(stair_choice));
	}

	return minTime;
}

int main(int argc, char** argv) {
	int test_case;
	int T;
	cin >> T;
	for (test_case = 1; test_case <= T; ++test_case) {
		init();    
		input();   
		int result = solve(); 
		cout << "#" << test_case << " " << result << "
"; 
	}
	return 0;
}
```

### 소스 코드 (cpp)
```cpp
#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>

using namespace std;

int arr[4][8];
int K;
vector<pair<int, int>> changes;
int res;

// 왼쪽 회전 함수
void leftRotate(int gear, int k) {
	int temp[8];
	//회전
	for (int i = 0; i < 8; ++i) {
		temp[(i - k + 8) % 8] = arr[gear][i];
	}
	//적용
	for (int i = 0; i < 8; ++i) {
		arr[gear][i] = temp[i];
	}
}

// 오른쪽 회전 함수
void rightRotate(int gear, int k) {
	int temp[8];
	//회전
	for (int i = 0; i < 8; ++i) {
		temp[(i + k) % 8] = arr[gear][i];
	}
	//적용
	for (int i = 0; i < 8; ++i) {
		arr[gear][i] = temp[i];
	}
}

void solve() {

	for (int i = 0; i < changes.size(); i++)
	{
		int gear = changes[i].first;
		int dir = changes[i].second;
		vector<pair<int, int>> changeGears;//바꿀기어 넣어둘곳
		changeGears.push_back({ gear,dir });//처음기어

		// 기어 오른쪽확인
		for (int rightGear = gear; rightGear < 3; rightGear++)
		{
			if (arr[rightGear][2] != arr[rightGear + 1][6]) {
				dir = (dir == 1 ? -1 : 1);
				changeGears.push_back({ rightGear + 1, dir });
			}
			else break;
		}

		dir = changes[i].second;//처음기어 방향 복구

		// 기어 왼쪽확인
		for (int leftGear = gear; leftGear > 0; leftGear--)
		{
			if (arr[leftGear][6] != arr[leftGear - 1][2]) {
				dir = (dir == 1 ? -1 : 1);
				changeGears.push_back({ leftGear - 1, dir });
			}
			else break;
		}

		//기어 회전
		for (int j = 0; j < changeGears.size(); j++)
		{
			int changeGearNum = changeGears[j].first;
			int changeDir = changeGears[j].second;
			(changeDir > 0) ? rightRotate(changeGearNum, changeDir) : leftRotate(changeGearNum, -changeDir);
		}
	}

	for (int j = 0; j < 4; j++)
	{
		res += (arr[j][0] << (j));
	}
}

int main()
{
	int T;
	cin >> T;
	for (int tc = 1; tc <= T; tc++)
	{
		changes.clear();
		res = 0;
		cin >> K;
		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 8; j++)
			{
				cin >> arr[i][j];
			}
		}
		for (int i = 0; i < K; i++)
		{
			int a, b;
			cin >> a >> b;
			changes.push_back({ a - 1,b });
		}

		solve();

		cout << "#" << tc << " " << res << "
";
	}
	return 0;
}
```

### 소스 코드 (cpp)
```cpp
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

int N;
vector<vector<int>> arr;
vector<vector<bool>> visited;
vector<pair<int, int>> cores;

int res;
int maxConnected;

int dy[4] = { -1,0,1,0 };
int dx[4] = { 0,1,0,-1 };


void dfs(int cnt, int len, int connected) {
    // 최소 길이 갱신
    if (maxConnected == connected) {
        res = min(res, len);
    }
    else if (maxConnected < connected) {
        maxConnected = connected;
        res = len;
    }

    if (cnt >= cores.size()) {
        return;
    }   

    if (connected + (cores.size() - cnt) < maxConnected) {
        return;
    }

    pair<int, int> cur = cores[cnt];

    // 모서리에 위치한 코어는 이미 연결된 것으로 간주하고 다음 코어 탐색
    if (cur.first == 0 || cur.second == 0 || cur.first == N - 1 || cur.second == N - 1) {
        dfs(cnt + 1, len, connected + 1);
        return;
    }

    // 4방향으로 시도
    for (int i = 0; i < 4; i++) {
        bool isCross = false;
        int ny = cur.first;
        int nx = cur.second;
        int nl = 0;
        vector<pair<int, int>> path;  // 전선 경로 저장

        // 전선 설치 시도
        while (true) {
            ny += dy[i];
            nx += dx[i];

            // 범위를 벗어나면 전원이 연결된 것
            if (ny < 0 || nx < 0 || ny >= N || nx >= N) {
                break;
            }

            // 다른 전선이나 코어와 교차하는 경우
            if (visited[ny][nx]) {
                isCross = true;
                break;
            }

            // 전선 설치 가능 경로로 간주
            path.push_back({ ny, nx });
            nl++;
        }

        // 교차 발생 시 다음 방향으로 시도
        if (isCross) {
            continue;
        }

        // 전선 설치
        for (auto& p : path) {
            visited[p.first][p.second] = true;
        }

        // 다음 코어 탐색
        dfs(cnt + 1, len + nl, connected + 1);

        // 전선 원상복구
        for (auto& p : path) {
            visited[p.first][p.second] = false;
        }
    }

    // 해당 코어를 연결하지 않고 넘어가는 경우
    dfs(cnt + 1, len, connected);
}


int main() {
    int T;
    cin >> T;
    for (int tc = 1; tc <= T; tc++) {
        cin >> N;

        arr.clear(); arr.resize(N, vector<int>(N));
        visited.clear(); visited.resize(N, vector<bool>(N, false));
        cores.clear();
        res = 21e8; maxConnected = 0;

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                cin >> arr[i][j];
                // 코어 좌표 저장
                if (arr[i][j] == 1) {
                    cores.push_back({ i, j });
                    visited[i][j] = true;
                }
            }
        }

        dfs(0, 0, 0);

        cout << "#" << tc << " " << res << "
";
    }
}
```

### 소스 코드 (cpp)
```cpp
#include <iostream>
#include<algorithm>
#include<vector>
#include<string>

using namespace std;


int main(int argc, char** argv)
{
	int n;
	cin >> n;

	for (int i = 1; i < n + 1; i++)
	{
		int a = i;
		int cnt = 0;
		while (a) {
			int b = a % 10;
			a /= 10;
			if (b == 3 || b == 6 || b == 9) cnt++;
		}
		if (cnt) {
			while (cnt--)
				cout << "-";

			cout << " ";
		}
		else {
			cout << i << " ";
		}
	}

	return 0;
}
```

### 소스 코드 (cpp)
```cpp
#include <iostream>
#include <vector>

using namespace std;



int main()
{
	int test_case;
	int T;	
	cin >> T;
	for (test_case = 1; test_case <= T; ++test_case)
	{
		long long result = 0;
		int max_price = 0;

		int N;
		cin >> N;

		vector<int> days(N);

		for (int i = 0; i < N; i++)
		{
			cin >> days[i];
		}

		for (auto iter = days.rbegin(); iter != days.rend(); ++iter)
		{
			if (*iter > max_price) {
				max_price = *iter;
			}
			result += max_price - *iter;
		}

		cout << "#" << test_case << " " << result << endl;
	}
}
```

### 소스 코드 (cpp)
```cpp
#include <iostream>
#include <vector>

using namespace std;

int arr[9];

int main(int argc, char** argv)
{
	int test_case;
	int T;
	cin >> T;
	for (test_case = 1; test_case <= T; ++test_case)
	{
		double result = 0.0;
		int n;

		cin >> n;
		for (int i = 0; i < n; i++)
		{
			cin >> arr[i];
		}

		int subsetCount = 1 << n; // 부분집합 개수

		for (int mask = 1; mask < subsetCount; ++mask) { // 공집합 제외
			double subsetSum = 0.0;
			int subsetSize = 0;

			for (int i = 0; i < n; ++i) {
				if (mask & (1 << i)) {
					subsetSum += arr[i];
					subsetSize++;
				}
			}

			double subsetAverage = subsetSum / subsetSize;
			result += subsetAverage;
		}

		result /= subsetCount - 1; // 공집합 제외

		cout << fixed;
		cout.precision(18);
		cout << "#" << test_case << " " << result << "
";
	}
	return 0;
}
```

### 소스 코드 (cpp)
```cpp
#include<iostream>

using namespace std;

int n;
int cnt;
bool col[13];
bool ldru[25];//n*2
bool lurd[25];

void func(int rw) {
	if (rw == n) {
		cnt++;
		return;
	}
	for (int cl = 0; cl < n; cl++)
	{
		if (col[cl]) continue;
		if (ldru[rw + cl]) continue;
		if (lurd[rw - cl + n - 1]) continue;
		col[cl] = true;
		ldru[rw + cl] = true;
		lurd[rw - cl + n - 1] = true;
		func(rw + 1);
		col[cl] = false;
		ldru[rw + cl] = false;
		lurd[rw - cl + n - 1] = false;
	}
}

int main() {
	int tc;
	cin >> tc;
	for (int i = 1; i <= tc; i++){
		cin >> n;
		cnt = 0;
		func(0);
		cout << "#" << i << " " << cnt << "
";
	}
	return 0;
}
```

### 소스 코드 (cpp)
```cpp
#include <vector>
#include <iostream>

using namespace std;

vector<int> solution(vector<int> arr) 
{
    vector<int> answer (1,10);

    for(int i=0;i<arr.size();i++){
        if(answer.back()!=arr[i])
            answer.push_back(arr[i]);
    }
    answer.erase(answer.begin());

    return answer;
}
```

### 소스 코드 (cpp)
```cpp
#include <bits/stdc++.h>

using namespace std;

vector<int> solution(vector<string> park, vector<string> routes) {
    vector<int> answer={0,0};
    queue<string> q;
    
    for(int i=0;i<park.size();i++){ 
        for(int j=0;j<park[i].size();j++){
            if(park[i][j]=='S'){
                answer[0]=i;
                answer[1]=j;
            }
        }
    }
    
    while(!routes.empty()){
        string r = routes.front();
        char op = r[0]; r.erase(r.begin());
        int n = stoi(r);
        bool flag=true;
        
        if(op=='E'){
            for(int i=1;i<=n;i++){
                if(answer[1]+i>=park[answer[0]].size()){
                    flag=false;
                    break;
                }
                else if(park[answer[0]][answer[1]+i]=='X') {
                    flag=false;
                    break;
                }
            }
            if(flag) answer[1]+=n;
        }
        else if(op=='W'){
            for(int i=1;i<=n;i++){
                if(answer[1]-i<0){
                    flag=false;
                    break;
                }
                else if(park[answer[0]][answer[1]-i]=='X') {
                    flag=false;
                    break;
                }
            }
            if(flag) answer[1]-=n;
        }
        else if(op=='S'){
            for(int i=1;i<=n;i++){
                if(answer[0]+i>=park.size()){
                    flag=false;
                    break;
                }
                else if(park[answer[0]+i][answer[1]]=='X') {
                    flag=false;
                    break;
                }
            }
            if(flag) answer[0]+=n;
        }
        else if(op=='N'){
            for(int i=1;i<=n;i++){
                if(answer[0]-i<0){
                    flag=false;
                    break;
                }
                else if(park[answer[0]-i][answer[1]]=='X') {
                    flag=false;
                    break;
                }
            }
            if(flag) answer[0]-=n;
        }
        routes.erase(routes.begin());
    }
    
    
    
    return answer;
}
```

### 소스 코드 (cpp)
```cpp
#include <string>
#include <vector>
#include <algorithm>
using namespace std;

string solution(vector<string> participant, vector<string> completion) {
    string answer = "";
    unordered_map<string, int> temp;
    for (string name : participant)
    {  
        temp[name]++;
    }
    for (string name : completion)
    {
        temp[name]--;
    }
    for (auto pair : temp)
    {
        if (pair.second > 0)
        {
            answer = pair.first;
            break;
        }
    }
    return answer;
}
```

### 소스 코드 (cpp)
```cpp
#include <vector>
#include <map>

using namespace std;

int solution(vector<int> nums)
{
    int answer = 0;
    map<int,int> m;
    
    for(const auto& n:nums){
        m[n]++;
    }
    
    int size=nums.size()/2;
    
    if(m.size()<size)
        return m.size();
    else 
        return size;
}
```

### 소스 코드 (cpp)
```cpp
#include <string>
#include <vector>
#include <map>
#include <algorithm>

using namespace std;

vector<int> solution(vector<string> id_list, vector<string> report, int k) {
    vector<int> answer(id_list.size(),0);
    map<string,vector<string>> m;
    map<string,int> reportmap;
    
    sort(report.begin(),report.end());
    report.erase(unique(report.begin(),report.end()),report.end());

    for(const auto& rp:report){
        int space=rp.find(" ");
        string a=rp.substr(0,space);
        string b=rp.substr(space+1,rp.size()-1);
        m[a].emplace_back(b);
        reportmap[b]++;
    }
    
    for(int i=0;i<id_list.size();i++){
        int sum=0;
        for(auto id:m[id_list[i]]){
            if(reportmap[id]>=k)
                sum++;
        }
        answer[i]=sum;
    }

    
    return answer;
}
```

### 소스 코드 (cpp)
```cpp
#include <string>
#include <vector>

using namespace std;

vector<vector<int>> solution(int n) {
    vector<vector<int>> answer(n, vector<int>(n));

    int num = 1; 
    int x = 0, y = 0; 
    int direction = 0; 
    
    for (int i = 0; i < n * n; i++) {
        answer[y][x] = num; 

        if (direction == 0) {
            if (x == n - 1 || answer[y][x + 1] != 0) {
                direction = 1;
                y++;
            } else {
                x++;
            }
        } else if (direction == 1) {
            if (y == n - 1 || answer[y + 1][x] != 0) {
                direction = 2;
                x--;
            } else {
                y++;
            }
        } else if (direction == 2) {
            if (x == 0 || answer[y][x - 1] != 0) {
                direction = 3;
                y--;
            } else {
                x--;
            }
        } else if (direction == 3) {
            if (y == 0 || answer[y - 1][x] != 0) {
                direction = 0;
                x++;
            } else {
                y--;
            }
        }

        num++;
    }

    return answer;
}
```

### 소스 코드 (cpp)
```cpp
#include <string>
#include <vector>
#include <map>

using namespace std;

vector<int> solution(vector<int> progresses, vector<int> speeds) {
    vector<int> answer;
    vector<int> arr;
    for(int i=0;i<progresses.size();i++){
        int day=1;
        int p=progresses[i];
        while(p<100){
            p+=speeds[i];
            day++;
        }
        arr.push_back(day);
        
    }
    while(!arr.empty()){
        int cnt=1;
        int day=arr.front();
        arr.erase(arr.begin());
        
        while(arr.front()<=day && !arr.empty()){
            arr.erase(arr.begin());
            cnt++;
        }
        answer.push_back(cnt);
    }
    return answer;
}
```

### 소스 코드 (cpp)
```cpp
#include <iostream>
#include <vector>
#include <unordered_map>
#include <stack>
#include <algorithm>

using namespace std;

int arr[101][101];

int answer = 21e8;
int firstGroupCnt = 0;

void dfs(int start, int n)
{
	firstGroupCnt = 0;
	bool visited[101] = { false };
	stack<int> s;
	s.push(start);
	while (!s.empty())
	{
		int current = s.top();
		s.pop();

		if (visited[current])
			continue;
		visited[current] = true;
		firstGroupCnt++;
		for (int next = 0; next < n; next++)
		{
			if (arr[current][next] == 1 && !visited[next])
			{
				s.push(next);
			}
		}
	}

	// 두 번째 그룹은 전체 노드 수에서 첫 번째 그룹의 노드 수를 뺀 값
	int SecondGroupCount = n - firstGroupCnt;

	answer = min(answer, abs(firstGroupCnt - SecondGroupCount));
}

int solution(int n, vector<vector<int>> wires)
{

	// 노드 생성
	for (const auto &wire : wires)
	{
		arr[wire[0] - 1][wire[1] - 1] = 1;
		arr[wire[1] - 1][wire[0] - 1] = 1;
	}

	// 와이어 삭제 후 송전탑 개수 계산
	for (int from = 0; from < n; ++from)
	{
		for (int to = from + 1; to < n; ++to)
		{
			if (arr[from][to] == 1)
			{
				// 연결된 와이어 제거
				arr[from][to] = 0;
				arr[to][from] = 0;

				// 첫 번째 그룹 탐색
				dfs(from, n);

				// 제거했던 와이어 다시 복구
				arr[from][to] = 1;
				arr[to][from] = 1;
			}
		}
	}

	return answer;
}
```

### 소스 코드 (cpp)
```cpp
#include <string>
#include <vector>
#include <iostream>

using namespace std;

int answer = 0;

void solve(int cur, int sum, int target, const vector<int> &numbers){
    if(cur==numbers.size()){
        if(target==sum) answer++;
        return;
    }
    int num = numbers[cur];
    solve(cur+1,sum+num,target,numbers);
    solve(cur+1,sum-num,target,numbers);
}

int solution(vector<int> numbers, int target) {
    
    solve(0,0,target,numbers);
    
    return answer;
}
```

### 소스 코드 (cpp)
```cpp
#include <string>
#include <vector>
#include <algorithm>

using namespace std;

bool solution(vector<string> phone_book) {
    bool answer = true;
    
    sort(phone_book.begin(),phone_book.end());

    for(int i=0;i<phone_book.size()-1;i++){
        for(int j=i+1;j<phone_book.size();j++){
            auto key= phone_book[j].find(phone_book[i]);
            
            if(key==0) return false;
            else break;
        }
    }
    
    return answer;
}
```

### 소스 코드 (cpp)
```cpp
#include <string>
#include <vector>
#include <map>

using namespace std;

int solution(vector<vector<string>> clothes) {
    int answer = 1;
    map<string,int> m;
    for(int i=0;i<clothes.size();i++){
        m[clothes[i][1]]++;
    }
    
    for(auto iter=m.begin();iter!=m.end();iter++){
        answer*=1 + iter->second;
    }
    
    answer--;
    
    return answer;
}
```

### 소스 코드 (cpp)
```cpp
#include <bits/stdc++.h>

using namespace std;

long long solution(int r1, int r2) {
   long long answer=0;
    
    answer -= r2-r1+1;
    
    long long R1 = (long long)r1*r1;
    long long R2 = (long long)r2*r2;
    
    for(int x=0;x<=r2;x++){
        long long xx = (long long)x*x;
        int t1, t2 = sqrt(R2-xx);
        if(x<r1){
            double td = sqrt(R1-xx);
            int ti = sqrt(R1-xx);
            t1 = td>(double)ti?ti+1:ti;
        }else{
            t1 = 0;
        }
        answer += t2 - t1 + 1;
    }
    
    return answer*4;
}
```

### 소스 코드 (cpp)
```cpp
#include<string>
#include <iostream>

using namespace std;

bool solution(string s)
{
    int count = 0;  
    for (const auto &c : s) {
        if (c == '(') {
            count++;
        } else {
            count--;
            if (count < 0) return false;
        }       
    }
    return count==0?true:false;
}
```

### 소스 코드 (cpp)
```cpp
#include <bits/stdc++.h>

using namespace std;

int solution(vector<vector<int>> targets) {
    int answer = 0;
  
    sort(targets.begin(), targets.end());

    int maxEnd = 0;

    for (const auto& target : targets) {
        if (target[0] >= maxEnd) {
            answer++;
            maxEnd = target[1];
        }
        else {
            maxEnd = min(maxEnd, target[1]);
        }
    }

    return answer;
}
```

### 소스 코드 (cpp)
```cpp
#include <string>
#include <vector>
#include <queue>
#include <iostream>

using namespace std;

vector<vector<int>> map (1000,(vector<int>(1000,0)));

void insertMap(int n){
    int dir_type = 0;
    int count = 1;
    int sum = 0;
    
    for(int i=1;i<=n;i++){
        sum+=i;
    }
    
    queue<pair<int,int>> q;
    
    map[0][0]=1;
    if(n==1) return;
    
    q.push({0,0});
    
    while(!q.empty()){
        if(count == sum) break;
        
        auto [x,y]= q.front();
        q.pop();
        
        switch(dir_type){
            case 0:
                x ++;
                if(map[x][y]!= 0 || x == n){
                    dir_type = 1;
                    q.push({x-1,y++});
                }
                else {
                    map[x][y]=++count;
                    q.push({x,y});
                }
                break;
            case 1:
                y ++;
                if(map[x][y]!= 0 || y == n){
                    dir_type = 2;
                    q.push({x,y-1});
                }
                else {
                    map[x][y]=++count;
                    q.push({x,y});
                }
                break;
            case 2: 
                x --; y--;
                if(map[x][y]!= 0){
                    dir_type = 0;
                    q.push({x+1 ,y+1});
                }
                else {
                    map[x][y]=++count;
                    q.push({x,y});
                }
                break;
        }
    }
}

vector<int> solution(int n) {
    vector<int> answer;
    insertMap(n);
    for(int i=0; i<n; i++){
        for(int j=0; j<=i; j++){ 
            answer.push_back(map[i][j]);
        }
    }
    
    return answer;
}
```

### 소스 코드 (cpp)
```cpp
#include<vector>
#include<queue>
#include <iostream>
using namespace std;

int dy[4]={1,0,-1,0};
int dx[4]={0,1,0,-1};
int N,M;

int bfs(const vector<vector<int>> &maps){
    queue<pair<int,int>> q;
    int visited[101][101]={0};
    int res = 21e8;
    
    q.push({0,0});
    visited[0][0]=1;
    cout<<N << " "<<M<<endl;
    while(!q.empty()){
        pair<int,int> cur =q.front();
        q.pop();
        if(cur.first == N-1 && cur.second == M-1){
            res=visited[cur.first][cur.second];
            break;
        }
        for(int i=0; i<4; i++){
            int ny = cur.first + dy[i];
            int nx = cur.second + dx[i];
            
            if(ny<0||nx<0||ny>=N||nx>=M)continue;
            if(maps[ny][nx]==0) continue;
            if(visited[ny][nx]) continue;
            
            visited[ny][nx]=visited[cur.first][cur.second]+1;
            q.push({ny,nx});
        }
    }
    
    if(res==21e8) res=-1;
    
    return res;
}

int solution(vector<vector<int>> maps)
{
    N = maps.size(); M = maps[0].size();
    int answer = 0;
    answer = bfs(maps);
    return answer;
}
```

### 소스 코드 (cpp)
```cpp
#include <string>
#include <vector>

using namespace std;

int solution(int n, vector<vector<int>> results) {
    int answer = 0;
    vector<vector<int>> graph (n+1,vector<int>(n+1,-1));
    
    // 자기 자신과 경기 0
    for(int i=1; i<n+1; i++){
        graph[i][i]=0;
    }
    
    // 승리 1, 패배 2
    for(auto res : results){
        graph[res[0]][res[1]]=1;
        graph[res[1]][res[0]]=2;
    }
    
    for(int k=1; k<= n; k++){
        for(int win=1; win<=n ; win++){
            for(int lose=1; lose<=n; lose++){
                if (graph[win][k] == 1 && graph[k][lose] == 1){
                    graph[win][lose] = 1;
                }
                else if (graph[k][win] == 2 && graph[win][lose] == 2){
                    graph[k][lose] = 2;
                }
            }
        }
    }
        
    for(int k=1; k<= n; k++){
        for(int win=1; win<=n ; win++){
            for(int lose=1; lose<=n; lose++){
                if (graph[win][k] == 1 && graph[k][lose] == 1){
                    graph[win][lose] = 1;
                }
                else if (graph[k][win] == 2 && graph[win][lose] == 2){
                    graph[k][lose] = 2;
                }
            }
        }
    }
    
    for(int win=1; win<n+1; win++){
        bool isFill = true;
        for(int lose=1; lose<n+1; lose++){
            if(graph[win][lose] == -1) isFill = false;
        }
        if(isFill) answer++;
    }
    
    return answer;
}
```

### 소스 코드 (cpp)
```cpp
#include <string>
#include <vector>

using namespace std;

vector<vector<int>> map;

int solution(int n, int s, int a, int b, vector<vector<int>> fares) {
    int answer = 0;
    int INF = n*100000 + 1;
    map.assign(n+1, vector<int>(n+1, INF));
    
    for (int i = 0; i < fares.size(); ++i) {
        map[fares[i][0]][fares[i][1]] = fares[i][2];
        map[fares[i][1]][fares[i][0]] = fares[i][2];
    }
    
    for (int i = 1; i <= n ; ++i) {
        map[i][i] = 0;
    }

    for (int k = 1; k <= n; ++k) {
        for (int i = 1; i <= n; ++i) {
            for (int j = 1; j <= n; ++j) {
                if (map[i][j] > map[i][k] + map[k][j]){
                    map[i][j] =  map[i][k] + map[k][j];
                }
            }
        }
    }

    answer = map[s][a] + map[s][b];
    
    for (int i = 1; i <= n; ++i) { 
        if (i!=s){
            if (answer > map[s][i] + map[i][a] + map[i][b]){
                answer = map[s][i] + map[i][a] + map[i][b];
            }
        }
    }
    
    return answer;
}
```

### 소스 코드 (cpp)
```cpp
#include <string>
#include <vector>
#include <algorithm>

using namespace std;
int answer = 21e8;
bool visited[51];

bool checkString(const string& a, const string& b) {
    int count = 0;

    for (int i = 0; i < a.length(); ++i) {
        if (a[i] == b[i]) {
            count++;
        }
    }
    if(count == a.length() - 1) return true;
    return false;
}

void solve(string cur, string target,int cnt, const vector<string> &words){
    if(cnt>=answer) return;
    if(cur==target){
        answer=min(cnt,answer);
    }
    for(int i =0; i<words.size();i++){
        if(visited[i]) continue;
        string word = words[i];
        if(checkString(cur,word)){
            visited[i]=true;
            solve(word,target,cnt+1,words);
            visited[i]=false;
        }
    }
}

int solution(string begin, string target, vector<string> words) {
    if(find(words.begin(),words.end(),target)==words.end()) return 0;
    solve(begin,target,0,words);
    return answer;
}
```

### 소스 코드 (cpp)
```cpp
#include <vector>

using namespace std;

int union_find[201];

int find(int tar)
{
	if (tar == union_find[tar])
		return tar;

	int ret = find(union_find[tar]);
	union_find[tar] = ret;
	return ret;
}

void setUnion(int a, int b)
{
	int t1 = find(a);
	int t2 = find(b);

	if (t1 == t2)
		return;

	union_find[t2] = t1;
}

int solution(int n, vector<vector<int>> computers) {
	int answer = 0;

	//초기화
	for (int i = 0; i < n; i++) {
		union_find[i] = i;
	}

	//union-find 설정
	for (int computer = 0; computer < n; computer++)
	{
		for (int connect = 0; connect < n; connect++)
		{
			if (computers[computer][connect]) {
				setUnion(computer, connect);
			}
		}
	}

	// root node 확인
	vector<bool> visited(n, false);

	for (int i = 0; i < n; i++) {
		int root = find(i);
		if (!visited[root]) {
			answer++;
			visited[root] = true;
		}
	}

	return answer;
}
```

