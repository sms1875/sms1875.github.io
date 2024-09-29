---
title: "# CodingTestStudy"
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
#include <bits/stdc++.h>

using namespace std;

int solution(vector<int> scoville, int K) {
    int answer = 0;
    priority_queue<int, vector<int>, greater<int>> pq;
    for(const auto& i:scoville){
        pq.push(i);
    }
    while(pq.top()<K){
        if(pq.size()<2) return -1;
        int a = pq.top(); pq.pop();
        int b = pq.top(); pq.pop();
        pq.push(a+b*2);
        answer++;
    }
    
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

### 소스 코드 (cc)
```cc
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <queue>

using namespace std;

int bfs(int start, const unordered_map<int, int>& ladders, const unordered_map<int, int>& snakes) {
    vector<bool> visited(101, false); // 1~100까지 방문 여부 체크
    queue<pair<int, int>> q; // 현재 위치와 주사위 굴림 횟수를 저장
    q.push({start, 0});
    visited[start] = true;

    while (!q.empty()) {
        int cur = q.front().first;
        int cnt = q.front().second;
        q.pop();

        // 목표 지점에 도달했을 때
        if (cur >= 100) {
            return cnt;
        }

        for (int i = 1; i <= 6; i++) {
            int next = cur + i;
            if (next > 100) continue;

            // 사다리가 있는 경우
            if (ladders.count(next)) {
                next = ladders.at(next);
            }
            // 뱀이 있는 경우
            else if (snakes.count(next)) {
                next = snakes.at(next);
            }

            if (!visited[next]) {
                visited[next] = true;
                q.push({next, cnt + 1});
            }
        }
    }

    return -1; // 이론적으로 이 부분에 도달할 수 없음
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(0); cout.tie(0);

    int N, M;
    cin >> N >> M;
    unordered_map<int, int> ladders;
    unordered_map<int, int> snakes;

    for (int i = 0; i < N; i++) {
        int from, to;
        cin >> from >> to;
        ladders[from] = to;
    }

    for (int i = 0; i < M; i++) {
        int from, to;
        cin >> from >> to;
        snakes[from] = to;
    }

    int result = bfs(1, ladders, snakes);
    cout << result << "
";

    return 0;
}
```

### 소스 코드 (cc)
```cc
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

int main() {
	ios_base::sync_with_stdio(false);
	cin.tie(0);

	int N, K;
	cin >> N >> K;  // N: 물건의 개수, K: 배낭의 최대 무게

	vector<int> W(N + 1);  // 물건들의 무게
	vector<int> V(N + 1);  // 물건들의 가치

	for (int i = 1; i <= N; i++) {
		cin >> W[i] >> V[i];
	}

	// DP 테이블 선언: dp[i][w]는 i번째 물건까지 고려했을 때 배낭의 최대 무게 w일 때의 최대 가치
	vector<vector<int>> dp(N + 1, vector<int>(K + 1, 0));

	// DP 계산
	for (int i = 1; i <= N; i++) {
		for (int w = 0; w <= K; w++) {
			// 물건을 배낭에 넣지 않는 경우
			dp[i][w] = dp[i - 1][w];

			// 물건을 배낭에 넣는 경우
			if (w >= W[i]) {
				dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - W[i]] + V[i]);
			}
		}
	}

	// 결과 출력: 배낭에 넣을 수 있는 최대 가치
	cout << dp[N][K] << "
";

	return 0;
}
```

### 소스 코드 (cc)
```cc
#include <iostream>
#include <queue>
#include<algorithm>

using namespace std;

int M, N;
int res = 0;
vector<vector<int>> arr;
vector<vector<bool>> visited;
int dx[4] = { -1, 0, 0, 1 };
int dy[4] = { 0, -1, 1, 0 };

void solve(pair<int, int> cur, int cnt, int sum) {
	if (cnt == 4) { 
		res = max(res, sum);
		return;
	}

	for (int i = 0; i < 4; i++)
	{
		int ny = cur.first + dy[i];
		int nx = cur.second + dx[i];
		if (ny <= 0 || nx <= 0 || ny > N || nx > M || visited[ny][nx]) continue;
		visited[ny][nx] = true;
		solve({ ny,nx }, cnt + 1, sum + arr[ny][nx]);
		visited[ny][nx] = false;
	}
}

int main() {
	cin >> N >> M;

	arr.resize(N + 2, vector<int>(M + 2, 0));
	visited.resize(N + 2, vector<bool>(M + 2, false));

	for (int i = 1; i <= N; i++)
	{
		for (int j = 1; j <= M; j++)
		{
			cin >> arr[i][j];
		}
	}
	for (int i = 1; i <= N; i++)
	{
		for (int j = 1; j <= M; j++)
		{
			visited[i][j] = true;
			solve({ i,j }, 1, arr[i][j]);
			visited[i][j] = false;

			int cross = arr[i][j] + arr[i - 1][j] + arr[i + 1][j] + arr[i][j - 1] + arr[i][j + 1];
			cross -= min(min(arr[i - 1][j], arr[i + 1][j]), min(arr[i][j - 1], arr[i][j + 1]));//가장작은값
			res = max(res, cross);
		}
	}
	cout << res;
}
```

### 소스 코드 (cc)
```cc
#include <iostream>
#include <vector>

using namespace std;

const long long MOD = 1000000007;  // 피보나치 수를 나눌 큰 소수

// 2x2 행렬의 곱셈을 수행하는 함수
vector<vector<long long>> matrixMultiply(const vector<vector<long long>>& a, const vector<vector<long long>>& b) {
	vector<vector<long long>> result(2, vector<long long>(2, 0));  // 결과 행렬을 초기화
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2; j++) {
			for (int k = 0; k < 2; k++) {
				result[i][j] = (result[i][j] + a[i][k] * b[k][j]) % MOD;  // 행렬 곱셈 후 MOD 연산
			}
		}
	}
	return result;
}

// 2x2 행렬을 n번 곱하는 함수
vector<vector<long long>> matrixPower(vector<vector<long long>> base, long long exp) {
	vector<vector<long long>> result = { {1, 0}, {0, 1} };  // 항등 행렬로 초기화
	while (exp > 0) {
		if (exp % 2 == 1) {  // 지수가 홀수이면
			result = matrixMultiply(result, base);  // 결과에 현재 행렬을 곱함
		}
		base = matrixMultiply(base, base);  // 행렬을 제곱
		exp /= 2;  // 지수를 반으로 줄임
	}
	return result;
}

// n번째 피보나치 수를 계산하는 함수
long long fibonacci(long long n) {
	if (n == 0) return 0;
	vector<vector<long long>> base = { {1, 1}, {1, 0} };  // 피보나치 행렬 초기값 설정
	vector<vector<long long>> result = matrixPower(base, n - 1);  // 행렬 거듭제곱을 통해 피보나치 수 계산
	return result[0][0];  // 피보나치 수 반환
}

int main() {
	ios_base::sync_with_stdio(false); cin.tie(0); cout.tie(0);

	long long dist;
	cin >> dist;

	cout << fibonacci(dist) << "
";  // n번째 피보나치 수 출력

	return 0;
}
```

### 소스 코드 (cc)
```cc
#include<iostream>
#include<vector>
#include<queue>
#include<algorithm>

using namespace std;

int dy[4] = { -1, 1, 0, 0 };
int dx[4] = { 0, 0, -1, 1 };

void solve(vector<vector<int>>& arr, const pair<int, int>& st) {
	int N = arr.size();
	int time = 0;
	int babySharkSize = 2;
	int eatCnt = 0;
	pair<int, int> curSharkPos = st;

	while (true) {
		pair<int, pair<int, int>> nextPos = { 0, {-1, -1} };

		vector<vector<bool>> visited(N, vector<bool>(N, false));
		queue<pair<int, pair<int, int>>> q;
		q.push({ 0,curSharkPos });
		visited[curSharkPos.first][curSharkPos.second] = true;

		vector<pair<int, pair<int, int>>> fish;  // 물고기 후보들 저장

		while (!q.empty()) {
			pair<int, pair<int, int>> cur = q.front();
			q.pop();

			int curDist = cur.first;
			int curY = cur.second.first;
			int curX = cur.second.second;

			// 물고기를 찾음
			if (arr[curY][curX] >= 1 && arr[curY][curX] < babySharkSize) {
				fish.push_back({ curDist, {curY, curX} });
			}

			for (int i = 0; i < 4; i++) {
				int ny = curY + dy[i];
				int nx = curX + dx[i];
				if (ny < 0 || nx < 0 || ny >= N || nx >= N) continue;
				if (visited[ny][nx]) continue;
				if (arr[ny][nx] > babySharkSize) continue; // 상어보다 큰 물고기는 지나갈 수 없음
				visited[ny][nx] = true;
				q.push({ curDist + 1, {ny, nx} });
			}
		}

		// 후보들 중에서 가장 가까운 물고기를 선택
		if (!fish.empty()) {
			// 거리, 위쪽, 왼쪽 순서로 정렬
			sort(fish.begin(), fish.end(), [](pair<int, pair<int, int>>& a, pair<int, pair<int, int>>& b) {
				if (a.first == b.first) {
					if (a.second.first == b.second.first) {
						return a.second.second < b.second.second;  // x 좌표가 작은 것이 우선
					}
					return a.second.first < b.second.first;  // y 좌표가 작은 것이 우선
				}
				return a.first < b.first;  // 거리가 가까운 것이 우선
			});

			// 가장 가까운 물고기를 선택하여 이동
			nextPos = fish[0];
			curSharkPos = nextPos.second; // 상어 위치 이동
			time += nextPos.first;  // 이동한 거리 추가
			eatCnt++;  // 물고기를 먹음
			arr[curSharkPos.first][curSharkPos.second] = 0; // 물고기를 먹었으니 빈 칸으로 설정

			// 상어 크기 증가 체크
			if (eatCnt == babySharkSize) {
				babySharkSize++;
				eatCnt = 0;
			}
		}
		else {
			// 더 이상 먹을 물고기가 없는 경우
			break;
		}
	}

	cout << time;
}

int main() {
	int N;
	cin >> N;
	vector<vector<int>> arr(N, vector<int>(N));
	pair<int, int> babySharkPos;

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			cin >> arr[i][j];
			if (arr[i][j] == 9) {
				babySharkPos.first = i;
				babySharkPos.second = j;
				arr[i][j] = 0;  
			}
		}
	}

	solve(arr, babySharkPos);

	return 0;
}
```

### 소스 코드 (cc)
```cc
#include <iostream>
#include <vector>
#include <queue>
using namespace std;

int N, M;

int main() {
	std::ios::sync_with_stdio(0);
	std::cin.tie(0); std::cout.tie(0);

	cin >> N >> M;

	vector<vector<int>> adj(N + 1);
	vector<int> degree(N + 1);

	for (int i = 0; i < M; i++)
	{
		int a, b;
		cin >> a >> b;
		adj[a].push_back(b);
		degree[b]++;
	}

	queue<int> q;
	for (int i = 1; i <= N; i++)
	{
		if (degree[i] == 0) {
			q.push(i);
		}
	}

	while (!q.empty()) {
		int cur = q.front(); q.pop();

		cout << cur << " ";

		for (int next : adj[cur])
		{
			degree[next]--;
			if (degree[next] == 0) {
				q.push(next);
			}
		}
	}

	return 0;
}
```

### 소스 코드 (cc)
```cc
#include <iostream>
#include <queue>

using namespace std;

int N, M, H;
vector<vector<vector<int>>> arr; // [h][y][x]
vector<pair<int, pair<int, int>>> st;
int cnt = 0;

int dh[4] = { 1, -1 };
int dy[4] = { 0,1,0,-1 };
int dx[4] = { 1,0,-1,0 };

void solve() {
	int minV = 0;

	if (cnt == 0) {
		cout << 0;
		return;
	}

	queue<pair<int, pair<int, int>>> q;
	for (const auto& s : st) {
		q.push(s);
	}

	while (!q.empty()) {
		pair<int, pair<int, int>> cur = q.front();
		q.pop();
		// 옆
		for (int i = 0; i < 4; i++)
		{
			int ny = cur.second.first + dy[i];
			int nx = cur.second.second + dx[i];
			if (ny < 0 || nx < 0 || ny >= N || nx >= M) continue;
			if (arr[cur.first][ny][nx] != 0) continue;
			arr[cur.first][ny][nx] = arr[cur.first][cur.second.first][cur.second.second] + 1;
			cnt--;
			q.push({ cur.first, {ny, nx } });
			minV = max(minV, arr[cur.first][ny][nx]);
		}
		// 위
		for (int i = 0; i < 2; i++)
		{
			int nh = cur.first + dh[i];
			if (nh < 0 || nh >= H) continue;
			if (arr[nh][cur.second.first][cur.second.second] != 0) continue;
			arr[nh][cur.second.first][cur.second.second] = arr[cur.first][cur.second.first][cur.second.second] + 1;
			cnt--;
			q.push({ nh, {cur.second.first, cur.second.second } });
			minV = max(minV, arr[nh][cur.second.first][cur.second.second]);
		}
	}
	cout << (cnt == 0 ? minV - 1 : -1);
}

int main() {
	ios_base::sync_with_stdio(false);
	cin.tie(0); cout.tie(0);

	cin >> M >> N >> H;

	arr.resize(H, vector<vector<int>>(N, vector<int>(M)));

	for (int i = 0; i < H; i++)
	{
		for (int j = 0; j < N; j++)
		{
			for (int k = 0; k < M; k++)
			{
				cin >> arr[i][j][k];
				if (arr[i][j][k] == 0)
					cnt++;
				else if (arr[i][j][k] == 1)
					st.push_back({ i,{ j,k } });
			}
		}
	}
	solve();
}
```

### 소스 코드 (cc)
```cc
#include <iostream>
using namespace std;

int n;
int result = 0;

void solve(int row, int col_mask, int diag1_mask, int diag2_mask) {
	if (row == n) {
		result++;
		return;
	}

	// 가능한 열 위치를 비트마스크로 표현
	int available = ((1 << n) - 1) & ~(col_mask | diag1_mask | diag2_mask);

	while (available) {
		int bit = available & -available;  // 가장 오른쪽의 1비트 추출
		available -= bit;  // 현재 선택한 열을 제거
		solve(row + 1, col_mask | bit, (diag1_mask | bit) << 1, (diag2_mask | bit) >> 1);
	}
}

int main() {
	cin >> n;
	solve(0, 0, 0, 0);
	cout << result << endl;
	return 0;
}
```

### 소스 코드 (cc)
```cc
#include <iostream>
#include <queue>

using namespace std;

int M, N;
vector<vector<int>> arr;
vector<pair<int, int>> st;
int cnt = 0;

void solve() {
	int dy[4] = { 0,1,0,-1 };
	int dx[4] = { 1,0,-1,0 };
	int minV = 0;

	if (cnt == 0) {
		cout << 0;
		return;
	}

	queue<pair<int, int>> q;
	for (const auto& s : st) {
		q.push(s);
	}

	while (!q.empty()) {
		pair<int, int> cur = q.front();
		q.pop();
		for (int i = 0; i < 4; i++)
		{
			int ny = cur.first + dy[i];
			int nx = cur.second + dx[i];
			if (ny < 0 || nx < 0 || ny >= N || nx >= M) continue;
			if (arr[ny][nx] != 0) continue;
			arr[ny][nx] = arr[cur.first][cur.second] + 1;
			cnt--;
			q.push({ ny, nx });
			minV = max(minV, arr[ny][nx]);
		}
	}
	cout << (cnt == 0 ? minV - 1 : -1);
}

int main() {
	cin >> M >> N;

	arr.resize(N, vector<int>(M));

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < M; j++)
		{
			cin >> arr[i][j];
			if (arr[i][j] == 0) 
				cnt++;
			else if (arr[i][j] == 1) 
				st.push_back({ i,j });
		}
	}
	solve();
}
```

### 소스 코드 (cc)
```cc
#include <bits/stdc++.h>
using namespace std;

void func(int a, int b, int n){
  if(n == 1){
    cout << a << ' ' << b << '
';
    return;
  }
  func(a, 6-a-b, n-1);
  cout << a << ' ' << b << '
';
  func(6-a-b, b, n-1);
}

int main(void){
  ios::sync_with_stdio(0);
  cin.tie(0);
  int k;
  cin >> k;
  cout << (1<<k) - 1 << '
';
  func(1, 3, k);
}
```

### 소스 코드 (cc)
```cc
#include <iostream>
#include <vector>
#include <queue>

using namespace std;

int dy[4] = { 1, 0, -1, 0 };
int dx[4] = { 0, -1, 0, 1 };

void bfs(int y, int x, vector<vector<bool>>& visited, const vector<vector<char>>& arr, bool colorblind) {
	queue<pair<int, int>> q;
	q.push({ y, x });
	visited[y][x] = true;
	char currentColor = arr[y][x];

	while (!q.empty()) {
		pair<int, int> cur = q.front();
		q.pop();

		for (int i = 0; i < 4; i++) {
			int ny = cur.first + dy[i];
			int nx = cur.second + dx[i];

			if (ny < 0 || nx < 0 || ny >= arr.size() || nx >= arr.size()) continue;
			if (visited[ny][nx]) continue;

			if (colorblind) {
				// 적록색약의 경우 'R'과 'G'를 같은 색상으로 간주
				if ((currentColor == 'R' || currentColor == 'G') && (arr[ny][nx] == 'R' || arr[ny][nx] == 'G')) {
					visited[ny][nx] = true;
					q.push({ ny, nx });
				}
				else if (currentColor == arr[ny][nx]) {
					visited[ny][nx] = true;
					q.push({ ny, nx });
				}
			}
			else {
				if (currentColor == arr[ny][nx]) {
					visited[ny][nx] = true;
					q.push({ ny, nx });
				}
			}
		}
	}
}

void solve(const vector<vector<char>>& arr) {
	int cntNormal = 0;
	int cntColorblind = 0;

	vector<vector<bool>> visitedNormal(arr.size(), vector<bool>(arr.size(), false));
	vector<vector<bool>> visitedColorblind(arr.size(), vector<bool>(arr.size(), false));

	// 일반 사람의 경우
	for (int i = 0; i < arr.size(); i++) {
		for (int j = 0; j < arr.size(); j++) {
			if (!visitedNormal[i][j]) {
				bfs(i, j, visitedNormal, arr, false);
				cntNormal++;
			}
		}
	}

	// 적록색약의 경우
	for (int i = 0; i < arr.size(); i++) {
		for (int j = 0; j < arr.size(); j++) {
			if (!visitedColorblind[i][j]) {
				bfs(i, j, visitedColorblind, arr, true);
				cntColorblind++;
			}
		}
	}

	cout << cntNormal << " " << cntColorblind << "
";
}

int main() {
	ios_base::sync_with_stdio(false); cin.tie(0); cout.tie(0);

	int N;
	cin >> N;

	vector<vector<char>> arr(N, vector<char>(N));

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			cin >> arr[i][j];
		}
	}

	solve(arr);

	return 0;
}
```

### 소스 코드 (cc)
```cc
#include <iostream>
#include <queue>
#include <unordered_map>

using namespace std;

int main() {
	ios_base::sync_with_stdio(false);
	cin.tie(0); cout.tie(0);

	int T;
	cin >> T;
	while (T--) {
		unordered_map<int, int> map;
		priority_queue<int> maxpq; // 최대 힙
		priority_queue<int, vector<int>, greater<int>> minpq; // 최소 힙
		int Q;
		cin >> Q;
		int numCnt = 0;

		while (Q--) {
			char op;
			cin >> op;
			int n;
			cin >> n;
			if (op == 'I') {
				maxpq.push(n);
				minpq.push(n);
				map[n]++; // 해당 숫자의 삽입 횟수 증가
				numCnt++;
			}
			else {
				if (numCnt == 0) continue; // 유효한 숫자가 없는 경우 무시
				numCnt--;

				if (n == 1) { // 최대값 삭제
					if (maxpq.empty()) continue;
					while (map[maxpq.top()] == 0) {
						maxpq.pop();
					}
					map[maxpq.top()]--;
					maxpq.pop();
				}
				else { // 최소값 삭제
					if (minpq.empty()) continue;
					while (map[minpq.top()] == 0) {
						minpq.pop();
					}
					map[minpq.top()]--;
					minpq.pop();
				}
			}
		}

		// 남아있는 유효한 최댓값과 최솟값을 확인
		while (!maxpq.empty() && map[maxpq.top()] == 0) {
			maxpq.pop();
		}
		while (!minpq.empty() && map[minpq.top()] == 0) {
			minpq.pop();
		}

		if (numCnt == 0 || maxpq.empty() || minpq.empty()) {
			cout << "EMPTY
";
		}
		else {
			cout << maxpq.top() << " " << minpq.top() << "
";
		}
	}

	return 0;
}
```

### 소스 코드 (cc)
```cc
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

// 이진 탐색 함수: 정렬된 배열에서 num 이상의 첫 번째 위치를 찾음
// lower_bound 함수를 통해서 사용 가능
int binary_search(vector<int>& lis, int num) {
	int low = 0, high = lis.size();

	while (low < high) {
		int mid = low + (high - low) / 2;
		if (lis[mid] < num) {
			low = mid + 1;
		}
		else {
			high = mid;
		}
	}
	return low;
}

int main() {
	ios_base::sync_with_stdio(false);
	cin.tie(0);
	cout.tie(0);

	int N;
	cin >> N;

	vector<int> arr(N);  // 입력된 수열
	vector<int> lis;  // LIS를 저장할 벡터
	vector<int> pos(N);  // 각 숫자가 LIS에서 들어간 위치를 기록할 배열
	vector<int> trace(N, -1);  // 이전 위치 추적을 위한 배열

	// 입력 받기
	for (int i = 0; i < N; i++) {
		cin >> arr[i];
	}

	// LIS 구하기
	for (int i = 0; i < N; i++) {
		int num = arr[i];
		int pos_in_lis = binary_search(lis, num);  // 이진 탐색을 통해 LIS에서 들어갈 위치를 찾음
		

		if (pos_in_lis == lis.size()) {
			lis.push_back(num);  // 새로운 숫자를 LIS의 끝에 추가
		}
		else {
			lis[pos_in_lis] = num;  // 해당 위치의 값을 갱신
		}

		pos[i] = pos_in_lis;  // 해당 숫자가 들어간 위치 기록

		// trace 배열로 이전 요소 추적
		if (pos_in_lis > 0) {
			trace[i] = pos_in_lis - 1;
		}
	}

	// LIS 복원하기
	vector<int> actual_lis(lis.size());
	int current_pos = lis.size() - 1;
	for (int i = N - 1; i >= 0; i--) {
		if (pos[i] == current_pos) {
			actual_lis[current_pos] = arr[i];
			current_pos--;
		}
	}

	cout << lis.size();

	return 0;
}
```

### 소스 코드 (cc)
```cc
#include <iostream>
#include <string>
#include <vector>
#include <sstream>

using namespace std;

// 배열 문자열을 파싱하여 벡터로 변환하는 함수
vector<int> parseArrayString(const string& arrStr) {
	vector<int> v;

	// 배열 문자열에서 대괄호 제거 후 내부의 숫자들을 추출
	if (arrStr.length() > 2) {  // 빈 배열이 아닌 경우에만 파싱
		string innerArrayStr = arrStr.substr(1, arrStr.length() - 2);
		istringstream sstream(innerArrayStr);  // 문자열을 스트림으로 변환
		string token;

		// 쉼표를 기준으로 문자열을 나누어 숫자로 변환 후 벡터에 추가
		while (getline(sstream, token, ',')) {
			v.push_back(stoi(token));  // 문자열을 정수로 변환하여 벡터에 추가
		}
	}

	return v;
}

int main() {
	ios_base::sync_with_stdio(false);
	cin.tie(0); cout.tie(0);
	int T;
	cin >> T;
	while (T--) {
		string ops;
		int len;
		string arrStr;

		cin >> ops >> len >> arrStr;  // 함수 문자열, 배열 길이, 배열 문자열 입력

		vector<int> arr = parseArrayString(arrStr);
		bool isReverse = false;
		int front = 0, back = 0;
		bool isError = false;
		for (auto op : ops) {
			if (op == 'R') {
				isReverse = !isReverse;  // 뒤집기 연산 시 플래그만 변경
			}
			else if (op == 'D') {
				if (front + back >= arr.size()) {
					cout << "error
";
					isError = true;
					break;
				}
				if (isReverse) {
					back++;  // 뒤집어진 상태에서는 뒤에서 제거
				}
				else {
					front++;  // 일반 상태에서는 앞에서 제거
				}
			}
		}

		if (isError) {
			continue;
		}

		// 결과 출력
		cout << '[';
		if (isReverse) {
			for (int i = arr.size() - back - 1; i >= front; --i) {
				cout << arr[i];
				if (i != front) cout << ',';
			}
		}
		else {
			for (int i = front; i < arr.size() - back; ++i) {
				cout << arr[i];
				if (i != arr.size() - back - 1) cout << ',';
			}
		}
		cout << "]
";
	}

	return 0;
}
```

### 소스 코드 (cc)
```cc
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <queue>

using namespace std;

string bfs(int start, int target) {
	queue<pair<int, string>> q;
	vector<bool> visited(10000, false);  // 각 숫자에 대한 방문 여부를 기록

	q.push({ start, "" });
	visited[start] = true;

	while (!q.empty()) {
		int cur = q.front().first;
		string op = q.front().second;
		q.pop();

		// 목표 숫자에 도달했을 때
		if (cur == target) {
			return op;
		}

		int next;

		// D 연산
		next = (cur * 2) % 10000;
		if (!visited[next]) {
			visited[next] = true;
			q.push({ next, op + "D" });
		}

		// S 연산
		next = cur == 0 ? 9999 : cur - 1;
		if (!visited[next]) {
			visited[next] = true;
			q.push({ next, op + "S" });
		}

		// L 연산
		next = (cur % 1000) * 10 + (cur / 1000);
		if (!visited[next]) {
			visited[next] = true;
			q.push({ next, op + "L" });
		}

		// R 연산
		next = (cur % 10) * 1000 + (cur / 10);
		if (!visited[next]) {
			visited[next] = true;
			q.push({ next, op + "R" });
		}
	}
}

int main() {
	ios_base::sync_with_stdio(false);
	cin.tie(0); cout.tie(0);

	int T;
	cin >> T;
	while (T--) {
		int cur, dest;
		cin >> cur >> dest;
		string ans = bfs(cur, dest);
		cout << ans << "
";
	}

	return 0;
}
```

### 소스 코드 (cc)
```cc
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

### 소스 코드 (cc)
```cc
#include <iostream>
#include <stack>
#include <vector>
#include <string>

using namespace std;

int precedence(char op) {
	if (op == '^') return 3;
	if (op == '*' || op == '/') return 2;
	if (op == '+' || op == '-') return 1;
	return 0;
}

vector<string> shuntingYard(const string& str) {
	stack<char> operators;
	vector<string> output;
	string token = "";

	// 문자열을 순차적으로 읽어서 토큰화
	for (size_t i = 0; i < str.size(); i++) {
		char current = str[i];

		// 여는 괄호는 스택에 추가
		if (current == '(') {
			operators.push(current);
		}
		// 닫는 괄호는 여는 괄호를 만날 때까지 스택에서 연산자를 출력
		else if (current == ')') {
			while (!operators.empty() && operators.top() != '(') {
				output.push_back(string(1, operators.top()));
				operators.pop();
			}
			operators.pop(); // 여는 괄호 제거
		}
		// 피연산자 확인
		else if (precedence(current)==0) {
			token += current;
			output.push_back(token); 
			token = "";
		}
		// 연산자 처리
		else {
			while (!operators.empty() &&
				(precedence(operators.top()) > precedence(current) ||(precedence(operators.top()) == precedence(current)))) {output.push_back(string(1, operators.top()));
				operators.pop();
			}
			operators.push(current);
		}
	}

	// 스택에 남은 연산자 모두 출력
	while (!operators.empty()) {
		output.push_back(string(1, operators.top()));
		operators.pop();
	}

	return output;
}

int main() {
	string str; 
	cin >> str;

	vector<string> result = shuntingYard(str);

	for (const string& token : result) {
		cout << token ;
	}
	cout << endl;

	return 0;
}
```

### 소스 코드 (cc)
```cc
#include<bits/stdc++.h>
using namespace std;

int main(){
    int n,m;
    cin >>n>>m;
    
    vector<vector<int>> d (n+2,vector<int>(n+2,10000001));
    while(m--){
        int a,b,c;
        cin >>a>>b>>c;
        d[a][b]=min(d[a][b],c);
    }
    for(int i=1;i<=n;i++){
        d[i][i]=0;
    }
    for(int i=1;i<=n;i++){
        for(int j=1;j<=n;j++){
            for(int k=1;k<=n;k++){
                d[j][k]=min(d[j][k],d[j][i]+d[i][k]);
            }
        }
    }
    for(int i=1;i<=n;i++){
        for(int j=1;j<=n;j++){
            if(d[i][j]>10000000) cout <<"0 ";
            else cout <<d[i][j]<<' ';
        }
        cout <<'
';
    }
}
```

### 소스 코드 (cc)
```cc
#include <bits/stdc++.h>

using namespace std;

int main(){
    priority_queue<int, vector<int>, greater<int>> pq;
    int n;
    cin >> n;
    while(n--){
        int input;
        cin >> input;
        pq.push(input);
    }
    int sum=0;
    while(pq.size()>1){
        int a=pq.top(); pq.pop();
        int b=pq.top(); pq.pop();
        sum+=a+b;
        pq.push(a+b);
    }
    cout << sum;
}
```

### 소스 코드 (cc)
```cc
#include <iostream>
#include <cmath>
#include <cstdio>
using namespace std;
int main() {
    int num;
    scanf("%d",&num);
    for(int i = 0; i < num; i++)
    {
        long long x,y;
        long long move,max = 0;
        cin >>x>>y;
        while(max*max <= y-x)
            max++;
        max--;
        move = 2*max -1;
        long long rem = y-x - max*max;
        rem = (long long)ceil((double)rem / (double)max);
        move += rem;
        printf("%lld
",move);
    }
}
```

### 소스 코드 (cc)
```cc
#include <iostream>
#include <string>
#include <algorithm>
#include <unordered_map>

using namespace std;

unordered_map<string, string> union_find;
unordered_map<string, int> group_size;

string find(string tar)
{
	// 현재 노드가 루트가 아니면, 루트를 찾을 때까지 재귀적으로 호출
	if (tar != union_find[tar]) {
		// 경로 압축: 루트를 찾고, 현재 노드의 상위 노드를 루트로 갱신
		union_find[tar] = find(union_find[tar]);
	}
	return union_find[tar];
}

void setUnion(string a, string b)
{
	string t1 = find(a);
	string t2 = find(b);

	if (t1 != t2)
	{
		// Union by Size
		if (group_size[t1] < group_size[t2]) {
			swap(t1, t2); // t1이 항상 큰 그룹이 되도록 보장
		}
		union_find[t2] = t1;
		group_size[t1] += group_size[t2];
	}
}

int main()
{
	ios::sync_with_stdio(false); cin.tie(NULL); cout.tie(NULL);
	int tc;
	cin >> tc;
	while (tc--) {
		int n;
		cin >> n;

		// 초기화
		union_find.clear();
		group_size.clear();

		while (n--) {
			string parent, child;
			cin >> parent >> child;

			// 처음 입력된 parent
			if (union_find.find(parent) == union_find.end()) {
				union_find[parent] = parent;
				group_size[parent] = 1;
			}

			// 처음 입력된 child
			if (union_find.find(child) == union_find.end()) {
				union_find[child] = child;
				group_size[child] = 1;
			}

			setUnion(parent, child);

			// 합친 후 부모의 그룹 크기를 출력
			cout << group_size[find(parent)] << '
';
		}
	}
	return 0;
}
```

### 소스 코드 (cc)
```cc
#include <iostream>
#include <vector>
#include <queue>
using namespace std;

int T, N, M;

int main() {
	std::ios::sync_with_stdio(0);
	std::cin.tie(0); std::cout.tie(0);

	cin >> T;

	while (T--) {
		cin >> N;

		vector<vector<int>> adj(N + 1);
		vector<int> degree(N + 1);
		vector<int> lastYear(N + 1);  // 작년 순위 저장

		// 작년 순위 입력
		for (int i = 1; i <= N; i++) {
			cin >> lastYear[i];
		}

		// 작년 순위에 맞춰 간선 및 진입 차수 설정
		for (int i = 1; i <= N; i++) {
			for (int j = i + 1; j <= N; j++) {
				adj[lastYear[i]].push_back(lastYear[j]);
				degree[lastYear[j]]++;
			}
		}

		// 역전 정보 입력
		cin >> M;
		for (int i = 0; i < M; i++) {
			int a, b;
			cin >> a >> b;

			// a -> b 방향이 있었다면 b -> a로 뒤집어야 하고,
			// b -> a 방향이 있었다면 a -> b로 뒤집어야 함
			bool swapped = false;
			for (int j = 0; j < adj[a].size(); j++) {
				if (adj[a][j] == b) {
					adj[a].erase(adj[a].begin() + j);
					degree[b]--;
					adj[b].push_back(a);
					degree[a]++;
					swapped = true;
					break;
				}
			}

			if (!swapped) {
				for (int j = 0; j < adj[b].size(); j++) {
					if (adj[b][j] == a) {
						adj[b].erase(adj[b].begin() + j);
						degree[a]--;
						adj[a].push_back(b);
						degree[b]++;
						break;
					}
				}
			}
		}

		// 위상 정렬 수행
		queue<int> q;
		vector<int> result;

		for (int i = 1; i <= N; i++) {
			if (degree[i] == 0) {
				q.push(i);
			}
		}

		bool certain = true; // 순위가 확실한지 체크
		bool cycle = false;  // 사이클 발생 여부 체크

		for (int i = 0; i < N; i++) {
			if (q.size() == 0) {
				cycle = true; // 큐가 비어있으면 사이클 발생
				break;
			}
			if (q.size() > 1) {
				certain = false; // 큐에 여러 팀이 있으면 순위를 확정할 수 없음
			}

			int cur = q.front();
			q.pop();
			result.push_back(cur);

			for (int next : adj[cur]) {
				degree[next]--;
				if (degree[next] == 0) {
					q.push(next);
				}
			}
		}

		if (cycle) {
			cout << "IMPOSSIBLE
";
		} else if (!certain) {
			cout << "?
";
		} else {
			for (int x : result) {
				cout << x << " ";
			}
			cout << "
";
		}
	}

	return 0;
}
```

### 소스 코드 (cc)
```cc
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

int main() {
	ios_base::sync_with_stdio(false);
	cin.tie(0);
	
	int N, M;  
	cin >> N >> M;
	vector<vector<int>> arr(N + 2,vector<int>(N + 2, 0));
	vector<vector<int>> sum(N + 2, vector<int>(N + 2, 0));

	for (int i = 1; i <= N; i++)
	{
		for (int j = 1; j <= N; j++)
		{
			cin >> arr[i][j];
			sum[i][j] = arr[i][j] + sum[i - 1][j] + sum[i][j - 1] - sum[i - 1][j - 1];
		}
	}

	for (int i = 0; i < M; i++)
	{
		int fromX, fromY, toX, toY;
		cin >> fromX >> fromY >> toX >> toY;
		cout << sum[toX][toY] - sum[fromX - 1][toY] - sum[toX][fromY - 1] + sum[fromX - 1][fromY - 1] << "
";
    }
	
	return 0;
}
```

### 소스 코드 (cc)
```cc
#include <bits/stdc++.h>

using namespace std;

int main()
{
  int count;
  cin >> count;
  while (count--)
  {
    string str;
    cin >> str;

    list<char> editer;
    auto cursor = editer.end();

    for (auto c : str)
    {
      switch (c)
      {
      case '<':
      {
        if (cursor != editer.begin())
          cursor--;
        break;
      }
      case '>':
      {
        if (cursor != editer.end())
          cursor++;
        break;
      }
      case '-':
      {
        if (cursor != editer.begin())
        {
          cursor--;
          cursor = editer.erase(cursor);
        }
        break;
      }
      default:
      {
        editer.insert(cursor, c);
        break;
      }
      }
    }
    for (auto c: editer) cout << c;
    cout << endl;
  }
}
```

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

### 소스 코드 (cc)
```cc
#include <iostream>
#include <vector>
#include <queue>

using namespace std;

void bfs(int st, const vector<vector<int>>& node, vector<bool>& visited) {
	queue<int> q;
	q.push(st);
	visited[st] = true;

	while (!q.empty()) {
		int cur = q.front();
		q.pop();

		for (int i = 0; i < node[cur].size(); i++) {
			int next = node[cur][i];
			if (visited[next]) continue;
			visited[next] = true;
			q.push(next);
		}
	}
}

int main() {
	ios_base::sync_with_stdio(false); cin.tie(0); cout.tie(0);

	int N, M;
	cin >> N >> M;

	vector<vector<int>> node(N + 1);
	vector<bool> visited(N + 1, false);

	int res = 0;

	for (int i = 0; i < M; i++) {
		int from, to;
		cin >> from >> to;
		node[from].push_back(to);
		node[to].push_back(from);
	}

	for (int i = 1; i <= N; i++) {
		if (!visited[i]) {
			bfs(i, node, visited);
			res++;
		}
	}

	cout << res << "
";

	return 0;
}
```

### 소스 코드 (cc)
```cc
#include <bits/stdc++.h>
using namespace std;

int dx[4] = {1, 0, -1, 0};
int dy[4] = {0, 1, 0, -1};

int main()
{
  int Tcase;
  cin >> Tcase;
  while (Tcase--)
  {
    int res = 0;

    int x, y, cnt;
    cin >> x >> y >> cnt;

    int board[50][50] = {0};
    bool vis[50][50] = {false};

    while (cnt--)
    {
      int a, b;
      cin >> a >> b;
      board[a][b] = 1;
    }
    
    for (int i = 0; i < x; i++)
    {
      for (int j = 0; j < y; j++)
      {
        if (board[i][j] == 1 && vis[i][j] == 0)
        {
          res++;
          queue<pair<int, int>> q;
          vis[i][j] = 1;
          q.push({i, j});
          int size = 0;
          while (!q.empty())
          {
            size++;
            auto cur = q.front();
            q.pop();
            for (int dir = 0; dir < 4; dir++)
            {
              int nx = cur.first + dx[dir];
              int ny = cur.second + dy[dir];
              if (nx < 0 || nx >= x || ny < 0 || ny >= y)
                continue;
              if (vis[nx][ny] || board[nx][ny] != 1)
                continue;
              vis[nx][ny] = 1;
              q.push({nx, ny});
            }
          }
        }
      }
    }
    cout << res << '
';
  }
}
```

### 소스 코드 (cc)
```cc
#include<iostream>
#include<vector>
#include<string>
#include<algorithm>
#include<unordered_map>


using namespace std;

int res = 21e8;

void solve(int num, int opCnt) {
	if (num <= 0) return;
	if (num == 1) {
		res = min(opCnt, res);
		return;
	}
	if (opCnt >= res) return;

	if (num % 3 == 0) solve(num / 3, opCnt + 1);
	if (num % 2 == 0) solve(num / 2, opCnt + 1);
	solve(num - 1, opCnt + 1);
}

int main() {
	int num;
	cin >> num;
	solve(num, 0);
	cout << res;
	return 0;
}
```

### 소스 코드 (cc)
```cc
#include <iostream>
#include <algorithm>
#include <vector>
using namespace std;
int n;
vector <pair<int,int>> v;
vector <int> ans(1000001);
int main(){
    cin >> n;
    for(int i =0; i<n; i++){
        int x;
        cin >> x;
        v.push_back({x,i});
    }
    sort(v.begin(),v.end());
    
    int pivot = v[0].first;
    int cnt = 0;
    ans[v[0].second] = 0;

    for(int i = 1; i < n ; i++){
        if(pivot==v[i].first){
            ans[v[i].second] = cnt;
        }else {
            ans[v[i].second] = ++cnt;
            pivot = v[i].first;
        }
    }
    for(int i = 0; i < n ; i++){
        cout << ans[i] << ' ';
    }
}
```

### 소스 코드 (cc)
```cc
#include <bits/stdc++.h>
using namespace std;

int board[502][502]; 
bool vis[502][502]; 
int dx[4] = {1,0,-1,0};
int dy[4] = {0,1,0,-1}; 

int main() {
    int n,m;
    int cnt=0;
    int max_size=0;
    
    cin >> n >> m;
    
    for(int i = 0; i < n; i++)
        for(int j = 0; j < m; j++)
            cin >> board[i][j];
    
    for(int i = 0; i < n; i++){
        for(int j = 0; j < m; j++){
            if(board[i][j]==1 && vis[i][j]==0){
                cnt++;
                queue<pair<int,int> > q;
                vis[i][j]=1;
                q.push({i,j});
                int size=0;
                while(!q.empty()){
                    size++;
                    auto cur = q.front();
                    q.pop();
                    for(int dir=0; dir<4; dir++){
                        int nx= cur.first+dx[dir];
                        int ny= cur.second+dy[dir];
                        if(nx < 0 || nx >= n || ny < 0 || ny >= m) continue;
                        if(vis[nx][ny] || board[nx][ny] != 1) continue; 
                        vis[nx][ny]=1;
                        q.push({nx,ny});
                    }
                }
                max_size=max(size,max_size);
            }
        }
    }
     cout << cnt << '
' << max_size;
}
```

### 소스 코드 (cc)
```cc
#include <iostream>
#include <set>

using namespace std;

int main() {
	ios_base::sync_with_stdio(false); cin.tie(0); cout.tie(0);

	int N, M;
	cin >> N >> M;
	multiset<int> set;
	while (M + N--) {
		int num;
		cin >> num;
		set.insert(num);
	}

	for (auto num : set)
	{
		cout << num << " ";
	}
}
```

### 소스 코드 (cc)
```cc
#include<iostream>
#include<vector>
#include<string>
#include<algorithm>
#include<unordered_map>


using namespace std;

int main() {
	int a, b;
	cin >> a >> b;
	string person;
	unordered_map<string,int> people;
	vector<string> ans;

	for (int i = 0; i < a; i++) {
		cin >> person;
		people[person]++;
	}
	for (int i = 0; i < b; i++)
	{
		cin >> person;
		if (people[person])
			ans.push_back(person);
	}
	sort(ans.begin(), ans.end());
	cout << ans.size() << "
";
	for (int i = 0; i < ans.size(); i++)
	{
		cout << ans[i] << "
";
	}
}
```

### 소스 코드 (cc)
```cc
#include<iostream>
#include<vector>
#include<string>
#include<algorithm>

using namespace std;

int main() {
	int ans = 0;
	int N;
	cin >> N;
	vector<int> time(N);
	for (int i = 0; i < N; i++)
	{
		cin >> time[i];
	}
	sort(time.begin(), time.end());
	for (int i = N-1; i >= 0; i--)
	{
		ans += time[i] * (N-i);
	}
	cout << ans;
	return 0;
}
```

### 소스 코드 (cc)
```cc
#include <iostream>
#include <map>

using namespace std;

int main() {
	ios_base::sync_with_stdio(false); cin.tie(0); cout.tie(0);

	int N;
	cin >> N;
	map<int, int> map;
	while (N--) {
		int num;
		cin >> num;
		map[num]++;
	}

	for (auto i : map)
	{
		while (i.second--) {
			cout << i.first << "
";
		}
	}

}
```

### 소스 코드 (cc)
```cc
#include <iostream>
#define MAX 9
using namespace std;

int n,m;
int arr[MAX] = {0,};
bool visited[MAX] = {0,};

void dfs(int cnt)
{
    if(cnt == m)
    {
        for(int i = 0; i < m; i++)
            cout << arr[i] << ' ';
        cout << '
';
        return;
    }
    for(int i = 1; i <= n; i++)
    {
        if(!visited[i])
        {
            visited[i] = true;
            arr[cnt] = i;
            dfs(cnt+1);
            visited[i] = false;
        }
    }
}

int main() {
    cin >> n >> m;
    dfs(0);
}
```

### 소스 코드 (cc)
```cc
#include <iostream>

using namespace std;

int N, M;
int from, to;
int res = 0;
int arr[100002];
int sum[100002];

int main() {
	ios::sync_with_stdio(false);
	cin.tie(NULL); cout.tie(NULL);

	cin >> N >> M;
	arr[0] = 0;
	sum[0] = 0;
	for (int i = 1; i <= N; i++)
	{
		cin >> arr[i];
		sum[i] = sum[i - 1] + arr[i];
	}
	for (int i = 0; i < M; i++)
	{
		cin >> from >> to;
		cout << sum[to] - sum[from - 1] << '
';
	}
}
```

### 소스 코드 (cc)
```cc
#include <iostream>
#include <string>

using namespace std;

int bitmask = 0; // 비트 연산을 위한 변수

int main() {
	ios::sync_with_stdio(0);
	cin.tie(0);

	int n;
	cin >> n;
	while (n--) {
		string command;
		int num;
		cin >> command;

		if (command == "add") {
			cin >> num;
			int bit_to_set = (1 << (num - 1)); // num 번째 비트를 1로 설정하기 위한 비트 마스크 계산
			bitmask = bitmask | bit_to_set; // 비트 연산: OR 연산을 통해 num 번째 비트를 1로 설정
		}
		else if (command == "remove") {
			cin >> num;
			int bit_to_clear = (1 << (num - 1)); // num 번째 비트를 0으로 설정하기 위한 비트 마스크 계산
			bitmask = bitmask & (~bit_to_clear); // 비트 연산: AND 연산과 NOT 연산을 통해 num 번째 비트를 0으로 설정
		}
		else if (command == "check") {
			cin >> num;
			int bit_to_check = (1 << (num - 1)); // num 번째 비트를 확인하기 위한 비트 마스크 계산
			int check_result = (bitmask & bit_to_check); // 비트 연산: AND 연산을 통해 num 번째 비트가 1인지 확인
			int is_set = (check_result >> (num - 1)); // 비트 연산: 오른쪽으로 쉬프트하여 비트 값 추출 (0 또는 1)
			cout << is_set << "
"; // 비트 값 출력 (1 또는 0)
		}
		else if (command == "toggle") {
			cin >> num;
			int bit_to_toggle = (1 << (num - 1)); // num 번째 비트를 반전시키기 위한 비트 마스크 계산
			bitmask = bitmask ^ bit_to_toggle; // 비트 연산: XOR 연산을 통해 num 번째 비트를 반전
		}
		else if (command == "all") {
			bitmask = 0xFFFFF; // 모든 비트를 1로 설정 (20 비트의 모든 비트를 1로 설정)
		}
		else if (command == "empty") {
			bitmask = 0; // 모든 비트를 0으로 설정
		}
	}
	return 0;
}
```

### 소스 코드 (cc)
```cc
#include<iostream>

using namespace std;

int main(){
    

    char a;
    cin>>a;
    cout<<static_cast<int>(a)<<'
';

    return 0;}
```

### 소스 코드 (cc)
```cc
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

int main() {
    int n, m;
    cin >> n >> m;  // n: 바구니의 개수, m: 명령의 개수

    vector<int> baskets(n);

    // 바구니 초기화
    for (int i = 0; i < n; ++i) {
        baskets[i] = i + 1;
    }

    // m개의 명령 처리
    for (int i = 0; i < m; ++i) {
        int begin, end, mid;
        cin >> begin >> end >> mid;
        begin--;  // 인덱스는 0부터 시작하므로 1 감소
        end--;   // 인덱스는 0부터 시작하므로 1 감소
        mid--;   // 인덱스는 0부터 시작하므로 1 감소

        // 부분 배열 회전
        rotate(baskets.begin() + begin, baskets.begin() + mid, baskets.begin() + end + 1);
    }

    // 최종 배열 출력
    for (int i = 0; i < n; ++i) {
        cout << baskets[i] << " ";
    }
    cout << endl;

    return 0;
}
```

### 소스 코드 (cc)
```cc
#include <iostream>
#include <cmath>
using namespace std;

int main() {
    ios::sync_with_stdio(false); cin.tie(NULL); cout.tie(NULL);
    int T,H,W,N,room; cin>>T;
    while(T!=0){
    cin>>H>>W>>N;
    if(N%H==0)
        room=H*100+ceil((double)N/H);
    else
        room=(N%H)*100+ceil((double)N/H);
    cout<<room<<'
';
    T--;
    }
}
```

### 소스 코드 (cc)
```cc
#include<iostream>
using namespace std;


int main (){
ios_base :: sync_with_stdio(false);
cin.tie(NULL);
cout.tie(NULL);


    int tmp,res=0; 
    int list[10]={};
    for(int i=0;i<10;i++){
        cin>>tmp;
        list[i]=tmp%42;
    }
    for(int i=0;i<10;i++){
        for(int j=i+1;j<10;j++){
            if(list[i]==list[j]){
                res++;
                list[j]=-j;
            }
        }
    }
    
    cout<<10-res;
    return 0;
}
```
