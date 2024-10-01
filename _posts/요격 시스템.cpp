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