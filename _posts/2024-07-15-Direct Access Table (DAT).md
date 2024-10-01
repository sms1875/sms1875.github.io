---
layout: post
title: "Direct Access Table (DAT)"
date: 2024-07-25 14:39:00
categories: [Study, Data Structure]
tags: [DAT, Direct Access Table]
---
## DAT(Direct Access Table)란?
DAT는 값을 인덱스로 활용하는 자료구조<br>
Java의 객체나 파이썬의 딕셔너리처럼, DAT를 사용하면 키의 존재 유무 확인, 개수 카운트 등이 가능<br>
C++에서는 아래와 같은 방식으로 사용

```cpp
int bucket[200]; 
bucket['A'] = 1;
```

## DAT의 활용 예시

배열에 어떤 종류의 알파벳이 있는지 찾아내는 문제

```
예시 배열: A D B F A D

출력 결과: ABDF
```

일반적인 방법은 중첩 for문을 사용

```cpp
#include <iostream>
using namespace std;

char arr[10] = "ADBFAD";

int main() {
    for (int i = 0; arr[i] != '\0'; i++) {
        bool isUnique = true;
        for (int j = 0; j < i; j++) {
            if (arr[i] == arr[j]) {
                isUnique = false;
                break;
            }
        }
        if (isUnique) {
            cout << arr[i];
        }
    }
    return 0;
}​
```

하지만 DAT를 사용하면 중첩 for문 없이도 문제를 해결할 수 있음.

```cpp
#include <iostream>
using namespace std;

int main() {
    int bucket[200] = { 0 };
    char vect[7] = "ABDFAD";

    for (int i = 0; i < 6; i++) {
        bucket[vect[i]] = 1;
    }

    for (int i = 0; i < 200; i++) {
        if (bucket[i] == 1) cout << (char)(i);
    }
    return 0;
}
```

DAT의 특징으로는 자동으로 오름차순 정렬이 됨

## DAT의 응용

DAT는 문자 인덱스뿐만 아니라 숫자 인덱스 카운트 등이 가능

```
예시 배열: 4 1 1 1 5 4

출력 결과:

1: 3개
4: 2개
5: 1개
```

DAT 활용 예시

```cpp
#include <iostream>
using namespace std;

int main() {
    int bucket[200] = { 0 };
    int vect[6] = { 4, 1, 1, 1, 5, 4 };

    for (int i = 0; i < 6; i++) {
        bucket[vect[i]]++;
    }

    for (int i = 0; i < 200; i++) {
        if (bucket[i] > 0) {
            cout << i << " : " << bucket[i] << "개\n";
        }
    }
    return 0;
}
```

DAT는 플래그 용도뿐만 아니라 카운트를 저장하는데도 사용가능
