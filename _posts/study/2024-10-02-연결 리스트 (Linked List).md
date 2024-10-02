---
layout: post
title: "연결 리스트 (Linked List)"
date: 2024-10-02 10:24:00+0900
categories: [Study, Data Structure]
tags: [Linked List]
---

## 연결 리스트란?

**각 요소가 다른 요소를 가리키는 포인터를 포함하는** 선형 자료구조로, 배열과 달리 **비연속적인 메모리**에 저장됨  
각 요소를 **노드(Node)**라고 하며, 각 노드는 데이터를 저장하는 필드와 다음 노드를 가리키는 포인터로 구성  

### 주요 특징
1. **동적 크기**: 연결 리스트는 크기가 동적으로 조정되어, 배열과 달리 미리 크기를 선언할 필요가 없음
2. **포인터 기반 연결**: 각 노드가 다른 노드를 가리키며, 비연속적인 메모리 위치에서 데이터를 관리할 수 있음 
3. **순차 접근**: 배열과 달리 인덱스를 사용한 직접 접근이 불가능하며, 순차적으로 접근해야 함

### 장점
1. **동적 메모리 관리**: 필요한 만큼의 메모리만 할당할 수 있어, 메모리 낭비가 적음
2. **삽입 및 삭제의 용이성**: 배열은 삽입과 삭제 시 데이터 이동이 필요하지만, 연결 리스트는 노드 연결만 변경하면 되므로 효율적

### 단점
1. **순차 접근**: 임의의 위치에 접근하기 위해서는 처음부터 순차적으로 탐색해야 하므로, 배열보다 접근 시간이 느림
2. **추가 메모리 사용**: 각 노드가 데이터 외에도 다음 노드를 가리키는 포인터를 저장해야 하므로, 추가적인 메모리 공간이 필요

### Array와 비교

| 배열 | 연결 리스트 |
|:----:|:----------:|
| 정적 크기 | 동적 크기 |
| 인덱스를 통한 빠른 접근 O(1) | 순차 접근 O(n) |
| 삽입/삭제 시 데이터 이동 O(n) | 삽입/삭제가 효율적 O(1) |
| 메모리 사용 효율적 | 포인터 때문에 추가 메모리 필요 |
  

> 연결 리스트는 동적 크기를 지원하며, 삽입과 삭제가 빈번하게 일어나는 상황에 적합한 자료구조  
{: .prompt-tip}  

## 주요 연산
1. **삽입 (Insert)**:
    - 연결 리스트의 처음 또는 중간에 새로운 노드를 삽입
    - 시간 복잡도: O(1) (처음에 삽입), O(n) (중간에 삽입)
  
2. **삭제 (Delete)**:
    - 리스트의 특정 노드를 삭제
    - 시간 복잡도: O(n) (노드를 찾고 연결을 재설정)

3. **탐색 (Search)**:
    - 리스트에서 특정 값을 가진 노드를 찾음
    - 시간 복잡도: O(n)

4. **업데이트 (Update)**:
    - 리스트의 특정 노드의 값을 변경
    - 시간 복잡도: O(n)

## 종류  

1. **단일 연결 리스트 (Singly Linked List)**: 각 노드가 하나의 다음 노드만을 가리킴
2. **이중 연결 리스트 (Doubly Linked List)**: 각 노드가 다음 노드와 이전 노드를 모두 가리키므로, 양방향 탐색이 가능
3. **원형 연결 리스트 (Circular Linked List)**: 마지막 노드가 처음 노드를 가리켜, 리스트의 끝과 시작이 연결

## 구현 예시  

### 단일 연결 리스트  

```cpp
#include <iostream>
using namespace std;

struct Node {
    int data;
    Node* next;
};

class LinkedList {
public:
    Node* head;

    LinkedList() {
        head = nullptr;
    }

    // 새로운 노드를 리스트 앞에 삽입
    void insert(int newData) {
        Node* newNode = new Node();
        newNode->data = newData;
        newNode->next = head;
        head = newNode;
    }

    // 리스트 출력
    void printList() {
        Node* temp = head;
        while (temp != nullptr) {
            cout << temp->data << " -> ";
            temp = temp->next;
        }
        cout << "null" << endl;
    }
};

int main() {
    LinkedList list;
    list.insert(10);
    list.insert(20);
    list.insert(30);
    list.printList();  // 30 -> 20 -> 10 -> null

    return 0;
}
```


### 이중 연결 리스트  

```cpp
#include <iostream>
using namespace std;

struct Node {
    int data;
    Node* next;
    Node* prev;  // 이전 노드를 가리키는 포인터
};

class DoublyLinkedList {
public:
    Node* head;

    DoublyLinkedList() {
        head = nullptr;
    }

    // 리스트의 끝에 새로운 노드 삽입
    void append(int newData) {
        Node* newNode = new Node();
        newNode->data = newData;
        newNode->next = nullptr;
        newNode->prev = nullptr;

        if (head == nullptr) {  // 리스트가 비어있는 경우
            head = newNode;
        } else {
            Node* temp = head;
            while (temp->next != nullptr) {
                temp = temp->next;  // 리스트의 끝으로 이동
            }
            temp->next = newNode;  // 새로운 노드를 끝에 연결
            newNode->prev = temp;  // 이전 노드를 설정
        }
    }

    // 리스트 출력 (앞에서 뒤로)
    void printList() {
        Node* temp = head;
        while (temp != nullptr) {
            cout << temp->data << " <-> ";
            temp = temp->next;
        }
        cout << "null" << endl;
    }

    // 리스트 역순 출력 (뒤에서 앞으로)
    void printReverse() {
        Node* temp = head;
        while (temp && temp->next != nullptr) {
            temp = temp->next;  // 리스트의 끝으로 이동
        }
        while (temp != nullptr) {
            cout << temp->data << " <-> ";
            temp = temp->prev;  // 이전 노드로 이동
        }
        cout << "null" << endl;
    }
};

int main() {
    DoublyLinkedList list;
    list.append(10);
    list.append(20);
    list.append(30);
    list.printList();       // 10 <-> 20 <-> 30 <-> null
    list.printReverse();    // 30 <-> 20 <-> 10 <-> null

    return 0;
}
```


###  원형 연결 리스트  

```cpp
#include <iostream>
using namespace std;

struct Node {
    int data;
    Node* next;
};

class CircularLinkedList {
public:
    Node* head;

    CircularLinkedList() {
        head = nullptr;
    }

    // 리스트의 끝에 새로운 노드 삽입
    void append(int newData) {
        Node* newNode = new Node();
        newNode->data = newData;

        if (head == nullptr) {  // 리스트가 비어있는 경우
            head = newNode;
            newNode->next = head;  // 자기 자신을 가리킴 (원형 구조)
        } else {
            Node* temp = head;
            while (temp->next != head) {  // 마지막 노드로 이동
                temp = temp->next;
            }
            temp->next = newNode;  // 새로운 노드를 끝에 연결
            newNode->next = head;  // 마지막 노드가 처음 노드를 가리킴
        }
    }

    // 리스트 출력
    void printList() {
        if (head == nullptr) return;  // 리스트가 비어있으면 종료

        Node* temp = head;
        do {
            cout << temp->data << " -> ";
            temp = temp->next;
        } while (temp != head);  // 처음 노드로 돌아올 때까지 반복
        cout << "(head)" << endl;
    }
};

int main() {
    CircularLinkedList list;
    list.append(10);
    list.append(20);
    list.append(30);
    list.printList();  // 10 -> 20 -> 30 -> (head)

    return 0;
}
```

### 라이브러리

>C++의 표준 라이브러리 <list>를 사용하면 쉽게 이중 연결 리스트를 구현할 수 있음   
**std::list** 는 이중 연결 리스트로 구현되어 있어, 양방향 순회가 가능하고 효율적인 삽입과 삭제 연산을 제공
{: .prompt-info}  

```cpp
#include <iostream>
#include <list>
using namespace std;

int main() {
    // 이중 연결 리스트 생성
    list<int> myList;

    // 요소 추가
    myList.push_back(10);
    myList.push_back(20);
    myList.push_front(5);  // 리스트의 앞에 추가

    // 리스트 순회 및 출력
    cout << "리스트 내용: ";
    for (const auto& item : myList) {
        cout << item << " ";
    }
    cout << endl;

    // 특정 위치에 요소 삽입
    auto it = myList.begin();
    advance(it, 2);  // 두 번째 위치로 이동
    myList.insert(it, 15);

    // 요소 삭제
    myList.pop_front();  // 첫 번째 요소 삭제

    // 수정된 리스트 출력
    cout << "수정된 리스트: ";
    for (const auto& item : myList) {
        cout << item << " ";
    }
    cout << endl;

    // 리스트 크기
    cout << "리스트 크기: " << myList.size() << endl;

    return 0;
}
```
