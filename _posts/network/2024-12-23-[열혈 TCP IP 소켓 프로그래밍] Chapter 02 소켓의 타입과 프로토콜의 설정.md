---
layout: post
title: "[열혈 TCP IP 소켓 프로그래밍] Chapter 02 소켓의 타입과 프로토콜의 설정"
date: 2024-12-23 20:25:00+0900
categories: [Study, Network]
tags: [열혈 TCP IP 소켓 프로그래밍, Socket, Network, TCP/IP]
---
## **프로토콜과 소켓**

### **프로토콜이란?**

- 컴퓨터 간 데이터를 송수신하기 위한 통신 규약
- 소켓을 생성할 때, 데이터 전송 방식을 결정하는 주요 요소
- **구성**
    - **domain**: 사용되는 프로토콜 체계 (예: IPv4, IPv6)
    - **type**: 데이터 전송 방식 (예: TCP, UDP)
    - **protocol**: 특정 전송 방식 (예: IPPROTO_TCP, IPPROTO_UDP)

### **소켓 생성 함수**

```c
#include <sys/socket.h>

int socket(int domain, int type, int protocol);
```

- **성공 시**: 소켓 파일 디스크립터 반환
- **실패 시**: -1 반환
- `domain`, `type`, `protocol`로 소켓의 데이터 송수신 방법을 정의


### **프로토콜 체계 (Protocol Family)**

- 프로토콜 종류에 따라 체계화된 분류
- **대표적인 프로토콜 체계**
    
    
    | **이름** | **설명** |
    | --- | --- |
    | PF_INET | IPv4 인터넷 프로토콜 체계 |
    | PF_INET6 | IPv6 인터넷 프로토콜 체계 |
    | PF_LOCAL | 로컬 통신을 위한 UNIX 체계 |
    | PF_PACKET | Low Level 소켓 체계 |
    | PF_IPX | IPX 노벨 프로토콜 체계 |


### **소켓의 타입**

- **소켓 타입**은 데이터를 송수신하는 방식에 따라 결정
- **대표적인 타입**
    - **연결 지향형 (TCP)**: 안정적 데이터 전송
    - **비연결 지향형 (UDP)**: 빠르고 효율적인 전송


## **TCP와 UDP의 비교**

### **TCP 소켓**

- **특징**
    - 데이터 전송 보장 및 순서 보장
    - 데이터를 **스트림(stream)**으로 처리하여 경계 없음
    - **1대1 연결 구조**
    - 한 번 보낸 데이터와 읽은 데이터의 호출 횟수가 다를 수 있음
- **사용 사례**: HTTP, 파일 전송, 이메일

### **UDP 소켓**

- **특징**
    - 데이터 전송 보장 없음
    - 데이터를 **패킷(packet)** 단위로 처리하며 경계가 존재
    - 순서 보장 없음
    - 빠른 전송 속도
    - 데이터 크기 제한이 있음
- **사용 사례**: 스트리밍, VoIP, 온라인 게임

| **특성** | **TCP** | **UDP** |
| --- | --- | --- |
| 데이터 전송 보장 | O | X |
| 데이터 순서 보장 | O | X |
| 데이터 경계 존재 여부 | X | O |
| 속도 | 느림 | 빠름 |
| 연결 구조 | 1대1 연결 | 연결 없음 |


### **TCP 소켓 생성**

```c
int tcp_socket = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP);
```

- `PF_INET`: IPv4 체계
- `SOCK_STREAM`: 연결 지향형
- `IPPROTO_TCP`: TCP 프로토콜

### **UDP 소켓 생성**

```c
int udp_socket = socket(PF_INET, SOCK_DGRAM, IPPROTO_UDP);
```

- `SOCK_DGRAM`: 비연결 지향형
- `IPPROTO_UDP`: UDP 프로토콜


### **TCP 서버 코드**

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>

void error_handling(char *message);

int main(int argc, char *argv[])
{
    int serv_sock, clnt_sock;
    struct sockaddr_in serv_addr, clnt_addr;
    socklen_t clnt_addr_size;

    char message[] = "Hello World!";

    serv_sock = socket(PF_INET, SOCK_STREAM, 0);
    bind(serv_sock, (struct sockaddr*)&serv_addr, sizeof(serv_addr));
    listen(serv_sock, 5);

    clnt_addr_size = sizeof(clnt_addr);
    clnt_sock = accept(serv_sock, (struct sockaddr*)&clnt_addr, &clnt_addr_size);

    write(clnt_sock, message, sizeof(message));
    close(clnt_sock);
    close(serv_sock);

    return 0;
}
```

### **TCP 클라이언트 코드**

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>

void error_handling(char *message);

int main(int argc, char *argv[])
{
    int sock;
    struct sockaddr_in serv_addr;
    char message[30];
    int str_len = 0, idx = 0, read_len = 0;

    sock = socket(PF_INET, SOCK_STREAM, 0);
    connect(sock, (struct sockaddr*)&serv_addr, sizeof(serv_addr));

    while ((read_len = read(sock, &message[idx++], 1)) > 0)
        str_len += read_len;

    printf("Message from server: %s \n", message);
    close(sock);

    return 0;
}
```


### **윈도우 기반 TCP 소켓 구현**

```c
#include <winsock2.h>

SOCKET socket(int af, int type, int protocol);
```

- **성공 시**: 소켓 핸들 반환
- **실패 시**: `INVALID_SOCKET` 반환
