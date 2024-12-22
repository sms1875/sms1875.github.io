---
layout: post
title: "[열혈 TCP IP 소켓 프로그래밍] Chapter 01 네트워크 프로그래밍과 소켓의 이해"
date: 2024-12-22 22:33:00+0900
categories: [Study, Network]
tags: [열혈 TCP IP 소켓 프로그래밍, Socket, Network, TCP/IP]
---
## 네트워크 프로그래밍이란?

- **네트워크로 연결된 둘 이상의 컴퓨터 사이에서 데이터를 송수신**하기 위해 소켓을 기반으로 프로그래밍하는 것
- **소켓 프로그래밍**이라고도 부른다
- **모바일 앱 개발 등** 최근 프로그래밍 환경에서는 네트워크 요소가 거의 모든 영역에 포함된다

### 소켓(Socket)

- 네트워크 상에서 데이터를 주고받기 위해 사용되는 **표준화된 소프트웨어 모듈**
- **인터넷 연결과 데이터 송수신**을 도와주는 도구로, 내부 동작 방식을 몰라도 통신이 가능
- 프로그래머가 데이터 송수신에 대한 **물리적/소프트웨어적 세부 사항**을 신경 쓰지 않도록 추상화된 구조를 제공

## 서버 소켓의 생성 과정

소켓 프로그래밍에서 **서버 소켓**은 연결 요청을 수락하기 위한 소켓을 의미

1. **소켓 생성**: `socket()` 함수 호출
2. **IP와 포트 번호 할당**: `bind()` 함수 호출
3. **연결 요청 대기 상태로 전환**: `listen()` 함수 호출
4. **연결 요청 수락**: `accept()` 함수 호출

서버는 클라이언트보다 **먼저 실행**되어야 하며, 실행 과정이 더 복잡하다

### 1. 소켓 생성

TCP 소켓은 **전화기**에 비유할 수 있음

소켓은 `socket()` 함수를 통해 생성

```c
#include <sys/socket.h>
int socket(int domain, int type, int protocol);
// 성공 시 파일 디스크립터, 실패 시 -1 반환
```

### 예제

```c
int serv_sock = socket(PF_INET, SOCK_STREAM, 0);
if (serv_sock == -1)
    error_handling("socket() error");
```

- **PF_INET**: IPv4 프로토콜 사용
- **SOCK_STREAM**: TCP 소켓 생성
- **0**: 기본 프로토콜 사용

### 2. 소켓 주소 할당

소켓에도 **주소**(IP + Port 번호)가 필요함

전화기에 번호를 부여하는 과정과 비슷하다

```c
#include <sys/socket.h>
int bind(int sockfd, struct sockaddr *myaddr, socklen_t addrlen);
// 성공 시 0, 실패 시 -1 반환
```

### 예제

```c
struct sockaddr_in serv_addr;
memset(&serv_addr, 0, sizeof(serv_addr));
serv_addr.sin_family = AF_INET;
serv_addr.sin_addr.s_addr = htonl(INADDR_ANY);
serv_addr.sin_port = htons(8080);

if (bind(serv_sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) == -1)
    error_handling("bind() error");
```

### 3. 연결 가능 상태로 변경

소켓을 **연결 요청 대기 상태**로 전환하기 위해 `listen()` 함수를 호출

```c
#include <sys/socket.h>int listen(int sockfd, int backlog);
// 성공 시 0, 실패 시 -1 반환
```

### 예제

```c
if (listen(serv_sock, 5) == -1)
    error_handling("listen() error");
```

- **backlog**: 대기 가능한 연결 요청 수

### 4. 연결 요청 수락

연결 요청이 들어오면 `accept()` 함수로 수락하고, 새로운 **클라이언트 소켓**이 생성

```c
#include <sys/socket.h>
int accept(int sockfd, struct sockaddr *addr, socklen_t *addrlen);
// 성공 시 파일 디스크립터, 실패 시 -1 반환
```

### 예제

```c
struct sockaddr_in clnt_addr;
socklen_t clnt_addr_size = sizeof(clnt_addr);

int clnt_sock = accept(serv_sock, (struct sockaddr *)&clnt_addr, &clnt_addr_size);
if (clnt_sock == -1)
    error_handling("accept() error");
```

### 서버 전체 코드

```c

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>

void error_handling(char *message);

int main(int argc, char *argv[]) {
    int serv_sock, clnt_sock;
    struct sockaddr_in serv_addr, clnt_addr;
    socklen_t clnt_addr_size;
    char message[] = "Hello World!";

    if (argc != 2) {
        printf("Usage: %s <port>\n", argv[0]);
        exit(1);
    }

    serv_sock = socket(PF_INET, SOCK_STREAM, 0);
    if (serv_sock == -1)
        error_handling("socket() error");

    memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    serv_addr.sin_port = htons(atoi(argv[1]));

    if (bind(serv_sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) == -1)
        error_handling("bind() error");

    if (listen(serv_sock, 5) == -1)
        error_handling("listen() error");

    clnt_addr_size = sizeof(clnt_addr);
    clnt_sock = accept(serv_sock, (struct sockaddr *)&clnt_addr, &clnt_addr_size);
    if (clnt_sock == -1)
        error_handling("accept() error");

    write(clnt_sock, message, sizeof(message));
    close(clnt_sock);
    close(serv_sock);
    return 0;
}

void error_handling(char *message) {
    fputs(message, stderr);
    fputc('\n', stderr);
    exit(1);
}

```

## 클라이언트 소켓의 구현

클라이언트는 **연결 요청**을 보내는 역할

과정이 간단함

### 클라이언트 연결 과정

1. **소켓 생성**: `socket()` 함수 호출
2. **서버 연결 요청**: `connect()` 함수 호출

```c
#include <sys/socket.h>
int connect(int sockfd, struct sockaddr *serv_addr, socklen_t addrlen);
// 성공 시 0, 실패 시 -1 반환
```

### 클라이언트 예제 코드

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>

void error_handling(char *message);

int main(int argc, char *argv[]) {
    int sock;
    struct sockaddr_in serv_addr;
    char message[30];
    int str_len;

    if (argc != 3) {
        printf("Usage: %s <IP> <port>\n", argv[0]);
        exit(1);
    }

    sock = socket(PF_INET, SOCK_STREAM, 0);
    if (sock == -1)
        error_handling("socket() error");

    memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = inet_addr(argv[1]);
    serv_addr.sin_port = htons(atoi(argv[2]));

    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) == -1)
        error_handling("connect() error");

    str_len = read(sock, message, sizeof(message) - 1);
    if (str_len == -1)
        error_handling("read() error");

    printf("Message from server: %s\n", message);
    close(sock);
    return 0;
}

void error_handling(char *message) {
    fputs(message, stderr);
    fputc('\n', stderr);
    exit(1);
}
```

### 실행 결과

![image.png](assets/img/posts/network/1/image.png)

## 리눅스 기반 파일 조작

리눅스에서는 **소켓도 파일로 간주되며,** 파일 입출력 함수를 사용하여 데이터를 송수신

이를 저수준 파일 입출력 방식이라고 하며, 운영체제가 제공하는 파일 디스크립터를 활용한다

---

### 1. 파일 디스크립터와 소켓

운영체제는 파일과 소켓을 구분하지 않고 동일하게 파일 디스크립터를 사용하여 관리

```markdown
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/socket.h>

int main(void) {
    int fd1 = socket(PF_INET, SOCK_STREAM, 0);
    int fd2 = open("test.dat", O_CREAT | O_WRONLY | O_TRUNC);
    int fd3 = socket(PF_INET, SOCK_DGRAM, 0);

    printf("file descriptor 1: %d\n", fd1);
    printf("file descriptor 2: %d\n", fd2);
    printf("file descriptor 3: %d\n", fd3);

    close(fd1);
    close(fd2);
    close(fd3);
    return 0;
}
```

실행 결과를 통해 파일과 소켓이 동일한 파일 디스크립터로 관리됨을 확인할 수 있다

![image.png](assets/img/posts/network/1/image%201.png)

### 2. 파일 열기

리눅스에서는 `open()` 함수를 사용하여 파일을 연다

필요한 경우 파일을 생성하거나 기존 내용을 삭제할 수도 있다

```c
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

int open(const char *path, int flag);
// 성공 시 파일 디스크립터 반환, 실패 시 -1 반환
```

- O_CREAT: 파일이 없으면 생성
- O_TRUNC: 기존 내용을 모두 삭제
- O_APPEND: 기존 데이터 보존, 이어서 뒤에 저장
- O_RDONLY: 읽기 전용으로 파일 오픈
- O_WRONLY: 쓰기 전용으로 파일 오픈
- O_RDWR: 읽기/쓰기 겸용 모드로 열기

### 3. 파일 데이터 쓰기

`write()` 함수를 사용하여 파일에 데이터를 기록

```c
#include <unistd.h>

ssize_t write(int fd, const void *buf, size_t nbytes);
// 성공 시 기록된 바이트 수, 실패 시 -1 반환
```

예제

```c
int fd = open("data.txt", O_CREAT | O_WRONLY | O_TRUNC);
char buf[] = "Hello, Linux!";

if (write(fd, buf, sizeof(buf)) == -1)
    perror("write error");

close(fd)
```

![image.png](assets/img/posts/network/1/image%202.png)

### 4. 파일 데이터 읽기

`read()` 함수는 파일에서 데이터를 읽어온다

```c
#include <unistd.h>

ssize_t read(int fd, void *buf, size_t nbytes);
// 성공 시 읽어온 바이트 수(EOF는 0), 실패 시 -1 반환
```

### 예제

```c
char buf[100];
int fd = open("data.txt", O_RDONLY);

if (read(fd, buf, sizeof(buf)) == -1)
    perror("read error");

printf("File content: %s\n", buf);
close(fd);
```

---

![image.png](assets/img/posts/network/1/image%203.png)

## 윈도우 기반 소켓 구현

윈도우에서는 소켓을 별도의 리소스로 관리하며, 파일과 구분한다

**WinSock2** 라이브러리를 사용하여 소켓 프로그래밍을 수행한다

---

### 1. WinSock 초기화

윈도우에서는 소켓을 사용하기 전에 **WSAStartup** 함수를 통해 WinSock 라이브러리를 초기화해야 한다

```c
#include <winsock2.h>

int WSAStartup(WORD wVersionRequested, LPWSADATA lpWSAData);
// 성공 시 0, 실패 시 에러 코드 반환
```

### 초기화 코드

```c
WSADATA wsaData;
if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
    printf("WSAStartup() error!\n");
    return 1;
}
```

---

### 2. 소켓 생성 및 연결

윈도우에서는 `socket()` 함수와 `connect()` 함수를 사용하여 소켓을 생성하고 서버에 연결 요청을 보냄

```c
#include <winsock2.h>

SOCKET socket(int af, int type, int protocol);
int connect(SOCKET s, const struct sockaddr *name, int namelen);
```

### 클라이언트 예제

```c
#include <stdio.h>
#include <winsock2.h>

int main(int argc, char *argv[]) {
    WSADATA wsaData;
    SOCKET hSocket;
    struct sockaddr_in servAddr;
    char message[30];
    int strLen;

    if (argc != 3) {
        printf("Usage: %s <IP> <port>\n", argv[0]);
        return 1;
    }

    WSAStartup(MAKEWORD(2, 2), &wsaData);
    hSocket = socket(PF_INET, SOCK_STREAM, 0);

    memset(&servAddr, 0, sizeof(servAddr));
    servAddr.sin_family = AF_INET;
    servAddr.sin_addr.s_addr = inet_addr(argv[1]);
    servAddr.sin_port = htons(atoi(argv[2]));

    connect(hSocket, (struct sockaddr *)&servAddr, sizeof(servAddr));
    strLen = recv(hSocket, message, sizeof(message) - 1, 0);

    message[strLen] = 0;
    printf("Message from server: %s\n", message);

    closesocket(hSocket);
    WSACleanup();
    return 0;
}
```

---

### 3. 소켓 종료

윈도우에서는 소켓 종료 시 `closesocket()` 함수를 호출하고 `WSACleanup()`으로 WinSock 리소스를 해제한다

```c
#include <winsock2.h>

int closesocket(SOCKET s);
int WSACleanup(void);
```

---

## 정리

네트워크 프로그래밍은 OS에 따라 세부 구현 방식이 다를 수 있지만, **TCP/IP 프로토콜과 소켓이라는 공통 기반**을 사용하기 때문에 대부분의 개념이 일치

리눅스는 **파일 시스템을 활용한 접근 방식**, 윈도우는 **WinSock 기반의 API**로 접근하는 차이점
