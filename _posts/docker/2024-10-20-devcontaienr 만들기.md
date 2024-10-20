---
layout: post
title: "devcontainer 만들기"
date: 2024-10-20 12:02:00+0900
categories: [Study, Docker]
tags: [Docker, Devcontainer]
---

## DevContainer 란?

- DevContainer는 **개발 환경을 코드로 정의**하고, 이를 기반으로 **일관된 개발 환경을 제공**하는 도구
- Visual Studio Code의 확장 기능으로, 개발 환경을 Docker 컨테이너 내에 구성
- **Docker**와 같은 가상화 기술을 이용해 **독립된 환경**에서 작업할 수 있으며, 환경 설정이 자동화되므로 개발 환경 차이를 최소화할 수 있다

## DevContainer 구조

### **로컬 시스템 (Local OS)**

- **VS Code**: 사용자가 개발을 위해 실행하는 애플리케이션으로, 주로 테마 및 UI 확장(Theme/UI Extension)과 같은 UI 관련 기능을 로컬에서 실행한다
- **소스 코드**: 실제 소스 코드는 로컬에 저장되며, 이는 컨테이너 내부로 마운트됨. 로컬의 파일 변경 사항이 즉시 컨테이너에서 반영된다.

### **컨테이너 (Container)**

- **VS Code 서버**: 컨테이너 내부에서 실행되며, VS Code의 일부 기능(파일 시스템 접근, 터미널 실행, 디버깅 등)을 처리
- **워크스페이스 확장(Workspace Extension)**: 프로젝트와 관련된 확장은 컨테이너 내에서 실행됨. 이는 코드 실행과 디버깅에 주로 사용되며, 로컬의 UI 확장과는 별도로 동작
- **파일 시스템, 터미널, 애플리케이션, 디버거**: 소스 코드와 개발 도구들이 컨테이너 내부에서 관리된다. 개발 환경 내 모든 작업은 컨테이너 내에서 처리

### 동작 방식

1. **Exposed Port**: 로컬 VS Code와 컨테이너 내 VS Code 서버 간의 통신을 위해 포트가 노출된다. 이를 통해 로컬에서 컨테이너로 요청을 보내고, 작업이 처리될 수 있도록 함
2. **Volume Mount**: 로컬의 소스 코드 디렉터리가 컨테이너 내부로 마운트된다. 이를 통해 로컬에서 코드가 변경되면, 변경 사항이 즉시 컨테이너에서 반영된다.
3. **VS Code Server**: 컨테이너 내부에서 실행되며, 파일 시스템에 접근하고 터미널을 실행하며, 애플리케이션을 실행하거나 디버깅할 수 있는 환경을 제공한다.
4. **Workspace Extensions**: 프로젝트에 필요한 확장 기능들은 컨테이너 내부에 설치되어 실행되며, 이를 통해 컨테이너 내부에서 일관된 개발 환경을 유지할 수 있다.

![image.png](assets/img/posts/docker/devcontainer 만들기/image.png)

### DevContainer 구성 파일

```
project-root/
│
├── .devcontainer/          # DevContainer 설정 디렉토리
│   ├── devcontainer.json    # DevContainer 환경 설정 파일
│   ├── Dockerfile           # (선택적) 컨테이너에 설치할 패키지 및 설정을 정의하는 Docker 이미지
│   ├── post-create.sh       # (선택적) 컨테이너가 생성된 후 실행될 스크립트 파일
│   └── docker-compose.yml   # (선택적) 다중 컨테이너 환경 정의
│
├── src/                    # 프로젝트 소스 코드 디렉토리
│   └── main.py              # 예시 소스 코드 파일
│
└── README.md               # 프로젝트 설명 파일
```

- **devcontainer.json**

```
{
  // 개발 컨테이너의 이름
  "name": "Python 3.9 Development Environment",

  // 사용할 Docker 이미지 또는 Dockerfile 지정
  "image": "python:3.9-slim",
  // "build": {
  //   "dockerfile": "Dockerfile",
  //   "context": "..",
  //   "args": { "VARIANT": "3.9" }
  // },

  // 컨테이너 생성 후 실행할 명령어
  "postCreateCommand": "pip install --user -r requirements.txt",

  // 컨테이너에 마운트할 볼륨 설정
  "mounts": [
    "source=${localWorkspaceFolder},target=/workspace,type=bind,consistency=cached"
  ],

  // 컨테이너 내에서 사용할 기본 사용자 설정
  "remoteUser": "vscode",

  // VS Code 관련 설정
  "customizations": {
    "vscode": {
      // 컨테이너에 설치할 VS Code 확장 프로그램
      "extensions": [
        "ms-python.python",
        "ms-python.vscode-pylance",
        "ms-toolsai.jupyter"
      ],
      // VS Code 설정 오버라이드
      "settings": {
        "python.defaultInterpreterPath": "/usr/local/bin/python",
        "python.linting.enabled": true,
        "python.linting.pylintEnabled": true
      }
    }
  },

  // 포워딩할 포트 설정
  "forwardPorts": [8000, 8080],

  // 컨테이너 생성 시 실행할 명령어
  "onCreateCommand": "echo 'Container created!' > /tmp/created.txt",

  // 컨테이너 시작 시 실행할 명령어
  "updateContentCommand": "pip install -r requirements.txt",

  // 컨테이너 실행 후 실행할 명령어
  "postStartCommand": "echo 'Container started!' > /tmp/started.txt",

  // 컨테이너에 추가할 기능 (feature) 설정
  "features": {
    "ghcr.io/devcontainers/features/node:1": {
      "version": "lts"
    },
    "ghcr.io/devcontainers/features/git:1": {
      "version": "latest",
      "ppa": false
    }
  },

  // 환경 변수 설정
  "remoteEnv": {
    "MY_VARIABLE": "my-value"
  },

  // 컨테이너 실행 시 추가할 Docker 실행 인자
  "runArgs": ["--cap-add=SYS_PTRACE", "--security-opt", "seccomp=unconfined"],

  // 개발 컨테이너에 추가할 사용자 또는 그룹 설정
  "containerUser": "vscode",
  "containerEnv": {
    "HOME": "/home/vscode"
  },

  // Docker Compose 사용 시 설정
  // "dockerComposeFile": "../docker-compose.yml",
  // "service": "app",
  // "workspaceFolder": "/workspace",

  // 셧다운 액션 설정 (none, stopContainer, stopCompose)
  "shutdownAction": "stopContainer",

  // 개발 컨테이너 내에서 사용할 기본 셸 설정
  "overrideCommand": false,

  // 추가 컨테이너 호스트 요구사항 (예: GPU 지원)
  // "hostRequirements": {
  //   "cpus": 4,
  //   "memory": "8gb",
  //   "storage": "32gb"
  // },

  // 원격 머신에서 개발 컨테이너 실행 시 설정
  // "remoteHost": "ssh://user@host",

  // 개발 컨테이너에 추가할 라벨
  "containerLabel": {
    "com.example.label": "value"
  },

  // 아이디어 공유 또는 문제 보고를 위한 설정
  // "ideShare": {
  //   "enabled": true
  // }
}
```

- post-create.sh

```
#!/bin/bash

# 필요한 파이썬 패키지 설치
pip install flake8 black

# Git 사용자 정보 설정
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# 기타 초기화 작업 수행
echo "DevContainer is ready!"
```

## DevContainer 만들어보기

ROS2 개발 가상환경을 만들어볼것이다

vscode에서 Dev container 확장을 추가

![image.png](assets/img/posts/docker/devcontainer 만들기/image%201.png)

프로젝트 루트에 .devcontainer 파일을 만들어준다

![image.png](assets/img/posts/docker/devcontainer 만들기/image%202.png)

devcontainer.json 파일을 만든다

![image.png](assets/img/posts/docker/devcontainer 만들기/image%203.png)

이전에 ros2 컨테이너를 만들 때 사용한 **osrf/ros:humble-desktop-full 이미지를 사용할것이다**

![image.png](assets/img/posts/docker/devcontainer 만들기/image%204.png)

설정해야할 부분이 너무 많다…

devcontainer.json 파일은 ChatGPT를 이용해서 만들었다

```json
{
  "name": "ROS2 Humble", // DevContainer의 이름. VS Code에서 DevContainer를 구분할 때 사용.
  "image": "osrf/ros:humble-desktop-full", // 사용할 Docker 이미지. 여기서는 ROS2 Humble Desktop Full 이미지 사용.
  "runArgs": [
    "--privileged", // 컨테이너가 호스트 시스템의 모든 하드웨어 리소스에 접근할 수 있게 함.
    "--env=DISPLAY=host.docker.internal:0", // GUI 프로그램 실행을 위해 X11 디스플레이 설정.
    "--name=ros2-humble" // 생성될 컨테이너의 이름을 'ros2-humble'로 명시.
  ],
  "workspaceMount": "source=${localWorkspaceFolder},target=/root/ros2-ws,type=bind",
  // 호스트의 현재 작업 디렉토리를 컨테이너의 ~/ros2-ws에 마운트.
  "workspaceFolder": "/root/ros2-ws", // 컨테이너 내에서 작업할 기본 폴더를 ~/ros2-ws로 설정.
  "mounts": [
    "source=${localEnv:HOME}${localEnv:USERPROFILE}/.Xauthority,target=/root/.Xauthority,type=bind"
    // 호스트의 .Xauthority 파일을 컨테이너의 /root/.Xauthority에 마운트하여,
    // X11 인증을 통해 GUI 프로그램을 실행할 수 있게 함.
  ],
  "customizations": {
    "vscode": {
      "settings": {
        "terminal.integrated.shell.linux": "/bin/bash" // VS Code의 기본 터미널 셸을 Bash로 설정.
      },
      "extensions": [
        "ms-vscode.cpptools", // C++ 개발을 위한 확장.
        "ms-python.python", // Python 개발을 위한 확장.
        "ms-vscode.cmake-tools", // CMake 프로젝트를 관리하고 빌드하기 위한 확장.
        "twxs.cmake", // CMake 파일에 대한 추가적인 문법 강조와 지원을 제공하는 확장.
        "ms-iot.vscode-ros" // ROS (Robot Operating System) 개발을 위한 확장.
      ]
    }
  }
}
```

bashrc 에 설정하기 위해 커맨드 추가

devcontainer.json

```json
{
	... ,
 "postCreateCommand": "chmod +x ~/ros2-ws/.devcontainer/setup.sh && /bin/bash ~/ros2-ws/.devcontainer/setup.sh" // setup.sh 파일
}
```

setup.sh

```json
#!/bin/bash

# ROS2 환경 설정
echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc
echo 'export ROS_DOMAIN_ID=13' >> ~/.bashrc
echo 'alias sb="source ~/.bashrc && echo \"bashrc is reloaded\""' >> ~/.bashrc

# 환경설정 적용
source ~/.bashrc

# 워크스페이스 빌드 및 환경 설정
cd ~/ros2-ws
colcon build

# 빌드 후 환경 설정
echo 'source ~/ros2-ws/install/local_setup.bash' >> ~/.bashrc
source ~/.bashrc

echo "ROS2 Humble 환경 설정 완료"`
```

![image.png](assets/img/posts/docker/devcontainer 만들기/image%205.png)

이제 vscode로 프로젝트를 열면 알람이 나온다

Reopen in Container를 누르면 컨테이너를 생성한다

![image.png](assets/img/posts/docker/devcontainer 만들기/image%206.png)

show log를 누르면 진행상황도 볼 수 있다

![image.png](assets/img/posts/docker/devcontainer 만들기/image%207.png)

![image.png](assets/img/posts/docker/devcontainer 만들기/image%208.png)

기다리면 결과가 나온다

그런데 문제가 발생했다

![image.png](assets/img/posts/docker/devcontainer 만들기/image%209.png)

GPT 검색 결과 줄바꿈 형식 문제인듯 하여 에디터 형식을 LF로 변경해줬다

![image.png](assets/img/posts/docker/devcontainer 만들기/image%2010.png)

![image.png](assets/img/posts/docker/devcontainer 만들기/image%2011.png)

변경 후에 다시 빌드해보았다

![image.png](assets/img/posts/docker/devcontainer 만들기/image%2012.png)

성공

![image.png](assets/img/posts/docker/devcontainer 만들기/image%2013.png)

![image.png](assets/img/posts/docker/devcontainer 만들기/image%2014.png)
