---
layout: post
title: "[ROS2] package와 Node 만들기"
date: 2024-10-11 03:08:00+0900
categories: [Study, ROS2]
tags: [ROS2]
---

{% include embed/youtube.html id='dlTzRYS99UA' %}

ws의 src 폴더로 이동한다

![image.png](assets/img/posts/ros2/package와 Node 만들기/image.png)

패키지와 노드를 생성해보자

```bash
ros2 pkg create --build-type ament_python --node-name my_first_node my_first_package

# --build-type ament_python : 패키지의 빌드 유형을 ament_python으로 지정하여 Python 기반 ROS2 패키지를 생성
# --node-name my_first_node : 생성할 기본 노드의 이름을 my_first_node로 지정
# my_first_package          : 생성할 패키지의 이름을 my_first_package로 설정
```

![image.png](assets/img/posts/ros2/package와 Node 만들기/image%201.png)

build type은 python, cmake 등 원하는 형식으로 지정이 가능하다

> node란?  
> ROS에서 실행 가능한 최소 단위  
> ros2 run [pkg name][node name] : 노드 실행  
> ros2 node list : 실행중인 노드 리스트  
> ros2 node info [node name] : Subscribers, Publishers, Servcie Servers 등 제공하는 기능들  
> {: .prompt-info}

src 폴더 안에 package가 생성되었는지 확인한다

![image.png](assets/img/posts/ros2/package와 Node 만들기/image%202.png)

ws로 돌아가 빌드를 실행한다

```bash
cd ..
colcon build
```

![image.png](assets/img/posts/ros2/package와 Node 만들기/image%203.png)

빌드한 node를 실행해보자

```bash
ros2 run my_first_package my_first_node
```

![image.png](assets/img/posts/ros2/package와 Node 만들기/image%204.png)

**작동이 되지 않는다 !!**

워크스페이스의 install 파일을 보면

local_setup.bash라는 파일이 있다

이를 불러와야 인식할 수 있다

```bash
ls install/
source ./install/local_setup.bash
ros2 run my_first_package my_first_node
```

![image.png](assets/img/posts/ros2/package와 Node 만들기/image%205.png)

![image.png](assets/img/posts/ros2/package와 Node 만들기/image%206.png)

# 워크스페이스 환경설정

local_setup.bash 등록을 터미널을 실행할 때 자동으로 시켜보자

```bash
echo "source ~/ros_ws/install/local_setup.bash" >> /root/.bashrc
```

![image.png](assets/img/posts/ros2/package와 Node 만들기/image%207.png)

터미널을 다시 실행하고 node를 실행해본다

**“Hi from my_first_package.”**가 출력된다

![image.png](assets/img/posts/ros2/package와 Node 만들기/image%208.png)
