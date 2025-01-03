---
layout: post
title: "[ROS2] Subscriber, Publisher"
date: 2024-10-12 21:24:00+0900
categories: [Study, ROS2]
tags: [ROS2]
---

## SubScriber 만들기

새로운 노드 파일을 만들어보자

src/my_first_package/my_first_package 경로에 my_subscriber.py를 추가한다

![image.png](assets/img/posts/study/ros2/subscriber, publisher/image.png)

my_subscriber.py에서는 turtlesim의 Pose를 읽을것이다

```python
import rclpy as rp
from rclpy.node import Node
from turtlesim.msg import Pose  # 구독할 topic의 type

class TurtlesimSubscriber(Node):  # Node 상속
    def __init__(self):
        super().__init__("turtlesim_subscriber")  # Node의 init
        self.subscription = self.create_subscription(  # subscriptions 생성
            Pose,  # type
            "/turtle1/pose",  # topic
            self.callback,  # callback
            10,  # 데이터 queue 크기
        )

    def callback(self, msg):
        print("X: ", msg.x, ", Y: ", msg.y)  # 좌표 출력

def main():
    rp.init()

    turtlesim_subscriber = TurtlesimSubscriber()
    rp.spin(turtlesim_subscriber)

    turtlesim_subscriber.destroy_node()
    rp.shutdown()

if __name__ == "__main__":
    main()

```

create_subscription의 메소드에서 각 arg를 확인할 수 있다

![image.png](assets/img/posts/study/ros2/subscriber, publisher/image%201.png)

setup.py에 entry point를 추가해준다

![image.png](assets/img/posts/study/ros2/subscriber, publisher/image%202.png)

이제 ws에서 build를 해보자

`ros2 run my_first_package my` 까지 입력 후 tap 2번 입력할 때 노드 목록이 나오면 성공이다

```bash
colcon build
sb # local_setup.bash 새로고침
ros2 run my_first_package my_subscriber
```

![image.png](assets/img/posts/study/ros2/subscriber, publisher/image%203.png)

이제 새로운 터미널에서 topic list를 확인해보자

> `ros2 topic list` 에서 **/turtle1/pose** 는 published가 아니라 subscribed 이므로 헷갈릴 수 있다.
>
> -v로 자세히 확인해봐야 한다
{: .prompt-warning}

![image.png](assets/img/posts/study/ros2/subscriber, publisher/image%204.png)

## Publisher 만들기

이제 Publisher를 만들것이다

src/my_first_package/my_first_package에 my_publisher.py를 추가한다

이 코드는 /turtle1/cmd_vel topic을 500ms마다 linear.x=2, angular.z=2로 설정하고 발행한다

```python
import rclpy as rp
from rclpy.node import Node

from geometry_msgs.msg import Twist

class TurtlesimPublisher(Node):  # Node 상속
    def __init__(self):
        super().__init__("turtlesim_publisher")  # Node의 init
        self.publisher = self.create_publisher(  # publisher 생성
            Twist,  # type
            "/turtle1/cmd_vel",  # topic
            10,  # 데이터 queue 크기
        )
        timer_period = 0.5  # 500ms 마다 발행
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        msg = Twist()
        msg.linear.x = 2.0
        msg.angular.z = 2.0
        self.publisher.publish(msg)

def main(args=None):
    rp.init(args=args)

    turtlesim_publisher = TurtlesimPublisher()
    rp.spin(turtlesim_publisher)

    turtlesim_publisher.destroy_node()
    rp.shutdown()

if __name__ == "__main__":
    main()
```

이제 subscriber와 동일하게 entry point에 추가하고 build 해보자

![image.png](assets/img/posts/study/ros2/subscriber, publisher/image%205.png)

```bash
colcon build
sb
ros2 run my_first_package  my_publisher
```

![image.png](assets/img/posts/study/ros2/subscriber, publisher/image%206.png)

## publish and subscribe

이제 turtlesim, publisher, subscriber를 모두 실행한다

```bash
ros2 run turtlesim turtlesim_node
ros2 run my_first_package  my_publisher
ros2 run my_first_package  my_subscriber
```

거북이가 빙글빙글 돌면서 위치가 출력된다

참고로 거북이가 움직일때마다 토픽을 구독해서 subscriber 로그가 빠르게 올라온다

![image.png](assets/img/posts/study/ros2/subscriber, publisher/image%207.png)

rqt로 구조를 확인할 수 있다

![image.png](assets/img/posts/study/ros2/subscriber, publisher/image%208.png)

다음 노드 관계는 아래와 같다

1. publisher가 cmd_vel 토픽을 발행하고 turtlesim이 구독
2. turtlesim이 pose 토픽을 발행하고 subscriber가 구독
