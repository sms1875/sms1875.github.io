---
layout: post
title: "Jenkins 튜토리얼 따라하기 : Guided Tour (1)"
date: 2025-04-25 14:08:00+0900
categories: [Study, CICD]
tags: [CICD, Jenkins]
---

### **Jenkins 초기 설정**

Jenkins의 **password key**를 확인하기 위해 다음 명령어를 입력

```bash
docker exec jenkins cat /var/jenkins_home/secrets/initialAdminPassword 
```

아래와 같이 직접 Container에 접속해서 확인할 수도 있다

```bash
docker ps
docker logs <docker-container-name>
```

![image.png](assets/img/posts/study/cicd/Docker + Guided Tour (1)/image%201.png)

![image.png](assets/img/posts/study/cicd/Docker + Guided Tour (1)/image%202.png)

[http://localhost:8080/](http://localhost:8080/) 에서 Jenkins UI를 확인할 수 있다

password key를 붙여넣는다

![image.png](assets/img/posts/study/cicd/Docker + Guided Tour (1)/image%203.png)

`Install suggested plugins` 선택

![image.png](assets/img/posts/study/cicd/Docker + Guided Tour (1)/image%204.png)

![image.png](assets/img/posts/study/cicd/Docker + Guided Tour (1)/image%205.png)

관리자 계정 생성

![image.png](assets/img/posts/study/cicd/Docker + Guided Tour (1)/image%206.png)

Jenkins URL 설정

지금은 별다른 변경 없이 그대로 진행하였다

![image.png](assets/img/posts/study/cicd/Docker + Guided Tour (1)/image%207.png)

설정 완료

![image.png](assets/img/posts/study/cicd/Docker + Guided Tour (1)/image%208.png)

## **Pipeline 구현**

### **Docker Pipeline plugin 설치**

우측 상단의 `Manage Jenkins` 를 클릭하여 관리 페이지로 간 후, `Plugins` 클릭

![image.png](assets/img/posts/study/cicd/Docker + Guided Tour (1)/image%209.png)

![image.png](assets/img/posts/study/cicd/Docker + Guided Tour (1)/image%2010.png)

`Available plugins` 클릭 → **Docker Pipeline** 입력 → `Install`을 체크하고 설치를 진행한다

![image.png](assets/img/posts/study/cicd/Docker + Guided Tour (1)/image%2011.png)

설치 과정이 표시되면 아래로 스크롤해서 `Jenkins 재시작` 을 선택한다

![image.png](assets/img/posts/study/cicd/Docker + Guided Tour (1)/image%2012.png)

![image.png](assets/img/posts/study/cicd/Docker + Guided Tour (1)/image%2013.png)

`Installed plugins` 에서 Docker Pipeline plugin이 설치된 것을 확인할 수 있다

![image.png](assets/img/posts/study/cicd/Docker + Guided Tour (1)/image%2014.png)

### **Repository 설정**

repo의 root 경로에 `Jenkinsfile` 을 생성한다

**Jenkinsfile example link** : [https://www.jenkins.io/doc/pipeline/tour/hello-world/#examples](https://www.jenkins.io/doc/pipeline/tour/hello-world/#examples)

```groovy
pipeline {
    agent { docker { image 'node:22.14.0-alpine3.21' } }
    stages {
        stage('build') {
            steps {
                sh 'node --version'
            }
        }
    }
}
```

![image.png](assets/img/posts/study/cicd/Docker + Guided Tour (1)/image%2015.png)

### **Item 추가**

Jenkins 에서 `New Item`을 클릭한다

![image.png](assets/img/posts/study/cicd/Docker + Guided Tour (1)/image%2016.png)

적당히 아이템 이름을 입력 한 뒤 `Multibranch Pipeline` 선택

![image.png](assets/img/posts/study/cicd/Docker + Guided Tour (1)/image%2017.png)

`Add source` 를 누르고 Github 선택

![image.png](assets/img/posts/study/cicd/Docker + Guided Tour (1)/image%2018.png)

github repo의 주소 입력후 `Validate`  를 클릭해서 repo와 연결을 진행한다

이후 `Save` 를 클릭하면 log에서 repo와 연결 상태를 확인할 수 있다

![image.png](assets/img/posts/study/cicd/Docker + Guided Tour (1)/image%2019.png)

![image.png](assets/img/posts/study/cicd/Docker + Guided Tour (1)/image%2020.png)

`빌드 기록`에서 #1 빌드가 성공적으로 수행되었다

로그를 확인하면 `node --version` 실행 결과를 확인할 수 있다

![image.png](assets/img/posts/study/cicd/Docker + Guided Tour (1)/image%2021.png)

![image.png](assets/img/posts/study/cicd/Docker + Guided Tour (1)/image%2022.png)

## **Reference**

1. [https://www.jenkins.io/doc/book/installing/docker/](https://www.jenkins.io/doc/book/installing/docker/)
2. [https://www.jenkins.io/doc/pipeline/tour/getting-started/](https://www.jenkins.io/doc/pipeline/tour/getting-started/)
3. [https://shanepark.tistory.com/500](https://shanepark.tistory.com/500)
