---
layout: post
title: "Jenkins 튜토리얼 따라하기 : Jenkins 튜토리얼 따라하기  : Docker + Guided Tour (1)"
date: 2025-04-25 14:08:00+0900
categories: [Study, CICD]
tags: [CICD, Jenkins]
---
## **Docker 에서 Jenkins 실행하기**

### **Docker Image & Container**

> Window11 환경에서 진행하였다.   
> Linux, Mac에서 Docker run 명령어는 [Jenkins Handbook](https://www.jenkins.io/doc/book/installing/docker/) 참고
{: .prompt-warning}

**Jenkins Container**와 **Docker deamon Container**간의 통신을 위해 **Docker 네트워크**를 설정해 준다 

```bash
docker network create jenkins
```

Jenkins Container 안에서 Docker 명령을 실행하기 위해 **`docker:dind`** 이미지를 다운로드하고 실행한다

```bash
docker run --name jenkins-docker --rm --detach ^
  --privileged --network jenkins --network-alias docker ^
  --env DOCKER_TLS_CERTDIR=/certs ^
  --volume jenkins-docker-certs:/certs/client ^
  --volume jenkins-data:/var/jenkins_home ^
  --publish 2376:2376 ^
  docker:dind
```

- `--privileged` : dind가 제대로 작동하기 위해 권한 설정
- `--volume jenkins-docker-certs:/certs/client` : TLS 인증서를 위한 볼륨 마운트
- `--volume jenkins-data:/var/jenkins_home` : Jenkins 데이터를 위한 볼륨 마운트

공식 Jenkins 이미지를 기반으로  Docker CLI를 추가한 Dockerfile을 만든다

```docker
FROM jenkins/jenkins
USER root
RUN apt-get update && apt-get install -y lsb-release
RUN curl -fsSLo /usr/share/keyrings/docker-archive-keyring.asc \
  https://download.docker.com/linux/debian/gpg
RUN echo "deb [arch=$(dpkg --print-architecture) \
  signed-by=/usr/share/keyrings/docker-archive-keyring.asc] \
  https://download.docker.com/linux/debian \
  $(lsb_release -cs) stable" > /etc/apt/sources.list.d/docker.list
RUN apt-get update && apt-get install -y docker-ce-cli
USER jenkins
```

Dockerfile을 기반으로 **`myjenkins`** 라는 이름으로 Docker 이미지를 빌드하고 실행시킨다

```bash
docker build -t myjenkins .
```

```bash
docker run --name jenkins --restart=on-failure --detach ^
  --network jenkins --env DOCKER_HOST=tcp://docker:2376 ^
  --env DOCKER_CERT_PATH=/certs/client --env DOCKER_TLS_VERIFY=1 ^
  --volume jenkins-data:/var/jenkins_home ^
  --volume jenkins-docker-certs:/certs/client:ro ^
  --publish 8080:8080 --publish 50000:50000 myjenkins
```

Docker Container 실행 모습

![image.png](assets/img/posts/study/cicd/Docker + Guided Tour (1)/image.png)

### **두 개의 Container를 별도로 띄우는 이유**

* Jenkins 자체는 **Container 안에서 돌아가는 일반 앱**   
  -> Jenkins가 실제로 **빌드하고 테스트할 애플리케이션을 Docker Container로 띄우려면** `Docker 엔진(docker CLI)`이 필요하다
* **문제는 Docker안에 또 Docker를 설치하거나, 소켓을 공유**해야 한다는 건데, 보안이나 권한 문제 등이 존재
* 이를 해결하기 위해 Jenkins가 사용할 Docker deamon을 따로 `docker:dind`(Docker in Docker)로 만들어서, Jenkins가 그 Container에 접근해서 작업을 진행한다
* **작동 과정**
  1. **웹 UI에서 파이프라인 생성**
      - GitHub 저장소 연결 또는 Dockerfile 기반 프로젝트 설정
  2. **파이프라인 실행 시**
      - Jenkins 내부에서 `docker` CLI 명령어 실행
      - 환경 변수 `DOCKER_HOST=tcp://docker:2376` 설정에 따라 명령은 `jenkins-docker` Container의 Docker deamon에 전달
  3. **Container 생성 및 실행**
      - `jenkins-docker`는 명령을 받아 **Docker Container를 실행**
  4. **작업 완료 후 결과 보고**
      - Jenkins가 로그/빌드 결과를 수집하여 UI에 표시
    

```
Host (Local)
├─ Docker Network: jenkins
│   ├─ jenkins-blueocean      ← Jenkins 앱 (명령 내림)
│   └─ jenkins-docker         ← Docker deamon (명령 실행)
└─ 빌드 Container          ← 실제로 여기 생성됨
```

> **Jenkins에서 작업하는 Container는 실제로 Jenkins Container 내부나, Docker deamon Container 내부가 아닌, "호스트 머신(Local)"에 띄워진다**  
{: .prompt-info} 

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
