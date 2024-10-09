---
layout: post
title: "WSL2 + Docker Desktop 세팅"
date: 2024-10-03 03:07:00+0900
categories: [Study, Docker]
tags: [WSL2, Docker, Docker Desktop]
math: true
---

## WSL2 설치하기

> WSL란?   
> Windows에서 별도의 가상 머신이나 듀얼 부팅 설정 없이 리눅스 배포판(Ubuntu, Debian 등)을 설치하고 실행할 수 있는 환경  
{: .prompt-info}   


관리자 권한으로 터미널을 실행한다  

![image.png](assets/img/posts/docker/WSL2 + Docker Desktop 세팅/image.png)  

기다리면 설치가 됨  
설치가 완료되면 재부팅한다   

![image.png](assets/img/posts/docker/WSL2 + Docker Desktop 세팅/image 1.png)  


wsl 버전 확인   

```powershell  
# WSL의 버전 정보  
wsl -v  

# 설치된 WSL 버전과 리눅스 배포판 목록 확인  
wsl -l -v
```

![image.png](assets/img/posts/docker/WSL2 + Docker Desktop 세팅/image 2.png)  

## Docker Desktop 설치하기  


[Docker 홈페이지](https://www.docker.com/)에서 install 파일 다운 및 실행

![image.png](assets/img/posts/docker/WSL2 + Docker Desktop 세팅/image 3.png)  

![image.png](assets/img/posts/docker/WSL2 + Docker Desktop 세팅/image 4.png)  

설치가 완료되면 재부팅한다  

![image.png](assets/img/posts/docker/WSL2 + Docker Desktop 세팅/image 5.png)  

재부팅 후 설치된 Docker Desktop 실행  

![image.png](assets/img/posts/docker/WSL2 + Docker Desktop 세팅/image 6.png)  

계정을 만드거나 로그인 해준다  

![image.png](assets/img/posts/docker/WSL2 + Docker Desktop 세팅/image 7.png)  

설문은 스킵해도 무방하다  

![image.png](assets/img/posts/docker/WSL2 + Docker Desktop 세팅/image 8.png)  

설치 완료 화면  

![image.png](assets/img/posts/docker/WSL2 + Docker Desktop 세팅/image 9.png)  

## Docker 실행

Docker가 정상적으로 실행되는지 테스트해보자  
터미널에서 아래 명령어를 입력해서 이미지를 다운받자   
docker desktop 터미널로 입력했지만 윈도우 터미널로 해도 상관없다  

```powershell
docker pull nginx
```

> Docker 이미지  
> 애플리케이션과 그 실행 환경이 포함된 정적 파일로, 컨테이너를 생성하는 데 사용됨  
{: .prompt-info}     

![image.png](assets/img/posts/docker/WSL2 + Docker Desktop 세팅/image 10.png)  

기다리면 다운이 되고 이미지에 nginx가 추가됨

![image.png](assets/img/posts/docker/WSL2 + Docker Desktop 세팅/image 11.png)  

다운받은 이미지로 컨테이너를 실행해보자  
아래 명령어 입력  

```powershell
docker run --name webserver -p 8080:80 nginx 
```
<details>  
    <summary><b>각 옵션의 의미 (From GPT)</b></summary>
    <ul>
        <li>
            <b>docker run</b><br>
            - `docker run`은 새로운 컨테이너를 생성하고, 이미지를 기반으로 해당 컨테이너를 실행하는 명령어입니다.<br>
            - `docker`는 Docker CLI(명령어 인터페이스)를 호출하는 기본 명령어이고, `run`은 컨테이너를 실행하기 위한 서브 커맨드입니다.
        </li>
        <li>
            <b>--name webserver</b><br>
            - `--name` 옵션은 생성할 컨테이너의 이름을 지정합니다.<br>
            - 여기서는 `webserver`라는 이름을 설정했습니다.<br>
            - 이 옵션을 사용하지 않으면, 도커는 임의의 이름(예: `adoring_elion` 등)을 컨테이너에 할당합니다.
        </li>
        <li>
            <b>-p 8080:80</b><br>
            - `-p` 옵션은 컨테이너의 포트를 호스트의 특정 포트와 매핑(포트 포워딩)하는 역할을 합니다.<br>
            - 형식: `호스트포트:컨테이너포트`<br>
            - 호스트 포트: 호스트(사용자의 로컬 머신)에서 사용할 포트 번호.<br>
            - 컨테이너 포트: 컨테이너 내부에서 사용하는 포트 번호.<br>
            - 예시에서 `8080:80`은 <b>호스트의 8080 포트</b>를 <b>컨테이너의 80 포트</b>와 연결하라는 의미입니다.
        </li>
        <li>
            <b>nginx</b><br>
            - `nginx`는 사용할 Docker 이미지 이름입니다.<br>
            - 이 명령어는 도커 허브(Docker Hub)에서 `nginx` 이미지를 가져와 컨테이너를 생성하고 실행합니다.<br>
            - 이미 로컬에 이미지가 존재하지 않으면, 자동으로 Docker Hub에서 다운로드하여 사용합니다.
        </li>
    </ul>
</details>  


> Docker 컨테이너  
> 이미지를 기반으로 실행되는 애플리케이션 인스턴스로, 각각 독립된 환경에서 애플리케이션을 실행함    
{: .prompt-info}     

![image.png](assets/img/posts/docker/WSL2 + Docker Desktop 세팅/image 12.png)  

webserver 컨테이너가 추가된 것을 확인할 수 있다   

![image.png](assets/img/posts/docker/WSL2 + Docker Desktop 세팅/image 13.png)  

새 터미널에서 `docker ps`를 통해 실행 중인 컨테이너 목록을 확인할 수 있다  

![image.png](assets/img/posts/docker/WSL2 + Docker Desktop 세팅/04afe2ca-32ff-4a92-a8e6-81d94c8c6b74.png)  


localhost:8080 으로 접속해서 아래화면이 보이면 성공이다  

![image.png](assets/img/posts/docker/WSL2 + Docker Desktop 세팅/image 14.png)  

## Container 및 Image 삭제  

docker desktop에서 테스트에 사용한 nginx 컨테이너와 이미지를 삭제해보자   

stop 버튼을 눌러 컨테이너를 종료한다  

![image.png](assets/img/posts/docker/WSL2 + Docker Desktop 세팅/image 15.png)  

delete 버튼을 눌러 컨테이너 삭제한다

![image.png](assets/img/posts/docker/WSL2 + Docker Desktop 세팅/image 16.png)  

![image.png](assets/img/posts/docker/WSL2 + Docker Desktop 세팅/image 17.png)  

delete 버튼을 눌러 이미지를 삭제

![image.png](assets/img/posts/docker/WSL2 + Docker Desktop 세팅/image 18.png)   
