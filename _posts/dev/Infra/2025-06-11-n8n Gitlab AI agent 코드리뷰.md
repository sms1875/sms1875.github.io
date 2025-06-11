---
layout: post
title: "n8n Gitlab AI agent 코드리뷰"
date: 2025-06-11 23:28:00+0900
categories: [Dev, Infra]
tags: [Infra, Ant Media Server, RTMP, Docker, Streaming]
---

## 프로젝트 설정

- docker compose
    
    ```yaml
    services:
      n8n:
        image: n8nio/n8n:latest
        container_name: n8n`
        user: root
        ports:
          - "127.0.0.1:5678:5678"
        volumes:
          - ./n8n_data:/home/node/.n8n
        environment:
          - TZ=Asia/Seoul
          - N8N_PROTOCOL=http
          - N8N_HOST=n8n.gaemibot.duckdns.org
          - N8N_PORT=5678
          - WEBHOOK_URL=https://n8n.gaemibot.duckdns.org
          - N8N_EDITOR_BASE_URL=https://n8n.gaemibot.duckdns.org
          - GENERIC_TIMEZONE=Asia/Seoul
        restart: unless-stopped
    
    ```
    
    - `volumes`
        - `./n8n_data:/home/node/.n8n`: 현재 경로의 `n8n_data` 폴더를 컨테이너 내부의 `/home/node/.n8n` 경로에 연결하여 데이터를 영구적으로 저장
    - `environment`
        - `N8N_HOST=n8n.gaemibot.duckdns.org`: n8n에 접속할 때 사용할 호스트 주소를 설정
        - `N8N_PORT=5678`: n8n이 내부적으로 사용할 포트 번호를 `5678`로 설정
        - `WEBHOOK_URL=https://n8n.gaemibot.duckdns.org`: n8n 웹훅이 사용할 기본 URL을 설정
        - `N8N_EDITOR_BASE_URL=https://n8n.gaemibot.duckdns.org`: n8n 편집기 접속 시 사용할 기본 URL을 설정
        - `GENERIC_TIMEZONE=Asia/Seoul`: n8n 내부에서 일반적으로 사용할 시간대를 'Asia/Seoul'로 설정

`N8N_HOST` 에 설정한 `https://n8n.gaemibot.duckdns.org` 에 접속

![image.png](/assets/img/posts/dev/Infra/n8n Gitlab commit/image.png)

계정 생성 및 설문 조사를 하면 라이센스 키를 email로 발송해준다

![image.png](/assets/img/posts/dev/Infra/n8n Gitlab commit/image%201.png)

![image.png](/assets/img/posts/dev/Infra/n8n Gitlab commit/image%202.png)

라이센스 키를 활성화시키면 Communty Edition 등록을 확인할 수 있다

![image.png](/assets/img/posts/dev/Infra/n8n Gitlab commit/image%203.png)

![image.png](/assets/img/posts/dev/Infra/n8n Gitlab commit/image%204.png)

## Workflow 생성

메인 화면에 가면 Workflow 페이지를 확인할 수 있다

**`Create Workflow`** 로 새로운 workflow 생성

![image.png](/assets/img/posts/dev/Infra/n8n Gitlab commit/image%205.png)

![image.png](/assets/img/posts/dev/Infra/n8n Gitlab commit/image%206.png)

tamplate : [https://n8n.io/workflows/2167-chatgpt-automatic-code-review-in-gitlab-mr/](https://n8n.io/workflows/2167-chatgpt-automatic-code-review-in-gitlab-mr/)

- MR 에서 특정 comment를 작성하면 ai agent를 이용하여 코드 리뷰를 수행하고 결과를 comment에 추가해준다
- 목표 : 이를 commit → push 를 할 때마다 자동으로 수행

**`Use for Free`** → **`Copy template to clipboard`** 후 workflow에 붙여넣기를 하면 템플릿이 추가된다

## SSAFY Gitlab Repo 연동

**`Webhook node`**에서 Test URL을 복사 후 **`Listen for test event`** 

![image.png](/assets/img/posts/dev/Infra/n8n Gitlab commit/image%207.png)

![image.png](/assets/img/posts/dev/Infra/n8n Gitlab commit/image%208.png)

Gitlab에서 새로운 Webhook을 만든다

- URL에 Test URL 입력
- Trigger에 Push events 설정

![image.png](/assets/img/posts/dev/Infra/n8n Gitlab commit/image%209.png)

![image.png](/assets/img/posts/dev/Infra/n8n Gitlab commit/image%2010.png)

Test → Push events 

`Webhook node` 에서 결과를 확인 가능하다

![image.png](/assets/img/posts/dev/Infra/n8n Gitlab commit/image%2011.png)

![image.png](/assets/img/posts/dev/Infra/n8n Gitlab commit/image%2012.png)

기존의 `Need review`는 MR Comment 에서 특정 단어를 찾아서 리뷰 필요성을 분기해주는 노드이다

새로운 플로우에는 필요 없으므로 삭제

![image.png](/assets/img/posts/dev/Infra/n8n Gitlab commit/image%2013.png)

새로운 `Split commit node` 를 추가해준다

- Fields To Split Out : `body.commits`

![image.png](/assets/img/posts/dev/Infra/n8n Gitlab commit/image%2014.png)

![image.png](/assets/img/posts/dev/Infra/n8n Gitlab commit/image%2015.png)

테스트 결과 commit이 개별 item으로 분리되었다

![image.png](/assets/img/posts/dev/Infra/n8n Gitlab commit/image%2016.png)

`Get Commit Diff node`를 SSAFY Gitlab 구조에 맞게 수정한다

- URL : `https://lab.ssafy.com/api/v4/projects/&lbrace;&lbrace; $('Push Webhook').first().json.body.project_id &rbrace;&rbrace;/repository/commits/&lbrace;&lbrace; $json.id &rbrace;&rbrace;/diff`
- 각 commit에서 diff를 가져온다
- Header에 SSAFY Gitlab token을 설정해준다
    - PRIVATE-TOKEN : <your-private-token>

![image.png](/assets/img/posts/dev/Infra/n8n Gitlab commit/image%2017.png)

각 diff를 item으로 가져온다

![image.png](/assets/img/posts/dev/Infra/n8n Gitlab commit/image%2018.png)

현재 diff에는 어떤 commit의 diff인지 알 수 없다

이를 해결하기 위해 `Add Commit SHA to Diff Item node` 를 만든다

- `Set node`
- commit의 sha를 field에 추가해준다
- Fields to Set
    - name : commit_sha
    - type : String
    - value : `&lbrace;&lbrace; $('Split Commit').item.json.id &rbrace;&rbrace;`

![image.png](/assets/img/posts/dev/Infra/n8n Gitlab commit/image%2019.png)

![image.png](/assets/img/posts/dev/Infra/n8n Gitlab commit/image%2020.png)

item에 새로 추가된 `commit_sha` field를 확인할 수 있다

![image.png](/assets/img/posts/dev/Infra/n8n Gitlab commit/image%2021.png)

`Skip File Changes`, `Parse Last Diff Line`, `Code` node는 그대로 유지해준다

![image.png](/assets/img/posts/dev/Infra/n8n Gitlab commit/image%2022.png)

- 참고 : Skip File Changes 노드는 단순한 경로나 이름 변경, 이미지 파일등은 예외처리한다

![image.png](/assets/img/posts/dev/Infra/n8n Gitlab commit/image%2023.png)

`Basic LLM Chain` 에서 원하는 AI Model을 추가해준다

나는 무료로 사용할 수 있는 Gemini api를 사용하였다

![image.png](/assets/img/posts/dev/Infra/n8n Gitlab commit/image%2024.png)

Prompt도 설정 가능하다

````
File path：
&lbrace;&lbrace; $node['Skip File Changes'].json.new_path &rbrace;&rbrace;
```Original code
 &lbrace;&lbrace; $json.originalCode &rbrace;&rbrace;
```
change to
```New code
 &lbrace;&lbrace; $json.newCode &rbrace;&rbrace;
```
Please review the code changes in this section:
````

![image.png](/assets/img/posts/dev/Infra/n8n Gitlab commit/image%2025.png)

AI 코드 리뷰 결과

![image.png](/assets/img/posts/dev/Infra/n8n Gitlab commit/image%2026.png)

`Post Discussions` 의 값도 SSAFY Gitlab에 맞도록 수정해준다

- URL : `https://lab.ssafy.com/api/v4/projects/&lbrace;&lbrace; $('Push Webhook').first().json.body.project_id &rbrace;&rbrace;/repository/commits/&lbrace;&lbrace; $node['Skip File Changes'].json.commit_sha &rbrace;&rbrace;/comments`
- Header에 SSAFY Gitlab token을 설정해준다
    - PRIVATE-TOKEN : `<your-private-token>`
- body
    - Parameter Type : Form Data
    - name : note
    - value : `&lbrace;&lbrace; $json.text &rbrace;&rbrace;`

![image.png](/assets/img/posts/dev/Infra/n8n Gitlab commit/image%2027.png)

![image.png](/assets/img/posts/dev/Infra/n8n Gitlab commit/image%2028.png)

commit 내역에 코드 리뷰 결과 comment를 확인할 수 있다

![image.png](/assets/img/posts/dev/Infra/n8n Gitlab commit/image%2029.png)

![image.png](/assets/img/posts/dev/Infra/n8n Gitlab commit/image%2030.png)

## Branch 분기 처리 TroubleShooting

새로운 feature branch를 checkout 한 다음 처음 push하면 

origin branch의 commit들에도 comment가 달리는 현상이 있었다

![image.png](/assets/img/posts/dev/Infra/n8n Gitlab commit/image%2031.png)

### **원인**

새로운 branch를 처음 push할 때 webhook의 `body.before` 값이 `00000000...` (null commit)이 되고, `body.commits` 배열에는 해당 브랜치의 전체 history가 포함되었다

### **해결 방안**

- `Push Webhook` 노드 바로 다음에 `IF` 노드를 추가한다
- 조건: `&lbrace;&lbrace; $('Push Webhook').first().json.body.before === '0000000000000000000000000000000000000000' &rbrace;&rbrace;`
    - 이 조건이 `true`이면 skip
    - 이 조건이 `false`이면 기존 workflow를 수행하도록 하였다

![image.png](/assets/img/posts/dev/Infra/n8n Gitlab commit/image%2032.png)

![image.png](/assets/img/posts/dev/Infra/n8n Gitlab commit/image%2033.png)

초기 목표는 체크아웃 후 첫 commit부터 가져오는것이였지만, GitLab Webhook이나 API만으로 Branch checkout 시점의 origin branch를 직접적으로 알 수 있는 방법은 없다고 한다
그래서 우선 새로 생긴 브런치 푸쉬만 예외처리를 하였다

## Reference

1. [https://n8n.io/workflows/2167-chatgpt-automatic-code-review-in-gitlab-mr/](https://n8n.io/workflows/2167-chatgpt-automatic-code-review-in-gitlab-mr/)
2. [https://docs.gitlab.com/api/commits/#post-comment-to-commit](https://docs.gitlab.com/api/commits/#post-comment-to-commit)
