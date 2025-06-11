---
layout: post
title: "Jenkins 튜토리얼 따라하기 : Guided Tour (3)"
date: 2025-05-05 16:54:00+0900
categories: [Study, CICD]
tags: [CICD, Jenkins]
---
## **파이프라인 종료 후 처리 및 알림**

- **`post`**
    - `pipeline` 블록 안에서 사용하는 **특수 블록**으로, 파이프라인이 **끝난 후 실행할 작업을 정의**한다
    - `always`, `success`, `failure`, `unstable`, `aborted`, `changed` 등 다양한 조건에 따라 후처리를 지정할 수 있다
    - 주로 **정리(cleanup)**, **알림(notification)**, **테스트 결과 수집**, **산출물 저장** 등에 활용된다

### **Jenkinsfile**

```groovy
pipeline {
    agent any
    stages {
        stage('No-op') {
            steps {
                sh 'ls'
            }
        }
    }
    post {
        always {
            echo 'One way or another, I have finished'
            deleteDir() /* clean up our workspace */
        }
        success {
            echo 'I succeeded!'
        }
        unstable {
            echo 'I am unstable :/'
        }
        failure {
            echo 'I failed :('
        }
        changed {
            echo 'Things were different before...'
        }
    }
}
```
  
![image.png](assets/img/posts/study/cicd/Guided Tour (3)/image.png)

### **Mattermost 알림 전송**

- **Webhook URL**을 이용하여 전송

```groovy
pipeline {
    agent any
    stages {
        stage('No-op') {
            steps {
                sh 'ls'
            }
        }
    }
    post {
        always {
            echo 'One way or another, I have finished'
            deleteDir() /* clean up our workspace */
        }
        success {
            sh """
            curl -X POST -H 'Content-Type: application/json' \
            -d '{
                "text": "✅ 파이프라인 ${env.JOB_NAME} #${env.BUILD_NUMBER} 이(가) 성공적으로 완료되었습니다."
            }' https://mattermost.example.com/hooks/YOUR_WEBHOOK_TOKEN
            """
        }
        unstable {
            echo 'I am unstable :/'
        }
        failure {
            echo 'I failed :('
        }
        changed {
            echo 'Things were different before...'
        }
    }
}
```

![image.png](assets/img/posts/study/cicd/Guided Tour (3)/image%201.png)

- Mattermost 채널에서 메세지를 확인할 수 있다

![image.png](assets/img/posts/study/cicd/Guided Tour (3)/image%202.png)

### **Jenkins Credentials 사용**

- Webhook URL을 공개된 소스코드나 저장소(GitHub 등)에 노출시키면 안된다
    - Mattermost Webhook URL은 **인증 없이 누구나** 해당 주소로 POST 요청을 보내면 메시지를 보낼 수 있다
    - 따라서, Webhook **URL 자체가 인증 수단이며, 유출 시 외부인이 채널에 스팸이나 악성 메시지를 보낼 수 있다**
- **설정 방법**
    1. Jenkins → **"자격 증명(Credentials)"** 메뉴로 이동
    2. **"Secret Text"** 추가
        - ID 예시: `MATTERMOST_WEBHOOK`

### Jenkinsfile

```groovy
post {
    success {
        withCredentials([string(credentialsId: 'MATTERMOST_WEBHOOK', variable: 'HOOK_URL')]) {
            sh """
            curl -X POST -H 'Content-Type: application/json' \\
            -d '{"text": "✅ 파이프라인 ${env.JOB_NAME} #${env.BUILD_NUMBER} 성공"}' $HOOK_URL
            """
        }
    }
}
```

- `withCredentials`: Jenkins에서 보안 정보를 안전하게 참조
- `env.JOB_NAME`, `env.BUILD_NUMBER`: Jenkins 내장 환경 변수

# 배포

- 가장 기본적인 **continuous delivery Pipeline**은 최소 세 단계로 `Jenkinsfile` 에서 구성된다
    - **Build**
    - **Test**
    - **Deploy**

```groovy
pipeline {
    agent any
    options {
        skipStagesAfterUnstable()
    }
    environment {
        IMAGE_NAME = "my-vite-app"
        IMAGE_TAG  = "latest"
    }
    stages {
        stage('Build') {
            steps {
                echo '✅ 빌드 시작'
                sh 'docker build -t $IMAGE_NAME:$IMAGE_TAG .'
            }
        }
        stage('Test') {
            steps {
                echo '✅ 테스트 (예: 이미지 검사)'
                sh 'docker run --rm $IMAGE_NAME:$IMAGE_TAG sh -c "npm test || exit 0"'
            }
        }
        stage('Deploy') {
            steps {
                echo '✅ 배포'
                sh '''
                   docker rm -f vite-prod || true
                   docker run -d --name vite-prod -p 8000:80 $IMAGE_NAME:$IMAGE_TAG
                '''
            }
        }
    }
    post {
        always {
            echo '✅ 파이프라인 종료'
        }
    }
}
```

![image.png](assets/img/posts/study/cicd/Guided Tour (3)/image%203.png)

## 배포 환경별 단계 확장

- `staging`, `production` 같은 배포 환경을 캡처하기 위해 stage를 늘리는 패턴이 자주 사용된다

```groovy
pipeline {
    agent any
    options {
        skipStagesAfterUnstable()
    }
    environment {
        IMAGE_NAME = "my-vite-app"
        IMAGE_TAG  = "latest"
    }
    stages {
        stage('Build') {
            steps {
                echo '✅ 빌드 시작'
                sh 'docker build -t $IMAGE_NAME:$IMAGE_TAG .'
            }
        }
        stage('Test') {
            steps {
                echo '✅ 테스트 (예: 이미지 검사)'
                sh 'docker run --rm $IMAGE_NAME:$IMAGE_TAG sh -c "npm test || exit 0"'
            }
        }
        stage('Deploy - Staging') {
            steps {
                echo '✅ 스테이징 배포'
                sh '''
                   docker rm -f vite-staging || true
                   docker run -d --name vite-staging -p 5173:5173 $IMAGE_NAME:$IMAGE_TAG
                '''
                // sh './run-smoke-tests'
            }
        }
        stage('Deploy - Production') {
            steps {
                echo '✅ 프로덕션 배포'
                sh '''
                   docker rm -f vite-prod || true
                   docker run -d --name vite-prod -p 8000:80 $IMAGE_NAME:$IMAGE_TAG
                '''
            }
        }
    }
    post {
        always {
            echo '✅ 파이프라인 종료'
        }
    }
}
```

- **스모크 테스트 추가 예시**
    
    `Deploy - Staging` 단계에서 `./run-smoke-tests` 스크립트를 호출해 간단한 검증을 수행한 뒤, 결과에 따라 프로덕션 배포 여부를 결정할 수 있다
    
- **Continuous Delivery vs Continuous Deployment**
    - **Continuous Deployment:** 위와 같이 자동으로 프로덕션까지 배포하는 파이프라인
    - **Continuous Delivery :** 조직마다 검토·승인 절차가 필요할 수 있으므로, 이 과정을 수작업 승인(input) 단계로 대체

![image.png](assets/img/posts/study/cicd/Guided Tour (3)/image%204.png)

### 사용자 승인 요청

- `input` step을 이용하면 stage간 전환 시 **사용자의 입력**을 통한 과정을 진행할 수 있다

```groovy
pipeline {
    agent any
    options {
        skipStagesAfterUnstable()
    }
    environment {
        IMAGE_NAME = "my-vite-app"
        IMAGE_TAG  = "latest"
    }
    stages {
        stage('Build') {
            steps {
                echo '✅ 빌드 시작'
                sh 'docker build -t $IMAGE_NAME:$IMAGE_TAG .'
            }
        }
        stage('Test') {
            steps {
                echo '✅ 테스트 (예: 이미지 검사)'
                sh 'docker run --rm $IMAGE_NAME:$IMAGE_TAG sh -c "npm test || exit 0"'
            }
        }
        stage('Deploy - Staging') {
            steps {
                echo '✅ 스테이징 배포'
                sh '''
                   docker rm -f vite-staging || true
                   docker run -d --name vite-staging -p 5173:5173 $IMAGE_NAME:$IMAGE_TAG
                '''
            }
        }
        stage('Sanity Check') {
            steps {
                input message: "스테이징 배포 확인 후 Continue", ok: "Continue"
            }
        }
        stage('Deploy - Production') {
            steps {
                echo '✅ 프로덕션 배포'
                sh '''
                   docker rm -f vite-prod || true
                   docker run -d --name vite-prod -p 8000:80 $IMAGE_NAME:$IMAGE_TAG
                '''
            }
        }
    }
    post {
        always {
            echo '✅ 파이프라인 종료'
        }
    }
}
```

- `Sanity check` : 사용자가 `Continue` 또는 `Abort` 를 입력할 때까지 작업을 대기한다

![image.png](assets/img/posts/study/cicd/Guided Tour (3)/image%205.png)

- Abort를 선택하면 Production Stage를 건너뛴다

![image.png](assets/img/posts/study/cicd/Guided Tour (3)/image%206.png)

## Reference

1. [https://www.jenkins.io/doc/pipeline/tour/post/](https://www.jenkins.io/doc/pipeline/tour/post/)
2. [https://www.jenkins.io/doc/pipeline/tour/deployment/](https://www.jenkins.io/doc/pipeline/tour/deployment/)