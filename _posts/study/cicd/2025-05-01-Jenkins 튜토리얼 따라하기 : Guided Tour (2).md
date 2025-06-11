---
layout: post
title: "Jenkins 튜토리얼 따라하기 : Guided Tour (2)"
date: 2025-05-01 22:14:00+0900
categories: [Study, CICD]
tags: [CICD, Jenkins]
---
## **실행 환경 정의**

- **`agent`**
    - Jenkins가 파이프라인이나 그 일부를 어디에서 어떻게 실행할지를 정의한다
    - 모든 파이프라인에서 필수적이다
- `agent`가 내부적으로 동작하는 방식
    1. `agent` 블록 내에 포함된 모든 단계들이 Jenkins에 의해 실행 대기열에 등록된다. 
    2. 실행 가능한 [executor](https://www.jenkins.io/doc/book/glossary//#executor)가 준비되면, 이 단계들이 실행된다
    3. 소스 제어에서 체크아웃된 파일들과 파이프라인에서 작업에 필요한 추가 파일들을 포함된  [workspace](https://www.jenkins.io/doc/book/glossary//#workspace)가 할당된다.
- **`agent docker`**
    - Jenkins 파이프라인이 Docker 컨테이너 내에서 실행되도록 지정한다
    - 도구 및 환경 구성을 Docker 이미지로 캡슐화하여, 에이전트의 시스템 도구나 종속성을 수동으로 설정하지 않아도 된다
- **Docker를 사용하는 이유**
    1. 파이프라인이 필요한 실행 환경(언어, 프레임워크, CLI 등)을 Docker 이미지 하나로 정의할 수 있다
    2. 다양한 시스템 도구와 라이브러리를 미리 설치한 Docker 이미지를 사용하면, 별도 환경 설정 없이 일관된 실행이 가능하다
    3. Docker 컨테이너 안에서 실행 가능한 도구라면 무엇이든 사용할 수 있다
    

### **Jenkinsfile**

```groovy
pipeline {
    agent {
        docker { image 'node:22.15.0-alpine3.21' }
    }
    stages {
        stage('Test') {
            steps {
                sh 'node --eval "console.log(process.arch,process.platform)"'
            }
        }
    }
}
```

- **`agent`**:  **Docker** 이미지를 지정하여, 해당 이미지 내에서 파이프라인이 실행되도록 설정
    - **`docker { image 'node:22.15.0-alpine3.21' }`**: Docker 이미지를 사용하여 `node:22.15.0-alpine3.21` 이미지를 기반으로 파이프라인을 실행
- **`stages`**: 파이프라인 내의 여러 단계들을 정의
    - **`stage('Test')`**: 'Test'라는 이름의 단계를 정의하고, 그 안에서 테스트를 수행
    - **`steps`**: 각 단계에서 실행할 명령어들을 포함
        - `sh 'node --eval "console.log(process.arch,process.platform)"'`: Docker 컨테이너 내에서 `node` 명령어를 실행하고, 그 결과로 현재 노드 프로세스의 아키텍처와 플랫폼 정보를 출력

### **결과**

![image.png](assets/img/posts/study/cicd/Guided Tour (2)/image.png)

## **환경 변수 사용**

- **`environment`**
    - 파이프라인 전역 또는 특정 단계(`stage`) 내에서 환경 변수를 정의할 수 있다
    - 특정 단계 내에 정의하면, 해당 단계에만 적용된다
- **활용 예시**
    - `Makefile` 또는 셸 스크립트 실행 시, Jenkins 환경에 맞게 동작을 제어하는 용도로 사용
    - 외부 플러그인이 설정한 환경 변수도 존재하므로, 사용하는 플러그인 문서를 참고해야 함
    - 자격 증명 정보를 하드코딩하지 않고 Jenkins Credentials 기능을 통해 안전하게 참조 가능


### **Jenkinsfile**

```groovy
pipeline {
    agent {
        label '!windows'
    }

    environment {
        DISABLE_AUTH = 'true'
        DB_ENGINE    = 'sqlite'
    }

    stages {
        stage('Build') {
            steps {
                echo "Database engine is ${DB_ENGINE}"
                echo "DISABLE_AUTH is ${DISABLE_AUTH}"
                sh 'printenv'
            }
        }
    }
}
```

- **`environment`**: 파이프라인 실행 시 사용할 환경 변수들을 선언하는 블록
    - **`DISABLE_AUTH`**, **`DB_ENGINE`**: 파이프라인 전역에서 접근 가능한 환경 변수
- **`agent`**: `!windows` 라벨이 아닌 모든 에이전트에서 실행되도록 지정
- **`stage('Build')`**: 환경 변수를 사용하는 빌드 단계
    - `echo`: 설정된 환경 변수 값을 출력
    - `sh 'printenv'`: 컨테이너나 셸 내에서 환경 변수 전체를 출력

### **결과**

![image.png](assets/img/posts/study/cicd/Guided Tour (2)/image%201.png)

## **테스트 및 산출물(Artifacts) 기록**

- 테스트는 지속적 통합(CI)의 핵심이지만, 콘솔 출력에서 오류를 찾는 것은 비효율적이다
- Jenkins는 테스트 결과 파일을 기반으로 **자동으로 테스트 결과를 수집하고 UI에 표시**할 수 있다
- 일반적으로는 `JUnit` XML 포맷이 사용되며, 다른 포맷을 위한 플러그인도 존재

### **테스트 코드 및 Jenkinsfile**

![image.png](assets/img/posts/study/cicd/Guided Tour (2)/image%202.png)

```groovy
pipeline {
  agent {
    docker {
      image 'node:18-alpine'
    }
  }

  stages {
    stage('Test') {
      steps {
        sh 'npm install'
        sh 'npm test'
      }
    }
  }

  post {
    always {
      junit 'build/reports/test-results.xml'
    }
  }
}
```

- `post { always { ... } }`: 테스트 실행 여부와 관계없이 항상 실행
- `junit 'build/reports/test-results.xml'`: 해당 XML 파일을 파싱하여 Jenkins에 결과 반영 (Fail/Pass 수, trend 등)

### **결과**

![image.png](assets/img/posts/study/cicd/Guided Tour (2)/image%203.png)

![image.png](assets/img/posts/study/cicd/Guided Tour (2)/image%204.png)

### **실패 테스트 케이스 추가 및 결과**

- 실패 결과를 확인할 수 있다

![image.png](assets/img/posts/study/cicd/Guided Tour (2)/image%205.png)

![image.png](assets/img/posts/study/cicd/Guided Tour (2)/image%206.png)

![image.png](assets/img/posts/study/cicd/Guided Tour (2)/image%207.png)

### **실패 상황을 `UNSTABLE`로 처리하기**

- 일부 테스트 실패는 **"품질 이슈"**로 처리하고 싶을 수 있다
- 이때 `catchError`를 활용하면 **빌드를 실패시키지 않고 `UNSTABLE` 상태로 유지**할 수 있음
- `Status` 비교
    
    
    | 상태       | 의미                                                      |
    | ---------- | --------------------------------------------------------- |
    | `SUCCESS`  | 모든 스테이지 및 테스트가 성공                            |
    | `FAILURE`  | 빌드 자체 실패 (스크립트 오류, 의존성 문제 등)            |
    | `UNSTABLE` | 빌드는 성공했지만 일부 테스트 실패 또는 품질 경고 발생 등 |
    
    ```groovy
    pipeline {
      agent {
        docker { image 'node:18-alpine' }
      }
      stages {
        stage('Test') {
          steps {
            sh 'npm install'
            // npm test 에서 실패를 잡아서 UNSTABLE 로 전환
            catchError(buildResult: 'UNSTABLE', stageResult: 'UNSTABLE') {
              sh 'npm test'
            }
          }
        }
      }
      post {
        always {
          junit 'build/reports/test-results.xml'
        }
      }
    }
    ```
    

![image.png](assets/img/posts/study/cicd/Guided Tour (2)/image%208.png)

### **테스트 결과 및 산출물 저장**

- 테스트 결과 외에도 **산출물(artifacts)**을 저장하여 추후 다운로드나 분석에 활용할 수 있다

```groovy
pipeline {
  agent {
    docker { image 'node:18-alpine' }
  }
  stages {
    stage('Install') {
      steps {
        sh 'npm install'
      }
    }
    stage('Test') {
      steps {
        // 테스트 실패 시 UNSTABLE 처리
        catchError(buildResult: 'UNSTABLE', stageResult: 'UNSTABLE') {
          sh 'npm test'
        }
      }
    }
  }
  post {
    always {
      junit 'build/reports/test-results.xml'
      archiveArtifacts artifacts: 'build/reports/test-results.xml', fingerprint: true
    }
  }
}
```

- `archiveArtifacts`: 지정한 파일을 Jenkins에 저장
    - `fingerprint: true`: 파일의 해시를 기록하여 추적 가능 (빌드 간 비교 등)
- 테스트 결과와 함께 보고서를 **Jenkins UI에서 다운로드 가능**

![image.png](assets/img/posts/study/cicd/Guided Tour (2)/image%209.png)

![image.png](assets/img/posts/study/cicd/Guided Tour (2)/image%2010.png)


## **Reference**
1. [https://www.jenkins.io/doc/pipeline/tour/agents/](https://www.jenkins.io/doc/pipeline/tour/agents/)
2. [https://www.jenkins.io/doc/pipeline/tour/environment/](https://www.jenkins.io/doc/pipeline/tour/environment/)
3. [https://www.jenkins.io/doc/pipeline/tour/tests-and-artifacts/](https://www.jenkins.io/doc/pipeline/tour/tests-and-artifacts/)
