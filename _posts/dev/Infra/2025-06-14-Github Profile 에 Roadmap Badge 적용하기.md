---
layout: post
title: "Github Profile 에 Roadmap.sh Badge 적용하기"
date: 2025-06-14 16:28:00+0900
categories: [Dev, Infra]
tags: [Infra, Roadmap.sh, Github Profile, Badge]
---

## Roadmap.sh

새로운 언어를 공부하는데 그냥 공부하니까 너무 재미가 없어서

공부법을 찾아보면 한번씩 보이던 [https://roadmap.sh/](https://roadmap.sh/) 를 이용하였다

![image.png](/assets/img/posts/dev/Infra/Roadmap Badge/image.png)

깃허브에서 Stars Rank가 무려 6등이다

![image.png](/assets/img/posts/dev/Infra/Roadmap Badge/image%201.png)

다양한 기술이나 분야에 대해 로드맵이 있다

자신이 관심있는 기술이나 분야를 북마크해서 보면 된다

![image.png](/assets/img/posts/dev/Infra/Roadmap Badge/image%202.png)

로드맵에는 각 항목에 대해 중요도를 표시해준다

![image.png](/assets/img/posts/dev/Infra/Roadmap Badge/image%203.png)

항목을 클릭하면 관련된 유료, 무료 학습 자료들을 볼 수 있는데

공부자체는 꼭 적힌 링크가 아니라 AI Tutor에 질문하거나 구글 검색으로 해도 된다

공부하면서 진행 상태를 변경해주면 된다

![image.png](/assets/img/posts/dev/Infra/Roadmap Badge/image%204.png)

**Roadmap.sh**에서는

내 진도율을 마크다운, html 등으로 Badge로 등록할 수 있도록 제공해준다

깃허브 프로필에 내 학습 상태가 표시되니까 동기부여가 되는거 같다

![image.png](/assets/img/posts/dev/Infra/Roadmap Badge/image%205.png)

## Badge 만들기

**Visit Profile → Edit Profile → Road Card**
`HTML`, `Markdown`, `SVG` 등 다양한 형식으로 Badge를 만들 수 있다 

![image.png](/assets/img/posts/dev/Infra/Roadmap Badge/image%206.png)

![image.png](/assets/img/posts/dev/Infra/Roadmap Badge/image%207.png)

![image.png](/assets/img/posts/dev/Infra/Roadmap Badge/image%208.png)

## Workflow로 Badge Update (Github Profile)

Github에서 자체 Caching을 통해 이미지를 저장해두기 때문에

Roadmap.sh에서 학습을 진행해도 깃허브 프로필에는 반영이 잘 되지 않는다

자세한 내용은 [**Issue#7537**](https://github.com/kamranahmedse/developer-roadmap/issues/7537)에서 확인할 수 있다  

**Cache busting**을 통해 이를 해결할 수 있는데

Workflow에서 timestamp를 추가하는 방식으로 이미지의 주소를 계속 업데이트해준다

### 1. Workflow 생성

![image.png](/assets/img/posts/dev/Infra/Roadmap Badge/image%209.png)

```yaml
name: Update Roadmap.sh Badge

on:
  schedule:
    - cron: '0 0 * * *'  # 00:00 자동 실행
  workflow_dispatch:     # 수동 실행

jobs:
  update:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      # README.md의 뱃지 URL 업데이트
      - name: Update roadmap.sh badge URLs 
        run: |
          TIMESTAMP=$(date +%s)
          README_FILE="README.md"

          echo "Updating badges in $README_FILE with timestamp $TIMESTAMP"

          TEMP_FILE=$(mktemp)

          # 1단계: 기존의 t=... 파라미터를 모두 제거하여 초기화합니다.
          # [?&]는 '?' 또는 '&'를 의미하며, 기존 타임스탬프를 깔끔하게 지웁니다.
          sed -E 's/([?&])t=[0-9]*//g' "$README_FILE" > "$TEMP_FILE"

          # 2단계: 모든 roadmap.sh URL 뒤에 '&t=타임스탬프'를 추가합니다.
          # 이 단계에서는 URL에 '?'가 있든 없든 일단 '&'로 추가합니다.
          # sed의 구분자로 '|'를 사용하여 URL의 '/'와 충돌하지 않도록 합니다.
          sed -E "s|(https://roadmap.sh/card/[^)]+)|\\1\&t=$TIMESTAMP|g" "$TEMP_FILE" > "${TEMP_FILE}.step2"

          # 3단계: 잘못 추가된 '&'를 '?'로 수정합니다.
          # '/card/' 뒤에 '?'가 없는데 '&t='가 붙은 경우, 첫 '&'를 '?'로 바꿉니다.
          # 이 명령으로 ?가 없는 순수한 URL에 타임스탬프가 올바르게 추가됩니다.
          sed -E 's|(/card/[^?)]*)\&t=|\1?t=|g' "${TEMP_FILE}.step2" > "$README_FILE"

          # 임시 파일 삭제
          rm "$TEMP_FILE" "${TEMP_FILE}.step2"

          echo "Badge update process complete."
          
      - name: Commit changes
        run: |
          git config --local user.email "action@github.com"
          git config --local user.name "GitHub Action"
          git add README.md
          git diff --staged --quiet || git commit -m "Update Roadmap.sh Badge [$(date)]"
          git push
```

Workflow 파일을 만들고 Actions 탭으로 가면 생성한 workflow를 확인할 수 있다

![image.png](/assets/img/posts/dev/Infra/Roadmap Badge/image%2010.png)

### 2. Workflow Permission 설정

workflow는 UTC 기준 00시(한국 09시)에 자동으로 작동한다

하지만 지금은 테스트를 위해 `Run workflow`로 바로 작동해보았다

![image.png](/assets/img/posts/dev/Infra/Roadmap Badge/image%2011.png)

권한 설정이 되지 않아 commit에 실패했다

![image.png](/assets/img/posts/dev/Infra/Roadmap Badge/image%2012.png)

Setting → Action → General 에서 `Read and write permissions` 권한을 설정해준다

![image.png](/assets/img/posts/dev/Infra/Roadmap Badge/image%2013.png)

다시 워크플로우를 실행하여 업데이트 되는지 확인한다

![image.png](/assets/img/posts/dev/Infra/Roadmap Badge/image%2014.png)

깃허브 프로필에도 제대로 적용되는지 확인한다

![image.png](/assets/img/posts/dev/Infra/Roadmap Badge/image%2015.png)

커밋 내용에서도 업데이트가 제대로 반영된걸 확인할 수 있다

![image.png](/assets/img/posts/dev/Infra/Roadmap Badge/image%2016.png)

![image.png](/assets/img/posts/dev/Infra/Roadmap Badge/image%2017.png)
