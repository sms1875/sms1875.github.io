---
layout: post
title: "Ant Media Server + RTMP 영상 재생"
date: 2025-06-09 23:04:00+0900
categories: [Dev, Infra]
tags: [Infra, Ant Media Server, RTMP, Docker, Streaming]
---

## Setting

### 설치

Dockerfile과 docker-compose 파일을 다운받는다

```bash
wget https://raw.githubusercontent.com/ant-media/Scripts/master/docker/docker-compose.yml
wget https://raw.githubusercontent.com/ant-media/Scripts/master/docker/Dockerfile_Process -O Dockerfile
```

github에서 community 버전 zip 파일을 다운받는다

[https://github.com/ant-media/Ant-Media-Server/releases](https://github.com/ant-media/Ant-Media-Server/releases)

![image.png](/assets/img/posts/dev/Infra/Ant Media Server + RTMP 영상 재생/image.png)

### docker compose 설정

- args
    - AntMediaServer : github 에서 다운받은 zip 이름
- windows 에서는 아직 `network_mode: host` 이 지원이 되지 않아서 port를 설정해주었다

```yaml
services:
  antmedia:
    build:
      context: ./
      dockerfile: ./Dockerfile
      args:
        AntMediaServer: "ant-media-server-community-2.13.2.zip"
    ports:
      - "5080:5080" # HTTP
      - "5443:5443" # HTTPS
      - "1935:1935" # RTMP
    container_name: antmedia
    restart: always
    entrypoint: /usr/local/antmedia/start.sh
#   network_mode: host
#   volumes:
#      - antmedia_vol:/usr/local/antmedia/
#volumes:
#  antmedia_vol:
#    external: true
#    name:
#      antmedia_volume

```

명령어를 실행한다

```bash
docker-compose build
docker-compose up -d
```

[http://localhost:5080/](http://localhost:5080/) 에 접속해서 Ant Media Server 관리 화면이 나오는지 확인한다

![image.png](/assets/img/posts/dev/Infra/Ant Media Server + RTMP 영상 재생/image%201.png)

계정을 만들고 로그인을 하면 대시보드를 확인할 수 있다

![image.png](/assets/img/posts/dev/Infra/Ant Media Server + RTMP 영상 재생/image%202.png)

## Streaming Test

### OBS Studio 송출

OBS Studio설정에서 방송 → 서비스 : 사용자 지정 → 서버와 스트림 키 입력

스트림 키는 자신이 원하는 것으로 입력하면 된다

![image.png](/assets/img/posts/dev/Infra/Ant Media Server + RTMP 영상 재생/image%203.png)

대시보드를 확인하면 Active Live Streams 가 1로 추가된것을 확인할 수 있다

![image.png](/assets/img/posts/dev/Infra/Ant Media Server + RTMP 영상 재생/image%204.png)

### 영상 확인

왼쪽 메뉴의 APPLICATIONS에서 LiveAPP 클릭하면 Stream 목록을 확인할 수 있다

![image.png](/assets/img/posts/dev/Infra/Ant Media Server + RTMP 영상 재생/image%205.png)

![image.png](/assets/img/posts/dev/Infra/Ant Media Server + RTMP 영상 재생/image%206.png)

![image.png](/assets/img/posts/dev/Infra/Ant Media Server + RTMP 영상 재생/image%207.png)

## Ant Media Server 설정

Community 에서는 영상 저장, api security, 푸쉬 알람 등이 설정 가능하다

![image.png](/assets/img/posts/dev/Infra/Ant Media Server + RTMP 영상 재생/image%208.png)

### 자동 영상 저장 설정

obs에서 방송을 종료하면 VoD에 저장된것을 확인할 수 있다

![image.png](/assets/img/posts/dev/Infra/Ant Media Server + RTMP 영상 재생/image%209.png)

![image.png](/assets/img/posts/dev/Infra/Ant Media Server + RTMP 영상 재생/image%2010.png)

S3계정이 있으면 S3 에도 저장이 가능하다

![image.png](/assets/img/posts/dev/Infra/Ant Media Server + RTMP 영상 재생/image%2011.png)

### API Security

JWT, IP 화이트리스트 등을 통해 보안 설정이 가능하다

지금은 원할한 영상 재생을 위해 보안을 해제했다

![image.png](/assets/img/posts/dev/Infra/Ant Media Server + RTMP 영상 재생/image%2012.png)

## API 통신 테스트

[https://antmedia.io/rest/](https://antmedia.io/rest/) 에서 api 목록을 확인할 수 있다

- 현재 방송 중인 목록 가져오기
    - Request
        
        ```bash
        curl -X 'GET' 'http://localhost:5080/LiveApp/rest/v2/broadcasts/list/0/10?type_by=liveStream' -H 'accept: application/json'
        ```
        
    - Response
        
        ```bash
        [{"streamId":"test","status":"broadcasting","playListStatus":null,"type":"liveStream","publishType":"RTMP","name":null,"description":null,"publish":true,"date":1747187095739,"plannedStartDate":0,"plannedEndDate":0,"duration":60054,"endPointList":null,"playListItemList":null,"publicStream":true,"is360":false,"listenerHookURL":null,"category":null,"ipAddr":null,"username":null,"password":null,"quality":null,"speed":0.997,"streamUrl":null,"originAdress":"172.21.0.2","mp4Enabled":1,"webMEnabled":0,"seekTimeInMs":0,"conferenceMode":null,"subtracksLimit":-1,"expireDurationMS":0,"rtmpURL":"rtmp://172.21.0.2/LiveApp/test","zombi":true,"pendingPacketSize":1,"hlsViewerCount":0,"dashViewerCount":0,"webRTCViewerCount":0,"rtmpViewerCount":0,"startTime":1747187096837,"receivedBytes":18774200,"bitrate":2503224,"width":1280,"height":720,"encoderQueueSize":0,"dropPacketCountInIngestion":0,"dropFrameCountInEncoding":0,"packetLostRatio":0.0,"packetsLost":0,"jitterMs":0,"rttMs":0,"userAgent":"N/A","remoteIp":null,"latitude":null,"longitude":null,"altitude":null,"mainTrackStreamId":"","subTrackStreamIds":[],"absoluteStartTimeMs":0,"webRTCViewerLimit":-1,"hlsViewerLimit":-1,"dashViewerLimit":-1,"subFolder":null,"currentPlayIndex":0,"metaData":"","playlistLoopEnabled":true,"updateTime":1747187156891,"role":"","hlsParameters":null,"autoStartStopEnabled":false,"encoderSettingsList":null,"virtual":false,"anyoneWatching":false}]
        ```
        
        ![image.png](/assets/img/posts/dev/Infra/Ant Media Server + RTMP 영상 재생/image%2013.png)
        

## 방송 화면 표시하기

### 실시간 영상 확인

- html
    - `broadcasts/list/0/10?type_by=liveStream` :  StreamId를 가져온다
    - StreamId를 포함한 src를 구성해서 frame에서 재생한다
    
    ```html
    <!DOCTYPE html>
    <html lang="ko">
      <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>Ant Media Server - 방송 목록</title>
        <style>
          body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
              Oxygen, Ubuntu, Cantarell, "Open Sans", "Helvetica Neue", sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f4f7f6;
            color: #333;
          }
          .main-title {
            text-align: center;
            color: #2c3e50;
            margin-bottom: 30px;
          }
          .section {
            background-color: #fff;
            padding: 20px;
            margin-bottom: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
          }
          .section-title {
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
            margin-top: 0;
            margin-bottom: 20px;
            color: #3498db;
            font-size: 1.8em;
          }
          .item-container {
            margin-bottom: 25px;
            padding: 15px;
            border: 1px solid #ecf0f1;
            border-radius: 5px;
            background-color: #fdfdfd;
          }
          .item-container h3 {
            margin-top: 0;
            margin-bottom: 10px;
            color: #333;
            font-size: 1.4em;
          }
          .item-container iframe {
            width: 100%;
            max-width: 640px;
            height: auto; /* 자동 높이 조절, width에 맞춰 비율 유지 */
            aspect-ratio: 16 / 9; /* 16:9 비율 유지 */
            border: none;
            border-radius: 4px;
            background-color: #000; /* 비디오 로딩 중 배경색 */
          }
          .item-details {
            font-size: 0.85em;
            color: #555;
            margin-bottom: 10px;
            line-height: 1.6;
          }
          .item-details span {
            margin-right: 15px;
          }
          .loading-message,
          .error-message {
            text-align: center;
            font-size: 1.1em;
            color: #7f8c8d;
            padding: 20px;
          }
          .error-message {
            color: #e74c3c;
          }
          .no-content {
            text-align: center;
            color: #7f8c8d;
            padding: 20px;
            font-style: italic;
          }
        </style>
      </head>
      <body>
        <h1 class="main-title">Ant Media Server 연동 페이지</h1>
    
        <div class="section">
          <h2 class="section-title">실시간 방송 목록</h2>
          <div id="loadingMessageLive" class="loading-message">
            실시간 방송 목록을 불러오는 중...
          </div>
          <div id="streamsListContainer"></div>
        </div>
    
        <script>
          // --- 설정 (사용자 환경에 맞게 수정해주세요) ---
          const antMediaServerUrl = "http://localhost:5080"; // Ant Media Server 주소
          const applicationName = "LiveApp"; // 사용하는 애플리케이션 이름
          // -----------------------------------------
    
          // 헬퍼 함수: 밀리초를 MM:SS 또는 HH:MM:SS 형식으로 변환
          function formatDuration(ms) {
            if (ms === null || typeof ms === "undefined" || ms < 0) return "N/A";
            let seconds = Math.floor(ms / 1000);
            let minutes = Math.floor(seconds / 60);
            let hours = Math.floor(minutes / 60);
            seconds %= 60;
            minutes %= 60;
            const pad = (num) => String(num).padStart(2, "0");
            if (hours > 0) {
              return `${pad(hours)}:${pad(minutes)}:${pad(seconds)}`;
            }
            return `${pad(minutes)}:${pad(seconds)}`;
          }
    
          // 헬퍼 함수: 타임스탬프를 YYYY-MM-DD HH:MM:SS 형식으로 변환
          function formatTimestamp(ts) {
            if (ts === null || typeof ts === "undefined" || ts <= 0) return "N/A";
            const date = new Date(ts);
            const pad = (num) => String(num).padStart(2, "0");
            return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(
              date.getDate()
            )} ${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(
              date.getSeconds()
            )}`;
          }
    
          // 실시간 방송 목록 가져와서 표시하는 함수
          async function fetchAndDisplayLiveStreams() {
            const streamsContainer = document.getElementById(
              "streamsListContainer"
            );
            const loadingMessage = document.getElementById("loadingMessageLive");
            streamsContainer.innerHTML = ""; // 기존 목록 초기화
            loadingMessage.style.display = "block";
    
            try {
              const response = await fetch(
                `${antMediaServerUrl}/${applicationName}/rest/v2/broadcasts/list/0/10?type_by=liveStream`,
                {
                  method: "GET",
                  headers: { Accept: "application/json" },
                }
              );
    
              if (!response.ok) {
                throw new Error(
                  `API 호출 실패 (Live): ${response.status} ${response.statusText}. 서버 응답을 확인하세요.`
                );
              }
              const broadcasts = await response.json();
              const liveStreams = broadcasts.filter(
                (stream) => stream.status === "broadcasting"
              );
    
              loadingMessage.style.display = "none";
    
              if (liveStreams.length === 0) {
                streamsContainer.innerHTML =
                  '<p class="no-content">현재 진행 중인 실시간 방송이 없습니다.</p>';
                return;
              }
    
              liveStreams.forEach((stream) => {
                const streamId = stream.streamId;
                const streamName = stream.name || streamId; // 스트림 이름이 없으면 ID 사용
    
                const streamDiv = document.createElement("div");
                streamDiv.className = "item-container"; // 공통 클래스 사용
                streamDiv.innerHTML = `<h3>${streamName}</h3>`;
    
                const iframeSrc = `${antMediaServerUrl}/${applicationName}/play.html?name=${streamId}&autoplay=true&mute=true&playType=webrtc,hls`; // WebRTC 우선, HLS fallback
                const iframeElement = document.createElement("iframe");
                iframeElement.setAttribute("src", iframeSrc);
                iframeElement.setAttribute("frameborder", "0");
                iframeElement.setAttribute("allowfullscreen", "");
                iframeElement.setAttribute(
                  "allow",
                  "autoplay; fullscreen;encrypted-media;picture-in-picture"
                );
    
                streamDiv.appendChild(iframeElement);
                streamsContainer.appendChild(streamDiv);
              });
            } catch (error) {
              console.error("실시간 방송 정보 로딩 오류:", error);
              loadingMessage.style.display = "none";
              streamsContainer.innerHTML = `<p class="error-message">실시간 방송 목록 로딩 오류: ${error.message}<br>CORS 설정 및 Ant Media Server 상태를 확인하세요.</p>`;
            }
          }
    
          // 페이지 로드 시 모든 함수 실행
          document.addEventListener("DOMContentLoaded", () => {
            fetchAndDisplayLiveStreams();
            fetchAndDisplayVoDs();
          });
    
          // (선택 사항) 주기적으로 실시간 방송 목록 갱신
          // setInterval(fetchAndDisplayLiveStreams, 30000); // 30초마다
        </script>
      </body>
    </html>
    
    ```
    
- 결과
    
    ![image.png](/assets/img/posts/dev/Infra/Ant Media Server + RTMP 영상 재생/image%2014.png)
    

### 저장된 영상 확인

- html
    - `vods/list/0/20?sort_by=date&order_by=desc` :  VoD 객체 목록을 가져온다
    - VoD 객체에서 경로를 가져와서 vide에서 재생한다
    
    ```html
    <!DOCTYPE html>
    <html lang="ko">
      <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>Ant Media Server - 방송 및 VoD 목록</title>
        <style>
          body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
              Oxygen, Ubuntu, Cantarell, "Open Sans", "Helvetica Neue", sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f4f7f6;
            color: #333;
          }
          .main-title {
            text-align: center;
            color: #2c3e50;
            margin-bottom: 30px;
          }
          .section {
            background-color: #fff;
            padding: 20px;
            margin-bottom: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
          }
          .section-title {
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
            margin-top: 0;
            margin-bottom: 20px;
            color: #3498db;
            font-size: 1.8em;
          }
          .item-container {
            /* stream-container, vod-container 공통 스타일 */
            margin-bottom: 25px;
            padding: 15px;
            border: 1px solid #ecf0f1;
            border-radius: 5px;
            background-color: #fdfdfd;
          }
          .item-container h3 {
            margin-top: 0;
            margin-bottom: 10px;
            color: #333;
            font-size: 1.4em;
          }
          .item-container iframe,
          .item-container video {
            width: 100%;
            max-width: 640px;
            height: auto; /* 자동 높이 조절, width에 맞춰 비율 유지 */
            aspect-ratio: 16 / 9; /* 16:9 비율 유지 */
            border: none;
            border-radius: 4px;
            background-color: #000; /* 비디오 로딩 중 배경색 */
          }
          .item-details {
            font-size: 0.85em;
            color: #555;
            margin-bottom: 10px;
            line-height: 1.6;
          }
          .item-details span {
            margin-right: 15px;
          }
          .loading-message,
          .error-message {
            text-align: center;
            font-size: 1.1em;
            color: #7f8c8d;
            padding: 20px;
          }
          .error-message {
            color: #e74c3c;
          }
          .no-content {
            text-align: center;
            color: #7f8c8d;
            padding: 20px;
            font-style: italic;
          }
        </style>
      </head>
      <body>
        <h1 class="main-title">Ant Media Server 연동 페이지</h1>
    
        <div class="section">
          <h2 class="section-title">실시간 방송 목록</h2>
          <div id="loadingMessageLive" class="loading-message">
            실시간 방송 목록을 불러오는 중...
          </div>
          <div id="streamsListContainer"></div>
        </div>
    
        <div class="section">
          <h2 class="section-title">저장된 영상 (VoD)</h2>
          <div id="loadingMessageVoD" class="loading-message">
            저장된 영상 목록을 불러오는 중...
          </div>
          <div id="vodsListContainer"></div>
        </div>
    
        <script>
          // --- 설정 (사용자 환경에 맞게 수정해주세요) ---
          const antMediaServerUrl = "http://localhost:5080"; // Ant Media Server 주소
          const applicationName = "LiveApp"; // 사용하는 애플리케이션 이름
          // -----------------------------------------
    
          // 헬퍼 함수: 밀리초를 MM:SS 또는 HH:MM:SS 형식으로 변환
          function formatDuration(ms) {
            if (ms === null || typeof ms === "undefined" || ms < 0) return "N/A";
            let seconds = Math.floor(ms / 1000);
            let minutes = Math.floor(seconds / 60);
            let hours = Math.floor(minutes / 60);
            seconds %= 60;
            minutes %= 60;
            const pad = (num) => String(num).padStart(2, "0");
            if (hours > 0) {
              return `${pad(hours)}:${pad(minutes)}:${pad(seconds)}`;
            }
            return `${pad(minutes)}:${pad(seconds)}`;
          }
    
          // 헬퍼 함수: 타임스탬프를 YYYY-MM-DD HH:MM:SS 형식으로 변환
          function formatTimestamp(ts) {
            if (ts === null || typeof ts === "undefined" || ts <= 0) return "N/A";
            const date = new Date(ts);
            const pad = (num) => String(num).padStart(2, "0");
            return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(
              date.getDate()
            )} ${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(
              date.getSeconds()
            )}`;
          }
    
          // 실시간 방송 목록 가져와서 표시하는 함수
          async function fetchAndDisplayLiveStreams() {
            const streamsContainer = document.getElementById(
              "streamsListContainer"
            );
            const loadingMessage = document.getElementById("loadingMessageLive");
            streamsContainer.innerHTML = ""; // 기존 목록 초기화
            loadingMessage.style.display = "block";
    
            try {
              const response = await fetch(
                `${antMediaServerUrl}/${applicationName}/rest/v2/broadcasts/list/0/10?type_by=liveStream`,
                {
                  method: "GET",
                  headers: { Accept: "application/json" },
                }
              );
    
              if (!response.ok) {
                throw new Error(
                  `API 호출 실패 (Live): ${response.status} ${response.statusText}. 서버 응답을 확인하세요.`
                );
              }
              const broadcasts = await response.json();
              const liveStreams = broadcasts.filter(
                (stream) => stream.status === "broadcasting"
              );
    
              loadingMessage.style.display = "none";
    
              if (liveStreams.length === 0) {
                streamsContainer.innerHTML =
                  '<p class="no-content">현재 진행 중인 실시간 방송이 없습니다.</p>';
                return;
              }
    
              liveStreams.forEach((stream) => {
                const streamId = stream.streamId;
                const streamName = stream.name || streamId; // 스트림 이름이 없으면 ID 사용
    
                const streamDiv = document.createElement("div");
                streamDiv.className = "item-container"; // 공통 클래스 사용
                streamDiv.innerHTML = `<h3>${streamName}</h3>`;
    
                const iframeSrc = `${antMediaServerUrl}/${applicationName}/play.html?name=${streamId}&autoplay=true&mute=true&playType=webrtc,hls`; // WebRTC 우선, HLS fallback
                const iframeElement = document.createElement("iframe");
                iframeElement.setAttribute("src", iframeSrc);
                iframeElement.setAttribute("frameborder", "0");
                iframeElement.setAttribute("allowfullscreen", "");
                iframeElement.setAttribute(
                  "allow",
                  "autoplay; fullscreen;encrypted-media;picture-in-picture"
                );
    
                streamDiv.appendChild(iframeElement);
                streamsContainer.appendChild(streamDiv);
              });
            } catch (error) {
              console.error("실시간 방송 정보 로딩 오류:", error);
              loadingMessage.style.display = "none";
              streamsContainer.innerHTML = `<p class="error-message">실시간 방송 목록 로딩 오류: ${error.message}<br>CORS 설정 및 Ant Media Server 상태를 확인하세요.</p>`;
            }
          }
    
          // 저장된 영상(VoD) 목록 가져와서 표시하는 함수
          async function fetchAndDisplayVoDs() {
            const vodsContainer = document.getElementById("vodsListContainer");
            const loadingMessage = document.getElementById("loadingMessageVoD");
            vodsContainer.innerHTML = "";
            loadingMessage.style.display = "block";
    
            try {
              const response = await fetch(
                `${antMediaServerUrl}/${applicationName}/rest/v2/vods/list/0/20?sort_by=date&order_by=desc`,
                {
                  // 최근 20개, 최신순 정렬
                  method: "GET",
                  headers: { Accept: "application/json" },
                }
              );
    
              if (!response.ok) {
                throw new Error(
                  `API 호출 실패 (VoD): ${response.status} ${response.statusText}. 서버 응답을 확인하세요.`
                );
              }
              const vods = await response.json();
    
              loadingMessage.style.display = "none";
    
              if (!vods || vods.length === 0) {
                vodsContainer.innerHTML =
                  '<p class="no-content">저장된 영상이 없습니다.</p>';
                return;
              }
    
              vods.forEach((vod) => {
                const vodId = vod.vodId;
                const vodDisplayName = vod.vodName || vodId;
                const vodFileName = vod.vodName; // API 응답의 vodName이 실제 파일명 (예: "my_video.mp4")이라고 가정
    
                if (!vodFileName) {
                  // vodName이 없는 경우 건너뛰거나 다른 ID 사용
                  console.warn("VoD 이름이 없어 건너뜁니다:", vod);
                  return;
                }
    
                const creationDateFormatted = formatTimestamp(vod.creationDate);
                const durationFormatted = formatDuration(vod.duration);
                const fileSizeMB = vod.fileSize
                  ? (vod.fileSize / (1024 * 1024)).toFixed(2) + " MB"
                  : "N/A";
    
                const vodDiv = document.createElement("div");
                vodDiv.className = "item-container"; // 공통 클래스 사용
    
                vodDiv.innerHTML = `
                            <h3>${vodDisplayName}</h3>
                            <div class="item-details">
                                <span>ID: ${vodId}</span>
                                <span>생성일: ${creationDateFormatted}</span>
                                <span>길이: ${durationFormatted}</span>
                                <span>크기: ${fileSizeMB}</span>
                            </div>
                        `;
    
                // HTML5 <video> 태그 사용
                const videoSrc = `${antMediaServerUrl}/${applicationName}/streams/${vodFileName}`;
                const videoElement = document.createElement("video");
                videoElement.setAttribute("controls", "");
                videoElement.setAttribute("preload", "metadata"); // 메타데이터만 미리 로드
                videoElement.setAttribute("src", videoSrc);
    
                videoElement.addEventListener("error", function (e) {
                  console.error(`VoD '${vodDisplayName}' 재생 중 오류:`, e);
                  const errorMsgElement = document.createElement("p");
                  errorMsgElement.className = "error-message";
                  errorMsgElement.style.textAlign = "left";
                  errorMsgElement.textContent =
                    "영상을 불러오는 데 실패했습니다. 파일이 서버에 정확한 경로로 존재하고 접근 가능한지 확인해주세요.";
                  if (videoElement.parentNode) {
                    videoElement.parentNode.insertBefore(
                      errorMsgElement,
                      videoElement.nextSibling
                    );
                  } else {
                    vodDiv.appendChild(errorMsgElement);
                  }
                  videoElement.style.display = "none";
                });
    
                vodDiv.appendChild(videoElement);
                vodsContainer.appendChild(vodDiv);
              });
            } catch (error) {
              console.error("저장된 영상 정보 로딩 오류:", error);
              loadingMessage.style.display = "none";
              vodsContainer.innerHTML = `<p class="error-message">저장된 영상 목록 로딩 오류: ${error.message}<br>CORS 설정 및 Ant Media Server 상태를 확인하세요.</p>`;
            }
          }
    
          // 페이지 로드 시 모든 함수 실행
          document.addEventListener("DOMContentLoaded", () => {
            fetchAndDisplayLiveStreams();
            fetchAndDisplayVoDs();
          });
    
          // (선택 사항) 주기적으로 실시간 방송 목록 갱신
          // setInterval(fetchAndDisplayLiveStreams, 30000); // 30초마다
        </script>
      </body>
    </html>
    ```
    
- 결과
    
    ![image.png](/assets/img/posts/dev/Infra/Ant Media Server + RTMP 영상 재생/image%2015.png)
    

## Docker Volume Mount

현재 설정으로는 Docker Container가 종료 또는 삭제되면 영상을 확인할 수 없다

**`Volume Mount`**를 통해 데이터를 보존해야한다

- docker volume 생성
    
    ```bash
    docker volume create antmedia_volume
    ```
    
- volumes 주석 해제

![image.png](/assets/img/posts/dev/Infra/Ant Media Server + RTMP 영상 재생/image%2016.png)

- docker compose를 다시 실행해준다
    
    ```bash
    docker compose down
    docker compose build
    docker compose up -d
    ```
    

## Reference

1. [https://antmedia.io/docs/guides/clustering-and-scaling/docker/docker-and-docker-compose-installation/#1-download-docker-compose-and-dockerfile-files](https://antmedia.io/docs/guides/clustering-and-scaling/docker/docker-and-docker-compose-installation/#1-download-docker-compose-and-dockerfile-files)
