---
layout: post
title: "[Flutter] flutter secure storage NDK 설정"
date: 2025-03-14 23:29:00+0900
categories: [Dev, Troubleshooting]
tags: [Troubleshooting, flutter]
---
## 문제

flutter secure storage를 사용하기 위해 `minSdkVersion` 을 설정해야 함

## 해결법

### 1. ndk 버전 설정

검색 시에는 `minSdkVersion`  버전을 “18”로 설정하라는 정보가 많았는데

`Flutter = 3.29.1` , `flutter_secure_storage: ^9.2.4` 버전으로 27 이상이 필요한것 같다

```bash
Your project is configured with Android NDK 26.3.11579264, but the following plugin(s) depend on a different Android NDK version:
- flutter_secure_storage requires Android NDK 27.0.12077973
- path_provider_android requires Android NDK 27.0.12077973
Fix this issue by using the highest Android NDK version (they are backward compatible).
Add the following to C:\Users\SSAFY\Documents\S12P21C206\clientend\android\app\build.gradle.kts:

    android {
        ndkVersion = "27.0.12077973"
        ...
    }

C:\Users\SSAFY\Documents\S12P21C206\clientend\android\app\src\main\AndroidManifest.xml:15:23-67 Error:
	Attribute data@scheme at AndroidManifest.xml:15:23-67 requires a placeholder substitution but no value for <YOUR_NATIVE_APP_KEY> is provided.
C:\Users\SSAFY\Documents\S12P21C206\clientend\android\app\src\main\AndroidManifest.xml:47:21-65 Error:
	Attribute data@scheme at AndroidManifest.xml:47:21-65 requires a placeholder substitution but no value for <YOUR_NATIVE_APP_KEY> is provided.
C:\Users\SSAFY\Documents\S12P21C206\clientend\android\app\src\debug\AndroidManifest.xml Error:
	Validation failed, exiting

FAILURE: Build failed with an exception.

* What went wrong:
Execution failed for task ':app:processDebugMainManifest'.
> Manifest merger failed with multiple errors, see logs

* Try:
> Run with --stacktrace option to get the stack trace.
> Run with --info or --debug option to get more log output.
> Run with --scan to get full insights.
> Get more help at https://help.gradle.org.

BUILD FAILED in 5s
Error: Gradle task assembleDebug failed with exit code 1
```

android\app\build.gradle.kts  

ndkVersion 설정

숫자로 적으면 안되고 문자열로 넣어야 했다

![image.png](assets/img/posts/dev/Troubleshooting/flutter secure storage NDK 설정/image.png)

### 2. SDK Tools 설치

ndk가 설치되지 않았다고 오류가 발생

sdk manager에서 맞는 NDK 버전을 설치했다

```bash
FAILURE: Build failed with an exception.

* Where:
Build file 'C:\Users\SSAFY\Documents\S12P21C206\clientend\android\build.gradle.kts' line: 16

* What went wrong:
A problem occurred configuring project ':app'.
> [CXX1101] NDK at C:\Users\SSAFY\AppData\Local\Android\sdk\ndk\27.0.12077973 did not have a source.properties file

* Try:
> Run with --stacktrace option to get the stack trace.
> Run with --info or --debug option to get more log output.
> Run with --scan to get full insights.
> Get more help at https://help.gradle.org.

BUILD FAILED in 2s
Error: Gradle task assembleDebug failed with exit code 1
```

SDK Tools -> Show Package Details를 선택하면 버전을 확인할 수 있다

필요한 NDK 버전을 선택하고 apply를 눌러서 설치하면 해결되었다

![image.png](assets/img/posts/dev/Troubleshooting/flutter secure storage NDK 설정/image%201.png)

![image.png](assets/img/posts/dev/Troubleshooting/flutter secure storage NDK 설정/image%202.png)

## 후기

ndk는 하위 호환이 된다고 하는데 

이미 상위 ndk가 있는데 왜 추가적으로 설치해야 하는지 이해가 잘 안된다
