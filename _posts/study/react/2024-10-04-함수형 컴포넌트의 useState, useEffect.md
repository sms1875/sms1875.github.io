---
layout: post
title: "함수형 컴포넌트의 useState, useEffect"
date: 2024-10-04 06:51:00+0900
categories: [Study, React]
tags: [React, Frontend, Web]
---
## React Hook이란?
React 16.8부터 도입된 React Hook은 함수형 컴포넌트에서도 상태(state)와 라이프사이클(lifecycle) 기능을 사용할 수 있도록 제공하는 새로운 방식  
useState, useEffect를 통해 기존 클래스형 컴포넌트에서만 가능했던 기능을 함수형 컴포넌트에서도 손쉽게 구현  

> React Hook을 사용하면 함수형 컴포넌트에서도 상태 관리와 부수 효과 관리를 할 수 있으며, 코드가 더 간결하고 직관적으로 작성 가능  
{: .prompt-info}

### 주요 Hook 종류  
1. useState: 컴포넌트의 상태를 관리  
2. useEffect: 컴포넌트의 라이프사이클을 관리 (마운트, 업데이트, 언마운트 시 실행)  
3. useContext, useReducer, useMemo, useCallback 등 다양한 고급 Hook 제공  

### useState 사용법  

useState는 컴포넌트의 상태 값을 관리하기 위해 사용되는 Hook  
초기 상태 값을 인수로 전달하고, 상태 값과 그 값을 업데이트하는 함수를 반환  


```jsx
// 기본 사용법
const [state, setState] = useState(initialState);  
state: 현재 상태 값
setState: 상태를 업데이트하는 함수
initialState: 초기 상태 값
``` 

```jsx
// 예시: 카운터 컴포넌트
import React, { useState } from 'react';  

function Counter() {  
  // `count`라는 상태 값을 정의하고 초기값은 0  
  const [count, setCount] = useState(0);  

  return (  
    <div>  
      <h2>Counter: {count}</h2>  
      <button onClick={() => setCount(count + 1)}>증가</button>  
      <button onClick={() => setCount(count - 1)}>감소</button>  
    </div>  
  );  
}  

export default Counter;  
```

1. useState(0): 카운터의 초기값을 0으로 설정
2. setCount(count + 1): 증가 버튼을 클릭할 때 count 값을 1 증가
3. setCount(count - 1): 감소 버튼을 클릭할 때 count 값을 1 감소

> setState 함수가 호출되면, 해당 상태 값만 업데이트되며 React는 컴포넌트를 다시 렌더링하여 변경된 UI를 반영  
{: .prompt-tip}

### useEffect 사용법
useEffect는 **부수 효과(side effect)**를 관리하기 위해 사용되는 Hook  
컴포넌트가 마운트(생성), 업데이트, 언마운트(제거)될 때 특정 코드를 실행하도록 설정  


```jsx
// 기본 사용법
useEffect(() => {  
  // 실행하고자 하는 코드  
  return () => {  
    // cleanup 함수 (언마운트 시 실행)  
  };  
}, [dependencyArray]);  
``` 

* Effect 함수: 마운트 또는 업데이트 시 실행될 코드
* Cleanup 함수: 언마운트 시 실행될 코드
* dependencyArray: 이 배열에 포함된 값이 변경될 때만 effect가 재실행됨
   * 빈 배열 []: 컴포넌트가 처음 마운트될 때만 실행
   * dependencyArray가 없으면 매번 렌더링 될 때마다 useEffect 실행  

```jsx
// 예시: 간단한 타이머 컴포넌트
import React, { useState, useEffect } from 'react';  

function Timer() {  
  const [seconds, setSeconds] = useState(0);  

  useEffect(() => {  
    // 1초마다 seconds를 1씩 증가시키는 타이머 설정  
    const interval = setInterval(() => {  
      setSeconds(s => s + 1);  
    }, 1000);  

    // Cleanup 함수: 컴포넌트 언마운트 시 타이머 정리  
    return () => clearInterval(interval);  
  }, []);  // 의존성 배열이 비어 있으므로, 컴포넌트가 처음 마운트될 때만 실행  

  return <h3>Elapsed Time: {seconds} seconds</h3>;  
}  

export default Timer;  
``` 

1. useEffect(() => { ... }, []);: 의존성 배열이 비어 있어 한 번만 실행됨
2. setInterval(() => setSeconds(s => s + 1), 1000);: 1초마다 seconds를 1씩 증가
3. return () => clearInterval(interval);: 컴포넌트가 언마운트될 때 타이머 정리

> 의존성 배열이 없는 경우([] 생략), 매번 렌더링될 때마다 useEffect가 실행됨  
{: .prompt-info}  

## useState와 useEffect 사용 예시

**예시 1: 다중 상태 관리**

```jsx
function UserProfile() {  
  const [name, setName] = useState('John');  
  const [age, setAge] = useState(25);  

  return (  
    <div>  
      <h3>Name: {name}</h3>  
      <h3>Age: {age}</h3>  
      <button onClick={() => setName('Alice')}>Change Name</button>  
      <button onClick={() => setAge(30)}>Change Age</button>  
    </div>  
  );  
}  
```

**예시 2: 의존성 배열을 사용한 조건부 실행**
```jsx
function RandomNumber() {  
  const [number, setNumber] = useState(0);  

  useEffect(() => {  
    console.log('Random number generated:', number);  
  }, [number]);  // `number`가 변경될 때마다 effect가 재실행됨  

  return (  
    <div>  
      <h3>Random Number: {number}</h3>  
      <button onClick={() => setNumber(Math.random())}>Generate</button>  
    </div>  
  );  
}  
```

### 결론: 언제 useState와 useEffect를 사용할까?

1. useState:
  * 컴포넌트의 상태를 관리하고, 그 상태에 따라 UI가 변경되어야 할 때 사용
  * 예: 입력 폼의 값, 버튼 클릭 횟수, 토글 상태 등

2. useEffect:
  * 부수 효과를 처리하고, 컴포넌트의 라이프사이클을 관리할 때 사용
  * 예: 데이터 요청, 이벤트 리스너 설정 및 정리, 타이머 등

> React Hook을 활용하면 함수형 컴포넌트에서도 상태와 라이프사이클을 쉽게 관리할 수 있으며, 코드의 가독성과 유지보수성이 크게 향상됨  
{: .prompt-tip}
