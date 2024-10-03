---
layout: post
title: "React 컴포넌트(Component)"
date: 2024-10-04 04:46:00+0900
categories: [Study, React]
tags: [React, Frontend, Web]
---
## 컴포넌트(Component)란?  

React 애플리케이션을 구성하는 독립적이고 재사용 가능한 코드의 블록   
각 컴포넌트는 자신만의 상태(state)와 속성(props)을 가지고, 특정 UI를 정의하고 기능을 담당   

React는 모듈화와 재사용성을 극대화하기 위해 컴포넌트 기반의 설계를 채택  
전체 애플리케이션을 작은 UI 조각으로 나누고, 이러한 컴포넌트들을 조합하여 복잡한 UI를 구성  

> React에서 컴포넌트는 HTML, CSS, JavaScript를 결합하여 하나의 독립적인 UI 요소를 만들 수 있음  
{: .prompt-info}  

### 장점
1. **재사용성**  
한 번 정의한 컴포넌트를 여러 곳에서 재사용할 수 있으므로, 중복 코드를 줄이고 유지보수를 용이하게 만듦

2. **독립성**  
각 컴포넌트는 자신만의 상태와 로직을 관리하므로, 다른 컴포넌트와 독립적으로 동작하고 수정 가능

3. **모듈화**  
컴포넌트가 명확하게 분리되어 있어, 팀 간 협업 시 각각의 컴포넌트를 독립적으로 개발하고 테스트 가능

4. **테스트 용이성**  
작은 단위의 컴포넌트는 독립적으로 테스트할 수 있으므로, 복잡한 UI의 테스트를 단순화할 수 있음


### 종류
* **함수형 컴포넌트 (Function Component)**  
간단한 형태의 React 컴포넌트로, 함수 형태로 정의  
상태와 라이프사이클 기능은 useState, useEffect와 같은 React Hook을 사용하여 관리
  
```jsx
function Greeting({ name }) {
  return <h1>Hello, {name}!</h1>;
}

// 사용 예시
<Greeting name="John" />
``` 

* **클래스형 컴포넌트 (Class Component)**   
ES6 클래스 문법을 사용하여 정의하며, 더 복잡한 상태와 라이프사이클 관리 기능 제공
함수형 컴포넌트가 도입되기 이전에 많이 사용되었으나, 현재는 Hook을 사용하는 함수형 컴포넌트가 더 권장됨  

```jsx
import React, { Component } from 'react';

class Greeting extends Component {
  render() {
    return <h1>Hello, {this.props.name}!</h1>;
  }
}

// 사용 예시
<Greeting name="John" />
``` 

* **컨테이너 컴포넌트 (Container Component)**  
UI를 렌더링하지 않고, 데이터나 상태 관리를 담당하는 컴포넌트  
다른 컴포넌트에 필요한 데이터를 전달하거나, 애플리케이션의 비즈니스 로직을 처리  

```jsx
function ContainerComponent({ children }) {
  const [count, setCount] = useState(0);

  return (
    <div>
      <h1>Counter: {count}</h1>
      <button onClick={() => setCount(count + 1)}>증가</button>
      {children}
    </div>
  );
}

// 사용 예시
<ContainerComponent>
  <Greeting name="Container User" />
</ContainerComponent>
``` 
### 구조
* **JSX를 사용한 렌더링**   

모든 React 컴포넌트는 JSX를 사용하여 UI를 정의  
JSX는 JavaScript와 HTML을 결합한 형태로, React 컴포넌트의 구조를 직관적으로 표현  

```jsx
function Welcome() {
  return (
    <div className="welcome">
      <h1>Welcome to React</h1>
    </div>
  );
}
```
 
* **props**  
props는 컴포넌트 간 데이터 전달을 위해 사용되는 속성  
부모 컴포넌트가 자식 컴포넌트에게 데이터를 전달할 때 사용되며, 자식 컴포넌트는 props를 읽기 전용으로 사용  

```jsx
function UserProfile({ name, age }) {
  return (
    <div>
      <h2>Name: {name}</h2>
      <p>Age: {age}</p>
    </div>
  );
}

// 사용 예시
<UserProfile name="Alice" age={25} />
``` 

* **state**  
state는 컴포넌트의 상태를 관리하는 객체  
상태는 컴포넌트 내에서 변경될 수 있으며, 상태가 변경되면 해당 컴포넌트와 하위 컴포넌트가 다시 렌더링  

```jsx
import React, { useState } from 'react';

function Counter() {
  const [count, setCount] = useState(0);

  return (
    <div>
      <p>현재 카운트: {count}</p>
      <button onClick={() => setCount(count + 1)}>증가</button>
    </div>
  );
}
``` 

> state는 React Hook인 useState를 통해 함수형 컴포넌트에서도 사용할 수 있음  
{: .prompt-tip}

## 컴포넌트 간의 데이터 흐름
React는 단방향 데이터 흐름을 사용하여 컴포넌트 간의 데이터가 부모에서 자식으로 전달  
상위 컴포넌트는 하위 컴포넌트에게 데이터를 전달하고, 하위 컴포넌트는 props를 통해 데이터를 수신  

### 데이터 전달 방식
* **props를 통한 데이터 전달**  
상위 컴포넌트에서 하위 컴포넌트에게 props를 이용하여 데이터를 전달  

```jsx
function ParentComponent() {
  const user = { name: 'John', age: 30 };
  return <ChildComponent user={user} />;
}

function ChildComponent({ user }) {
  return (
    <div>
      <h1>{user.name}</h1>
      <p>Age: {user.age}</p>
    </div>
  );
}
```

* **상태 끌어올리기 (Lifting State Up)**  
상위 컴포넌트와 하위 컴포넌트가 동일한 상태를 공유해야 할 때, 상태를 상위 컴포넌트로 끌어올리고 하위 컴포넌트에 props로 전달

```jsx
function ParentComponent() {
  const [count, setCount] = useState(0);

  return (
    <div>
      <ChildComponent count={count} onIncrement={() => setCount(count + 1)} />
    </div>
  );
}

function ChildComponent({ count, onIncrement }) {
  return (
    <div>
      <p>Count: {count}</p>
      <button onClick={onIncrement}>증가</button>
    </div>
  );
}
```

> 상태가 여러 컴포넌트에 걸쳐 필요할 때는 상태 끌어올리기 패턴을 사용하여 최상위 컴포넌트에서 관리  
{: .prompt-info}

## 컴포넌트 설계 시 주의사항

1. 작고 독립적인 컴포넌트 설계  
한 컴포넌트가 너무 많은 역할을 담당하지 않도록, 최대한 작고 독립적인 기능을 가진 컴포넌트로 설계  

2. 명확한 데이터 흐름 유지  
props와 state를 이용하여 데이터의 흐름을 명확하게 관리  
  
3. 컴포넌트의 책임 분리   
UI 렌더링, 데이터 관리, 이벤트 처리 등을 담당하는 컴포넌트를 명확하게 분리   