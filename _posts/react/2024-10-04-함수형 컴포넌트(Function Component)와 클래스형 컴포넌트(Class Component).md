---
layout: post
title: "함수형 컴포넌트(Function Component)와 클래스형 컴포넌트(Class Component)"
date: 2024-10-04 06:41:00+0900
categories: [Study, React]
tags: [React, Frontend, Web]
---
## 함수형 컴포넌트(Function Component)와 클래스형 컴포넌트(Class Component) 

### 함수형 컴포넌트란?
JavaScript 함수로 정의된 React 컴포넌트  
UI를 반환하고, props를 인수로 받아 렌더링 동작을 수행  

> 함수형 컴포넌트는 비교적 간단한 UI 컴포넌트를 정의할 때 사용되며, 상태(state)나 라이프사이클 메서드를 포함하지 않음  
{: .prompt-info}

```jsx
function Greeting({ name }) {  
  return <h1>Hello, {name}!</h1>;  
}

// 사용 예시  
<Greeting name="Alice" />  
```
## 클래스형 컴포넌트란?  

ES6 클래스로 정의된 React 컴포넌트  
React의 Component 클래스를 상속하고, 상태(state)와 라이프사이클 메서드를 통해 복잡한 UI 동작을 관리  

> 클래스형 컴포넌트는 상태를 가지고 있거나, 컴포넌트의 라이프사이클을 관리해야 할 때 사용됨  
{: .prompt-info}  

```jsx
import React, { Component } from 'react';  
  
class Greeting extends Component {  
  render() {  
    return <h1>Hello, {this.props.name}!</h1>;  
  }  
}

// 사용 예시  
<Greeting name="Alice" />  
``` 

## 주요 차이점
함수형 컴포넌트와 클래스형 컴포넌트는 기능적으로 동일한 역할을 하지만, 상태 관리와 코드의 간결성 측면에서 차이가 있음  

1. **상태 관리 (State Management)**  
함수형 컴포넌트는 기본적으로 상태를 관리하지 않지만, React Hook(useState, useEffect)을 통해 상태를 관리 가능  
클래스형 컴포넌트는 this.state를 사용하여 상태를 직접 관리  


```jsx
//함수형 컴포넌트에서 상태 관리  
import React, { useState } from 'react';  

function Counter() {  
  const [count, setCount] = useState(0);  

  return (  
    <div>  
      <p>Count: {count}</p>  
      <button onClick={() => setCount(count + 1)}>Increase</button>  
    </div>  
  );  
}
```

```jsx
//클래스형 컴포넌트에서 상태 관리
import React, { Component } from 'react';  

class Counter extends Component {  
  constructor(props) {  
    super(props);  
    this.state = { count: 0 };  
  }  

  render() {  
    return (  
      <div>  
        <p>Count: {this.state.count}</p>  
        <button onClick={() => this.setState({ count: this.state.count + 1 })}>Increase</button>  
      </div>  
    );  
  }  
}
```

2. **라이프사이클 메서드 (Lifecycle Methods)**  
함수형 컴포넌트는 React Hook(useEffect)을 사용하여 라이프사이클을 관리  
클래스형 컴포넌트는 componentDidMount, componentDidUpdate, componentWillUnmount와 같은 메서드를 사용하여 라이프사이클을 관리

```jsx 
// 함수형 컴포넌트에서 라이프사이클 관리  
import React, { useState, useEffect } from 'react';  

function Timer() {  
  const [seconds, setSeconds] = useState(0);  

  useEffect(() => {  
    const interval = setInterval(() => {  
      setSeconds(s => s + 1);  
    }, 1000);  

    // 컴포넌트 언마운트 시 타이머 정리  
    return () => clearInterval(interval);  
  }, []);  

  return <p>Seconds: {seconds}</p>;  
}  
```

``` jsx
// 클래스형 컴포넌트에서 라이프사이클 관리
import React, { Component } from 'react';  

class Timer extends Component {  
  constructor(props) {  
    super(props);  
    this.state = { seconds: 0 };  
  }  

  componentDidMount() {  
    this.interval = setInterval(() => {  
      this.setState({ seconds: this.state.seconds + 1 });  
    }, 1000);  
  }  

  componentWillUnmount() {  
    clearInterval(this.interval);  
  }  

  render() {  
    return <p>Seconds: {this.state.seconds}</p>;  
  }  
}
```

3. **코드의 간결성**  
함수형 컴포넌트는 더 간결한 문법으로 작성할 수 있음  
클래스형 컴포넌트는 this 키워드를 사용하여 상태와 메서드를 관리하므로, 코드가 복잡해질 수 있음  

```jsx
// 함수형 컴포넌트 예시
function WelcomeMessage({ name }) {  
  return <h1>Welcome, {name}!</h1>;  
}
```

```jsx
// 클래스형 컴포넌트 예시
import React, { Component } from 'react';  

class WelcomeMessage extends Component {  
  render() {  
    return <h1>Welcome, {this.props.name}!</h1>;  
  }  
}
```

4. **this 키워드**  
함수형 컴포넌트에서는 this 키워드를 사용할 필요 없음  
클래스형 컴포넌트에서는 this를 사용하여 상태와 메서드에 접근  

5. **성능**  
React 16.8 이후, 함수형 컴포넌트는 React Hook을 통해 성능이 크게 향상되었고, 컴포넌트의 상태와 라이프사이클을 처리할 수 있게 됨  
특히, 함수형 컴포넌트는 메모리 효율과 코드 최적화 측면에서 더 유리함  

> 최신 React 애플리케이션에서는 주로 함수형 컴포넌트를 사용하고, 클래스형 컴포넌트는 기존 코드베이스와의 호환성을 위해 유지  
{: .prompt-info}

### 결론: 언제 어떤 컴포넌트를 사용할까?  

**함수형 컴포넌트**
  * 간단한 UI 컴포넌트를 정의할 때  
  * 상태 관리가 필요할 때도 useState, useEffect 등의 Hook을 활용  
  * 최신 React 애플리케이션 개발 시 권장    

**클래스형 컴포넌트**  
  * 기존의 클래스형 컴포넌트가 사용된 코드와 호환이 필요할 때  
  * 매우 복잡한 로직을 포함한 경우, 클래스 컴포넌트에서의 메서드 관리가 더 직관적일 때  

> React Hook의 도입 이후, 클래스형 컴포넌트의 사용은 줄어들고 있으며, 함수형 컴포넌트가 새로운 표준으로 자리잡고 있음  
{: .prompt-tip}