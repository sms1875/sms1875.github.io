---
layout: post
title: 클래스형 컴포넌트의 constructor, state, render
date: 2024-10-05 16:54:00+0900
categories: [Study, React]
tags: [React, Frontend, Web]
---
## 클래스형 컴포넌트란?
React의 초기 버전에서 컴포넌트를 정의할 때 사용된 방식으로, ES6 클래스 문법을 기반으로 만들어진 컴포넌트  
클래스형 컴포넌트는 **상태(state)**와 라이프사이클 메서드를 사용할 수 있으며, 현재는 함수형 컴포넌트의 React Hook이 도입되기 전까지 React의 기본 컴포넌트 형태로 사용됨    

### 특징

* **상태(state)**와 props를 기반으로 UI를 렌더링  
* 클래스 메서드(render, componentDidMount 등)를 통해 컴포넌트의 라이프사이클 관리  
* this 키워드를 사용하여 컴포넌트 내의 상태와 메서드에 접근  

> 함수형 컴포넌트는 상태 관리와 부수 효과 관리를 useState 및 useEffect로 처리하지만, 클래스형 컴포넌트는 이를 this.state와 라이프사이클 메서드로 관리  
{: .prompt-info}  

### constructor 메서드  
클래스형 컴포넌트의 생성자 메서드로, 초기 상태를 정의하고 필요한 속성을 설정할 때 사용  
이 메서드는 클래스가 인스턴스화될 때 가장 먼저 호출되며, this.state를 사용하여 초기 상태를 정의  

* props: 부모 컴포넌트로부터 전달받은 속성  
* super(props): React.Component의 생성자를 호출하여 부모 클래스의 속성을 초기화  

```jsx
constructor(props) {  
  super(props);  
  // 컴포넌트의 상태 초기화  
  this.state = {  
    count: 0  
  };  
}  
```

**구현: 간단한 카운터 컴포넌트**  
constructor에서 this.state를 통해 초기 상태를 정의하고, setState를 사용하여 상태를 업데이트  

```jsx
// 간단한 카운터 컴포넌트
import React, { Component } from 'react';  

class Counter extends Component {  
  constructor(props) {  
    super(props);  
    // 초기 상태 값 정의  
    this.state = {  
      count: 0  
    };  
  }  

  render() {  
    return (  
      <div>  
        <h2>Counter: {this.state.count}</h2>  
        <button onClick={() => this.setState({ count: this.state.count + 1 })}>증가</button>  
        <button onClick={() => this.setState({ count: this.state.count - 1 })}>감소</button>  
      </div>  
    );  
  }  
}  

export default Counter;  
```

### state와 setState 메서드  
클래스형 컴포넌트의 상태 관리는 state 객체와 setState 메서드를 통해 이루어짐  
* state: 컴포넌트의 현재 상태를 저장하는 객체  
* setState: 상태 값을 업데이트하고, UI를 재렌더링  

```jsx
this.state = {  
  name: 'John',  
  age: 25  
};  
this.setState({ name: 'Alice' });  
```

> setState는 비동기적으로 실행되므로, setState가 호출된 직후에는 this.state가 즉시 반영되지 않을 수 있음  
{: .prompt-info}  

### render 메서드  
클래스형 컴포넌트에서 UI를 정의하는 필수 메서드  
render 메서드는 상태(this.state)와 속성(this.props)을 기반으로 JSX를 반환  

**특징**   
필수 메서드로, 반드시 JSX 또는 null을 반환해야 함  
상태가 변경될 때마다 React는 render() 메서드를 호출하여 UI를 업데이트  

```jsx
render() {  
  return <div>My Component</div>;  
}  
```


### this 키워드  
클래스형 컴포넌트에서는 this가 컴포넌트 인스턴스를 가리키며, 컴포넌트 내의 상태(this.state)와 메서드(this.setState)에 접근할 때 사용  
  
**메서드 바인딩**   
클래스 메서드에서 this를 사용할 때, this가 올바르게 참조되도록 메서드 바인딩을 해야 함   

```jsx
// 이벤트 핸들러에서 this.handleClick이 올바르게 this를 참조하도록 바인딩 필요   
constructor(props) {  
  super(props);  
  this.handleClick = this.handleClick.bind(this);  
}  
```

> 화살표 함수(=>)를 사용하면 자동으로 this 바인딩이 이루어짐  
{: .prompt-tip}  

**구현: 이벤트 핸들러에서 this 사용**  
constructor에서 이벤트 핸들러 this.handleClick을 this와 명시적으로 바인딩하여, 버튼 클릭 시 올바른 this를 참조하도록 함   

```jsx
import React, { Component } from 'react';  

class Toggle extends Component {  
  constructor(props) {  
    super(props);  
    this.state = {  
      isToggleOn: true  
    };  

    // `this` 바인딩  
    this.handleClick = this.handleClick.bind(this);  
  }  

  handleClick() {  
    this.setState(state => ({  
      isToggleOn: !state.isToggleOn  
    }));  
  }  

  render() {  
    return (  
      <button onClick={this.handleClick}>  
        {this.state.isToggleOn ? 'ON' : 'OFF'}  
      </button>  
    );  
  }  
}  

export default Toggle;  

```

### 클래스형 컴포넌트의 라이프사이클 메서드   

클래스형 컴포넌트는 다음과 같은 라이프사이클 메서드를 통해 컴포넌트의 생명주기를 관리  

* componentDidMount: 컴포넌트가 마운트된 직후 실행 (데이터 요청 등)  
* componentDidUpdate: 컴포넌트가 업데이트된 직후 실행  
* componentWillUnmount: 컴포넌트가 언마운트되기 직전 실행 (클린업 작업)  

```jsx
componentDidMount() {  
  console.log('컴포넌트가 마운트되었습니다.');  
}  
```

> 함수형 컴포넌트에서는 useEffect를 사용하여 클래스형 컴포넌트의 라이프사이클 메서드를 대체할 수 있음  
{: .prompt-tip}

## 결론
* 클래스형 컴포넌트는 constructor에서 상태를 초기화하고, render 메서드를 통해 UI를 정의하며, this를 사용하여 상태와 메서드를 관리  
* 함수형 컴포넌트와 React Hook의 등장으로 클래스형 컴포넌트는 사용 빈도가 줄었지만, 기존 코드 유지보수와 복잡한 UI 로직을 처리할 때 여전히 유용  
