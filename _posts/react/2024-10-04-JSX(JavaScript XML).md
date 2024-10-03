---
layout: post
title: "JSX(JavaScript XML)"
date: 2024-10-04 04:35:00+0900
categories: [Study, React]
tags: [React, Frontend, Web]
---
## JSX(JavaScript XML)란?
javaScript 코드 내에서 HTML과 유사한 문법을 사용하여 UI를 정의할 수 있는 React의 문법 확장 도구  
React 컴포넌트를 더욱 직관적이고 가독성 있게 작성할 수 있도록 지원  

JSX는 브라우저가 직접 이해할 수 있는 문법은 아니지만, Babel과 같은 컴파일러에 의해 JavaScript 코드로 변환되어 브라우저에서 실행  

```jsx
const element = <h1>Hello, world!</h1>;
```

위의 JSX 코드는 다음과 같은 JavaScript 코드로 변환됨

```javascript
const element = React.createElement('h1', null, 'Hello, world!');
``` 

### 장점
1. **가독성 향상**  
   HTML과 유사한 문법을 사용하므로, UI 구조를 더욱 직관적이고 이해하기 쉽게 표현 가능  

2. **컴포넌트 기반 설계**  
   각 UI 요소를 컴포넌트로 분리하여, 복잡한 애플리케이션을 작은 단위로 나눠 관리 가능  

3. **React의 기능과 결합**
   JavaScript와 결합하여 동적인 UI를 쉽게 생성하고, React의 상태 관리와 연동하여 UI를 효율적으로 업데이트 가능  

### JSX와 HTML의 차이점
1.  **class 대신 className 사용**  
    HTML에서는 요소의 클래스 속성을 지정할 때 class를 사용하지만, JSX에서는 className을 사용해야 함

```html
<!-- HTML -->
<div class="container"></div>
```

```jsx
// JSX
<div className="container"></div>
``` 

> React는 ```class```가 JavaScript의 예약어이기 때문에 ```className```이라는 대체 속성을 제공  
{: .prompt-info}   

2.  **태그 속성 이름의 차이**  
    HTML과는 달리 JSX에서는 camelCase 표기법을 사용하여 속성을 지정  

```html
<!-- HTML -->
<label for="username">사용자 이름</label>
```

```jsx
// JSX
<label htmlFor="username">사용자 이름</label>
```
> ```for```는 JavaScript의 예약어이기 때문에 ```htmlFor``` 이라는 대체 속성을 제공   
{: .prompt-info}  

3.  **self-closing 태그**  
    JSX에서는 input, img, br 등의 태그는 반드시 self-closing 태그로 작성해야 함 

```html
<!-- HTML -->
<input type="text">
```

```jsx
// JSX
<input type="text" />
```

4.  **JSX 표현식 사용**  
    JSX 내에서는 {} 중괄호를 사용하여 JavaScript 표현식을 삽입할 수 있음  
    JavaScript의 변수, 함수 호출, 삼항 연산자 등을 사용할 수 있으며, UI를 동적으로 구성 가능   

```jsx
const name = "React";
const element = <h1>Hello, {name}!</h1>;  // Hello, React! 출력
```

5.  **주석 작성 방식의 차이**  
    JSX에서는 JavaScript 주석과 HTML 주석이 결합된 형태로 주석을 작성  
    ```{/* */} ```형태를 사용하여 JSX 내에서 주석을 작성해야 하며, 일반 JavaScript 주석```(//, /* */)```은 JSX 블록 내에서는 오류를 발생시킴

```jsx
// JSX에서 주석 작성
<div>
  {/* 이 부분은 JSX 주석입니다. */}
  <h1>Hello, World!</h1>
</div>
```

### JSX의 구조와 규칙
1.  **태그는 반드시 하나의 부모 요소로 감싸야 함**  
JSX의 모든 태그는 하나의 부모 요소로 감싸져 있어야 하며, 그렇지 않으면 오류가 발생

```jsx
// 오류가 발생하는 예시
<h1>Title</h1>
<p>Paragraph</p>

{/* 위의 코드는 <h1>와 <p> 태그가 서로 독립적으로 존재하여 오류 발생 */}

// 수정된 예시
<div>
  <h1>Title</h1>
  <p>Paragraph</p>
</div>

{/* 이와 같이 모든 태그를 <div>, <Fragment> 등 하나의 부모 요소로 감싸야 함 */}
```

2. **JSX 내부에서는 조건문 대신 삼항 연산자 사용**  
   JSX 내부에서는 일반적인 if 문을 사용할 수 없으며, 대신 삼항 연산자를 이용하여 조건부 렌더링을 수행

```jsx
const isLoggedIn = true;
return (
  <div>
    {isLoggedIn ? <h1>Welcome back!</h1> : <h1>Please sign up.</h1>}
  </div>
);

{/* && 연산자를 이용하여 특정 조건을 만족할 때만 태그를 렌더링할 수도 있음 */}

const unreadMessages = ["Message1", "Message2"];
return (
  <div>
    {unreadMessages.length > 0 && <h2>You have {unreadMessages.length} unread messages.</h2>}
  </div>
);
```

3. JSX 내부에서 JavaScript 함수 호출  
   JSX 내에서는 함수를 호출하거나, map()과 같은 메서드를 사용하여 리스트를 렌더링 가능

```jsx
const numbers = [1, 2, 3, 4, 5];
const listItems = numbers.map((number) => <li key={number}>{number}</li>);

{/* map() 함수를 사용하여 리스트 아이템을 동적으로 생성하고, key 속성을 사용하여 각 아이템을 고유하게 구분  */}

return <ul>{listItems}</ul>;
```

> key 속성은 리스트 렌더링 시 각 요소를 구분하는 고유한 값으로 사용되며, 성능 최적화와 버그 방지를 위해 반드시 설정해야 함  
{: .prompt-tip}  

### 주의사항  

1. **대문자로 시작하는 컴포넌트 이름**  
   React 컴포넌트는 대문자로 시작해야 함. 그렇지 않으면 React가 일반 HTML 태그로 인식하여 제대로 렌더링되지 않을 수 있음  

```jsx
function myComponent() {  // 오류 발생
  return <h1>My Component</h1>;
}

function MyComponent() {  // 올바른 사용
  return <h1>My Component</h1>;
}
``` 

2. **중괄호 사용**  
   JSX에서 JavaScript 코드를 삽입할 때는 반드시 중괄호 {}로 감싸야 하며, 단순 문자열이나 숫자는 중괄호 없이 사용할 수 있음

```jsx
const text = "Hello";
return <h1>{text}</h1>;  // JSX 내에서 변수 사용

{/* 중괄호를 사용하여 함수 호출, 배열 접근, 객체 사용 등이 가능*/}
``` 

