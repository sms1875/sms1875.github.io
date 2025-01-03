---
layout: post
title: "React와 ReactDOM"
date: 2024-10-04 04:12:00+0900
categories: [Study, React]
tags: [React, Frontend, Web]
---
## ReactDOM이란?
React로 정의된 컴포넌트를 브라우저의 실제 DOM에 렌더링하거나, 서버 사이드 렌더링(SSR)을 통해 HTML 문자열로 변환할 때 사용되는 라이브러리  
ReactDOM은 UI의 렌더링 및 업데이트에 필요한 DOM 관련 메서드와 기능을 제공  

### 주요 기능
1. 컴포넌트의 DOM 렌더링    
   ReactDOM.createRoot 또는 ReactDOM.render를 사용하여, 컴포넌트를 특정 DOM 요소에 렌더링  
2. 서버 사이드 렌더링 지원    
   ReactDOMServer를 사용하여 서버 측에서 React 컴포넌트를 HTML 문자열로 변환 가능
3. 브라우저 DOM 조작  
   컴포넌트가 마운트되거나 언마운트될 때, 브라우저 DOM을 직접 조작

> ReactDOM은 React 컴포넌트를 실제 DOM과 결합하여, 브라우저나 서버에서 렌더링하는 일을 수행   
> React 18부터는 `ReactDOM.render` 대신 `ReactDOM.createRoot`를 사용하여, 컴포넌트를 특정 DOM 요소에 렌더링함  
{: .prompt-info}  

### React와 ReactDOM의 차이  

React와 ReactDOM은 모두 React 애플리케이션의 개발에 필수적인 라이브러리이지만, 그 역할은 명확하게 구분됨   

* React는 UI를 정의하고 상태 관리 및 데이터 흐름을 관리  
* ReactDOM은 정의된 UI를 실제 DOM에 렌더링하거나 서버 사이드 렌더링을 담당  

### React와 ReactDOM의 역할 분리  

React와 ReactDOM의 역할이 분리된 이유는 React가 플랫폼 독립적인 UI 라이브러리이기 때문  
이로 인해 React는 브라우저뿐만 아니라 모바일 애플리케이션(React Native), 서버 사이드 렌더링, 데스크탑 애플리케이션(Electron) 등 다양한 환경에서 사용할 수 있음  

> React는 모든 환경에서 사용될 수 있는 라이브러리이며, 렌더링과 관련된 작업은 ReactDOM 또는 React Native와 같은 환경별 라이브러리가 담당  
{: .prompt-tip}  

### 주요 메서드

* ReactDOM.createRoot
   * 컴포넌트를 특정 DOM 요소에 렌더링하기 위한 root를 생성하는 메서드
   * 비동기 렌더링과 향상된 성능을 제공
   * 한번 생성된 root는 재사용되며, 애플리케이션의 최상위 레벨에서 한 번만 호출  

```jsx
// index.jsx
import { createRoot } from 'react-dom/client';
import App from './App';

const container = document.getElementById('root');
const root = createRoot(container);
root.render(<App />);
```

* ReactDOM.createPortal
   * 부모 컴포넌트의 DOM 계층 구조 외부에 자식을 렌더링하는 메서드
   * 모달, 툴팁, 팝업과 같은 오버레이 UI를 구현할 때 유용
   * DOM 계층과 React 컴포넌트 계층을 독립적으로 구성 가능

```jsx
// Modal.jsx
import { createPortal } from 'react-dom';

function Modal({ isOpen, onClose, children }) {
  if (!isOpen) return null;

  return createPortal(
    <div className="modal-overlay">
      <div className="modal-content">
        {children}
        <button onClick={onClose}>닫기</button>
      </div>
    </div>,
    document.getElementById('modal-root') // index.html에 별도로 정의된 DOM 요소
  );
}

// App.jsx
function App() {
  const [isModalOpen, setIsModalOpen] = useState(false);

  return (
    <div>
      <button onClick={() => setIsModalOpen(true)}>모달 열기</button>
      <Modal 
        isOpen={isModalOpen} 
        onClose={() => setIsModalOpen(false)}
      >
        <h2>포털을 사용한 모달</h2>
        <p>이 내용은 root와 다른 DOM 노드에 렌더링됩니다.</p>
      </Modal>
    </div>
  );
}
```

* ReactDOM.flushSync
   * 상태 업데이트를 동기적으로 즉시 실행하도록 강제하는 메서드
   * 일반적으로는 권장되지 않지만, DOM을 즉시 업데이트해야 하는 특수한 경우에 사용
   * 성능에 영향을 줄 수 있으므로 신중하게 사용  

```jsx
import { flushSync } from 'react-dom';

function handleClick() {
  flushSync(() => {
    setCount(c => c + 1);
  });
  // 이 시점에서 DOM은 확실히 업데이트된 상태
  someElement.focus();
}
```

* ReactDOMServer 메서드들
   * renderToString: 컴포넌트를 HTML 문자열로 렌더링
   * renderToStaticMarkup: 추가적인 React 속성 없이 순수 HTML만 생성
   * renderToPipeableStream: 스트리밍 방식으로 HTML을 생성

```jsx
// server.js
import ReactDOMServer from 'react-dom/server';
import App from './App';

// 정적 HTML 생성
const html = ReactDOMServer.renderToString(<App />);

// 스트리밍 방식의 렌더링
const { pipe } = await ReactDOMServer.renderToPipeableStream(
  <App />,
  {
    bootstrapScripts: ['/client.js'],
    onShellReady() {
      res.setHeader('content-type', 'text/html');
      pipe(res);
    }
  }
);
```
