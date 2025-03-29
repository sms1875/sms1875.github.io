---
layout: post
title: "[Flutter] Tried to modify a provider while the widget tree was building"
date: 2025-03-29 18:14:00+0900
categories: [Dev, Troubleshooting]
tags: [Troubleshooting, flutter, Riverpod]
---

## 문제

페이지를 닫을 때 riverpod 상태를 초기화하는 과정에서 발생

```
======== Exception caught by widgets library =======================================================

The following StateNotifierListenerError was thrown while finalizing the widget tree:

At least listener of the StateNotifier Instance of 'RegisterNotifier' threw an exception

when the notifier tried to update its state.

The exceptions thrown are:

Tried to modify a provider while the widget tree was building.

If you are encountering this error, chances are you tried to modify a provider

in a widget life-cycle, such as but not limited to:

- build

- initState

- dispose

- didUpdateWidget

- didChangeDependencies

Modifying a provider inside those life-cycles is not allowed, as it could

lead to an inconsistent UI state. For example, two widgets could listen to the

same provider, but incorrectly receive different states.

To fix this problem, you have one of two solutions:

- (preferred) Move the logic for modifying your provider outside of a widget

life-cycle. For example, maybe you could update your provider inside a button's

onPressed instead.

- Delay your modification, such as by encapsulating the modification

in a `Future(() {...})`.

This will perform your update after the widget tree is done building.

```

## 원인

`dispose()` 중에는 위젯 트리가 빌드 중인데, 이 시점에서 `_controller.reset()`을 통해 provider 상태를 변경하려고 해서 Riverpod 에서 예외가 발생했다

- `initState() , dispose() , build() , didUpdateWidget()` 같은 위젯 생명주기 메서드 내에서 직접 provider 상태를 변경하면 발생
  1. 여러 위젯이 같은 provider를 구독하는 경우 서로 다른 상태를 받을 가능성이 있음
  2. 위젯이 빌드 중(building state) 이거나 해제 중(disposing state) 일 때 상태를 변경하면, 아직 빌드되지 않은 위젯이나 이미 삭제된 위젯에 대해 상태를 변경하려고 시도하게 됨

```dart
class MyScreen extends ConsumerStatefulWidget {
  @override
  ConsumerState<MyScreen> createState() => _MyScreenState();
}

class _MyScreenState extends ConsumerState<MyScreen> {
  late final MyController _controller;

  @override
  void dispose() {
    // 문제가 발생하는 부분
    _ref.read(myProvider.notifier).reset(); // 이 메서드가 내부적으로 프로바이더 상태를 변경
    super.dispose();
  }
  
  // ...
}
```

## 해결 방법 : **WidgetsBinding.instance.addPostFrameCallback**

`addPostFrameCallback`은 위젯 트리가 빌드된 후에 실행되는 콜백을 등록하는 메서드다.

이를 이용하면 `dispose()`가 호출된 이후, 다음 프레임에서 provider의 상태를 변경할 수 있다.

```dart
@override
void dispose() {
  WidgetsBinding.instance.addPostFrameCallback((_) {
    // 프레임이 완료된 후 실행되므로 안전하게 상태 변경 가능
    ref.read(myProvider.notifier).reset(); 
  });
  super.dispose();
}
```

### 추가 해결 방법

1. **이벤트 핸들러 내에서 상태 변경**
  
   - 위젯 트리 빌드 중에 상태를 변경하지 않고, 버튼 클릭과 같은 이벤트 핸들러 내에서 상태를 변경하도록 한다.

    ```dart
    ElevatedButton(
      onPressed: () {
        // 이벤트 핸들러 내에서는 안전하게 상태 변경 가능
        ref.read(myProvider.notifier).updateState();
      },
      child: Text('Update State'),
    )
    ```

2. **Future를 사용하여 상태 변경 지연**
   
   - Dart 이벤트 루프를 사용하여 상태 변경을 지연
  
    ```dart
    @override
    void dispose() {
    // Future를 사용하여 위젯 트리 빌드 이후로 상태 변경 지연
      Future.microtask(() {
        ref.read(myProvider.notifier).reset(); 
      });
      super.dispose();
    }
    ```

3. **StateNotifierProvider 사용**

   - StateNotifierProvider는 해당 provider를 구독하는 위젯이 모두 사라지면 자동으로 dispose된다. 
  
    ```dart
    final counterProvider = StateNotifierProvider<CounterNotifier, int>((ref) {
      return CounterNotifier();
    });

    class CounterNotifier extends StateNotifier<int> {
      CounterNotifier() : super(0);

      void increment() {
        state++;
      }
      
      // 위젯이 dispose될 때 자동으로 정리됨
      @override
      void dispose() {
        print("CounterNotifier disposed");
        super.dispose();
      }
    }
    ```

    ```dart
    class MyScreen extends ConsumerWidget {
      @override
      Widget build(BuildContext context, WidgetRef ref) {
        final counter = ref.watch(counterProvider);

        return Scaffold(
          appBar: AppBar(title: Text('Counter')),
          body: Center(child: Text('Count: $counter')),
          // FloatingActionButton이 사라지면 CounterNotifier도 자동으로 dispose됨
          floatingActionButton: FloatingActionButton(
            onPressed: () => ref.read(counterProvider.notifier).increment(), 
            child: Icon(Icons.add),
          ),
        );
      }
    }
    ```
