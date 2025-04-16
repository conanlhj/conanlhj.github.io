---
layout: post
title: "[기초계산수학] 1.5 Eigenvalues and Eigenvectors (Part II)"
date: 2025-04-16 12:27 +0900
description: Computational science and engineering 강의 내용 정리
author: shiggy
categories:
- 수업 내용 정리
- 기초계산수학
tag:
- [eigenvalues, eigenvectors]
pin: false
math: true
mermaid: true
toc: true
comments: true
render_with_liqueid: true
---

## 시작에 앞서..
> 이 포스팅 시리즈는 대학원 수업 [기초계산수학]의 내용을 바탕으로 정리한 글입니다. Gilbert strang 교수님의 책 "Computational science and engineering[^1]"을 참고하여 작성하였습니다.
{:.prompt-info}

---

## Eigenvalues and Eigenvectors

이번 장에서는 $ A x = \lambda x $ 형태를 다룬다. 이것은 eigenvector $ x $와 각각에 대응되는 eigenvalue $ \lambda $와 관련된 식이다. $ \text{det}(A - \lambda I) = 0 $ 부터 시작해서 위 식을 풀 수는 있지만.. 행렬이 커질수록 저 방법으로 푸는 것은 아주 고통스러울 것이다. 수치선형대수의 발전으로 eigenvalue를 계산하는 빠르고 안정적인 알고리즘이 등장했으나, 이번 장에서는 좀 특별한 행렬들을 다루기 때문에 수치적으로가 아닌 정확한 $ \lambda $와 $ x $를 구하는 방법을 다를 것이다. 이번 장은 다음의 두 파트로 나뉠 수 있다.

1. Part I. 행렬 $ A $를 diagonalize하는데에 eigenvalue를 적용하여 $ u' = Au $를 푼다. ($ A = S \Lambda S^{-1} $)
2. Part II. $ K_n, T_n, B_n, C_n $ 행렬의 eigenvalues는 전부 $ \lambda = 2 - 2 \cos \theta $이다. ($ K = Q \Lambda Q^T $)

---

### Part II: Eigenvectors for Derivatives and Differences

이 챕터는 연속적인 문제(미분 방정식)와 이산적인 문제(행렬 방정식) 사이의 유사성에 대한 통찰을 제공한다. 특히, 2차 차분 행렬이 어떻게 미분 연산자와 유사한 구조를 가지는지, 그에 따라 고유벡터와 고유값이 어떤 형태를 가지는지를 탐구한다.

---

### 1. 연속적인 경우: 미분 방정식의 고유함수

2차 미분 방정식
$$
- \frac{d^2y}{dx^2} = \lambda y(x)
$$
은 주기함수 $ y(x) = \cos(\omega x), \sin(\omega x) $를 해로 가진다. 이 경우 고유값은 $ \lambda = \omega^2 $가 된다.

#### 예시 1: 고정-고정 경계 조건

$$
y(0) = 0,\quad y(1) = 0
$$
이 조건을 만족하는 고유함수는 다음과 같다.
$$
y_k(x) = \sin(k \pi x),\quad k = 1, 2, 3, \dots
$$
고유값은
$$
\lambda_k = (k \pi)^2
$$
이다.

---

### 2. 이산적인 경우: 행렬 $ K_n $의 고유벡터 (Discrete Sines)

행렬 $ K_n $은 중심에 2, 좌우에 -1이 위치한 삼각행렬로, 2차 차분을 수행한다.

이 행렬의 고유벡터는 다음과 같은 형태의 벡터들이다:
$$
y_k = \left( \sin\left(\frac{k \pi h}{n+1}\right), \sin\left(\frac{2k \pi h}{n+1}\right), \dots, \sin\left(\frac{nk \pi h}{n+1}\right) \right)
$$

여기서 $ h = \frac{1}{n+1} $, $ k = 1, 2, \dots, n $이다. 고유값은
$$
\lambda_k = 2 - 2\cos\left(\frac{k \pi}{n+1}\right)
$$
로 계산된다.

#### 예시 2: $ K_3 $의 고유값
$$
\lambda_1 = 2 - 2\cos\left(\frac{\pi}{4}\right) = 2 - \sqrt{2},\quad
\lambda_2 = 2,\quad
\lambda_3 = 2 + \sqrt{2}
$$

---

### 3. 행렬 $ B_n $: Discrete Cosine Transform (DCT)

$ B_n $은 경계에서 미분값이 0인 경우(자유 경계 조건)를 모사한다. 고유벡터는 다음과 같이 코사인 함수를 이산적으로 샘플링한 형태를 가진다.
$$
y_k = \left( \cos\left(\frac{(j - \frac{1}{2})k\pi}{n}\right) \right),\quad j = 1, \dots, n
$$

고유값은
$$
\lambda_k = 2 - 2\cos\left(\frac{k\pi}{n}\right),\quad k = 0, 1, \dots, n - 1
$$

#### 예시 3: $ B_3 $의 고유값
$$
\lambda_0 = 0,\quad \lambda_1 = 1,\quad \lambda_2 = 3
$$

고유벡터 $ y_0 = (1, 1, 1) $는 상수 함수에 대응되며, 이는 DC(0Hz) 성분이다.

---

### 4. 행렬 $ C_n $: Discrete Fourier Transform (DFT)

$ C_n $은 주기 경계 조건을 가진 순환 행렬이다. 이 행렬의 고유벡터는 복소 지수함수 $ e^{2\pi ikx} $를 이산적으로 샘플링한 형태를 가진다.

$$
y_k = \left( 1, w^k, w^{2k}, \dots, w^{(n-1)k} \right),\quad w = e^{2\pi i / n}
$$

고유값은
$$
\lambda_k = 2 - w^k - w^{-k} = 2 - 2\cos\left(\frac{2\pi k}{n}\right)
$$

#### 예시 4: $ C_4 $의 고유벡터

- $ y_0 = (1, 1, 1, 1) $
- $ y_1 = (1, i, -1, -i) $
- $ y_2 = (1, -1, 1, -1) $
- $ y_3 = (1, -i, -1, i) $

이 고유벡터들을 열로 갖는 행렬 $ F_n $은 푸리에 행렬이 되며, $ F_n^* F_n = nI $ 관계를 만족한다.

---

### 5. 정리: 세 가지 변환 비교

| 행렬    | 고유벡터 형태       | 고유값 예시                     | 대응하는 연속 해 |
| ------- | ------------------- | ------------------------------- | ---------------- |
| $ K_n $ | Sine                | $ 2 - 2\cos(\frac{k\pi}{n+1}) $ | $ \sin(k\pi x) $ |
| $ B_n $ | Cosine              | $ 2 - 2\cos(\frac{k\pi}{n}) $   | $ \cos(k\pi x) $ |
| $ C_n $ | Complex Exponential | $ 2 - 2\cos(\frac{2\pi k}{n}) $ | $ e^{2\pi ikx} $ |

이러한 이산 고유벡터들은 신호 처리, 수치 해석, 물리 시뮬레이션 등 다양한 분야에서 핵심적인 역할을 하며, 빠른 변환(Fast Fourier Transform, Fast Sine Transform 등)을 통해 효율적인 계산이 가능해진다.


---
## Reference
[^1]: Gilbert Strang, *Computational Science and Engineering*, Wellesley-Cambridge Press, 2007. DOI: [10.1137/1.9780961408817](https://epubs.siam.org/doi/abs/10.1137/1.9780961408817).
