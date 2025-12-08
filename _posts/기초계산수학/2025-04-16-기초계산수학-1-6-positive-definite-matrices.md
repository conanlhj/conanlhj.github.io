---
layout: post
title: "[기초계산수학] 1.6 Positive Definite Matrices"
date: 2025-04-16 15:14 +0900
description: Computational science and engineering 강의 내용 정리
author: shiggy
categories:
- 수업 내용 정리
- 기초계산수학
tag: [linear-algebra, positive-definite, matrix-decomposition, optimization, hessian-matrix, newtons-method, math-notes, numerical-linear-algebra]
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

## Positive Definite Matrices: 직관, 정의, 예제

Positive definite matrix(양의 정부호 행렬)는 수치 해석, 최적화, 머신러닝 등 다양한 분야에서 매우 중요한 개념이다. 이 글에서는 양의 정부호 행렬의 정의, 성질, 그리고 이를 판단하는 다양한 방법에 대해 예시를 통해 설명한다.

---

## 기본 개념: Positive Definiteness란?

**양의 정부호 행렬(Positive Definite Matrix)**은 다음 조건을 만족하는 **대칭(symmetric)** 행렬 $ K $이다.

$$
\forall u \neq 0,\quad u^T K u > 0
$$

이 조건은 행렬 $ K $에 대해 정의된 에너지 함수 $ P(u) = u^T K u $가 항상 양이라는 뜻이다. 즉, 벡터 $ u $가 원점이 아닐 경우, 항상 양의 값을 반환한다. 이는 수학적으로도 안정성과 최소값을 보장한다는 점에서 중요하다.

---

## 직관적 예시: 에너지 기반 정의

다음은 2×2 행렬에 대한 세 가지 예이다. 각 행렬에 대해 $ u = [u_1, u_2]^T $에 대해 $ u^T K u $를 계산해보자.

### 1. Positive Definite

$$
K = \begin{bmatrix} 2 & -1 \\ -1 & 2 \end{bmatrix}
$$
$$
u^T K u = 2u_1^2 - 2u_1u_2 + 2u_2^2 = u_1^2 + (u_1 - u_2)^2 + u_2^2 > 0
$$

이 함수는 모든 $ u \neq 0 $에서 양수이므로 양의 정부호이다.

---

### 2. Positive Semidefinite

$$
B = \begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix}
$$
$$
u^T B u = u_1^2 - 2u_1u_2 + u_2^2 = (u_1 - u_2)^2 \geq 0
$$

특정 벡터 ($ u_1 = u_2 $)에 대해 0이 되지만 음수는 아니므로 **semidefinite(반정부호)**이다.

---

### 3. Indefinite

$$
M = \begin{bmatrix} 1 & -3 \\ -3 & 1 \end{bmatrix}
$$
$$
u^T M u = u_1^2 - 6u_1u_2 + u_2^2 = (u_1 - 3u_2)^2 - 8u_2^2
$$

이 경우 $ u = [1, 1] $이면 결과가 음수(-4), $ u = [1, 10] $이면 양수(+41)이다. 즉, 방향에 따라 부호가 바뀌는 **indefinite(정부호 아님)** 행렬이다.

---

## 정의의 실질적 의미

양의 정부호 정의의 핵심은 다음과 같다.

- 그래프 형태로 보았을 때, $ u^T K u $는 원점을 기준으로 항상 위로 열린 **볼(bowl)** 형태이다.
- 최소값은 항상 $ u = 0 $에서 발생한다.
- 에너지 해석으로 해석하면, 모든 방향으로의 움직임에서 에너지가 증가함을 의미한다.

---

## Sum of Squares로 판별하기

양의 정부호인지 확인할 때 강력한 방법은 **제곱합 표현(sum of squares)**을 찾는 것이다.

예를 들어 위의 행렬 $ K $에 대해:

$$
u^T K u = 2u_1^2 - 2u_1u_2 + 2u_2^2 = (u_1 - u_2)^2 + u_1^2 + u_2^2
$$

모두 양수인 제곱항의 합으로 표현되므로 $ K $는 양의 정부호임이 보장된다.

### LDLᵗ 분해를 이용한 해석

$$
K = LDL^T = \begin{bmatrix} 1 & 0 \\ -0.5 & 1 \end{bmatrix} \begin{bmatrix} 2 & 0 \\ 0 & 1.5 \end{bmatrix} \begin{bmatrix} 1 & -0.5 \\ 0 & 1 \end{bmatrix}
$$

이렇게 **모든 pivot이 양수**라면 행렬은 양의 정부호이다.

---

## 다양한 표현: AᵗA, ATCA, QAQᵗ

### 1. $ K = A^T A $

- $ A $의 열들이 선형독립이면 $ K = A^T A $는 양의 정부호.
- 왜냐하면 $ u^T A^T A u = (A u)^T (A u) = \|A u\|^2 > 0 $

### 2. $ K = A^T C A $

- $ A $가 선형독립이고, $ C $가 양의 정부호라면, $ K $도 양의 정부호.

### 3. $ K = Q \Lambda Q^T $

- 모든 고유값 $ \lambda_i > 0 $이면 양의 정부호.
- 직교 행렬 $ Q $의 열벡터는 고유벡터.

---

## 예시: 3x3 Second Difference Matrix

$$
K = \begin{bmatrix} 2 & -1 & 0 \\ -1 & 2 & -1 \\ 0 & -1 & 2 \end{bmatrix}
$$

다음과 같은 다섯 가지 기준 모두를 만족하므로 positive definite이다.

1. 모든 피벗이 양수  
2. 모든 leading principal minor > 0  
3. 모든 고유값이 양수  
4. $ u^T K u > 0 $  
5. $ K = A^T A $ 형태 가능 (e.g. Cholesky factorization)

---

## 최소화 문제에서의 Positive Definiteness의 의미

Positive definite 행렬은 **최소화 문제(minimization problem)**와 깊은 연관이 있다. 특히 에너지 함수, 비용 함수, 손실 함수의 최소값을 찾을 때 핵심 역할을 한다.

---

## 2차 함수에서의 최소값: Quadratic Optimization

다음과 같은 형태의 비용 함수 $ P(u) $를 고려하자.

$$
P(u) = \frac{1}{2} u^T K u - u^T f
$$

여기서:
- $ K $는 대칭 행렬 (보통 positive definite),
- $ f $는 외력, 관측값 등의 벡터.

### 최소값을 찾는 조건

이 함수의 **gradient(기울기)**는 다음과 같다:

$$
\nabla P(u) = Ku - f
$$

이 식이 0이 되는 지점이 극값(최솟값 혹은 안장점)이다. 즉,
$$
Ku = f
$$

### 최소값의 해석적 표현

해당 지점이 **최소값**이 되려면 $ K $는 **positive definite**이어야 한다. 이 경우 최소값은 다음과 같이 계산된다.

$$
u^* = K^{-1}f,\quad
P_{\min} = -\frac{1}{2} f^T K^{-1} f
$$

---

## 최소값의 존재를 확인하는 방법

모든 $ u $에 대해 다음이 성립한다:

$$
P(u) - P(u^*) = \frac{1}{2}(u - u^*)^T K (u - u^*) \geq 0
$$

즉, $ u = u^* $를 제외하고는 항상 더 큰 값을 갖는다. 이는 $ K $가 positive definite일 때만 보장된다.

---

## 비선형 함수의 최소값: Hessian 조건

함수 $ P(u) $가 **비선형**이라면 어떻게 될까?

### Step 1: First Derivative Test

우선, 모든 편미분이 0인 점 $ \nabla P(u) = 0 $을 찾는다. 이 점이 극값 후보가 된다.

### Step 2: Second Derivative Test (Hessian)

해당 점에서 **Hessian 행렬 $ H $** (2차 도함수의 대칭 행렬)를 구하고 다음을 검사한다.

$$
u^T H u > 0\quad \forall u \neq 0
$$

즉, Hessian이 positive definite이면 **local minimum**임이 보장된다.

---

## 예시: 비선형 함수에서의 최소값 판별

다음과 같은 함수 $ P(u_1, u_2) $를 살펴보자.

$$
P(u) = 2u_1^2 + 3u_2^2 - u_1^4
$$

### Step 1: Gradient 계산

$$
\frac{\partial P}{\partial u_1} = 4u_1 - 4u_1^3,\quad
\frac{\partial P}{\partial u_2} = 6u_2
$$

이를 0으로 두면 세 개의 극점이 나온다:  
$$
(u_1, u_2) = (0, 0),\ (1, 0),\ (-1, 0)
$$

### Step 2: Hessian 분석

$$
H = \begin{bmatrix} 4 - 12u_1^2 & 0 \\ 0 & 6 \end{bmatrix}
$$

- $ (0, 0) $에서는 $ H = \begin{bmatrix} 4 & 0 \\ 0 & 6 \end{bmatrix} $ → **positive definite** → local minimum
- $ (1, 0) $과 $ (-1, 0) $에서는 $ H = \begin{bmatrix} -8 & 0 \\ 0 & 6 \end{bmatrix} $ → **indefinite** → saddle point

---

## Newton's Method for Minimization

비선형 함수의 최소점을 효율적으로 찾기 위한 대표적인 알고리즘이 **Newton’s Method**이다.

### 알고리즘 개요

1. 현재 위치 $ u^{(i)} $에서 gradient와 Hessian을 계산
2. 다음 위치 $ u^{(i+1)} $는 아래 식으로 업데이트

$$
u^{(i+1)} = u^{(i)} - H^{-1} \nabla P(u^{(i)})
$$

이는 2차 테일러 근사를 최소화하는 지점을 선택하는 방식이다.

### 주의점

- Hessian 계산 비용이 크거나, 양의 정부호가 아닐 경우 적용이 어려울 수 있다.
- 너무 큰 스텝은 발산 가능 → damping factor $ c < 1 $를 곱해서 스텝 조절

---

## 실용 요약: Positive Definiteness 확인법

행렬 $ K $가 양의 정부호인지 확인하는 5가지 주요 방법:

1. 모든 고유값이 양수인가?
2. 모든 피벗이 양수인가? (LDLᵗ)
3. $ K = A^T A $ 형태인가? (A의 열이 선형독립)
4. $ K = Q \Lambda Q^T $에서 $ \Lambda > 0 $?
5. 임의의 $ u \neq 0 $에 대해 $ u^T K u > 0 $?

---

## 마무리

양의 정부호 행렬은 단순한 수학적 정의를 넘어서, **최적화 문제의 최소값 존재 보장**, **수치적 안정성**, **행렬 분해(Cholesky, SVD, Eigen)** 등에 필수적인 조건이다. 어떤 형태의 문제를 다루든, positive definiteness는 일종의 **신뢰성 있는 기반**으로 작용한다.



---
## Reference
[^1]: Gilbert Strang, *Computational Science and Engineering*, Wellesley-Cambridge Press, 2007. DOI: [10.1137/1.9780961408817](https://epubs.siam.org/doi/abs/10.1137/1.9780961408817).
