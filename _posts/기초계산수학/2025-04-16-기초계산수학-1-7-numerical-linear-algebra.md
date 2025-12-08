---
layout: post
title: "[기초계산수학] 1.7 Numerical Linear Algebra: LU, QR, SVD"
date: 2025-04-16 15:23 +0900
description: Computational science and engineering 강의 내용 정리
author: shiggy
categories:
- 수업 내용 정리
- 기초계산수학
tag:
  - linear-algebra
  - matrix-decomposition
  - svd
  - qr-decomposition
  - pseudoinverse
  - numerical-linear-algebra
  - condition-number
  - math-notes
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

## Numerical Linear Algebra: LU, QR, SVD의 핵심

수학은 현실 문제를 식으로 바꾸는 일이고, 과학적 계산은 그 식을 실제로 **푸는 일**이다.  
그 중심에는 바로 **행렬 A를 어떻게 다룰 것인가**가 있다.

예를 들어 이런 식들이 있다:

- $ Ku = f $ : 선형 시스템
- $ Kx = \lambda x $ : 고유값 문제
- $ Mu'' + Ku = 0 $ : 진동 모델

이러한 문제들을 해결하려면 행렬 $ A $를 더 간단한 형태로 **쪼개는 과정**, 즉 **행렬 분해(factorization)**가 필요하다.  
그리고 이 분해는 대체로 **삼각행렬**, **직교행렬**, **희소행렬**의 곱으로 이루어진다.

---

## 우리가 다루게 될 세 가지 핵심 분해

우리는 앞으로 세 가지 분해를 자주 만나게 된다.  
각각은 조금씩 다른 목적과 특징을 가진다.

$$
A = LU,\quad A = QR,\quad A = U \Sigma V^T
$$

---

### 1. LU 분해

LU 분해는 가장 전통적인 방식이다.

$$
A = LU
$$

- $ L $: lower triangular (하삼각행렬)
- $ U $: upper triangular (상삼각행렬)

이건 가우시안 소거법과 같은 소거(elimination) 과정이라고 보면 된다.  
행을 아래로 내려가면서 위의 항들을 제거해나가는 방식이다.

LU 분해는 계산이 빠르고 직관적이지만, 수치적으로는 **불안정**할 수 있다.  
그래서 종종 행 교환(pivoting)이 필요한 경우도 있다. (뒤에서 다시 다룬다.)

---

### 2. QR 분해

QR 분해는 열벡터들을 **서로 직교**하고 **단위 벡터**로 만드는 방식이다.

$$
A = QR
$$

- $ Q $: 오쏘노멀(orthonormal) 열벡터로 구성된 행렬
- $ R $: upper triangular 행렬

이 방식은 계산 안정성이 뛰어나다. 특히, $ A^T A $ 같은 계산이 간단해지고, 수치 오차에도 강한 특성이 있다.  
나중에 최소제곱법(Least Squares) 문제를 풀 때 이 분해가 핵심 도구가 된다.

QR 분해는 두 가지 방식으로 계산할 수 있다:
- Gram-Schmidt 직교화
- Householder 반사 행렬 (이쪽이 더 안정적)

---

### 3. SVD (Singular Value Decomposition)

SVD는 모든 행렬을 회전–늘림–회전의 조합으로 분해하는 아주 강력한 방식이다.

$$
A = U \Sigma V^T
$$

- $ U $: AAT의 고유벡터 (좌측 특이벡터), 오쏘노멀
- $ \Sigma $: singular value들을 담은 대각행렬 (음이 아닌 값)
- $ V $: ATA의 고유벡터 (우측 특이벡터), 오쏘노멀

이건 모든 $ m \times n $ 행렬에 대해 정의되고, 대칭일 필요도 없다.  
특히 **데이터 축소, 압축, 잡음 제거** 같은 곳에서 핵심적으로 쓰인다.

SVD는 **행렬을 rank-1 단위로 쪼개서**, 의미 있는 정보부터 우선적으로 정리해준다.  
예를 들어 이미지 압축, 추천 시스템, PCA 분석 등에서 이게 사용된다.

---

## Positive Definite일 때 모든 게 하나로 모인다

특히, 행렬 $ K $가 **symmetric positive definite**라면 세 가지가 모두 연결된다.

$$
K = U \Sigma V^T = Q R = L L^T
$$

이 경우,
- $ U = Q $, $ V = Q $
- $ \Sigma $의 값들은 고유값과 같다
- $ K = Q \Lambda Q^T $ 형태로도 쓸 수 있다

즉, **모든 방향에서 잘 정의되고, 안정적인 계산이 가능**하다.

---

## Orthogonal matrix의 개념

QR과 SVD에서 공통적으로 등장하는 것이 바로 **orthogonal matrix**이다.  
이건 말 그대로, 열벡터들이 서로 **직교**하고 각 벡터의 크기가 1인 행렬을 말한다.

$$
Q^T Q = I
$$

이런 행렬을 곱해도 벡터의 크기(길이)는 변하지 않는다.  
수치 계산에서 **크기를 보존**하는 것은 아주 중요한 성질이다.

$$
\|Qx\| = \|x\|
$$

---

### 직관적인 예시들

#### 예시 1. 순열 행렬 (Permutation matrix)

행이나 열의 순서를 바꾸는 행렬이다. $ P^T = P^{-1} $이며, 크기를 바꾸지 않는다.

$$
P = 
\begin{bmatrix}
0 & 1 & 0 \\
1 & 0 & 0 \\
0 & 0 & 1 \\
\end{bmatrix}
$$

#### 예시 2. 회전 행렬 (Rotation)

벡터의 방향만 바꾼다. 크기는 그대로.

$$
Q = 
\begin{bmatrix}
\cos \theta & -\sin \theta \\
\sin \theta & \cos \theta \\
\end{bmatrix}
$$

#### 예시 3. 반사 행렬 (Householder reflection)

거울에 비춘 것처럼 반사시킨다. 크기 유지.

$$
H = I - 2uu^T
$$

이건 나중에 QR 분해할 때도 핵심 도구로 쓰인다.

## QR 분해: 직교화의 기술

QR 분해는 행렬 $ A $를 다음과 같이 나누는 것이다:

$$
A = QR
$$

- $ Q $: 열벡터들이 서로 직교하고, 각 벡터의 크기가 1인 **orthonormal matrix**
- $ R $: 위쪽 삼각형 형태의 **upper triangular matrix**

이렇게 쪼개는 이유는 단순하다.  
계산을 **안정적이고 효율적으로** 만들기 위해서다.  
특히 최소제곱 문제(least squares)처럼, 직접적으로 $ A^T A $를 계산하면 오차가 커지기 쉬운 문제에서 이 QR 분해가 큰 역할을 한다.

---

## Gram-Schmidt 방식

Gram-Schmidt는 일종의 "차례대로 직교화" 과정이다.  
벡터들을 하나씩 보면서 앞의 벡터들과 직교하도록 조정해 나가는 방식이다.

### 기본 아이디어

주어진 벡터들 $ a_1, a_2, \dots, a_n $에 대해서 다음과 같이 진행한다:

1. $ q_1 = \frac{a_1}{\|a_1\|} $
2. $ q_2 = \frac{a_2 - \text{proj}_{q_1}(a_2)}{\| \cdot \|} $
3. $ q_3 = \frac{a_3 - \text{proj}_{q_1}(a_3) - \text{proj}_{q_2}(a_3)}{\| \cdot \|} $
4. 계속...

여기서 projection은 다음과 같이 계산된다:

$$
\text{proj}_{q}(a) = (q^T a)q
$$

즉, $ a $ 벡터에서 $ q $ 방향의 성분을 제거하는 것이다.

---

### 예시: 손으로 따라가기

$$
A = 
\begin{bmatrix}
1 & 1 \\
1 & 0 \\
0 & 1 \\
\end{bmatrix}
$$

1. 첫 번째 열: $ a_1 = (1, 1, 0)^T $

$$
q_1 = \frac{a_1}{\|a_1\|} = \frac{1}{\sqrt{2}}(1, 1, 0)^T
$$

2. 두 번째 열: $ a_2 = (1, 0, 1)^T $

먼저 $ q_1 $ 방향 성분을 제거한다:

$$
\text{proj}_{q_1}(a_2) = (q_1^T a_2) q_1 = \frac{1}{2}(1, 1, 0)^T
$$

$$
u_2 = a_2 - \text{proj}_{q_1}(a_2) = \left( \frac{1}{2}, -\frac{1}{2}, 1 \right)^T
$$

정규화하면:

$$
q_2 = \frac{1}{\sqrt{\frac{3}{2}}}\left( \frac{1}{2}, -\frac{1}{2}, 1 \right)^T = \frac{1}{\sqrt{6}}(1, -1, 2)^T
$$

이런 식으로 $ Q $와 $ R $이 구성된다.

---

## Householder 방식

Gram-Schmidt는 직관적이지만, 수치적으로는 **불안정할 수 있다.**  
특히 벡터 간의 내적이 작거나, 거의 선형 종속일 경우 오차가 증폭된다.  
그래서 실제 계산에서는 **Householder 반사 행렬(reflection)**을 많이 쓴다.

### 핵심 아이디어

어떤 벡터 $ x $를 **첫 번째 좌표만 남고 나머지는 0이 되도록 반사**시키는 행렬을 만든다:

$$
H = I - 2uu^T,\quad u = \frac{x - \alpha e_1}{\|x - \alpha e_1\|},\quad \alpha = \pm\|x\|
$$

이런 $ H $를 순차적으로 $ A $에 곱해서 아래쪽을 모두 0으로 만든다.  
그 결과, $ A $는 upper triangular 형태인 $ R $이 되고, $ Q $는 이러한 $ H $들을 모두 곱한 것이다.

---

### Householder가 더 안정적인 이유

- 모든 연산이 벡터 하나와 그 반사 기준만으로 이루어진다.
- 직교성(orthogonality)을 수치적으로 정확하게 유지한다.
- MATLAB, NumPy 등 대부분의 라이브러리에서 기본 QR 방식이다.

---

## Q의 안정성

QR 분해에서 얻은 $ Q $는 **수치적으로 아주 안정적인 행렬**이다.  
벡터의 길이를 바꾸지 않기 때문에, 오차가 전파되거나 증폭되지 않는다.

예를 들어 $ Qx = b $를 푸는 상황에서는 다음이 성립한다:

$$
\|x\| = \|b\|,\quad \|\Delta x\| = \|\Delta b\|
$$

---

## 요약: Gram-Schmidt vs. Householder

| 기준                 | Gram-Schmidt            | Householder                     |
| -------------------- | ----------------------- | ------------------------------- |
| 직관성               | 매우 직관적             | 상대적으로 복잡                 |
| 수치적 안정성        | 낮음                    | 매우 높음                       |
| 계산 속도            | 빠름 (작은 문제에 적합) | 조금 느리지만 안정적            |
| 기본 라이브러리 채택 | 거의 사용되지 않음      | 대부분 사용 (예: MATLAB `qr()`) |

---

## SVD (Singular Value Decomposition): 행렬 분해의 끝판왕

SVD는 모든 $ m \times n $ 행렬에 대해 성립하는 아주 강력한 분해 방법이다.

$$
A = U \Sigma V^T
$$

- $ U $: AAT의 고유벡터들로 이루어진 $ m \times m $ orthogonal matrix  
- $ V $: ATA의 고유벡터들로 이루어진 $ n \times n $ orthogonal matrix  
- $ \Sigma $: singular values(특이값)만 포함된 $ m \times n $ 대각 행렬, 값은 항상 0 이상

이 분해의 의미는 간단히 말해서 **회전 – 늘림 – 다시 회전**이다.  
모든 행렬을 이러한 세 가지 단순한 연산의 조합으로 바꿀 수 있다는 말이다.

---

## SVD의 직관적인 해석

임의의 벡터 $ x $에 대해 $ Ax $를 계산한다고 하자.

$$
Ax = U \Sigma V^T x
$$

- 먼저 $ V^T $가 벡터를 회전시키고  
- $ \Sigma $가 각 축을 따라 늘리거나 줄이고  
- 마지막으로 $ U $가 다시 회전시킨다

결과적으로, $ A $는 벡터의 방향을 바꾸고 길이도 바꾼다.  
하지만 이 과정은 세 가지 단순한 연산으로 **분해 가능**하다는 점에서 아주 유용하다.

---

## Reduced SVD

대부분의 경우, $ A $의 rank가 $ r $이면 전체 $ U, \Sigma, V $를 다 쓸 필요는 없다.  
이럴 땐 다음과 같이 "축소된 형태(reduced form)"로 쓸 수 있다:

$$
A = U_r \Sigma_r V_r^T
$$

- $ U_r $: $ m \times r $  
- $ \Sigma_r $: $ r \times r $  
- $ V_r $: $ n \times r $

이 경우, $ A $는 다음과 같이 **rank-1 행렬들의 합**으로 표현된다:

$$
A = \sigma_1 u_1 v_1^T + \sigma_2 u_2 v_2^T + \dots + \sigma_r u_r v_r^T
$$

여기서 $ \sigma_i $는 singular value이고, 각 항은 rank-1 행렬이다.  
**가장 중요한 정보부터 차례로 정렬되어 있는 구조**라고 보면 된다.

---

## 예시: 아주 간단한 2x2 행렬

$$
A = 
\begin{bmatrix}
1 & 7 \\
\end{bmatrix}
$$

이 행렬은 rank 1이다.  
SVD를 적용하면 다음과 같이 쓸 수 있다:

$$
A = \frac{1}{\sqrt{50}} 
\begin{bmatrix}
1 \\
7 \\
\end{bmatrix}
\cdot 10 \cdot
\frac{1}{\sqrt{2}} 
\begin{bmatrix}
1 & 1 \\
\end{bmatrix}
$$

즉, $ A = u \sigma v^T $로 표현되며, 이건 하나의 rank-1 행렬이다.  
PCA나 이미지 압축에서도 이런 식으로 의미 있는 정보만 추출해서 사용한다.

---

## Pseudoinverse: 역행렬이 없어도 해를 구할 수 있다

보통 $ A^{-1} $은 정방행렬에만 정의되어 있다.  
하지만 대부분의 경우 $ A $는 정방이 아니거나, 심지어 역행렬이 존재하지 않기도 한다.

이럴 때 등장하는 것이 바로 **pseudoinverse $ A^+ $**이다.  
SVD를 이용하면 다음처럼 구할 수 있다:

$$
A^+ = V \Sigma^+ U^T
$$

- $ \Sigma^+ $는 $ \Sigma $에서 0이 아닌 singular value들에 대해서 역수를 취한 것
- 0인 값은 그냥 그대로 0으로 둔다

이 pseudoinverse는 다음 조건을 만족한다:

$$
AA^+A = A,\quad A^+AA^+ = A^+,\quad (AA^+)^T = AA^+,\quad (A^+A)^T = A^+A
$$

---

### 예시: pseudoinverse 계산

$$
A = 
\begin{bmatrix}
1 & 7 \\
\end{bmatrix}
\Rightarrow A^+ = \frac{1}{100}
\begin{bmatrix}
1 \\
7 \\
\end{bmatrix}
$$

이건 직관적으로도 맞는 결과다.  
$ A $가 $ uv^T $ 형태의 rank-1 행렬이면, $ A^+ = \frac{1}{\sigma} vu^T $ 형태가 된다.

---

## Condition Number와 Norm

마지막으로, 행렬의 **민감도(sensitivity)**를 측정하는 값이 있다.  
바로 **condition number**다.

### 정의

$$
\text{condition number } c(A) = \frac{\sigma_{\max}}{\sigma_{\min}}
$$

- 값이 크면 클수록, **작은 오차가 해에 큰 영향을 미친다.**
- 값이 1에 가까울수록, **문제가 안정적이고 잘 조건 지워진(well-conditioned)** 상태다.

---

### Norm이란?

행렬의 norm은 $ Ax $가 얼마나 늘어나는지를 나타낸다:

$$
\|A\| = \max_{x \ne 0} \frac{\|Ax\|}{\|x\|}
$$

이건 결국 가장 크게 늘어나는 방향(= singular vector 방향)의 **스케일링 비율**이다.  
즉, $ \|A\| = \sigma_{\max} $ 이고,  
$ \|A^{-1}\| = \frac{1}{\sigma_{\min}} $ 이다.

---

## 요약: SVD, Pseudoinverse, 그리고 Condition Number

SVD는 어떤 행렬이라도 세 개의 단순한 연산—회전, 스케일, 회전—으로 분해할 수 있도록 해준다.  
이 구조를 통해 우리는 행렬이 데이터를 어떻게 변형시키는지를 직관적으로 이해할 수 있고, 중요한 정보만을 남겨서 압축하거나 잡음을 제거할 수도 있다.

또한, 역행렬이 존재하지 않더라도 SVD를 활용하면 **pseudoinverse**를 구할 수 있다.  
이것은 선형 시스템의 해가 존재하지 않거나 유일하지 않을 때 **가장 적절한 해**, 특히 **최소 노름 해**를 구할 수 있게 해준다.

마지막으로, **condition number**는 주어진 행렬이 얼마나 "민감한가"를 나타내는 지표다.  
이는 결국 가장 크게 늘어나는 방향과 가장 작게 늘어나는 방향의 비율이며, 수치 계산의 안정성과 신뢰도에 큰 영향을 준다.  
Condition number가 클수록 작은 오차도 결과에 크게 영향을 미칠 수 있다.

한 문장으로 정리하자면,  
> **SVD는 행렬의 구조를 가장 근본적으로 이해할 수 있는 도구이며, 수치 해석의 거의 모든 문제에 연결된다.**

---
## Reference
[^1]: Gilbert Strang, *Computational Science and Engineering*, Wellesley-Cambridge Press, 2007. DOI: [10.1137/1.9780961408817](https://epubs.siam.org/doi/abs/10.1137/1.9780961408817).
