---
layout: post
title: "[기초계산수학] 1.1 Four Special Matrices"
date: 2025-03-20 15:44 +0900
description: Computational science and engineering 강의 내용 정리
author: shiggy
categories:
- 수업 내용 정리
- 기초계산수학
tag:
- [Toeplitz, Matrix, Applied Mathematics]
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


## **1. 서론: 행렬의 의미와 중요성**
행렬은 단순한 숫자의 배열이 아니라, **연산자(operator)** 로서의 의미를 가질 수 있다.  
예를 들어, 행렬 $A$ 가 벡터 $x$ 에 작용하여 $Ax$ 를 만든다고 할 때, 벡터의 원소들은 **물리적인 의미**(변위, 압력, 전압, 가격, 농도 등)를 가질 수 있으며, 행렬은 이를 변환하는 역할을 한다.

이 장에서는 네 가지 중요한 행렬 패밀리인 **Toeplitz 행렬**과 그 변형들을 다룬다.  
이들은 수학적 이론뿐만 아니라, **수치 해석, 신호 처리, 물리학 등 다양한 응용 분야**에서 핵심적인 역할을 한다.

## **2. Toeplitz 행렬 $K_n$**
Toeplitz 행렬은 **대각선이 일정한 값을 가지는 특수한 행렬**이다.  
그중에서도 **삼중 대각 행렬(Tridiagonal Matrix)** 형태를 가지는 **$K_n$** 행렬은 매우 중요한 성질을 갖는다.

### **2.1. $K_n$ 행렬의 형태**
$$
K_n =
\begin{bmatrix}
2 & -1 & 0 & 0 & \cdots & 0 \\
-1 & 2 & -1 & 0 & \cdots & 0 \\
0 & -1 & 2 & -1 & \cdots & 0 \\
\vdots & \vdots & \vdots & \vdots & \ddots & -1 \\
0 & 0 & 0 & 0 & -1 & 2
\end{bmatrix}
$$

이 행렬의 특징은 다음과 같다.

### **2.2. $K_n$ 의 특징**
1. **대칭 행렬(Symmetric Matrix)**  
   - 행렬의 전치 $K^T = K$ 이 성립함.
2. **희소 행렬(Sparse Matrix)**  
   - 대부분의 원소가 0이며, 비대각 원소는 총 $2(n-1)$ 개.
3. **삼중 대각 행렬(Tridiagonal Matrix)**  
   - 주대각선에 2, 첫 번째 상·하부 대각선에 -1.
4. **상수 계수를 가지는 행렬**  
   - Fourier 변환과 관련이 깊으며, 물리 시스템에서 **이산 라플라시안(Discrete Laplacian)** 역할을 함.
5. **역행렬이 존재하는 양의 정부호(Positive Definite) 행렬**  
   - 모든 고유값이 양수이므로, 선형 방정식 $K_n x = b$ 를 **안정적으로 해결**할 수 있음.

## **3. Circulant 행렬 $C_n$**
Toeplitz 행렬을 변형하여, **대각선이 순환(circular)하는 행렬**을 만들 수도 있다.

$$
C_4 =
\begin{bmatrix}
2 & -1 & 0 & -1 \\
-1 & 2 & -1 & 0 \\
0 & -1 & 2 & -1 \\
-1 & 0 & -1 & 2
\end{bmatrix}
$$

### **3.1. $C_n$ 행렬의 특징**
- Toeplitz 행렬과 유사하지만, **네 모서리(-1 값)가 추가**됨.
- **Circulant 행렬**이며, Fourier 변환을 사용하여 빠르게 대각화할 수 있음.
- **특이행렬(Singular Matrix)**: 행렬식이 0이므로, 역행렬이 존재하지 않음.

### **3.2. 왜 $C_n$ 은 특이행렬인가?**
행렬의 모든 행을 더하면, 결과가 **0 벡터**가 된다.  
즉, 모든 원소가 1인 벡터 $u = (1, 1, 1, 1)^T$ 가 **null space** 에 포함된다.  
이는 곧 **선형 종속이 존재**함을 의미하며, **역행렬이 존재하지 않는다는 것**을 의미한다.

## **4. 변형된 Toeplitz 행렬: $T_n$ 과 $B_n$**
Toeplitz 행렬의 변형으로, **특정한 경계 조건(Boundary Conditions)을 적용한 행렬**들이 존재한다.

### **4.1. $T_n$ 행렬 (변형된 Toeplitz 행렬)**
$$
T_n =
\begin{bmatrix}
1 & -1 & 0 & 0 \\
-1 & 2 & -1 & 0 \\
0 & -1 & 2 & -1 \\
0 & 0 & -1 & 2
\end{bmatrix}
$$

- 첫 번째 행의 (1,1) 원소가 **1로 변경됨**.
- **역행렬이 존재하며**, 모든 고유값이 양수인 **양의 정부호 행렬(Positive Definite Matrix)** 임.

### **4.2. $B_n$ 행렬**
$$
B_n =
\begin{bmatrix}
1 & -1 & 0 & 0 \\
-1 & 2 & -1 & 0 \\
0 & -1 & 2 & -1 \\
0 & 0 & -1 & 1
\end{bmatrix}
$$

- **첫 번째와 마지막 원소가 모두 1로 변경됨**.
- $C_n$ 과 마찬가지로 **특이행렬(Singular Matrix)**.


## **5. 결론**
Toeplitz 행렬은 수치 해석과 신호 처리에서 **핵심적인 역할을 하는 행렬**이다.  
특히, $K_n$, $C_n$, $T_n$, $B_n$ 같은 특정한 구조를 가진 행렬들은 **푸리에 변환, 미분 방정식, 물리적 시스템 해석 등에서 자주 사용**된다.

- **$K_n$**: **양의 정부호** 행렬로, **이산 라플라시안** 역할 수행. (*Non-Singular*)
- **$C_n$**: **특이 행렬**이며, **푸리에 변환과 관련**됨. (*Singular*)
- **$T_n$**: 경계 조건이 다른 Toeplitz 행렬, **양의 정부호**. (*Non-Singular*)
- **$B_n$**: 특이 행렬이며, **null space** 가 존재. (*Singular*)

이러한 행렬의 성질을 이해하면, **행렬 방정식 해법, 고유값 분석, 신호 처리 등 다양한 분야에서 효과적으로 활용할 수 있다**.

---

## Reference

[^1]: Gilbert Strang, *Computational Science and Engineering*, Wellesley-Cambridge Press, 2007. DOI: [10.1137/1.9780961408817](https://epubs.siam.org/doi/abs/10.1137/1.9780961408817).
