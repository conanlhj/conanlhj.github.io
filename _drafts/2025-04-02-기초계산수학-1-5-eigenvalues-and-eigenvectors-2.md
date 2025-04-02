---
layout: post
title: "[기초계산수학] 1.5 Eigenvalues and Eigenvectors (Part II)"
date: 2025-04-02 16:27 +0900
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

---
## Reference
[^1]: Gilbert Strang, *Computational Science and Engineering*, Wellesley-Cambridge Press, 2007. DOI: [10.1137/1.9780961408817](https://epubs.siam.org/doi/abs/10.1137/1.9780961408817).
