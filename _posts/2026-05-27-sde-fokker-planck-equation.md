---
layout: post
title: "[SDE] 기초개념: Fokker-Planck Equation - 확률 분포가 흐른다"
date: 2026-05-27 20:27 +0900
description: "SDE의 샘플 관점을 분포 관점으로 바꿔주는 방정식. 직관부터 유도, Diffusion model에서의 해석까지."
author: shiggy
categories: [공부, "확률미분방정식(SDE)"]
tags:
  [sde, fokker-planck, diffusion model, ornstein-uhlenbeck, score function, stochastic calculus]
pin: false
math: true
mermaid: true
toc: true
comments: true
render_with_liqueid: true
---

## 1. 핵심 질문

SDE를 공부하다 보면 이런 생각이 자연스럽게 든다.

> SDE는 **하나의 샘플**이 시간에 따라 어떻게 움직이는지를 기술한다. 그런데 **수많은 샘플들의 분포 전체**가 시간에 따라 어떻게 변하는지는 어떻게 기술할까?

다음 SDE를 생각해보자:

$$dx = f(x,t)\,dt + g(t)\,dw$$

샘플 $x(t)$ 하나는 drift $f$에 이끌리면서 노이즈 $g\,dw$에 흔들리며 돌아다닌다. 그런데 같은 초기 분포 $p_0(x)$에서 출발한 수많은 샘플들을 동시에 풀어놓으면, 그 전체의 밀도 함수 $p_t(x)$는 어떤 방정식을 따를까?

이것이 **Fokker-Planck equation**이 답하는 질문이다.

---

## 2. 직관

방 안에 연기 입자들이 가득 차 있다고 하자. 각 입자는 SDE를 따라 움직인다. 우리가 관심 있는 건 입자 하나의 궤적이 아니라, **"t초 후에 방의 어느 구역에 입자가 얼마나 밀집해 있는가"**, 즉 밀도 함수 $p_t(x)$다.

이 밀도가 변하는 원인은 두 가지다.

- **Drift $f$**: 바람처럼 입자들을 특정 방향으로 쓸어간다. 분포 전체가 이동한다.
- **Noise $g\,dw$**: 입자들을 사방으로 퍼뜨린다. 분포가 넓어진다.

Fokker-Planck equation은 이 두 효과를 하나의 PDE로 정확하게 기술한다.

---

## 3. 유도

임의의 smooth한 test function $\phi(x)$에 대해 기댓값의 시간 변화를 생각해보자:

$$\frac{d}{dt}\,\mathbb{E}[\phi(x(t))] = \frac{d}{dt}\int \phi(x)\,p_t(x)\,dx$$

좌변을 Itô's formula로 전개한다. $\phi(x(t))$에 Itô's formula를 적용하면:

$$d\phi = \nabla_x \phi \cdot dx + \frac{1}{2}g^2(t)\,\Delta\phi\,dt$$

$dx$에 SDE를 대입하면:

$$d\phi = \nabla_x \phi \cdot f\,dt + \nabla_x \phi \cdot g\,dw + \frac{1}{2}g^2(t)\,\Delta\phi\,dt$$

기댓값을 취하면 $dw$ 항은 평균이 0이라 사라지고:

$$\frac{d}{dt}\,\mathbb{E}[\phi] = \mathbb{E}\!\left[f \cdot \nabla_x \phi + \frac{1}{2}g^2(t)\,\Delta\phi\right]$$

이걸 분포로 표현하면:

$$\int \phi(x)\,\frac{\partial p_t}{\partial t}\,dx = \int \left[f(x,t)\cdot\nabla_x\phi(x) + \frac{1}{2}g^2(t)\,\Delta\phi(x)\right]p_t(x)\,dx$$

우변을 integration by parts로 정리해서 미분을 $\phi$에서 $p_t$로 옮긴다.

$$\int f \cdot \nabla_x\phi\,p_t\,dx = -\int \phi\,\nabla_x\cdot(f\,p_t)\,dx$$

$$\int \Delta\phi\,p_t\,dx = \int \phi\,\Delta p_t\,dx$$

따라서:

$$\int \phi(x)\,\frac{\partial p_t}{\partial t}\,dx = \int \phi(x)\left[-\nabla_x\cdot(f\,p_t) + \frac{1}{2}g^2(t)\,\Delta p_t\right]dx$$

이것이 임의의 $\phi$에 대해 성립해야 하므로 피적분함수끼리 같아야 한다:

$$\boxed{\frac{\partial p_t(x)}{\partial t} = -\nabla_x \cdot \left[f(x,t)\,p_t(x)\right] + \frac{1}{2}g^2(t)\,\Delta p_t(x)}$$

이것이 **Fokker-Planck equation**이다.

---

## 4. 두 항의 의미

$$\frac{\partial p_t}{\partial t} = \underbrace{-\nabla_x \cdot [f \cdot p_t]}_{\text{advection}} + \underbrace{\frac{1}{2}g^2(t)\,\Delta p_t}_{\text{diffusion}}$$

**Advection 항** $-\nabla_x \cdot [f \cdot p_t]$

Drift $f$가 입자들을 쓸어가는 효과다. $f$가 어느 방향을 가리키면 분포가 그쪽으로 밀린다. $f$가 한 점을 향해 수렴하는 형태라면 분포가 그 점으로 모여든다.

연속 방정식(continuity equation) $\partial_t \rho + \nabla \cdot (\rho v) = 0$과 같은 구조다. 확률 질량도 유체처럼 보존되면서 흐른다.

**Diffusion 항** $\frac{1}{2}g^2(t)\,\Delta p_t$

Noise가 입자들을 퍼뜨리는 효과다. Laplacian $\Delta p_t$는 주변 평균과의 차이를 측정하므로, 밀도가 높은 곳에서 낮은 곳으로 확산된다. $g$가 클수록 더 빠르게 퍼진다.

열 방정식(heat equation) $\partial_t u = \kappa\,\Delta u$와 같은 구조다.

---

## 5. Diffusion model에서의 해석

VP-SDE는 다음과 같이 정의된다:

$$dx = -\frac{1}{2}\beta(t)x\,dt + \sqrt{\beta(t)}\,dw$$

$f(x,t) = -\frac{1}{2}\beta(t)x$, $g(t) = \sqrt{\beta(t)}$를 Fokker-Planck equation에 대입하면:

$$\frac{\partial p_t}{\partial t} = \frac{1}{2}\beta(t)\,\nabla_x \cdot [x\,p_t] + \frac{1}{2}\beta(t)\,\Delta p_t$$

두 항의 역할이 명확하게 보인다.

- **Advection 항** $\frac{1}{2}\beta(t)\,\nabla_x \cdot [x\,p_t]$: $f = -\frac{1}{2}\beta(t)x$가 원점을 향하므로, 분포를 원점으로 **수축**시킨다.
- **Diffusion 항** $\frac{1}{2}\beta(t)\,\Delta p_t$: 분포를 **퍼뜨린다**.

수축과 확산이 균형을 이루면서, 어떤 초기 분포 $p_0$에서 출발하더라도 $t \to \infty$에서 $p_t \to \mathcal{N}(0, I)$로 수렴하게 된다. Forward process가 데이터를 Gaussian noise로 만드는 것은 이 두 효과의 합작이다.