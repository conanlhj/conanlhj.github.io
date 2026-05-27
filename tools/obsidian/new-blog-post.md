<%*
// =========================================================================
// new-blog-post.md — Templater 템플릿 (블로그용 새 글)
// =========================================================================
// 설치:
//   1) 이 파일을 <Vault>의 템플릿 폴더(예: 99.템플릿/)에 복사
//   2) Settings → Community plugins → Templater 활성화
//   3) Templater 설정 → Template folder location: 위 폴더 지정
//   4) (선택) Folder Templates에서 블로그 글 폴더에 이 템플릿 자동 적용
//
// 사용:
//   - 새 노트 생성 후 Cmd/Ctrl+P → "Templater: Replace templates in the active file"
//   - 또는 Templater hotkey 지정 (예: Cmd+T)
// =========================================================================

// 슬러그 입력 받기 (영문/숫자/하이픈만 권장 → URL이 됨)
const slug = await tp.system.prompt("Slug (예: sde-3-2-part2)");

// 카테고리 선택 (블로그의 _posts 폴더 구조와 맞춤)
const category = await tp.system.suggester(
  ["논문리뷰", "기초계산수학", "SDE 책", "기타"],
  ["논문리뷰", "기초계산수학", "SDE 책", "기타"]
);

// 서브카테고리 (SDE 책의 경우만 의미 있음)
let subcategory = "";
if (category === "SDE 책") {
  subcategory = await tp.system.suggester(
    ["1장", "2장", "3장", "참고자료", "(없음)"],
    ["1장", "2장", "3장", "참고자료", ""]
  );
}

const title = await tp.system.prompt("제목");
const date = tp.date.now("YYYY-MM-DD HH:mm:ss +0900");
const fileDate = tp.date.now("YYYY-MM-DD");

// 파일명 변경: YYYY-MM-DD-slug
await tp.file.rename(`${fileDate}-${slug}`);
-%>
---
title: <% title %>
description: 여기에 설명글 추가
author: shiggy
date: <% date %>
categories: [<% category %><% subcategory ? `, ${subcategory}` : "" %>]
tags: [태그, 추가]
pin: false
math: true
mermaid: true
toc: true
comments: true
render_with_liquid: true
# 발행 시 publish.py가 아래 필드를 사용함
slug: <% slug %>
---

<!--
=========================================================================
콜아웃 사용법 (발행 시 자동으로 Kramdown 박스로 변환됨)
=========================================================================

> [!definition] 정의 1.1
> σ-algebra란 다음 세 조건을 만족하는 집합 𝓕이다.

> [!theorem] 정리 2.3
> ...

> [!lemma] 보조정리 2.1
> ...

> [!example] 예시
> ...

> [!exercise] 2.11
> 다음을 증명하라.
> 1. ...
> 2. ...
> 3. ...

> [!problem] 문제
> ...

> [!paper]
> ICLR 2023
> Authors: ...

수식: $E[X]$, $$\int_0^\infty e^{-x^2} dx$$
머메이드:
```mermaid
graph LR
  A --> B
```

이미지: 그냥 붙여넣기 → publish.py가 assets/postimages/로 옮기고 경로 재작성
=========================================================================
-->

## 들어가며

여기서부터 본문을 작성합니다.
