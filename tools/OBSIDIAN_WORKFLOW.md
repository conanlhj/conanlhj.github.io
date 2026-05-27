# Obsidian → 블로그 발행 워크플로우

옵시디언에서 글을 쓰고 → 한 줄 명령으로 블로그에 발행하는 흐름.

---

## 1. 한 번만 하는 셋업

### (a) Obsidian Vault에 CSS 스니펫 설치

`tools/obsidian/blog-boxes.css`를 네 Vault의 다음 경로에 복사:

```
<네 Vault>/.obsidian/snippets/blog-boxes.css
```

그 다음 옵시디언에서:

1. Settings → **Appearance** → 맨 아래 **CSS snippets**
2. 새로고침 버튼 → `blog-boxes` 토글 ON

이걸 켜면 옵시디언에서도 `> [!definition] ...` 같은 콜아웃이 블로그와 동일한 색·테두리로 보임 (light/dark 모두 대응).

### (b) Templater 플러그인 + 새 글 템플릿

1. Settings → Community plugins → **Templater** 설치/활성화
2. Templater 설정:
   - **Template folder location**: Vault 안의 템플릿 폴더 (예: `99.템플릿/`)
3. `tools/obsidian/new-blog-post.md`를 위 템플릿 폴더에 복사
4. (선택) Hotkey 지정 → `Templater: Open Insert Template modal` 같은 명령에 단축키

이제 새 노트 만들고 단축키 → 템플릿 선택 → 슬러그/카테고리/제목 입력하면 frontmatter가 자동 채워짐.

### (c) Obsidian 설정 권장값

블로그 호환을 위해 아래 두 가지는 꼭 확인:

| 설정 | 위치 | 값 |
|---|---|---|
| Use [[Wikilinks]] | Files & Links | **OFF** (Jekyll에서 렌더 안 됨) |
| New link format | Files & Links | Relative path to file |
| Default location for new attachments | Files & Links | `attachments/` 같은 고정 폴더 권장 |

> **이미지 위키링크**(`![[image.png]]`)는 발행 스크립트가 알아서 표준 마크다운으로 바꿔주니까, 본문 위키링크(`[[다른 노트]]`)만 끄면 됨.

---

## 2. 매일 쓰는 흐름

```
Vault에서 새 노트 생성
   ↓ (Templater로 frontmatter 자동 채움)
글 작성 (콜아웃, 수식, 이미지 다 옵시디언 그대로)
   ↓
ready 되면 ↓ 한 줄
$ python tools/publish.py <vault의 .md 경로> --push
```

### 발행 명령 옵션

```bash
# 미리보기만 (파일 변경 X)
python tools/publish.py ~/Vault/Blog/SDE/이토적분.md --dry-run

# 파일만 변환해서 _posts에 쓰기 (커밋 X)
python tools/publish.py ~/Vault/Blog/SDE/이토적분.md

# 변환 + git add + commit
python tools/publish.py ~/Vault/Blog/SDE/이토적분.md --commit

# 변환 + commit + push (=즉시 블로그에 반영)
python tools/publish.py ~/Vault/Blog/SDE/이토적분.md --push

# 이미지 처리 건너뛰기 (이전에 한 번 발행한 글 재발행할 때)
python tools/publish.py ~/Vault/Blog/SDE/이토적분.md --no-images --push
```

푸시까지 가면 GitHub Actions(`.github/workflows/jekyll.yml`)가 알아서 빌드·배포해서 1~2분 안에 `blog.shiggy.kr`에 반영됨.

---

## 3. 콜아웃 ↔ 박스 매핑표

옵시디언에서 이렇게 쓰면 → 블로그에서 저 박스가 됨:

| 옵시디언 | 블로그 CSS 클래스 | 색상 |
|---|---|---|
| `> [!definition] 제목` | `{:.definition-box}` | 파랑 |
| `> [!theorem] 제목` | `{:.theorem-box}` | 초록 |
| `> [!lemma] 제목` | `{:.lemma-box}` | 청록 |
| `> [!example] 제목` | `{:.example-box}` | 노랑 |
| `> [!exercise] 제목` | `{:.exercise-box}` | 보라 |
| `> [!problem] 제목` | `{:.problem-box}` | 빨강 |
| `> [!paper]` | `{:.paper-info-box}` | 청색 액센트 |

축약형도 됨: `[!def]`, `[!thm]`, `[!lem]`, `[!ex]`, `[!exr]`, `[!prob]`.

**예시:**
```markdown
> [!exercise] 3.1
> 다음을 증명하라.
> 1. 첫 번째
> 2. 두 번째
```

→ 변환 후:
```markdown
> **3.1**
>
> 다음을 증명하라.
> 1. 첫 번째
> 2. 두 번째
{:.exercise-box}
```

---

## 4. 이미지 처리

옵시디언에서 이미지를 붙여넣으면 (보통 `attachments/Pasted image 20251204.png` 같은 이름으로 저장됨), 발행 시:

1. 자동으로 `assets/postimages/`로 복사
2. 파일명을 `<post-date>-<slug>-<원본명>.ext`로 변경
3. 본문 경로를 `../assets/postimages/...`로 재작성

따라서 옵시디언에서는 `![[image.png]]` 또는 `![alt](attachments/image.png)` 어떤 형태든 OK.

> **Tip**: Vault `Settings → Files & Links → Default location for new attachments`를 "In subfolder under current folder, named: `attachments`"로 두면 글마다 이미지가 분리돼서 깔끔함.

---

## 5. 카테고리 = 폴더 구조

frontmatter의 `categories` 배열이 그대로 `_posts/` 안의 폴더 경로가 됨:

```yaml
categories: [SDE 책, 3장]
```
→ `_posts/SDE 책/3장/<date>-<slug>.md`

```yaml
categories: [논문리뷰]
```
→ `_posts/논문리뷰/<date>-<slug>.md`

옵시디언 Vault 안의 폴더 구조와 무관하게, 어디서 작성하든 frontmatter만 맞으면 올바른 위치로 들어감.

---

## 6. 자주 빠지는 함정

| 증상 | 원인 / 해결 |
|---|---|
| 옵시디언에서 박스 색이 안 보임 | CSS 스니펫 토글 OFF / 옵시디언 재시작 |
| `[!definition]`이 변환 안 됨 | 콜아웃 첫 줄이 정확히 `> [!type]` 형식이어야 함 (대괄호·느낌표 누락 주의) |
| 이미지가 누락됐다고 warn 뜸 | Vault 안에 파일이 정말 있는지 확인. 첨부 폴더가 글 폴더와 멀면 `.obsidian/app.json`의 `attachmentFolderPath` 설정 확인 |
| 카테고리에 한글이 안 먹음 | frontmatter 인용 부호 없이 `[SDE 책, 3장]`처럼 쓰기 (또는 `["SDE 책", "3장"]`) |
| `slug` 필드가 발행본에 남음 | publish.py가 자동 제거함. 안 되면 버그 → 알려줘 |
| 글 수정 후 재발행했더니 이미지가 또 복사됨 | 동일 파일명·크기면 자동 skip. 그래도 싫으면 `--no-images` |

---

## 7. 알려진 자잘한 거

- `_config.yml`의 jekyll-compose 기본값에 `render_with_liqueid: true` 오타가 있음 (정식 키는 `render_with_liquid`). 현재 무시되고 있으나 동작에는 영향 없음. 고치고 싶으면 한 글자 수정.
- Exercise 박스의 `(a)`, `(b)` 자동 번호는 블로그에서는 작동하지만 옵시디언 Live Preview에서는 일부 제한적임 (Reading view에서는 OK).
