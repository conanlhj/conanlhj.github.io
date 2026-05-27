#!/usr/bin/env python3
"""
publish.py — Obsidian .md → Jekyll _posts 변환 + 이미지 이동

사용법:
    python tools/publish.py <vault에 있는 .md 경로>
    python tools/publish.py <...> --dry-run        # 미리보기만
    python tools/publish.py <...> --commit         # 변환 + git add/commit
    python tools/publish.py <...> --push           # 위 + git push

자동으로 처리되는 것:
  1) Obsidian 콜아웃 → Kramdown IAL 박스
       > [!definition] 정의 1.1        →   > **정의 1.1**
       > σ-algebra ...                       >
                                             > σ-algebra ...
                                             {:.definition-box}
  2) 이미지 (![[...]] 또는 ![](...)) → assets/postimages/ 로 복사하고
     경로를 ../assets/postimages/<post-date>-<slug>-<orig-name>.ext 로 교체
  3) Frontmatter의 categories를 폴더 경로로 변환:
       categories: [SDE 책, 3장]  →  _posts/SDE 책/3장/<date>-<slug>.md
  4) 내부용 필드(slug)는 발행본에서 제거
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
POSTS_DIR = REPO_ROOT / "_posts"
IMAGES_DIR = REPO_ROOT / "assets" / "postimages"

# Obsidian 콜아웃 type → 블로그 CSS 클래스
CALLOUT_MAP = {
    "definition": "definition-box",
    "def":        "definition-box",
    "theorem":    "theorem-box",
    "thm":        "theorem-box",
    "lemma":      "lemma-box",
    "lem":        "lemma-box",
    "example":    "example-box",
    "ex":         "example-box",
    "exercise":   "exercise-box",
    "exr":        "exercise-box",
    "problem":    "problem-box",
    "prob":       "problem-box",
    "paper":      "paper-info-box",
}


# --------------------------------------------------------------------------- #
# Vault / 첨부 위치 자동 인식
# --------------------------------------------------------------------------- #
def find_vault_root(start: Path) -> Path | None:
    cur = start.resolve()
    if cur.is_file():
        cur = cur.parent
    while cur != cur.parent:
        if (cur / ".obsidian").is_dir():
            return cur
        cur = cur.parent
    return None


def attachment_search_dirs(vault: Path, source: Path) -> list[Path]:
    """이미지 검색할 디렉토리들 (우선순위 순)."""
    dirs: list[Path] = [source.parent]  # 글이 있는 폴더부터

    # .obsidian/app.json 에서 attachmentFolderPath 읽기
    app_json = vault / ".obsidian" / "app.json"
    if app_json.exists():
        try:
            cfg = json.loads(app_json.read_text(encoding="utf-8"))
            ap = (cfg.get("attachmentFolderPath") or "").strip()
            if ap == "" or ap == "/":
                dirs.append(vault)
            elif ap.startswith("./"):
                dirs.append(source.parent / ap[2:])
            else:
                dirs.append(vault / ap)
        except Exception as e:
            print(f"  [warn] app.json 읽기 실패: {e}", file=sys.stderr)

    dirs.append(vault)  # 마지막 fallback
    # 중복 제거하면서 순서 유지
    seen, uniq = set(), []
    for d in dirs:
        key = str(d.resolve())
        if key not in seen:
            seen.add(key)
            uniq.append(d)
    return uniq


def resolve_image(name: str, search_dirs: list[Path]) -> Path | None:
    name = name.strip()
    p = Path(name)
    if p.is_absolute() and p.exists():
        return p
    for d in search_dirs:
        c = d / name
        if c.exists():
            return c
        c = d / p.name
        if c.exists():
            return c
    # 마지막 수단: vault 전체 재귀 검색
    if search_dirs:
        vault = search_dirs[-1]
        for found in vault.rglob(p.name):
            if found.is_file():
                return found
    return None


# --------------------------------------------------------------------------- #
# Frontmatter 파싱 / 렌더링
# --------------------------------------------------------------------------- #
def split_frontmatter(text: str) -> tuple[list[str], str]:
    """raw frontmatter 라인 리스트와 body를 반환."""
    if not text.startswith("---\n"):
        return [], text
    end = text.find("\n---\n", 4)
    if end == -1:
        # 파일 끝일 수도 있음
        end = text.find("\n---", 4)
        if end == -1:
            return [], text
        body = text[end + 4:]
    else:
        body = text[end + 5:]
    yaml_lines = text[4:end].splitlines()
    return yaml_lines, body


def fm_get(fm_lines: list[str], key: str) -> str:
    """간단한 top-level YAML 값 추출 (list/inline 모두 문자열로)."""
    pat = re.compile(rf"^{re.escape(key)}\s*:\s*(.*)$")
    for line in fm_lines:
        m = pat.match(line)
        if m:
            return m.group(1).strip()
    return ""


def fm_drop(fm_lines: list[str], keys: set[str]) -> list[str]:
    out = []
    skip_block_until_dedent = False
    for line in fm_lines:
        if skip_block_until_dedent:
            if line.startswith(" ") or line.startswith("\t") or line.startswith("-"):
                continue
            skip_block_until_dedent = False
        m = re.match(r"^([A-Za-z_][\w\-]*)\s*:", line)
        if m and m.group(1) in keys:
            # 값이 다음 줄에 인덴트되어 있을 수도 있으니 그것도 함께 건너뜀
            skip_block_until_dedent = True
            continue
        out.append(line)
    return out


def parse_inline_list(raw: str) -> list[str]:
    """'[a, b, c]' 또는 'a' 형식을 리스트로."""
    raw = raw.strip()
    if not raw:
        return []
    if raw.startswith("["):
        inner = raw[1:raw.rfind("]")] if "]" in raw else raw[1:]
        return [s.strip().strip('"').strip("'") for s in inner.split(",") if s.strip()]
    return [raw.strip().strip('"').strip("'")]


# --------------------------------------------------------------------------- #
# 콜아웃 변환
# --------------------------------------------------------------------------- #
CALLOUT_START_RE = re.compile(r"^>\s*\[!([A-Za-z]+)\][-+]?\s*(.*)$")


def convert_callouts(body: str) -> str:
    lines = body.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        m = CALLOUT_START_RE.match(lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue

        ctype = m.group(1).lower()
        title = m.group(2).strip()
        css = CALLOUT_MAP.get(ctype)
        if css is None:
            # 지원 안 하는 콜아웃 → 손대지 않고 그대로
            out.append(lines[i])
            i += 1
            continue

        # blockquote 본문 수집 (이어지는 모든 `>` 줄)
        i += 1
        body_lines: list[str] = []
        while i < len(lines) and lines[i].startswith(">"):
            body_lines.append(lines[i])
            i += 1

        # 출력
        if title:
            out.append(f"> **{title}**")
            if body_lines:
                out.append(">")
        out.extend(body_lines)
        out.append("{:" + f".{css}" + "}")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# 이미지 처리
# --------------------------------------------------------------------------- #
WIKILINK_IMG_RE = re.compile(r"!\[\[([^\]\|]+)(?:\|([^\]]*))?\]\]")
MD_IMG_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
SAFE_NAME_RE = re.compile(r"[^\w가-힣\-.]+")


def sanitize_image_basename(name: str) -> str:
    stem = Path(name).stem
    ext = Path(name).suffix.lower() or ".png"
    stem = SAFE_NAME_RE.sub("-", stem).strip("-") or "image"
    return f"{stem}{ext}"


def process_images(
    body: str,
    vault: Path,
    source: Path,
    post_date: str,
    slug: str,
    dry_run: bool,
) -> tuple[str, list[tuple[Path, Path]]]:
    search_dirs = attachment_search_dirs(vault, source)
    copies: list[tuple[Path, Path]] = []

    def make_dest(src: Path) -> Path:
        clean = sanitize_image_basename(src.name)
        return IMAGES_DIR / f"{post_date}-{slug}-{clean}"

    def queue(src: Path, dst: Path) -> None:
        for s, d in copies:
            if d == dst:
                return  # 중복 방지
        copies.append((src, dst))

    # 1) 위키링크 이미지 ![[name]]
    def repl_wikilink(m: re.Match) -> str:
        name = m.group(1).strip()
        alt = (m.group(2) or "").strip() or Path(name).stem
        src = resolve_image(name, search_dirs)
        if src is None:
            print(f"  [warn] 이미지 못 찾음: {name}", file=sys.stderr)
            return m.group(0)
        dst = make_dest(src)
        queue(src, dst)
        return f"![{alt}](../assets/postimages/{dst.name})"

    body = WIKILINK_IMG_RE.sub(repl_wikilink, body)

    # 2) 표준 마크다운 이미지 ![alt](path)
    def repl_md(m: re.Match) -> str:
        alt, path = m.group(1), m.group(2).strip()
        # 외부/이미 변환된 경로는 그대로
        if path.startswith(("http://", "https://", "data:", "/assets/", "../assets/")):
            return m.group(0)
        src = resolve_image(path, search_dirs)
        if src is None:
            print(f"  [warn] 이미지 못 찾음: {path}", file=sys.stderr)
            return m.group(0)
        dst = make_dest(src)
        queue(src, dst)
        return f"![{alt or Path(src).stem}](../assets/postimages/{dst.name})"

    body = MD_IMG_RE.sub(repl_md, body)

    if not dry_run and copies:
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        for src, dst in copies:
            if dst.exists() and dst.stat().st_size == src.stat().st_size:
                continue  # 이미 같은 크기면 skip
            shutil.copy2(src, dst)
            print(f"  [img] {src.name} → {dst.relative_to(REPO_ROOT)}")
    return body, copies


# --------------------------------------------------------------------------- #
# 메인
# --------------------------------------------------------------------------- #
def determine_target(fm_lines: list[str], source: Path) -> tuple[Path, str, str, list[str]]:
    raw_date = fm_get(fm_lines, "date")
    if raw_date:
        try:
            post_date = datetime.strptime(raw_date.split(" ")[0], "%Y-%m-%d").strftime("%Y-%m-%d")
        except ValueError:
            post_date = datetime.now().strftime("%Y-%m-%d")
    else:
        post_date = datetime.now().strftime("%Y-%m-%d")

    slug = fm_get(fm_lines, "slug").strip().strip('"').strip("'")
    if not slug:
        stem = source.stem
        m = re.match(r"\d{4}-\d{2}-\d{2}-(.+)", stem)
        slug = m.group(1) if m else stem

    cats = parse_inline_list(fm_get(fm_lines, "categories"))
    category_path = "/".join(cats) if cats else ""

    filename = f"{post_date}-{slug}.md"
    target = (POSTS_DIR / category_path / filename) if category_path else (POSTS_DIR / filename)
    return target, post_date, slug, cats


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Obsidian .md → Jekyll _posts publisher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("source", type=Path, help="Vault 안의 .md 경로")
    ap.add_argument("--dry-run", action="store_true", help="파일 안 쓰고 결과만 출력")
    ap.add_argument("--commit", action="store_true", help="git add + commit")
    ap.add_argument("--push", action="store_true", help="git push (--commit 포함)")
    ap.add_argument("--vault", type=Path, default=None, help="Vault 경로 강제 지정 (기본: 자동 감지)")
    ap.add_argument("--no-images", action="store_true", help="이미지 처리 건너뛰기")
    args = ap.parse_args()

    source = args.source.resolve()
    if not source.exists():
        print(f"ERROR: 파일이 없어: {source}", file=sys.stderr)
        sys.exit(1)

    vault = args.vault.resolve() if args.vault else find_vault_root(source)
    if vault is None:
        print(f"[warn] .obsidian/ 못 찾음. {source.parent}를 vault로 간주", file=sys.stderr)
        vault = source.parent

    print(f"Vault : {vault}")
    print(f"Source: {source.relative_to(vault) if vault in source.parents else source}")

    raw = source.read_text(encoding="utf-8")
    fm_lines, body = split_frontmatter(raw)

    target, post_date, slug, cats = determine_target(fm_lines, source)
    print(f"Target: _posts/{('/'.join(cats) + '/') if cats else ''}{target.name}")

    body = convert_callouts(body)
    copies: list[tuple[Path, Path]] = []
    if not args.no_images:
        body, copies = process_images(body, vault, source, post_date, slug, args.dry_run)

    # 내부 전용 필드 제거
    fm_out = fm_drop(fm_lines, {"slug"})
    output = "---\n" + "\n".join(fm_out) + "\n---\n" + body
    if not output.endswith("\n"):
        output += "\n"

    if args.dry_run:
        print("\n----- DRY RUN OUTPUT -----")
        print(output)
        print("--------------------------")
        print(f"\n이미지 {len(copies)}개가 복사될 예정:")
        for s, d in copies:
            print(f"  {s} → {d.relative_to(REPO_ROOT)}")
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(output, encoding="utf-8")
    print(f"[ok] 작성됨: {target.relative_to(REPO_ROOT)}")

    if args.commit or args.push:
        rel = [str(target.relative_to(REPO_ROOT))]
        for _, dst in copies:
            rel.append(str(dst.relative_to(REPO_ROOT)))
        subprocess.run(["git", "-C", str(REPO_ROOT), "add", "--", *rel], check=True)
        title = fm_get(fm_lines, "title").strip().strip('"').strip("'") or slug
        msg = f"post: {title}"
        subprocess.run(["git", "-C", str(REPO_ROOT), "commit", "-m", msg], check=True)
        print(f"[ok] commit: {msg}")
        if args.push:
            subprocess.run(["git", "-C", str(REPO_ROOT), "push"], check=True)
            print("[ok] pushed")


if __name__ == "__main__":
    main()
