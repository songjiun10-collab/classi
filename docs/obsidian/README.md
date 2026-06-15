# Obsidian 연동

Classi와 Obsidian을 잇는 두 가지 경로입니다.

## 1) 기출 분류 결과 → Obsidian 볼트

웹 앱(`frontend/classi_index.html`)에서 **설정 → 데이터 관리 → "Obsidian 내보내기"** 버튼을 누르면
분류된 문항이 `.zip`(노트 묶음)으로 내려받아집니다. 압축을 Obsidian 볼트에 풀면 됩니다.

생성 구조:

```
Classi/
├─ Classi.md            # 루트 MOC (과목 목록)
├─ 과목/{과목}.md        # 과목 허브 → 주제 노트로 링크
├─ 주제/{과목 · 주제}.md  # 주제 허브
└─ 문항/{과목}/…md       # 문항별 노트 (frontmatter·태그·[[위키링크]])
```

대시보드의 지식 그래프가 Obsidian 그래프 뷰에 그대로 재현됩니다.

### 분류할 때마다 자동 저장 (선택)

같은 화면의 **"볼트 연결"** 버튼으로 Obsidian 볼트 폴더를 한 번 지정하면,
이후 **기출 분류가 끝날 때마다** 그 폴더에 노트가 자동으로 써집니다(File System Access API).
폴더 핸들은 브라우저(IndexedDB)에 보존되어 다음 방문에도 유지됩니다.

> **Chrome/Edge 등 Chromium 계열 브라우저 전용**입니다. Firefox·Safari에서는
> 버튼이 동작하지 않으니 위의 "Obsidian 내보내기"(zip)를 사용하세요. 보안 정책상
> 세션에 따라 폴더 접근 권한을 다시 허용해야 할 수 있습니다.

## 2) Claude Code 개발 로그 → Obsidian

`.claude/settings.json`의 **PostToolUse 훅**이 코드가 수정될 때마다
`scripts/hooks/obsidian_devlog.py`를 실행해 데브로그를 남깁니다.

- **(A) 레포 기록(원격 포함)**: `docs/obsidian/dev-log/YYYY-MM-DD.md`에 변경 항목이 쌓입니다.
  버전 파일(`package.json`, `VERSION`, `pyproject.toml`, `CHANGELOG.md` 등) 변경은 🔖 로 표시됩니다.
- **(B) 로컬 볼트 미러(선택)**: 환경변수 `OBSIDIAN_VAULT`에 로컬 Obsidian 볼트 경로를 지정하면
  같은 내용이 `<볼트>/Classi-DevLog/YYYY-MM-DD.md`에도 기록됩니다.

### 로컬 볼트 미러 켜기

```bash
export OBSIDIAN_VAULT="$HOME/Documents/MyVault"
```

위 변수가 없으면 (A) 레포 기록만 동작합니다 — 클라우드/원격 세션에서는 로컬 볼트에
접근할 수 없으므로 레포 기록만 남고, 그 폴더를 git으로 동기화해 볼트에서 열면 됩니다.

> 훅은 `docs/obsidian/`와 `scripts/hooks/` 경로 변경은 기록하지 않습니다(로그 루프 방지).
> 새로 등록한 훅은 `/hooks` 메뉴를 한 번 열거나 세션을 재시작해야 활성화됩니다.
