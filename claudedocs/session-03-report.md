# Session 3 작업 리포트

**프로젝트**: arXiv RAG v1
**세션**: Session 3 - Document Parsing Module
**완료일**: 2026-02-13
**상태**: ✅ 완료

---

## 1. 작업 목표

PLAN.md v2.2 기준 Session 3의 목표:
- LaTeX 소스 파싱 (우선)
- Marker 기반 PDF 파싱 (보조)
- 수식/그림/테이블 추출
- 노이즈 섹션 필터링
- 텍스트 품질 검증

---

## 2. 완료된 작업

### 2.1 구현된 모듈

```
src/parsing/
├── __init__.py              # ✅ 모듈 exports
├── models.py                # ✅ ParsedDocument, Section, Equation, Figure, Table
├── latex_parser.py          # ✅ LaTeX 소스 파싱 (pylatexenc)
├── marker_parser.py         # ✅ Marker PDF 파싱 (GPU fallback)
├── section_filter.py        # ✅ 노이즈 섹션 필터링
├── latex_cleaner.py         # ✅ LaTeX 명령어 정리
├── quality_checker.py       # ✅ 텍스트 품질 검증
├── equation_processor.py    # ✅ 수식 추출 + Gemini 설명
└── figure_processor.py      # ✅ 이미지/캡션 추출

scripts/
└── 02_parse.py              # ✅ 파싱 파이프라인 CLI
```

### 2.2 모듈별 상세

#### models.py
| 클래스 | 설명 |
|--------|------|
| `ParsedDocument` | 파싱된 문서 전체 구조 |
| `Section` | 섹션 (제목, 문단, 하위섹션) |
| `Paragraph` | 문단 단위 텍스트 |
| `Equation` | 수식 (LaTeX + 텍스트 설명) |
| `Figure` | 그림 (이미지 경로 + 캡션) |
| `Table` | 테이블 (마크다운 변환) |

#### latex_parser.py
- **중첩 브래킷 처리**: title 추출 시 balanced brace 알고리즘 사용
- **섹션 파싱**: `\section`, `\subsection`, `\subsubsection` 계층 구조
- **수식 추출**: equation, align, gather, multline 환경
- **그림/테이블 추출**: figure, table 환경

#### latex_cleaner.py
- **STRIP_COMMANDS_KEEP_CONTENT**: `\textbf`, `\raisebox`, `\scalebox` 등
- **REMOVE_COMMANDS_WITH_CONTENT**: `\cite`, `\ref`, `\includegraphics` 등
- **clean_paper_title()**: 논문 제목 전용 정리 (주석 제거, 줄바꿈 정규화)

#### section_filter.py
- **제외 섹션**: References, Acknowledgments, Author Contributions 등
- **EXCLUDED_SECTION_PATTERNS**: 25개 정규식 패턴
- **섹션 중요도 가중치**: Introduction(0.9), Methods(0.85), Results(0.85) 등

#### quality_checker.py
- **인코딩 검사**: `\ufffd` (대체 문자), NULL 문자
- **알파벳 비율 검사**: 50% 미만 시 경고
- **미변환 LaTeX 검사**: 5% 초과 시 경고

---

## 3. 테스트 결과

### 3.1 모듈 Import 테스트
```
✅ All parsing module imports successful
```

### 3.2 실제 LLM 논문 파싱 테스트

| 논문 | 인용수 | 섹션 | 수식 | 그림 | Title 추출 |
|------|--------|------|------|------|------------|
| DeepSeek-R1 | 5,471 | 15 | 8 | 2 | ✅ |
| Kimi k1.5 | 758 | 35 | 7 | 11 | ✅ |
| s1: test-time scaling | 990 | 32 | 22 | 10 | ⚠️ (DB fallback) |

### 3.3 파싱된 JSON 출력
```
data/parsed/
├── 2501.12948v2.json    # DeepSeek-R1
├── 2501.12599v4.json    # Kimi k1.5
├── 2501.19393v3.json    # s1
└── 2501.01945v2.json    # (테스트용)
```

---

## 4. 이슈 및 해결

| 이슈 | 해결 방법 |
|------|-----------|
| `pylatexenc` 미설치 | `pip install pylatexenc` |
| 중첩 브래킷 title 추출 실패 | balanced brace 알고리즘 구현 |
| `\raisebox`, `\includegraphics` 정리 | STRIP/REMOVE 패턴 추가 |
| LaTeX 주석(`%`) 제목에 포함 | `clean_paper_title()` 함수 추가 |
| ICML 등 학회 템플릿 title 미추출 | `\icmltitle`, `\neuripstitle` 등 8개 패턴 추가 |
| Inline math 미추출 | `inline_math_min_length` 파라미터 추가 (기본값 20자)

---

## 5. 파싱 전략 (구현됨)

```python
def parse_paper(arxiv_id: str) -> ParsedDocument:
    # 1. LaTeX 소스 시도 (우선)
    if latex_path_exists(arxiv_id):
        try:
            return parse_latex(arxiv_id)
        except LatexParseError:
            pass  # Fallback to Marker

    # 2. Marker로 PDF 파싱 (보조)
    return parse_pdf_with_marker(arxiv_id)
```

---

## 6. CLI 사용법

```bash
# 전체 파싱
python scripts/02_parse.py

# 특정 논문 파싱
python scripts/02_parse.py --arxiv-id 2501.12948v2

# LaTeX만 사용
python scripts/02_parse.py --latex-only

# 수식 설명 생성 (Gemini API)
python scripts/02_parse.py --with-equations

# 미리보기
python scripts/02_parse.py --dry-run
```

---

## 7. 현재 상태

```
다운로드 현황 (2026-02-13 10:55 기준):
  PDFs: 395개
  LaTeX: 393개

파싱 현황:
  Parsed: 5개 (테스트)
```

### 테스트 결과 (수정 후)

| 논문 | Title 추출 | Display Eq | Inline Eq | Total Eq |
|------|------------|------------|-----------|----------|
| DeepSeek-R1 | ✅ | 8 | 6 | 14 |
| Kimi k1.5 | ✅ | 7 | 22 | 29 |
| s1 | ✅ | 22 | 489 | 511 |

---

## 8. 다음 세션 (Session 4) 예정 작업

- `src/embedding/chunker.py`: 하이브리드 청킹
- `src/embedding/bge_embedder.py`: BGE-M3 임베더
- `src/embedding/openai_embedder.py`: OpenAI 임베더 (비교군)
- `scripts/03_embed.py`: 임베딩 파이프라인
- Supabase chunks 테이블 저장

---

## 9. 설치된 의존성

```bash
pip install pylatexenc
# marker-pdf는 requirements.txt에 이미 포함
```

---

*작성자: Claude Code*
*Session 3 완료: 2026-02-13 10:45 KST*
*Session 3 추가 수정: 2026-02-13 10:55 KST (Title 패턴 확장, Inline math 추출)*
