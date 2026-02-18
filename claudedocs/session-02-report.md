# Session 2 작업 리포트

**프로젝트**: arXiv RAG v1
**세션**: Session 2 - arXiv Data Collection Module
**완료일**: 2026-02-12
**상태**: ✅ 완료

---

## 1. 작업 목표

PLAN.md v2.2 기준 Session 2의 목표:
- arXiv API 클라이언트 구현 (키워드 필터링, Rate Limiting)
- Semantic Scholar API 클라이언트 구현 (인용수 조회)
- PDF/LaTeX 다운로드 매니저 구현
- Supabase 저장소 클라이언트 구현
- 수집 파이프라인 스크립트 구현

---

## 2. 완료된 작업

### 2.1 구현된 모듈

```
src/collection/
├── __init__.py          # ✅ 모듈 exports
├── models.py            # ✅ Paper, PaperStatus, CollectionStats
├── arxiv_client.py      # ✅ arXiv API wrapper + 2-stage filtering
├── semantic_scholar.py  # ✅ 인용수 조회 (rate limiting)
└── downloader.py        # ✅ PDF/LaTeX 비동기 다운로드

src/storage/
├── __init__.py          # ✅ 모듈 exports
└── supabase_client.py   # ✅ Papers CRUD + batch operations

scripts/
└── 01_collect.py        # ✅ 전체 수집 파이프라인
```

### 2.2 모듈별 상세

#### models.py
| 클래스 | 설명 |
|--------|------|
| `Paper` | Pydantic 모델, papers 테이블 매핑 |
| `PaperStatus` | Enum: pending/collected/parsed/embedded/failed |
| `ParseMethod` | Enum: latex/marker/none |
| `SearchQuery` | arXiv 검색 쿼리 파라미터 |
| `CollectionStats` | 수집 통계 (stage별 카운트) |

#### arxiv_client.py
- **2단계 필터링** 구현 (PLAN.md 2.5 기준)
  - Stage 1: Broad Recall (LLM 키워드 OR 검색)
  - Stage 2a: Rule-based filtering (Strong LLM indicators)
- **키워드 목록**: PRIMARY_KEYWORDS, MODEL_FAMILIES, TECHNIQUE_KEYWORDS 등
- **Rate Limiting**: 3초 간격 (settings 기반)
- **Retry 로직**: tenacity 기반 exponential backoff

#### semantic_scholar.py
- arXiv ID → Semantic Scholar 매핑
- 인용수/영향력 점수 조회
- **Rate Limiting**: 10 req/min (conservative, 429 방지)
- Batch 처리 + concurrency control

#### downloader.py
- **비동기 다운로드**: httpx + aiofiles
- PDF/LaTeX 동시 다운로드
- **진행 상황 추적**: ProgressLogger 통합
- Resume 지원 (skip_existing 옵션)

#### supabase_client.py
- papers 테이블 CRUD
- batch_insert_papers (upsert 모드)
- 상태별 조회, 통계 조회

### 2.3 Collection Pipeline (01_collect.py)

```
Stage 1: Broad Recall
    ↓ arXiv API 쿼리 (LLM 키워드)
Stage 2a: Rule-based Filtering
    ↓ Strong LLM indicators 체크
Stage 2b: LLM Verification (선택)
    ↓ Gemini로 edge cases 검증
Stage 3: Citation Enrichment & Ranking
    ↓ Semantic Scholar 인용수 조회
    ↓ 인용수 기준 정렬
Download & Save
    ↓ PDF/LaTeX 다운로드
    → Supabase 저장
```

**CLI 옵션**:
```bash
python scripts/01_collect.py --max-results 5000 --target 1000
python scripts/01_collect.py --dry-run          # 미리보기
python scripts/01_collect.py --skip-download    # 다운로드 스킵
python scripts/01_collect.py --max-verify 500   # LLM 검증 제한
```

---

## 3. 테스트 결과

### 3.1 모듈 Import 테스트
```
✅ from src.collection import Paper, ArxivClient, PaperStatus
✅ from src.storage import get_supabase_client
✅ Supabase configured: True
✅ Gemini configured: True
```

### 3.2 arXiv 검색 테스트
```
✅ Found 5 papers (quick_search)
✅ Stage 1: 20 papers fetched
✅ Stage 2a: 13 clearly LLM, 6 edge cases
```

### 3.3 다운로드 테스트
```
✅ 2602.11151v1.pdf (1.07 MB, 7 pages)
✅ 2602.11149v1.pdf (757 KB, 15 pages)
```

### 3.4 Supabase 저장 테스트
```
✅ Batch upserted 2 papers
✅ DB Stats: {total: 2, pending: 2}
```

---

## 4. 설치된 의존성

```bash
pip install arxiv httpx aiofiles tenacity google-generativeai
```

---

## 5. 이슈 및 해결

| 이슈 | 해결 방법 |
|------|-----------|
| `setup_logger` import 오류 | `setup_logging`으로 수정 |
| `ProgressLogger` 파라미터 불일치 | `name`, `log_every` 파라미터로 수정 |
| Semantic Scholar 429 오류 | Rate limit 10 req/min으로 감소 |
| `google.generativeai` deprecated 경고 | V2에서 `google.genai`로 마이그레이션 예정 |

---

## 6. 다음 세션 (Session 3) 예정 작업

- `src/parsing/latex_parser.py`: LaTeX 소스 파싱 (우선)
- `src/parsing/marker_parser.py`: Marker 기반 PDF 파싱 (보조)
- `src/parsing/equation_processor.py`: 수식 처리 + Gemini 설명
- `src/parsing/figure_processor.py`: 이미지/캡션 추출
- `src/parsing/section_filter.py`: Noise Section 필터링
- `src/parsing/quality_checker.py`: Text Quality 검증
- `scripts/02_parse.py`: 파싱 파이프라인

---

## 7. 현재 DB 상태

```
papers 테이블: 2 rows
  - 2602.11151v1: pending
  - 2602.11149v1: pending

다운로드된 파일:
  - data/pdfs/2602.11151v1.pdf (1.07 MB)
  - data/pdfs/2602.11149v1.pdf (757 KB)
```

---

## 8. 참고 사항

### Rate Limiting 설정
- arXiv: 3초 간격 (권장값)
- Semantic Scholar: 6초 간격 (10 req/min, conservative)
- Gemini: tenacity 기반 retry (2-10초 backoff)

### 키워드 필터링 전략
- **Strong LLM Indicators**: language model, llm, gpt, chatgpt, llama, rlhf, prompt, in-context learning
- **Negative Keywords**: robot, game, image classification 등 제외

---

*작성자: Claude Code*
*Session 2 완료: 2026-02-12 16:21 KST*
