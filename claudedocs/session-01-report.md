# Session 1 작업 리포트

**프로젝트**: arXiv RAG v1
**세션**: Session 1 - Project Initialization & Environment Setup
**완료일**: 2026-02-12
**상태**: ✅ 완료

---

## 1. 작업 목표

PLAN.md v2.2 기준 Session 1의 목표:
- 프로젝트 디렉토리 구조 생성
- 의존성 관리 파일 작성 (pyproject.toml, requirements.txt)
- 환경변수 설정 (.env, .env.example)
- Supabase 스키마 설계 및 배포
- 핵심 유틸리티 모듈 구현

---

## 2. 완료된 작업

### 2.1 프로젝트 구조

```
arxiv-rag-v1/
├── src/
│   ├── __init__.py
│   ├── arxiv/           # Session 2에서 구현
│   ├── parser/          # Session 3에서 구현
│   ├── embeddings/      # Session 4에서 구현
│   ├── retrieval/       # Session 5에서 구현
│   ├── generation/      # Session 6에서 구현
│   └── utils/
│       ├── __init__.py
│       ├── config.py    # ✅ 환경설정 관리
│       ├── logging.py   # ✅ 구조화된 로깅
│       └── gemini.py    # ✅ Gemini API 래퍼
├── scripts/
│   └── 00_init_supabase.sql  # ✅ DB 스키마
├── supabase/
│   └── migrations/
│       └── 20260212000000_init_schema.sql  # ✅ 배포됨
├── data/
│   ├── pdfs/            # PDF 저장 위치
│   └── processed/       # 처리된 데이터
├── tests/
├── notebooks/
├── pyproject.toml       # ✅
├── requirements.txt     # ✅
├── .env                 # ✅ (git 제외)
├── .env.example         # ✅
└── .gitignore           # ✅
```

### 2.2 의존성 (pyproject.toml)

| 카테고리 | 패키지 |
|----------|--------|
| arXiv API | arxiv>=2.1.0 |
| Database | supabase>=2.0.0 |
| Config | python-dotenv, pydantic, pydantic-settings |
| PDF Parsing | marker-pdf>=1.0.0, torch>=2.0.0, pylatexenc>=2.10 |
| Embeddings | FlagEmbedding>=1.2.0 |
| LLM | google-generativeai>=0.5.0, openai>=1.0.0 |
| API/UI | fastapi>=0.110.0, streamlit>=1.32.0 |

### 2.3 환경변수 설정

`.env` 파일에 설정된 항목:
- `SUPABASE_URL`: https://wfkectgpoifwbgyjslcl.supabase.co
- `SUPABASE_KEY`: (anon key 설정됨)
- `GEMINI_API_KEY`: (설정됨)
- `GEMINI_MODEL`: gemini-3-flash-preview
- `CHUNK_MAX_TOKENS`: 512
- `EMBEDDING_MODEL`: BAAI/bge-m3
- `LOG_LEVEL`: INFO

### 2.4 Supabase 스키마

**테이블 구조:**

| 테이블 | 설명 | 주요 컬럼 |
|--------|------|-----------|
| `papers` | 논문 메타데이터 | arxiv_id, title, authors, abstract, categories, pdf_path |
| `chunks` | 텍스트 청크 | paper_id, content, chunk_index, section_type, embedding (vector 1024) |
| `equations` | 수식 데이터 | paper_id, latex_source, text_description, context, embedding (vector 1024) |
| `figures` | 그림/테이블 | paper_id, figure_type, caption, file_path, embedding (vector 1024) |

**벡터 검색 함수:**
- `match_chunks(query_embedding, match_threshold, match_count)`: 청크 유사도 검색
- `match_equations(query_embedding, match_threshold, match_count)`: 수식 유사도 검색

**인덱스:**
- `chunks_embedding_idx`: ivfflat (lists=100)
- `equations_embedding_idx`: ivfflat (lists=100)

### 2.5 유틸리티 모듈

**config.py:**
- Pydantic Settings 기반 환경설정 관리
- `.env` 자동 로드
- `has_supabase`, `has_gemini`, `has_openai` 속성 제공

**logging.py:**
- 구조화된 로깅 (`setup_logger()`)
- 배치 작업용 `ProgressLogger` 클래스

**gemini.py:**
- `GeminiClient` 클래스
- `describe_equation(latex)`: 수식을 텍스트 설명으로 변환
- `verify_llm_relevance(title, abstract, keywords)`: 논문 관련성 검증 (2-stage filtering용)
- tenacity 기반 재시도 로직

---

## 3. 검증 결과

### 3.1 Supabase 연결 테스트
```
✅ papers 테이블 접근 성공
✅ chunks 테이블 접근 성공
✅ equations 테이블 접근 성공
✅ figures 테이블 접근 성공
```

### 3.2 Config 로드 테스트
```
✅ Supabase 설정 로드됨
✅ Gemini 설정 로드됨
✅ 기본값 (chunk_max_tokens=512) 적용됨
```

---

## 4. 이슈 및 해결

| 이슈 | 해결 방법 |
|------|-----------|
| `pydantic_settings` 모듈 없음 | `pip install pydantic-settings` 추가 |
| Supabase CLI 미설치 | Homebrew로 설치 (`brew install supabase/tap/supabase`) |
| WSL에서 CLI 인식 안됨 | WSL 재시작으로 PATH 갱신 |

---

## 5. 다음 세션 (Session 2) 예정 작업

- `src/arxiv/client.py`: arXiv API 클라이언트 구현
- `src/arxiv/keywords.py`: 키워드 목록 및 필터링 로직
- 2-stage 검색 구현 (Broad Recall → Relevance Verification)
- PDF 다운로드 및 papers 테이블 적재
- 배치 처리 및 Rate Limiting 구현

---

## 6. 참고 사항

### PLAN.md 버전 이력
- v2.0: 초기 계획 (GPT/Gemini 피드백 반영)
- v2.1: Quality Pipeline, Sparse 최적화, Reranker V1 포함 결정
- v2.2: 키워드 전략 정교화 (Model Families 분리, 2-stage filtering)

### 기술 결정 사항
1. **수식 임베딩**: LaTeX 원문은 source of truth로 저장, Gemini 텍스트 설명을 임베딩
2. **Sparse Vector**: Token ID 포맷 `{12345: 0.8}`, Top-128 필터링, App Layer 계산
3. **Reranker**: BGE-reranker-v2-m3를 V1에 포함 (V1.5 아님)
4. **키워드 필터링**: Hybrid 방식 (rule-based + LLM for edge cases)
