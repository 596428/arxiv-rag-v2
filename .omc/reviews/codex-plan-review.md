OpenAI Codex v0.63.0 (research preview)
--------
workdir: /home/ajh428/projects/arxiv-rag-v1
model: gpt-5.1-codex-max
provider: openai
approval: never
sandbox: read-only
reasoning effort: high
reasoning summaries: auto
session id: 019c4bba-69df-7f92-a652-b1a041bee38a
--------
user
# arXiv RAG v1 Implementation Plan Review Request

## 요청사항
아래 구현 계획서(PLAN.md)를 검토하고, **기획 단계에서 추가적으로 정의해야 할 사항**을 식별해주세요.

특히 다음 관점에서 검토해주세요:
1. **누락된 결정사항**: 구현 시 blocking issue가 될 수 있는 미정의 항목
2. **모호한 요구사항**: 더 명확한 정의가 필요한 부분
3. **기술적 리스크**: 추가 조사나 PoC가 필요한 영역
4. **의존성 문제**: 외부 서비스/라이브러리 관련 고려사항
5. **확장성 고려**: 나중에 변경하기 어려운 설계 결정

한국어로 답변해주세요.

---

## PLAN.md 내용

# arXiv RAG v1 - Implementation Plan

> 작성일: 2026-02-11
> 상태: Planning Complete - Ready for Execution

---

## 1. 프로젝트 개요

### 목표
arXiv에서 2025년 LLM 관련 논문을 수집하여 RAG 시스템 구축 (포트폴리오용)

### 핵심 결정사항

| 항목 | 결정 |
|------|------|
| 수집 기준 | LLM 키워드 필터 → 기간(2025.01~현재) → 인기도/인용수 정렬 |
| 수집량 | POC 기준 조정 (예상: 500~1000개) |
| PDF 저장 | 로컬 (`./data/pdfs/`) |
| LaTeX 소스 | 다운로드 (`./data/latex/`) |
| PDF 파서 | **Marker** (GPU 활용) |
| 임베딩 모델 | **BGE-M3** (primary), OpenAI text-embedding-3-large (비교군) |
| 벡터 DB | Supabase (pgvector) |
| 메타데이터 DB | MongoDB Atlas (기존) |
| 코드베이스 | 완전 신규 작성 |

---

## 2. 조사 결과 요약

### 2.1 arXiv 논문 수량 추정

| 카테고리 | 월간 평균 | 연간 추정 |
|----------|-----------|-----------|
| cs.LG (Machine Learning) | ~4,700+ | ~56,000+ |
| cs.CL (NLP/LLM) | ~2,000+ | ~24,000+ |
| cs.AI (AI) | ~1,200+ | ~14,000+ |
| stat.ML | ~400+ | ~5,000+ |

**결론**: LLM 키워드 필터링 + 인용수/인기도 기반 상위 선별 필수

### 2.2 Marker 기능 확인

```python
# Marker가 추출 가능한 블록 타입
BlockTypes.Table      # 테이블
BlockTypes.Equation   # 수식 (LaTeX)
BlockTypes.Figure     # 이미지/그림
BlockTypes.SectionHeader  # 섹션 헤더
BlockTypes.Form       # 폼

# 출력 형식
- Markdown (LaTeX 수식 $$로 감싸서 포함)
- JSON (구조화된 블록 데이터)
- 이미지 추출 (PIL.Image 딕셔너리)
```

### 2.3 BGE-M3 특성

| 특성 | 값 |
|------|-----|
| 차원 | 1024 (dense) |
| 최대 토큰 | 8192 |
| 검색 방식 | Dense + Sparse + ColBERT (하이브리드) |
| 다국어 | 100+ 언어 지원 |

```python
from FlagEmbedding import BGEM3FlagModel
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True, devices=['cuda:0'])

# Dense + Sparse 동시 추출
output = model.encode(texts, return_dense=True, return_sparse=True)
```

---

## 3. 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Data Collection                              │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐             │
│  │ arXiv API    │ → │ Semantic     │ → │ MongoDB      │             │
│  │ (metadata)   │   │ Scholar API  │   │ (metadata)   │             │
│  │              │   │ (citations)  │   │              │             │
│  └──────────────┘   └──────────────┘   └──────────────┘             │
│         ↓                                                            │
│  ┌──────────────┐   ┌──────────────┐                                │
│  │ PDF Download │ → │ Local Store  │  ./data/pdfs/{arxiv_id}.pdf   │
│  │ LaTeX Source │   │              │  ./data/latex/{arxiv_id}.tar.gz│
│  └──────────────┘   └──────────────┘                                │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                         Document Parsing                             │
│  ┌──────────────────────────────────────────────────────┐           │
│  │                    Marker (GPU)                       │           │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐     │           │
│  │  │ Text    │ │ Equation│ │ Figure  │ │ Table   │     │           │
│  │  │ Sections│ │ (LaTeX) │ │ (Image) │ │ (MD)    │     │           │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘     │           │
│  └──────────────────────────────────────────────────────┘           │
│                              ↓                                       │
│  ┌──────────────────────────────────────────────────────┐           │
│  │              Gemini API (수식/이미지 설명)             │           │
│  │  - Equation → 텍스트 설명 생성                        │           │
│  │  - Figure → 캡션 + 설명 생성 (선택적)                 │           │
│  └──────────────────────────────────────────────────────┘           │
│                              ↓                                       │
│  ┌──────────────┐                                                   │
│  │ Parsed JSON  │  ./data/parsed/{arxiv_id}.json                    │
│  └──────────────┘                                                   │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      Chunking & Embedding                            │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐             │
│  │ Hybrid       │ → │ BGE-M3       │ → │ Supabase     │             │
│  │ Chunking     │   │ (Dense+Sparse)│   │ (pgvector)   │             │
│  │ (섹션+문단)  │   │              │   │              │             │
│  └──────────────┘   └──────────────┘   └──────────────┘             │
│                              +                                       │
│                     ┌──────────────┐                                │
│                     │ OpenAI       │  (비교군)                       │
│                     │ embedding    │                                │
│                     └──────────────┘                                │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                          RAG Interface                               │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐             │
│  │ Search API   │ ← │ Hybrid       │ ← │ Reranker     │             │
│  │ (FastAPI)    │   │ Retrieval    │   │ (optional)   │             │
│  └──────────────┘   └──────────────┘   └──────────────┘             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. 디렉토리 구조

```
arxiv-rag-v1/
├── BLUEPRINT.md              # 아키텍처 설계 (기존)
├── PLAN.md                   # 구현 계획 (본 문서)
├── README.md                 # 프로젝트 소개
├── pyproject.toml            # Python 프로젝트 설정
├── requirements.txt          # 의존성
│
├── src/
│   ├── __init__.py
│   │
│   ├── collection/           # Phase 1: 데이터 수집
│   │   ├── __init__.py
│   │   ├── arxiv_client.py   # arXiv API 클라이언트
│   │   ├── semantic_scholar.py # 인용수 조회
│   │   ├── downloader.py     # PDF/LaTeX 다운로드
│   │   └── models.py         # 데이터 모델 (Paper, etc.)
│   │
│   ├── parsing/              # Phase 2: 문서 파싱
│   │   ├── __init__.py
│   │   ├── marker_parser.py  # Marker 기반 PDF 파싱
│   │   ├── equation_processor.py  # 수식 처리
│   │   ├── figure_processor.py    # 이미지 처리
│   │   └── models.py         # ParsedDocument, Section, etc.
│   │
│   ├── embedding/            # Phase 3: 임베딩
│   │   ├── __init__.py
│   │   ├── chunker.py        # 청킹 로직
│   │   ├── bge_embedder.py   # BGE-M3 임베더
│   │   ├── openai_embedder.py # OpenAI 임베더 (비교군)
│   │   └── models.py         # Chunk, EmbeddedChunk
│   │
│   ├── storage/              # 저장소
│   │   ├── __init__.py
│   │   ├── mongodb.py        # MongoDB 클라이언트
│   │   ├── supabase.py       # Supabase 클라이언트
│   │   └── local.py          # 로컬 파일 저장
│   │
│   ├── rag/                  # RAG 인터페이스
│   │   ├── __init__.py
│   │   ├── retriever.py      # 검색 로직
│   │   ├── reranker.py       # 재순위화 (optional)
│   │   └── api.py            # FastAPI 엔드포인트
│   │
│   └── utils/                # 유틸리티
│       ├── __init__.py
│       ├── config.py         # 설정 관리
│       ├── logging.py        # 로깅
│       └── gemini.py         # Gemini API 래퍼
│
├── scripts/                  # 실행 스크립트
│   ├── 01_collect.py         # 데이터 수집 실행
│   ├── 02_parse.py           # 파싱 실행
│   ├── 03_embed.py           # 임베딩 실행
│   └── 04_serve.py           # API 서버 실행
│
├── data/                     # 데이터 디렉토리 (gitignore)
│   ├── pdfs/                 # PDF 파일
│   ├── latex/                # LaTeX 소스
│   ├── parsed/               # 파싱 결과 JSON
│   ├── figures/              # 추출된 이미지
│   └── cache/                # 캐시
│
├── tests/                    # 테스트
│   ├── test_collection/
│   ├── test_parsing/
│   ├── test_embedding/
│   └── test_rag/
│
└── notebooks/                # 탐색용 노트북
    ├── 01_arxiv_exploration.ipynb
    ├── 02_marker_test.ipynb
    └── 03_embedding_comparison.ipynb
```

---

## 5. 작업 단위 분할 (Sub-Sessions)

### Session 1: 프로젝트 초기화 & 환경 설정
**예상 소요**: 1 session
**선행 작업**: 없음

```
[ ] 프로젝트 디렉토리 구조 생성
[ ] pyproject.toml / requirements.txt 작성
[ ] .gitignore 설정
[ ] 환경변수 템플릿 (.env.example)
[ ] Supabase 프로젝트 생성 및 스키마 설정
[ ] MongoDB 컬렉션 스키마 정의
[ ] 기본 config.py, logging.py 구현
```

**의존성 패키지**:
```
# Core
arxiv
pymongo
supabase
python-dotenv
pydantic
pydantic-settings

# Parsing
marker-pdf
torch
transformers

# Embedding
FlagEmbedding
openai

# API
fastapi
uvicorn

# Utils
httpx
aiofiles
tenacity
```

---

### Session 2: arXiv 데이터 수집 모듈
**예상 소요**: 1-2 sessions
**선행 작업**: Session 1

```
[ ] arxiv_client.py - arXiv API 래퍼
    - 카테고리별 검색 (cs.CL, cs.LG, cs.AI)
    - 날짜 범위 필터링
    - LLM 관련 키워드 필터링
    - Rate limiting (3초 간격)

[ ] semantic_scholar.py - 인용수 조회
    - arXiv ID → Semantic Scholar 매핑
    - 인용수/영향력 점수 조회
    - Batch 처리

[ ] models.py - Paper 데이터 모델
    - arxiv_id, title, authors, abstract
    - categories, published_date
    - citation_count, influence_score
    - pdf_url, latex_url
    - status (pending/collected/parsed/embedded)

[ ] downloader.py - 다운로드 매니저
    - PDF 다운로드 (비동기)
    - LaTeX 소스 다운로드
    - 진행 상황 추적
    - 실패 시 재시도

[ ] mongodb.py - MongoDB 저장
    - papers 컬렉션 CRUD
    - 상태 업데이트
    - 중복 체크

[ ] scripts/01_collect.py 구현
```

**API Rate Limits 참고**:
- arXiv: 3초 간격 권장
- Semantic Scholar: 100 requests/5min (무료)

---

### Session 3: PDF 파싱 모듈 (Marker)
**예상 소요**: 2 sessions
**선행 작업**: Session 1, 2

```
[ ] marker_parser.py - Marker 기반 파싱
    - GPU 설정 (4060ti)
    - PDF → 구조화된 문서 변환
    - 섹션/문단 추출
    - 수식/테이블/이미지 블록 분리

[ ] equation_processor.py - 수식 처리
    - LaTeX 수식 추출
    - Gemini API로 텍스트 설명 생성
    - 수식 메타데이터 (위치, 컨텍스트)

[ ] figure_processor.py - 이미지 처리
    - 이미지 추출 및 저장
    - 캡션 추출
    - (선택) Gemini Vision으로 설명 생성

[ ] models.py - 파싱 데이터 모델
    - ParsedDocument
    - Section, Paragraph
    - Equation, Figure, Table

[ ] scripts/02_parse.py 구현
```

**Marker 설정**:
```python
# GPU 메모리 최적화 (4060ti 16GB)
import torch
torch.cuda.set_per_process_memory_fraction(0.8)

# Marker 설정
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict

models = create_model_dict()
converter = PdfConverter(artifact_dict=models)
```

---

### Session 4: 청킹 & 임베딩 모듈
**예상 소요**: 2 sessions
**선행 작업**: Session 3

```
[ ] chunker.py - 하이브리드 청킹
    - 섹션 기반 1차 분할
    - 긴 섹션 문단 단위 2차 분할
    - 청크 메타데이터 부착
    - 토큰 수 계산 (tiktoken)

[ ] bge_embedder.py - BGE-M3 임베더
    - 모델 로드 (GPU)
    - Dense + Sparse 벡터 생성
    - Batch 처리
    - 메모리 최적화

[ ] openai_embedder.py - OpenAI 임베더
    - text-embedding-3-large
    - 비교 실험용
    - Rate limiting

[ ] supabase.py - Supabase 저장
    - chunks 테이블 CRUD
    - 벡터 저장 (pgvector)
    - Sparse 벡터 저장 (JSONB)

[ ] scripts/03_embed.py 구현
```

**청킹 파라미터**:
```python
CHUNK_CONFIG = {
    "max_tokens": 512,
    "overlap_tokens": 50,  # ~10%
    "min_chunk_tokens": 100,
}
```

---

### Session 5: RAG 검색 API
**예상 소요**: 1-2 sessions
**선행 작업**: Session 4

```
[ ] retriever.py - 검색 로직
    - Dense 검색 (코사인 유사도)
    - Sparse 검색 (BM25 스타일)
    - 하이브리드 검색 (가중 결합)
    - 필터링 (날짜, 카테고리, 저자)

[ ] reranker.py - 재순위화 (optional)
    - Cross-encoder 기반
    - BGE-reranker 또는 Cohere

[ ] api.py - FastAPI 엔드포인트
    - POST /search - 검색
    - GET /papers/{arxiv_id} - 논문 상세
    - GET /chunks/{chunk_id} - 청크 조회

[ ] scripts/04_serve.py 구현
```

---

### Session 6: 테스트 & 평가
**예상 소요**: 1 session
**선행 작업**: Session 5

```
[ ] 단위 테스트 작성
    - test_collection/
    - test_parsing/
    - test_embedding/
    - test_rag/

[ ] 임베딩 품질 비교
    - BGE-M3 vs OpenAI
    - 검색 정확도 측정
    - 응답 시간 비교

[ ] 노트북 정리
    - 탐색 과정 문서화
    - 결과 시각화
```

---

## 6. Supabase 스키마

```sql
-- Enable extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Papers metadata (summary from MongoDB)
CREATE TABLE papers (
    id SERIAL PRIMARY KEY,
    arxiv_id TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    authors TEXT[],
    abstract TEXT,
    categories TEXT[],
    published_date DATE,
    citation_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Text chunks with embeddings
CREATE TABLE chunks (
    id SERIAL PRIMARY KEY,
    chunk_id TEXT UNIQUE NOT NULL,
    paper_id TEXT NOT NULL REFERENCES papers(arxiv_id),
    content TEXT NOT NULL,
    section_title TEXT,
    chunk_type TEXT DEFAULT 'text',  -- text, equation, figure, table
    metadata JSONB,

    -- BGE-M3 embeddings
    embedding_dense vector(1024),
    embedding_sparse JSONB,  -- {token_id: weight, ...}

    -- OpenAI embedding (comparison)
    embedding_openai vector(3072),

    created_at TIMESTAMP DEFAULT NOW()
);

-- Equations
CREATE TABLE equations (
    id SERIAL PRIMARY KEY,
    paper_id TEXT NOT NULL REFERENCES papers(arxiv_id),
    equation_id TEXT UNIQUE NOT NULL,
    latex TEXT NOT NULL,
    text_description TEXT,  -- Gemini 생성
    section_id TEXT,
    embedding vector(1024),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Figures
CREATE TABLE figures (
    id SERIAL PRIMARY KEY,
    paper_id TEXT NOT NULL REFERENCES papers(arxiv_id),
    figure_id TEXT UNIQUE NOT NULL,
    image_path TEXT,
    caption TEXT,
    vlm_description TEXT,  -- Gemini Vision 생성
    embedding vector(1024),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_chunks_paper ON chunks(paper_id);
CREATE INDEX idx_chunks_embedding ON chunks
    USING ivfflat (embedding_dense vector_cosine_ops) WITH (lists = 100);
CREATE INDEX idx_chunks_content_trgm ON chunks
    USING gin (content gin_trgm_ops);
CREATE INDEX idx_papers_categories ON papers USING gin (categories);
CREATE INDEX idx_papers_date ON papers(published_date);
```

---

## 7. 환경 변수

```bash
# .env.example

# MongoDB
MONGODB_URI=mongodb+srv://...
MONGODB_DATABASE=arxiv_rag

# Supabase
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=eyJ...
SUPABASE_SERVICE_KEY=eyJ...  # For admin operations

# APIs
GEMINI_API_KEY_1=...
GEMINI_API_KEY_2=...
OPENAI_API_KEY=...
SEMANTIC_SCHOLAR_API_KEY=...  # Optional

# Paths
DATA_DIR=./data
PDF_DIR=./data/pdfs
LATEX_DIR=./data/latex
PARSED_DIR=./data/parsed
FIGURES_DIR=./data/figures

# Processing
BATCH_SIZE=10
MAX_CONCURRENT_DOWNLOADS=5
CHUNK_MAX_TOKENS=512
```

---

## 8. 이미지 처리 결정 필요

LLM 논문에서 가치 있는 이미지 유형:

| 유형 | 가치 | RAG 활용도 | 처리 방법 |
|------|------|-----------|----------|
| **아키텍처 다이어그램** | 높음 | 높음 | Gemini Vision 설명 |
| 학습 곡선 (Loss/Acc) | 중간 | 낮음 | 캡션만 |
| 벤치마크 비교 차트 | 높음 | 높음 | 테이블 변환 |
| Attention 히트맵 | 중간 | 중간 | Gemini Vision |
| 데이터셋 예시 | 낮음 | 낮음 | 스킵 가능 |

**권장**: MVP에서는 캡션만 추출, V2에서 아키텍처 다이어그램에 VLM 적용

---

## 9. 리스크 & 대응

| 리스크 | 영향 | 대응 |
|--------|------|------|
| arXiv API rate limit | 수집 지연 | 3초 간격 준수, 야간 배치 |
| Marker GPU OOM | 파싱 실패 | 배치 크기 조절, 메모리 관리 |
| Semantic Scholar 제한 | 인용수 누락 | 캐싱, 대체 지표 사용 |
| 대용량 PDF | 파싱 시간 | 페이지 제한, 타임아웃 |
| Supabase 용량 | 비용 증가 | 청크 수 최적화 |

---

## 10. 성공 지표

### MVP 완료 기준
- [ ] 500+ 논문 수집 완료
- [ ] 파싱 성공률 > 90%
- [ ] 검색 응답 시간 < 500ms
- [ ] 기본 검색 API 동작

### V1 완료 기준
- [ ] 1000+ 논문 수집
- [ ] 수식 텍스트 설명 포함
- [ ] BGE-M3 vs OpenAI 비교 결과
- [ ] 하이브리드 검색 구현
- [ ] 포트폴리오 문서화

---

## 11. 다음 단계

1. **Session 1 시작**: 프로젝트 초기화 & 환경 설정
2. Supabase 프로젝트 생성 (수동)
3. 환경 변수 설정

---

*Plan Version: 1.0*
*Last Updated: 2026-02-11*
mcp startup: no servers
warning: Repository snapshot encountered large untracked directories: .venv/lib/python3.12/site-packages/pygments/lexers (259 files), .venv/lib/python3.12/site-packages/pygments/lexers/__pycache__ (259 files). This can slow Codex; consider adding these paths to .gitignore or disabling undo in your config.

thinking
**Reviewing PLAN.md for missing decisions**

I'll analyze the PLAN.md content to identify any missing definitions or decisions across the five specified categories, then summarize these gaps in a concise bullet list referencing the relevant sections and file path.
codex
- API·데이터 수집
  - arXiv 검색 쿼리 스펙 미정(`PLAN.md`): LLM 키워드 목록/부정 키워드, 카테고리 가중치, 정렬 우선순위(인기도 vs 최근), 업데이트 주기/증분 수집 방식.
  - Semantic Scholar 사용 정책/쿼터 초과 시 폴백 전략 부재: 재시도 백오프, 로컬 캐시 키 설계(arxiv_id vs DOI), 키 없이 사용할 때의 제한 검증 필요.
  - 대용량/에러 PDF 처리 정책 미정: 최대 페이지 수, 실패/부분 파싱 시 상태 값, 재시도 횟수·타임아웃 기준.

- 저장소·스키마
  - 메타데이터 이중화: MongoDB vs Supabase `papers` 간 동기화 규칙/권한(쓰기 주체, 원장 DB) 불명확.
  - Supabase `embedding_sparse` JSONB 저장 구조 미정: 토큰 id/weight 포맷, 정규화 여부, 인덱스 전략(trgm? gin jsonb?) 정의 필요.
  - `chunk_type`와 equations/figures 테이블 관계 정의 부족: 별도 테이블을 언제 사용하는지(중복 저장?) 및 정합성 제약.
  - 멀티모델 임베딩 컬럼 null 처리/마이그레이션 정책 필요(OpenAI 없이 BGE만 입력 시 제약?).

- 파싱·처리 파이프라인
  - Marker 출력 포맷 선택 기준(마크다운 vs JSON)과 후처리 규칙 미정: 수식, 표, 그림을 청킹 전에 어떻게 텍스트화할지.
  - 수식/그림에 대한 Gemini 호출 정책: 호출 조건(길이/유형 필터), 실패 시 대체(원본 LaTeX만 저장?), 비용 상한 및 레이트리밋 관리 필요.
  - 이미지 저장/참조 방식 미정: 해상도/포맷, 파일명 규칙, 스토리지 위치(로컬만? CDN 업로드?), 중복 제거.

- 청킹·검색
  - 청킹 파라미터 고정값 외에 섹션/표/수식별 커스텀 규칙 부재: 표/수식은 별도 길이/overlap? 본문-캡션 결합 여부.
  - 하이브리드 검색 가중치, sparse 스코어 계산 방식, BM25 파라미터 미정; 재현 가능한 기본값 필요.
  - 재순위화(reranker) 도입 여부 기준과 모델 선택/리소스 요구사항 미정; 미사용 시 대안(단순 가중 합) 명시 필요.

- 운영·성능·보안
  - GPU/CPU 자원 계획과 배치 단위(동시 파싱/임베딩) 설정 기준 미정; OOM 발생 시 자동 스로틀링 규칙 필요.
  - 캐싱 전략 부재: arXiv 메타/다운로드, Marker 중간 산출물, 임베딩 재계산 방지 키 설계.
  - 비밀키 관리/회전 정책과 로깅 PII/키 마스킹 규칙 미정; API 호출 실패 로깅에 민감정보 포함 여부 점검 필요.
  - 비용 감시: Gemini/OpenAI 호출량 모니터링, 상한선 설정 및 초과 시 graceful degrade 정책 없음.

- 품질·테스트
  - 파싱/임베딩/검색 품질 평가 지표와 골든셋 정의 미정: 정밀·재현 측정 방법, 샘플 크기, 수동 라벨링 절차.
  - 회귀 테스트 전략 부재: 모델 버전 업데이트 시 재임베딩 필요 여부, 마이그레이션 계획/스크립트.
  - 타임라인: 초기 백필(500~1000편) vs 증분 업데이트 스케줄과 재처리 트리거(신규/수정 논문) 미정.

- 확장·변경 용이성
  - 멀티 테넌트/카테고리 확장 시 스키마·파이프라인 변경 계획 미정(예: 다른 연도/도메인 추가).
  - 대체 백엔드(벡터 DB) 스위치 가능성 고려 부족: Supabase 의존 API 추상화 레이어 필요성 검토.
  - 모델 교체(BGE-M3→차세대) 시 임베딩 컬럼 추가/교체 전략과 다운타임 최소화 절차 미정.
tokens used
6,789
- API·데이터 수집
  - arXiv 검색 쿼리 스펙 미정(`PLAN.md`): LLM 키워드 목록/부정 키워드, 카테고리 가중치, 정렬 우선순위(인기도 vs 최근), 업데이트 주기/증분 수집 방식.
  - Semantic Scholar 사용 정책/쿼터 초과 시 폴백 전략 부재: 재시도 백오프, 로컬 캐시 키 설계(arxiv_id vs DOI), 키 없이 사용할 때의 제한 검증 필요.
  - 대용량/에러 PDF 처리 정책 미정: 최대 페이지 수, 실패/부분 파싱 시 상태 값, 재시도 횟수·타임아웃 기준.

- 저장소·스키마
  - 메타데이터 이중화: MongoDB vs Supabase `papers` 간 동기화 규칙/권한(쓰기 주체, 원장 DB) 불명확.
  - Supabase `embedding_sparse` JSONB 저장 구조 미정: 토큰 id/weight 포맷, 정규화 여부, 인덱스 전략(trgm? gin jsonb?) 정의 필요.
  - `chunk_type`와 equations/figures 테이블 관계 정의 부족: 별도 테이블을 언제 사용하는지(중복 저장?) 및 정합성 제약.
  - 멀티모델 임베딩 컬럼 null 처리/마이그레이션 정책 필요(OpenAI 없이 BGE만 입력 시 제약?).

- 파싱·처리 파이프라인
  - Marker 출력 포맷 선택 기준(마크다운 vs JSON)과 후처리 규칙 미정: 수식, 표, 그림을 청킹 전에 어떻게 텍스트화할지.
  - 수식/그림에 대한 Gemini 호출 정책: 호출 조건(길이/유형 필터), 실패 시 대체(원본 LaTeX만 저장?), 비용 상한 및 레이트리밋 관리 필요.
  - 이미지 저장/참조 방식 미정: 해상도/포맷, 파일명 규칙, 스토리지 위치(로컬만? CDN 업로드?), 중복 제거.

- 청킹·검색
  - 청킹 파라미터 고정값 외에 섹션/표/수식별 커스텀 규칙 부재: 표/수식은 별도 길이/overlap? 본문-캡션 결합 여부.
  - 하이브리드 검색 가중치, sparse 스코어 계산 방식, BM25 파라미터 미정; 재현 가능한 기본값 필요.
  - 재순위화(reranker) 도입 여부 기준과 모델 선택/리소스 요구사항 미정; 미사용 시 대안(단순 가중 합) 명시 필요.

- 운영·성능·보안
  - GPU/CPU 자원 계획과 배치 단위(동시 파싱/임베딩) 설정 기준 미정; OOM 발생 시 자동 스로틀링 규칙 필요.
  - 캐싱 전략 부재: arXiv 메타/다운로드, Marker 중간 산출물, 임베딩 재계산 방지 키 설계.
  - 비밀키 관리/회전 정책과 로깅 PII/키 마스킹 규칙 미정; API 호출 실패 로깅에 민감정보 포함 여부 점검 필요.
  - 비용 감시: Gemini/OpenAI 호출량 모니터링, 상한선 설정 및 초과 시 graceful degrade 정책 없음.

- 품질·테스트
  - 파싱/임베딩/검색 품질 평가 지표와 골든셋 정의 미정: 정밀·재현 측정 방법, 샘플 크기, 수동 라벨링 절차.
  - 회귀 테스트 전략 부재: 모델 버전 업데이트 시 재임베딩 필요 여부, 마이그레이션 계획/스크립트.
  - 타임라인: 초기 백필(500~1000편) vs 증분 업데이트 스케줄과 재처리 트리거(신규/수정 논문) 미정.

- 확장·변경 용이성
  - 멀티 테넌트/카테고리 확장 시 스키마·파이프라인 변경 계획 미정(예: 다른 연도/도메인 추가).
  - 대체 백엔드(벡터 DB) 스위치 가능성 고려 부족: Supabase 의존 API 추상화 레이어 필요성 검토.
  - 모델 교체(BGE-M3→차세대) 시 임베딩 컬럼 추가/교체 전략과 다운타임 최소화 절차 미정.
