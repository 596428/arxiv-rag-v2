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
| 수집 기준 | LLM 키워드 필터 → 기간(2025.01~현재) → 인용수/다운로드수 정렬 |
| 수집량 | **1000개** (포트폴리오 적정 규모) |
| PDF 저장 | 로컬 (`./data/pdfs/`) |
| LaTeX 소스 | 다운로드 (`./data/latex/`) |
| 파싱 전략 | **LaTeX 우선 + PDF(Marker) 보조** |
| 임베딩 모델 | **BGE-M3** (primary), OpenAI text-embedding-3-large (비교군) |
| 통합 DB | **Supabase (pgvector)** - 메타데이터 + 벡터 단일화 |
| 검색 로직 | **RRF (Reciprocal Rank Fusion)** - Dense + Sparse 하이브리드 |
| UI | **Streamlit** (검색 데모) |
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

### 2.4 LLM 키워드 리스트 (v2 - 수정됨)

#### Primary Keywords (필수 포함)
- Large Language Model, LLM
- Language Model, Neural Language Model
- RLHF, Reinforcement Learning from Human Feedback
- Fine-tuning, Instruction Tuning
- Pre-trained Language Model, PLM

#### Model Families (오픈소스)
- LLaMA, Llama 2, Llama 3
- Qwen, Qwen2
- Mistral, Mixtral
- DeepSeek, DeepSeek-V2
- Falcon, Yi, Phi, Gemma, OLMo

#### Model Families (빅테크 연구)
- GPT, GPT-4, ChatGPT (OpenAI - 블로그만 공개)
- Gemini, PaLM, Bard (Google - arXiv 공개)
- Claude, Constitutional AI (Anthropic - 방법론 공개)

#### Technique Keywords (방법론)
- Prompt Engineering, Prompting
- RAG, Retrieval-Augmented Generation
- PEFT, LoRA, QLoRA, Adapter
- Quantization, Pruning
- Chain-of-Thought, CoT, Reasoning
- In-Context Learning, ICL
- Alignment, Safety
- Scaling Law
- Mixture of Experts, MoE

#### Refined Keywords (구체화 - 노이즈 방지)
- Self-Attention (NOT just "Attention")
- LLM Agent, Language Agent (NOT just "Agent")
- Text Embedding (NOT just "Embedding")
- Vision-Language Model, VLM (NOT just "Multimodal")
- Tool Use, Function Calling

#### Negative Keywords (Stage 2 필터링용)
- Robot, Robotics
- Game, Gaming
- Reinforcement Learning (단독 사용 시)
- Image Classification
- Object Detection
- Speech Recognition (단독)

### 2.5 논문 선별 알고리즘 (2단계 필터링)

#### Stage 1: Broad Recall (High Recall, Low Precision)
```
입력: arXiv API 쿼리
조건:
  - 카테고리: cs.CL, cs.LG, cs.AI, stat.ML
  - 기간: 2025.01.01 ~ 현재
  - 키워드: Primary + Model Families + Technique (OR 조합)
예상 결과: 5,000-10,000개
```

#### Stage 2: Relevance Verification (하이브리드 필터링)
```
Step 2a: 규칙 기반 필터링
  - 필수 조건: title 또는 abstract에 "language model" 또는 "LLM" 포함
  - Negative 키워드 제외: Robot, Game, Image Classification 등
  - 결과: 명확한 LLM 논문 + 경계선 케이스

Step 2b: LLM 검증 (경계선 케이스만)
  - 대상: Step 2a에서 분류 불확실한 논문
  - 방법: [model_name : gemini-3-flash-preview]로 abstract 분석
  - 프롬프트: "Is this paper primarily about LLM? Yes/No/Uncertain"
  - 비용: ~$0.5 (예상 1,000-2,000개 검증)
```

#### Stage 3: Ranking & Selection
```
정렬 우선순위:
  1. 인용수 (Semantic Scholar citation_count)
  2. 다운로드 수 (arXiv 인기도 지표, 인용수 0인 경우)
  3. 출판일 (최신순)

최종 선별: 상위 1000개
```

---

## 3. 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Data Collection                              │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐             │
│  │ arXiv API    │ → │ Semantic     │ → │ Supabase     │             │
│  │ (metadata)   │   │ Scholar API  │   │ (PostgreSQL) │             │
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
│  │          LaTeX Parser (우선) + Marker (보조)          │           │
│  │  ┌─────────────────────────────────────────────────┐ │           │
│  │  │ LaTeX 소스 있음 → LaTeX 파싱 시도                │ │           │
│  │  │ LaTeX 없음/실패 → Marker(GPU)로 PDF 파싱        │ │           │
│  │  └─────────────────────────────────────────────────┘ │           │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐     │           │
│  │  │ Text    │ │ Equation│ │ Figure  │ │ Table   │     │           │
│  │  │ Sections│ │ (LaTeX) │ │ (Image) │ │ (MD)    │     │           │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘     │           │
│  └──────────────────────────────────────────────────────┘           │
│                              ↓                                       │
│  ┌──────────────────────────────────────────────────────┐           │
│  │              Gemini API (수식 설명만)                 │           │
│  │  - Equation → 텍스트 설명 생성                        │           │
│  │  - Figure → 캡션만 추출 (VLM 미사용)                  │           │
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
│  │ Streamlit UI │ ← │ RRF Hybrid   │ ← │ Reranker     │             │
│  │ + FastAPI    │   │ Retrieval    │   │ (optional)   │             │
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
│   │   ├── latex_parser.py   # LaTeX 소스 파싱 (우선)
│   │   ├── marker_parser.py  # Marker 기반 PDF 파싱 (보조)
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
│   │   ├── supabase.py       # Supabase 클라이언트 (통합 DB)
│   │   └── local.py          # 로컬 파일 저장
│   │
│   ├── rag/                  # RAG 인터페이스
│   │   ├── __init__.py
│   │   ├── retriever.py      # RRF 하이브리드 검색
│   │   ├── reranker.py       # 재순위화 (optional)
│   │   └── api.py            # FastAPI 엔드포인트
│   │
│   ├── ui/                   # UI
│   │   ├── __init__.py
│   │   └── streamlit_app.py  # Streamlit 검색 데모
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
│   ├── 04_serve.py           # API 서버 실행
│   └── 05_ui.py              # Streamlit UI 실행
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
[ ] 기본 config.py, logging.py 구현
```

**의존성 패키지**:
```
# Core
arxiv
supabase
python-dotenv
pydantic
pydantic-settings

# Parsing
marker-pdf
torch
transformers
pylatexenc          # LaTeX 파싱

# Embedding
FlagEmbedding
openai

# API
fastapi
uvicorn

# UI
streamlit

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
    - 카테고리별 검색 (cs.CL, cs.LG, cs.AI, stat.ML)
    - 날짜 범위 필터링 (2025.01~현재)
    - LLM 키워드 필터링 (섹션 2.4 참조)
    - Rate limiting (3초 간격)

[ ] semantic_scholar.py - 인용수/다운로드수 조회
    - arXiv ID → Semantic Scholar 매핑
    - 인용수/영향력 점수 조회
    - 다운로드 수 조회 (2차 정렬용)
    - Batch 처리 + Exponential Backoff

[ ] models.py - Paper 데이터 모델
    - arxiv_id, title, authors, abstract
    - categories, published_date
    - citation_count, download_count
    - pdf_url, latex_url, pdf_path, latex_path
    - status (pending/collected/parsed/embedded)

[ ] downloader.py - 다운로드 매니저
    - PDF 다운로드 (비동기)
    - LaTeX 소스 다운로드
    - 진행 상황 추적
    - 실패 시 재시도

[ ] supabase.py - Supabase 저장 (통합 DB)
    - papers 테이블 CRUD
    - 상태 업데이트
    - 중복 체크

[ ] scripts/01_collect.py 구현
```

**API Rate Limits 참고**:
- arXiv: 3초 간격 권장
- Semantic Scholar: 100 requests/5min (무료) - Exponential Backoff 필수

---

### Session 3: 문서 파싱 모듈 (LaTeX 우선 + Marker 보조)
**예상 소요**: 2 sessions
**선행 작업**: Session 1, 2

```
[ ] latex_parser.py - LaTeX 소스 파싱 (우선)
    - .tar.gz 압축 해제
    - .tex 파일 탐색 및 파싱
    - 섹션/문단/수식/테이블 추출
    - pylatexenc 활용

[ ] marker_parser.py - Marker 기반 PDF 파싱 (보조)
    - GPU 설정 (4060ti)
    - LaTeX 없거나 실패 시 사용
    - PDF → 구조화된 문서 변환
    - 섹션/문단 추출, 수식/테이블/이미지 블록 분리

[ ] equation_processor.py - 수식 처리
    - LaTeX 수식 추출
    - Gemini API로 텍스트 설명 생성 (수식만)
    - 수식 메타데이터 (위치, 컨텍스트)
    - 실패 시 원본 LaTeX만 저장

[ ] figure_processor.py - 이미지 처리
    - 이미지 추출 및 저장
    - 캡션만 추출 (VLM 미사용)

[ ] models.py - 파싱 데이터 모델
    - ParsedDocument
    - Section, Paragraph
    - Equation, Figure, Table

[ ] scripts/02_parse.py 구현

[ ] section_filter.py - Noise Section Filter (신규)
    - EXCLUDED_SECTIONS 정의
    - 섹션 제목 매칭 로직
    - 제외 대상: acknowledgment, references, bibliography,
      author contributions, funding, ethics statement
    - 효과: 청크 20-30% 감소

[ ] latex_cleaner.py - LaTeX Cleanup 규칙 (신규)
    - 미변환 명령어 정리 (\textbf{text} → 내용만)
    - \cite, \ref 제거
    - 특수문자 변환 (\%, \& → %, &)

[ ] quality_checker.py - Text Quality Check (신규)
    - 깨진 인코딩 체크 (\ufffd 대체 문자)
    - 알파벳/숫자 비율 검증 (< 50% 비정상)
    - 연속 특수문자 10개 이상 필터링
```

**파싱 전략 우선순위**:
```python
def parse_paper(arxiv_id: str) -> ParsedDocument:
    # 1. LaTeX 소스 시도
    if latex_path_exists(arxiv_id):
        try:
            return parse_latex(arxiv_id)
        except LatexParseError:
            pass  # Fallback to Marker

    # 2. Marker로 PDF 파싱
    return parse_pdf_with_marker(arxiv_id)
```

**Marker GPU 설정**:
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
    - Sparse Vector Top-128 필터링 (저장 60% 절감)
    - Token ID 형식 저장 ({token_id: weight})
    - Sparsity 메트릭 로깅 (avg tokens/chunk)

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

### Session 5: RAG 검색 API + Streamlit UI
**예상 소요**: 1-2 sessions
**선행 작업**: Session 4

```
[ ] retriever.py - RRF 하이브리드 검색
    - Dense 검색 (코사인 유사도)
    - Sparse 검색 (BM25 스타일)
    - RRF 결합 (Reciprocal Rank Fusion)
    - 필터링 (날짜, 카테고리, 저자)

[ ] reranker.py - 재순위화 (V1 필수)
    - BGE-reranker-v2-m3 사용 (로컬, 비용 $0)
    - Cross-encoder 기반
    - Top-20 → Top-5 재순위화
    - MRR +10-20% 성능 향상

[ ] api.py - FastAPI 엔드포인트
    - POST /search - 검색
    - GET /papers/{arxiv_id} - 논문 상세
    - GET /chunks/{chunk_id} - 청크 조회

[ ] streamlit_app.py - Streamlit UI
    - 검색 입력 폼
    - 결과 표시 (논문 제목, 관련 청크)
    - 필터 옵션 (날짜, 카테고리)
    - 논문 상세 보기

[ ] scripts/04_serve.py 구현
[ ] scripts/05_ui.py 구현
```

**RRF 하이브리드 검색 구현**:
```python
def reciprocal_rank_fusion(
    dense_results: list[SearchResult],
    sparse_results: list[SearchResult],
    k: int = 60
) -> list[tuple[str, float]]:
    """
    RRF Score = Σ 1/(k + rank)
    k=60 is standard default (Cormack et al., 2009)
    """
    scores = {}
    for rank, doc in enumerate(dense_results):
        scores[doc.id] = scores.get(doc.id, 0) + 1/(k + rank + 1)
    for rank, doc in enumerate(sparse_results):
        scores[doc.id] = scores.get(doc.id, 0) + 1/(k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
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

## 6. Supabase 스키마 (통합 DB)

```sql
-- Enable extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Papers metadata (단일 저장소)
CREATE TABLE papers (
    id SERIAL PRIMARY KEY,
    arxiv_id TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    authors TEXT[],
    abstract TEXT,
    categories TEXT[],
    published_date DATE,
    citation_count INTEGER DEFAULT 0,
    download_count INTEGER DEFAULT 0,
    pdf_path TEXT,
    latex_path TEXT,
    parse_status TEXT DEFAULT 'pending',  -- pending/parsed/failed
    parse_method TEXT,  -- latex/marker
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
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

# Supabase (통합 DB)
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

# UI
STREAMLIT_PORT=8501
```

---

## 8. 이미지 처리 (확정)

LLM 논문에서 가치 있는 이미지 유형:

| 유형 | 가치 | RAG 활용도 | V1 처리 방법 |
|------|------|-----------|-------------|
| **아키텍처 다이어그램** | 높음 | 높음 | 캡션만 (V2에서 VLM) |
| 학습 곡선 (Loss/Acc) | 중간 | 낮음 | 캡션만 |
| 벤치마크 비교 차트 | 높음 | 높음 | 캡션만 |
| Attention 히트맵 | 중간 | 중간 | 캡션만 |
| 데이터셋 예시 | 낮음 | 낮음 | 스킵 |

**V1 결정**: 모든 이미지는 캡션만 추출 (VLM 미사용, 비용 절감)

### 8.1 파싱 전략 상세

#### LaTeX 우선 + PDF 보조 전략

| 단계 | 조건 | 처리 |
|------|------|------|
| 1 | LaTeX 소스 존재 | LaTeX 파싱 시도 |
| 2 | LaTeX 파싱 성공 | 수식/구조 추출 완료 |
| 3 | LaTeX 없음 or 실패 | Marker로 PDF 파싱 |
| 4 | 수식 추출 완료 | Gemini API로 텍스트 설명 생성 |

#### Gemini API 사용 정책
- **대상**: 수식(Equation) 블록만
- **이미지**: 캡션만 추출 (VLM 미사용)
- **실패 시**: 원본 LaTeX만 저장 (graceful degradation)
- **일일 한도**: 무료 티어 내 운영 (1500 RPD)

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
- [ ] 1000개 논문 수집 (목표)
- [ ] LaTeX 우선 파싱 + Marker 보조
- [ ] 수식 텍스트 설명 포함
- [ ] BGE-M3 vs OpenAI 비교 결과
- [ ] RRF 하이브리드 검색 구현
- [ ] Streamlit 검색 데모 UI
- [ ] 포트폴리오 문서화

---

## 11. 다음 단계

1. **Session 1 시작**: 프로젝트 초기화 & 환경 설정
2. Supabase 프로젝트 생성 (수동) - 완료
3. 환경 변수 설정

---

*Plan Version: 2.2*
*Last Updated: 2026-02-12*
*변경 이력:*
- *v2.0 (2026-02-11): GPT/Gemini 피드백 반영 - DB 단일화, LaTeX 우선 파싱, RRF 검색, Streamlit UI 추가*
- *v2.1 (2026-02-12): 4가지 기술 질문 분석 반영*
  - *Session 3: Quality Pipeline 추가 (section_filter, latex_cleaner, quality_checker)*
  - *Session 4: Sparse Vector Top-128 필터링, Token ID 저장 명세*
  - *Session 5: Reranker를 V1 필수로 변경 (BGE-reranker-v2-m3)*
  - *수식 저장 전략 확정: LaTeX 원본 보존 + 텍스트 설명 임베딩*
- *v2.2 (2026-02-12): 키워드 전략 및 검색 로직 개선*
  - *Risky 키워드 구체화: Attention→Self-Attention, Agent→LLM Agent, Embedding→Text Embedding*
  - *Model Families 카테고리 분리: 오픈소스(LLaMA, Qwen, Mistral...) + 빅테크(GPT, Gemini, Claude)*
  - *2단계 필터링 전략: Broad Recall → 하이브리드 Relevance Verification*
  - *Negative Keywords 추가: Robot, Game, Image Classification 등*
