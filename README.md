# arXiv RAG v2

AI/ML 연구 논문 검색 및 질의응답을 위한 RAG(Retrieval-Augmented Generation) 시스템

**[Live Demo](https://596428.github.io/arxiv-rag-v2)**

## Overview

arXiv에서 수집한 2,500개 AI/ML 분야 논문을 기반으로 한 검색 및 대화형 질의응답 시스템입니다. 10가지 검색 전략(Dense, Sparse, Hybrid + Reranker 조합)을 9,544개 벤치마크 쿼리로 평가하여 최적의 검색 성능을 검증했습니다.

## Key Features

- **Multi-Strategy Search**: Dense, Sparse, Hybrid 검색 전략 지원
- **Multi-Model Embeddings**: BGE-M3 (1024d) 및 OpenAI text-embedding-3-large (3072d)
- **Reranking**: BGE-reranker-v2-m3 cross-encoder
- **Interactive Chat**: Gemini 기반 대화형 논문 Q&A
- **Evaluation Dashboard**: 검색 성능 메트릭 시각화 (MRR, NDCG, Precision)
- **PDF Parsing**: marker-pdf를 활용한 고품질 논문 파싱

## Performance Benchmark

9,544개 평가 쿼리 기준 성능 (4 Query Styles × 3 Difficulty Levels):

| Model | MRR | NDCG@10 | P@10 | Latency |
|-------|-----|---------|------|---------|
| **Hybrid-3L+Rerank** | **0.857** | **0.853** | 0.813 | 234ms |
| 3-Large+Rerank | 0.846 | 0.842 | 0.801 | 198ms |
| Hybrid-3L | 0.829 | 0.823 | 0.782 | 79ms |
| Hybrid+Rerank | 0.823 | 0.818 | 0.778 | 187ms |
| Sparse+Rerank | 0.816 | 0.810 | 0.770 | 153ms |
| Sparse | 0.772 | 0.762 | 0.721 | 45ms |
| Dense+Rerank | 0.701 | 0.695 | 0.658 | 165ms |
| Dense | 0.621 | 0.630 | 0.528 | 25ms |

**Key Findings**:
- Hybrid-3L+Rerank 조합이 최고 성능 (NDCG@10: 0.853)
- Reranker가 conceptual/hard 쿼리에서 가장 큰 효과 (+62% 향상)
- Sparse 검색만으로도 keyword 쿼리에서 우수한 성능 (MRR: 0.903)
- OpenAI 3-large 임베딩이 BGE-M3 dense보다 유의미하게 우수

## Query Types & Difficulty

**4 Query Styles** (Gemini-3로 생성):
- **Keyword**: 기술 용어 4-7개 (모델명, 기법 포함)
- **Natural Short**: 간단한 질문 6-12 단어
- **Natural Long**: 상세 연구 질문 15-25 단어
- **Conceptual**: 패러프레이즈, 고유명사/약어 제외

**3 Difficulty Levels** (word overlap 기반 자동 분류):
- **Easy**: keyword 스타일 또는 50%+ 단어 중복
- **Medium**: 중간 수준의 단어 중복
- **Hard**: conceptual 스타일 또는 20% 미만 단어 중복

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  GitHub Pages (ajh428.github.io)                            │
│       │                                                     │
│       │ HTTPS                                               │
│       ▼                                                     │
│  ┌─────────────────────────────────────┐                   │
│  │  Cloudflare Tunnel                  │                   │
│  │  (api.acacia.chat)                  │                   │
│  └─────────────────────────────────────┘                   │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────────────────────────────────┐                   │
│  │  FastAPI Backend                    │                   │
│  │  ├── /api/v1/search → Qdrant       │                   │
│  │  ├── /api/v1/chat   → Qdrant+LLM   │                   │
│  │  └── /api/v1/papers → Supabase     │                   │
│  └─────────────────────────────────────┘                   │
│       │                 │                                   │
│       ▼                 ▼                                   │
│    Qdrant            Supabase                              │
│   (Vectors)         (Metadata)                             │
│   - Dense (BGE-M3)  - Paper info                           │
│   - Sparse          - Raw content                          │
│   - 3-Large                                                │
└─────────────────────────────────────────────────────────────┘
```

## Tech Stack

- **Backend**: Python 3.11+, FastAPI
- **Vector DB**: Qdrant (Dense + Sparse vectors)
- **Metadata DB**: Supabase (PostgreSQL)
- **Embedding**: BGE-M3 (FlagEmbedding), OpenAI text-embedding-3-large
- **Reranking**: BGE-reranker-v2-m3
- **PDF Parsing**: marker-pdf
- **Frontend**: Tailwind CSS, Chart.js, Vanilla JS
- **AI**: Google Gemini (chat generation)
- **Deployment**: Cloudflare Tunnel, Docker

## Project Structure

```
arxiv-rag-v1/
├── src/
│   ├── collection/     # arXiv 논문 수집
│   ├── parsing/        # PDF/LaTeX 파싱
│   ├── embedding/      # 임베딩 생성 (BGE-M3, OpenAI)
│   ├── storage/        # Supabase & Qdrant 클라이언트
│   ├── rag/            # 검색, Reranking, HyDE
│   └── api/            # FastAPI 엔드포인트
├── scripts/            # 파이프라인 스크립트
├── docs/               # GitHub Pages (Dashboard)
├── data/               # 논문 데이터, 벤치마크 결과
└── tests/              # 테스트 코드
```

## Data Collection Pipeline

1. **arXiv API 수집** (14개월: 2025.01 ~ 2026.02)
   - Categories: cs.CL, cs.AI, cs.LG, cs.CV, stat.ML, cs.IR, cs.NE
   - Positive Keywords: LLM, transformer, attention, etc.

2. **NG Keyword 필터링** (1,756개 키워드)
   - 9개 카테고리: biomedical, chemistry, earth_science, robotics, etc.

3. **Gemini 분류** (gemini-3-flash-preview)
   - SUITABLE / NOT SUITABLE 분류

4. **Semantic Filtering**
   - Anchor Query와 cosine similarity ≥ 0.55

5. **Multi-Score Ranking**
   - Citation + Recency + Semantic + Stratified

6. **Stratified Sampling** → Final 2,500 papers

## Installation

```bash
# Clone repository
git clone https://github.com/596428/arxiv-rag-v2.git
cd arxiv-rag-v2

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -e ".[dev]"

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

## Environment Variables

```bash
# Qdrant
QDRANT_URL=http://localhost:6333

# Supabase
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key

# AI APIs
GEMINI_API_KEY=your-gemini-api-key
OPENAI_API_KEY=your-openai-api-key
```

## Usage

### 1. Collect Papers
```bash
python scripts/01_collect.py --category cs.CL --max-papers 100
```

### 2. Download & Parse
```bash
python scripts/02_download.py
python scripts/02_parse.py
```

### 3. Generate Embeddings
```bash
python scripts/03_embed.py
```

### 4. Run Evaluation
```bash
# Generate synthetic benchmark queries
python scripts/08_generate_synthetic_benchmark.py

# Run evaluation
python scripts/06_evaluate.py
```

### 5. Run API Server
```bash
uvicorn src.api.main:app --reload
# API available at http://localhost:8000
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/search` | POST | 논문 검색 (dense/sparse/hybrid) |
| `/api/v1/chat` | POST | 대화형 Q&A |
| `/api/v1/papers` | GET | 논문 목록 조회 |
| `/health` | GET | 헬스 체크 |

## Live Demo

- **Dashboard**: [https://596428.github.io/arxiv-rag-v2](https://596428.github.io/arxiv-rag-v2)
- 10가지 모델 검색 성능 벤치마크 비교
- Query Type × Difficulty 분석
- 대화형 논문 Q&A

## License

MIT License

## Author

ajh428
