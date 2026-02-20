# arXiv RAG v1

AI/ML 연구 논문 검색 및 질의응답을 위한 RAG(Retrieval-Augmented Generation) 시스템

**[Live Demo](https://596428.github.io/arxiv-rag-v1)**

## Overview

arXiv에서 수집한 AI/ML 분야 논문들을 기반으로 한 검색 및 대화형 질의응답 시스템입니다. 다양한 임베딩 모델(BGE-M3, OpenAI)과 검색 전략(Dense, Sparse, Hybrid)을 벤치마킹하여 최적의 검색 성능을 제공합니다.

## Key Features

- **Multi-Strategy Search**: Dense, Sparse, Hybrid 검색 전략 지원
- **Multi-Model Embeddings**: BGE-M3 (multilingual) 및 OpenAI text-embedding-3-small
- **Interactive Chat**: Gemini 기반 대화형 논문 Q&A
- **Evaluation Dashboard**: 검색 성능 메트릭 시각화 (MRR, NDCG, Precision)
- **PDF Parsing**: marker-pdf를 활용한 고품질 논문 파싱 (수식, 그림 포함)

## Performance Benchmark

1,822개 평가 쿼리 기준 성능:

| Model | MRR | NDCG@10 | Latency (ms) |
|-------|-----|---------|--------------|
| **Sparse (BGE-M3)** | **0.772** | **0.762** | 2,599 |
| OpenAI | 0.757 | 0.756 | **225** |
| Hybrid | 0.614 | 0.681 | 3,578 |
| Dense (BGE-M3) | 0.433 | 0.434 | 440 |

**Key Findings**:
- Sparse 검색이 가장 높은 정확도 달성 (MRR 0.772)
- OpenAI 임베딩은 속도 대비 우수한 성능 (225ms, MRR 0.757)
- 학술 도메인에서 lexical matching의 강점 확인

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   arXiv     │────▶│   Parser    │────▶│  Embedder   │
│  Collector  │     │ (marker-pdf)│     │(BGE-M3/OAAI)│
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                                               ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Gemini    │◀────│  Retriever  │◀────│  Supabase   │
│   (Chat)    │     │ (pgvector)  │     │  (Vector DB)│
└─────────────┘     └─────────────┘     └─────────────┘
```

## Tech Stack

- **Backend**: Python 3.11+, FastAPI, Supabase (PostgreSQL + pgvector)
- **Embedding**: BGE-M3 (FlagEmbedding), OpenAI Embeddings
- **PDF Parsing**: marker-pdf, PyLaTeXenc
- **Frontend**: Tailwind CSS, Chart.js, Vanilla JS
- **AI**: Google Gemini (chat generation)

## Project Structure

```
arxiv-rag-v1/
├── src/
│   ├── collection/     # arXiv 논문 수집
│   ├── parsing/        # PDF/LaTeX 파싱
│   ├── embedding/      # 임베딩 생성 (BGE-M3, OpenAI)
│   ├── storage/        # Supabase 클라이언트
│   ├── rag/            # 검색 및 Reranking
│   └── ui/             # Streamlit UI
├── scripts/            # 파이프라인 스크립트
├── docs/               # GitHub Pages (Dashboard)
├── data/               # 논문 데이터 (PDF, 파싱 결과)
└── tests/              # 테스트 코드
```

## Installation

```bash
# Clone repository
git clone https://github.com/596428/arxiv-rag-v1.git
cd arxiv-rag-v1

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or .venv\Scripts\activate  # Windows

# Install dependencies
pip install -e ".[dev]"

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

## Environment Variables

```bash
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

### 4. Run API Server

```bash
python scripts/04_serve.py
# API available at http://localhost:8000
```

### 5. Launch UI

```bash
python scripts/05_ui.py
# Streamlit UI at http://localhost:8501
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/search` | POST | 논문 검색 (dense/sparse/hybrid) |
| `/chat` | POST | 대화형 Q&A |
| `/papers` | GET | 논문 목록 조회 |
| `/health` | GET | 헬스 체크 |

## Evaluation

```bash
# Generate synthetic benchmark queries
python scripts/08_generate_synthetic_benchmark.py

# Run evaluation
python scripts/06_evaluate.py
```

## Live Demo

- **Dashboard**: [https://596428.github.io/arxiv-rag-v1](https://596428.github.io/arxiv-rag-v1)
- 검색 성능 벤치마크 결과 및 대화형 논문 Q&A 제공

## License

MIT License

## Author

ajh428
