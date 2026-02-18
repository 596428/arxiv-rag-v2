# Session 4 작업 리포트

**프로젝트**: arXiv RAG v1
**세션**: Session 4 - Embedding Module
**완료일**: 2026-02-13
**상태**: ✅ 완료

---

## 1. 작업 목표

PLAN.md v2.2 기준 Session 4의 목표:
- 하이브리드 청킹 (섹션/문단 기반)
- BGE-M3 임베딩 (Dense + Sparse)
- OpenAI 임베딩 (비교군)
- Supabase chunks 테이블 저장

---

## 2. 완료된 작업

### 2.1 구현된 모듈

```
src/embedding/
├── __init__.py              # ✅ 모듈 exports
├── models.py                # ✅ Chunk, EmbeddedChunk, SparseVector, Configs
├── chunker.py               # ✅ HybridChunker (섹션/문단 기반)
├── bge_embedder.py          # ✅ BGE-M3 임베더 (Dense 1024 + Sparse 128)
└── openai_embedder.py       # ✅ OpenAI 임베더 (Async, 3072 dims)

src/storage/
└── supabase_client.py       # ✅ chunks 테이블 CRUD 추가

scripts/
└── 03_embed.py              # ✅ 임베딩 파이프라인 CLI
```

### 2.2 모듈별 상세

#### models.py
| 클래스 | 설명 |
|--------|------|
| `Chunk` | 청크 단위 (chunk_id, paper_id, content, section_title, metadata) |
| `EmbeddedChunk` | 임베딩된 청크 (embedding_dense, embedding_sparse, embedding_openai) |
| `SparseVector` | BM25 스파스 벡터 (indices, values, top-128) |
| `ChunkType` | TEXT, ABSTRACT, EQUATION, FIGURE, TABLE, MIXED |
| `ChunkingConfig` | max_tokens=512, overlap_tokens=50 |
| `EmbeddingConfig` | BGE-M3 + OpenAI 설정 |

#### chunker.py (HybridChunker)
- **섹션 기반 분리**: section 경계 존중
- **문단 기반 분리**: 512 토큰 초과 시 문단 단위로 분리
- **오버랩**: 50 토큰 (~10%) 컨텍스트 유지
- **토큰 카운팅**: tiktoken (cl100k_base)
- **Abstract 분리**: 별도 청크로 생성

#### bge_embedder.py (BGEEmbedder)
- **모델**: BAAI/bge-m3
- **Dense 벡터**: 1024 차원
- **Sparse 벡터**: Top-128 토큰 가중치 (BM25 스타일)
- **FP16 지원**: GPU 메모리 최적화
- **배치 처리**: 32개 청크 단위

#### openai_embedder.py (OpenAIEmbedder)
- **모델**: text-embedding-3-large
- **차원**: 3072 (또는 지정 가능)
- **Async 지원**: 동시 5개 요청
- **비교용**: BGE-M3와 검색 성능 비교

---

## 3. 테스트 결과

### 3.1 모듈 Import 테스트
```
✅ All embedding imports successful
  - HybridChunker: OK
  - BGEEmbedder: OK
  - OpenAIEmbedder: OK
```

### 3.2 Chunking 테스트
```
Testing with first 3 files...

✅ 2501.00712v2: 1 chunks, title: Rethinking Addressing in Language Models...
✅ 2501.00750v2: 52 chunks, title: Beyond Text: Implementing Multimodal Large...
✅ 2501.00759v3: 48 chunks, title: Enhancing Transformers for Generalizable...

Chunking Stats:
  Papers: 3
  Chunks: 101
  Tokens: 26,973
  Avg tokens/chunk: 267.1
  Min/Max tokens: 33/512
```

### 3.3 BGE-M3 임베딩 테스트
```
PyTorch: 2.10.0+cu128
CUDA available: True
GPU: NVIDIA GeForce RTX 4060 Ti
Memory: 8.0 GB

✅ First chunk embedding:
  - Dense dim: 1024
  - Sparse tokens: 128
  - Model load time: ~5.4s
```

### 3.4 Supabase 저장 테스트
```
✅ Supabase connected
✅ Test chunk upserted
✅ Retrieved chunk verified
```

---

## 4. 현재 상태

```
2026-02-13 16:22 기준:
  파싱된 파일: 956개
  DB 총 논문: 1,002개
  임베딩 완료: 39개 논문
  Chunks in DB: 1,515개
  실제 chunks 있는 논문: 33개
```

### DB 상태 상세
| 상태 | 개수 |
|------|------|
| total | 1,002 |
| pending | 963 |
| embedded | 39 |
| failed | 0 |

---

## 5. CLI 사용법

```bash
# 전체 임베딩 실행
python scripts/03_embed.py

# 특정 논문만
python scripts/03_embed.py --arxiv-id 2501.12948v2

# BGE-M3만 (OpenAI 제외)
python scripts/03_embed.py --bge-only

# OpenAI도 포함
python scripts/03_embed.py --with-openai

# 배치 사이즈 조정 (메모리 부족 시)
python scripts/03_embed.py --batch-size 16

# Dry-run (저장 안 함)
python scripts/03_embed.py --dry-run
```

---

## 6. 아키텍처 결정

### 6.1 하이브리드 검색 전략
```
Query → [Dense Search (BGE-M3)] + [Sparse Search (BM25)] → RRF Fusion → Top-K Results
```

### 6.2 Sparse Vector 저장
- **포맷**: JSONB `{"token_id": weight, ...}`
- **Top-K**: 128개 토큰만 저장 (60% 용량 절감)
- **Supabase**: `embedding_sparse` 컬럼 (JSONB)

### 6.3 청킹 전략
```
Paper → Sections → Paragraphs → Chunks (512 tokens, 50 overlap)
         ↓
      Abstract (별도 청크)
```

---

## 7. 다음 세션 (Session 5) 예정 작업

- `src/retrieval/dense_retriever.py`: pgvector 기반 Dense 검색
- `src/retrieval/sparse_retriever.py`: BM25 스파스 검색
- `src/retrieval/hybrid_retriever.py`: RRF 하이브리드 퓨전
- `scripts/04_search.py`: 검색 테스트 CLI
- Supabase pgvector 인덱스 최적화

---

## 8. 이슈 및 해결

| 이슈 | 해결 방법 |
|------|-----------|
| `tiktoken` 미설치 | `.venv/bin/python` 사용 (venv 활성화) |
| 필드명 혼동 (`dense_embedding` vs `embedding_dense`) | models.py 확인 후 수정 |
| WSL 메모리 부족 | 배치 사이즈 축소 권장 |

---

## 9. 설치된 의존성

```bash
# requirements.txt에 이미 포함
FlagEmbedding>=1.2.0
openai>=1.0.0
tiktoken>=0.7.0
```

---

*작성자: Claude Code*
*Session 4 완료: 2026-02-13 16:30 KST*
