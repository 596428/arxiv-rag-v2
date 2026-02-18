# Session 5 작업 리포트

**프로젝트**: arXiv RAG v1
**세션**: Session 5 - RAG Search API + Streamlit UI
**완료일**: 2026-02-13
**상태**: ✅ 완료

---

## 1. 작업 목표

PLAN.md v2.2 기준 Session 5의 목표:
- RRF 하이브리드 검색 (Dense + Sparse)
- BGE-reranker-v2-m3 재순위화
- FastAPI 검색 엔드포인트
- Streamlit 검색 데모 UI

---

## 2. 완료된 작업

### 2.1 구현된 모듈

```
src/rag/
├── __init__.py              # ✅ 모듈 exports
├── retriever.py             # ✅ RRF 하이브리드 검색 (Dense + Sparse)
├── reranker.py              # ✅ BGE-reranker-v2-m3 재순위화
└── api.py                   # ✅ FastAPI 검색 엔드포인트

src/ui/
├── __init__.py              # ✅ 모듈 exports
└── streamlit_app.py         # ✅ Streamlit 검색 데모 UI

scripts/
├── 04_serve.py              # ✅ FastAPI 서버 실행
└── 05_ui.py                 # ✅ Streamlit UI 실행

supabase/migrations/
└── 20260213000000_search_functions.sql  # ✅ 검색 RPC 함수
```

### 2.2 모듈별 상세

#### retriever.py
| 클래스 | 설명 |
|--------|------|
| `SearchResult` | 검색 결과 (chunk_id, content, scores, metadata) |
| `SearchResponse` | 검색 응답 (results, total, search_time) |
| `DenseRetriever` | Dense 벡터 검색 (pgvector 코사인 유사도) |
| `SparseRetriever` | Sparse 벡터 검색 (BM25 스타일) |
| `HybridRetriever` | RRF 하이브리드 퓨전 (k=60) |

**RRF 공식**:
```python
score = Σ weight / (k + rank)  # k=60 (Cormack et al., 2009)
```

#### reranker.py
| 클래스 | 설명 |
|--------|------|
| `BGEReranker` | Cross-encoder 재순위화 (BAAI/bge-reranker-v2-m3) |
| `LightweightReranker` | 경량 모델 (bge-reranker-base) |

#### api.py (FastAPI)
| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/` | GET | Health check |
| `/search` | POST | 하이브리드 검색 + 리랭킹 |
| `/search?q=` | GET | 검색 (쿼리 파라미터) |
| `/papers/{arxiv_id}` | GET | 논문 상세 조회 |
| `/papers/{arxiv_id}/chunks` | GET | 논문 청크 조회 |
| `/chunks/{chunk_id}` | GET | 청크 상세 조회 |
| `/stats` | GET | DB 통계 |
| `/papers` | GET | 논문 목록 (페이지네이션) |

#### streamlit_app.py
- 검색 입력 폼
- 검색 모드 선택 (hybrid/dense/sparse)
- 리랭킹 옵션
- 결과 표시 (스코어, 섹션, 컨텐츠)
- 예제 쿼리
- DB 통계 사이드바

---

## 3. 테스트 결과

### 3.1 Import 테스트
```
✅ RAG module imports successful
  - SearchResult, SearchResponse, HybridRetriever
  - BGEReranker, LightweightReranker
  - FastAPI app: 13 endpoints
```

### 3.2 End-to-End 검색 테스트
```
Query: "What is RLHF and how does it improve language models?"

✅ Found 10 results in 7.93s
   Search time (internal): 7928ms

Top 3 results:
  1. [2501.01031v3] score=0.016
     "Techniques such as Reinforcement Learning from Human Feedback (RLHF)..."
  2. [2501.01117v1] score=0.016
  3. [2501.01679v1] score=0.016
```

### 3.3 Supabase 검색 함수
```sql
-- Dense 검색
match_chunks_dense(query_embedding, match_count)

-- Sparse 검색
match_chunks_sparse(query_indices, query_values, match_count)

-- 하이브리드 검색 (서버사이드 RRF)
match_chunks_hybrid(query_embedding, query_indices, query_values, ...)
```

---

## 4. 알려진 이슈

### 4.1 Reranker CUDA OOM
- **증상**: BGE-M3 + Reranker 동시 로드 시 GPU 메모리 부족 (8GB)
- **영향**: Reranker 실패 시 원래 결과 반환 (graceful degradation)
- **해결 방안**:
  1. 검색 후 embedder.unload() 호출
  2. LightweightReranker (bge-reranker-base) 사용
  3. Reranker CPU 폴백

### 4.2 Sparse 검색 함수 ✅ 배포 완료
- **상태**: `supabase db push`로 마이그레이션 적용됨
- **함수**: `match_chunks_dense`, `match_chunks_sparse`, `match_chunks_hybrid`
- **테스트 결과**: 모두 정상 작동

---

## 5. CLI 사용법

### API 서버 실행
```bash
# 기본 실행
python scripts/04_serve.py

# 포트 변경
python scripts/04_serve.py --port 8000

# 개발 모드 (auto-reload)
python scripts/04_serve.py --reload
```

### Streamlit UI 실행
```bash
# 기본 실행
python scripts/05_ui.py

# 포트 변경
python scripts/05_ui.py --port 8501

# 헤드리스 모드
python scripts/05_ui.py --no-browser
```

---

## 6. 현재 상태

```
2026-02-13 17:42 기준:
  Papers: 1,002
  Embedded: 39
  Chunks: 1,515

검색 성능:
  - Dense 검색: ~7.9s (BGE-M3 로드 포함)
  - Reranker: CUDA OOM (graceful degradation)
```

---

## 7. 아키텍처 요약

```
Query
  ↓
[BGE-M3 Embedding]
  ↓
┌─────────────────┬─────────────────┐
│  Dense Search   │  Sparse Search  │
│  (pgvector)     │  (BM25-style)   │
└────────┬────────┴────────┬────────┘
         ↓                 ↓
      [RRF Fusion]
         ↓
   [BGE Reranker] (optional)
         ↓
      Results
```

---

## 8. 다음 단계 (Session 6)

- [ ] Supabase에 검색 함수 배포 (SQL 실행)
- [ ] 단위 테스트 작성
- [ ] 임베딩 품질 비교 (BGE-M3 vs OpenAI)
- [ ] 검색 성능 측정 (MRR, NDCG)
- [ ] 포트폴리오 문서화

---

## 9. 설치된 의존성

```bash
pip install fastapi uvicorn streamlit
```

---

*작성자: Claude Code*
*Session 5 완료: 2026-02-13 17:45 KST*
