# arXiv RAG v1 - Known Issues

## Open Issues

### ISSUE-001: Sparse Search O(n×m) Performance Problem

**Status:** Open
**Priority:** High
**Created:** 2026-02-19
**Category:** Performance

#### Problem

`match_chunks_sparse` 함수가 Full Table Scan + JSONB 연산으로 인해 극도로 느림.

- **현재 성능:** ~2,500ms/query
- **예상 성능:** ~100ms/query (Dense 수준)

#### Root Cause

```sql
-- supabase/migrations/20260213000000_search_functions.sql (lines 84-91)
SELECT COALESCE(SUM(
    (c.embedding_sparse->>key)::float * (query_sparse->>key)::float
), 0)
FROM jsonb_object_keys(c.embedding_sparse) AS key
WHERE query_sparse ? key
```

**복잡도 분석:**
- Full Table Scan: ~30,000 chunks
- Per-Row JSONB 연산: ~128 tokens/chunk
- 총 연산: **~3,840,000 JSON ops/query**
- 인덱스 사용 불가 (JSONB dot product)

#### Comparison

| Search Mode | Index | Complexity | Latency |
|-------------|-------|------------|---------|
| Dense | IVFFlat | O(log n) | 67ms |
| Sparse | None | O(n × m) | 2,500ms |

#### Potential Solutions

1. **pgvector svector 타입** (PostgreSQL 15+ / pgvector 0.6+)
   - Native sparse vector 지원
   - 인덱스 가능

2. **2-Stage Retrieval**
   - Dense로 후보군 필터링 (top 100)
   - Sparse로 재정렬 (100개만 계산)

3. **Vector DB 마이그레이션**
   - Qdrant: Native sparse vector 지원, 1GB 무료
   - Pinecone: Hybrid search 내장, 2GB 무료

4. **GIN Index on JSONB** (부분적 개선)
   ```sql
   CREATE INDEX idx_sparse_gin ON chunks USING GIN (embedding_sparse);
   ```
   - key 존재 여부 체크만 가속, dot product는 여전히 느림

#### Impact

- Benchmark 전체 시간 증가 (Hybrid = Dense + Sparse)
- 실시간 검색에서 사용자 경험 저하
- Sparse 검색 품질이 Dense보다 높음 (MRR 0.772 vs 0.433) - 최적화 가치 있음

#### Related Files

- `supabase/migrations/20260213000000_search_functions.sql`
- `src/rag/retriever.py` (SparseRetriever class)

---

## Closed Issues

(None yet)

---

## Notes

- 이슈 관리 방식 미정 (GitHub Issues, Linear, 또는 이 파일 유지)
- 벤치마크 결과 분석 후 우선순위 재조정 예정
