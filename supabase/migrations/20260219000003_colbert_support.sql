-- ===========================================
-- arXiv RAG v1 - ColBERT Support Migration
-- ===========================================
-- Adds ColBERT token-level embeddings for MaxSim retrieval
-- BGE-M3 produces ColBERT vectors alongside dense/sparse

-- -------------------------------------------
-- 1. Add ColBERT column (JSONB for variable token count)
-- -------------------------------------------
-- Format: {"token_embeddings": [[f1, f2, ...], ...], "token_count": N}
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding_colbert JSONB;

-- -------------------------------------------
-- 2. GIN index for JSONB operations
-- -------------------------------------------
CREATE INDEX IF NOT EXISTS idx_chunks_colbert_gin
ON chunks USING gin (embedding_colbert);

-- -------------------------------------------
-- 3. Comments
-- -------------------------------------------
COMMENT ON COLUMN chunks.embedding_colbert IS 'BGE-M3 ColBERT token embeddings for MaxSim retrieval. Format: {token_embeddings: [[1024-dim], ...], token_count: N}';
