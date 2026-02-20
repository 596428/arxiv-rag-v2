-- ===========================================
-- arXiv RAG v1 - OpenAI 1024-Dimension Migration
-- ===========================================
-- Fixes: pgvector IVFFlat/HNSW max 2000 dims limitation
-- Solution: Re-embed with OpenAI's MRL sweet spot (1024 dims)

-- -------------------------------------------
-- 1. Drop existing 3072-dim function (CASCADE drops dependencies)
-- -------------------------------------------
DROP FUNCTION IF EXISTS match_chunks_openai CASCADE;
DROP FUNCTION IF EXISTS match_chunks_openai_extended CASCADE;

-- -------------------------------------------
-- 2. Drop old 3072-dim column and recreate as 1024-dim
-- -------------------------------------------
ALTER TABLE chunks DROP COLUMN IF EXISTS embedding_openai;
ALTER TABLE chunks ADD COLUMN embedding_openai vector(1024);

-- -------------------------------------------
-- 3. Create IVFFlat index (now possible with 1024 dims)
-- -------------------------------------------
-- Note: 2000-dim limit was the blocker; 1024 dims works perfectly
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_openai
ON chunks USING ivfflat (embedding_openai vector_cosine_ops)
WITH (lists = 100);

-- -------------------------------------------
-- 4. Recreate search function with 1024-dim signature
-- -------------------------------------------
CREATE OR REPLACE FUNCTION match_chunks_openai(
    query_embedding vector(1024),
    match_count int DEFAULT 20
)
RETURNS TABLE (
    chunk_id TEXT,
    paper_id TEXT,
    content TEXT,
    section_title TEXT,
    chunk_type TEXT,
    metadata JSONB,
    similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        c.chunk_id,
        c.paper_id,
        c.content,
        c.section_title,
        c.chunk_type,
        c.metadata,
        1 - (c.embedding_openai <=> query_embedding) AS similarity
    FROM chunks c
    WHERE c.embedding_openai IS NOT NULL
    ORDER BY c.embedding_openai <=> query_embedding
    LIMIT match_count;
END;
$$;

-- -------------------------------------------
-- 5. Comments
-- -------------------------------------------
COMMENT ON COLUMN chunks.embedding_openai IS 'OpenAI text-embedding-3-large with MRL dimensionality reduction (1024 dims)';
COMMENT ON FUNCTION match_chunks_openai IS 'Dense vector search using OpenAI text-embedding-3-large (1024 dims, MRL reduced)';
