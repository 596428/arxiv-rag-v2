-- ===========================================
-- arXiv RAG v1 - OpenAI Embedding Search Function
-- ===========================================
-- Phase 3: OpenAI comparison experiment
-- Run in Supabase SQL Editor

-- -------------------------------------------
-- 1. Dense Vector Search (OpenAI text-embedding-3-large, 3072 dims)
-- -------------------------------------------
CREATE OR REPLACE FUNCTION match_chunks_openai(
    query_embedding vector(3072),
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
-- 2. Optional: Index for OpenAI embeddings
-- -------------------------------------------
-- Note: Only create after embedding data exists
-- Skip if free tier memory limit is an issue
-- CREATE INDEX IF NOT EXISTS idx_chunks_embedding_openai ON chunks
--     USING ivfflat (embedding_openai vector_cosine_ops) WITH (lists = 20);

-- -------------------------------------------
-- Comments
-- -------------------------------------------
COMMENT ON FUNCTION match_chunks_openai IS 'Dense vector search using OpenAI text-embedding-3-large (3072 dims)';
