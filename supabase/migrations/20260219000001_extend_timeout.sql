-- Extend statement timeout for OpenAI search (no index due to 3072 dims > 2000 limit)
-- This function wrapper sets a longer timeout for the search

CREATE OR REPLACE FUNCTION match_chunks_openai_extended(
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
    -- Set timeout to 60 seconds for this query
    SET LOCAL statement_timeout = '60s';
    
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

COMMENT ON FUNCTION match_chunks_openai_extended IS 'OpenAI search with extended timeout (60s) - no index due to 3072 dim limit';
