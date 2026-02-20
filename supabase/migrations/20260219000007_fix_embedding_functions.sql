-- ===========================================
-- arXiv RAG v1 - Fix Embedding Update Functions
-- ===========================================
-- Remove updated_at reference (column doesn't exist)

-- Update OpenAI embedding for a single chunk
CREATE OR REPLACE FUNCTION update_chunk_openai_embedding(
    p_chunk_id TEXT,
    p_embedding vector(1024)
)
RETURNS BOOLEAN
LANGUAGE plpgsql
AS $$
BEGIN
    -- Set extended timeout for this operation
    SET LOCAL statement_timeout = '30s';

    UPDATE chunks
    SET embedding_openai = p_embedding
    WHERE chunk_id = p_chunk_id;

    RETURN FOUND;
END;
$$;

-- Update ColBERT embedding for a single chunk
CREATE OR REPLACE FUNCTION update_chunk_colbert_embedding(
    p_chunk_id TEXT,
    p_embedding JSONB
)
RETURNS BOOLEAN
LANGUAGE plpgsql
AS $$
BEGIN
    -- Set extended timeout for this operation
    SET LOCAL statement_timeout = '30s';

    UPDATE chunks
    SET embedding_colbert = p_embedding
    WHERE chunk_id = p_chunk_id;

    RETURN FOUND;
END;
$$;
