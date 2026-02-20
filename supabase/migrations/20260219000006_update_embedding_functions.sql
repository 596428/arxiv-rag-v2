-- ===========================================
-- arXiv RAG v1 - Embedding Update Functions
-- ===========================================
-- Functions for updating embeddings with extended timeout

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

-- Comments
COMMENT ON FUNCTION update_chunk_openai_embedding IS 'Update OpenAI embedding with extended timeout';
COMMENT ON FUNCTION update_chunk_colbert_embedding IS 'Update ColBERT embedding with extended timeout';
