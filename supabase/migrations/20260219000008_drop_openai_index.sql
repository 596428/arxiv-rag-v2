-- ===========================================
-- arXiv RAG v1 - Drop OpenAI Index for Bulk Update
-- ===========================================
-- Temporarily drop the IVFFlat index to allow fast bulk updates.
-- After embedding is complete, recreate the index with a separate migration.

-- Drop the index
DROP INDEX IF EXISTS idx_chunks_embedding_openai;

-- Confirm index is dropped (this will show in the migration log)
DO $$
BEGIN
    RAISE NOTICE 'OpenAI embedding index dropped for bulk update';
END $$;
