-- ===========================================
-- arXiv RAG v2 - Remove Vector Columns Migration
-- ===========================================
-- This migration removes all vector columns from Supabase tables
-- as vectors will now be stored exclusively in Qdrant.
--
-- Storage strategy:
--   Supabase: Raw data + Metadata (papers, chunks, equations, figures)
--   Qdrant: Vector search (arxiv_chunks collection)
-- ===========================================

-- -------------------------------------------
-- 1. Drop Vector Search Functions
-- -------------------------------------------
-- These functions are no longer needed as search moves to Qdrant

DROP FUNCTION IF EXISTS match_chunks(vector(1024), float, int);
DROP FUNCTION IF EXISTS match_chunks_dense(vector(1024), int);
DROP FUNCTION IF EXISTS match_chunks_sparse(int[], float[], int);
DROP FUNCTION IF EXISTS match_chunks_openai(vector(1024), int);
DROP FUNCTION IF EXISTS match_chunks_openai(vector(3072), int);
DROP FUNCTION IF EXISTS match_chunks_colbert(text, int);
DROP FUNCTION IF EXISTS match_equations(vector(1024), float, int);

-- -------------------------------------------
-- 2. Drop Vector Indexes
-- -------------------------------------------

DROP INDEX IF EXISTS idx_chunks_embedding;
DROP INDEX IF EXISTS idx_chunks_embedding_openai;
DROP INDEX IF EXISTS idx_equations_embedding;
DROP INDEX IF EXISTS idx_figures_embedding;

-- -------------------------------------------
-- 3. Remove Vector Columns from chunks table
-- -------------------------------------------

-- Drop vector columns (v1 schema had 4 embedding columns)
ALTER TABLE chunks DROP COLUMN IF EXISTS embedding_dense;
ALTER TABLE chunks DROP COLUMN IF EXISTS embedding_sparse;
ALTER TABLE chunks DROP COLUMN IF EXISTS embedding_openai;
ALTER TABLE chunks DROP COLUMN IF EXISTS embedding_colbert;

-- -------------------------------------------
-- 4. Remove Vector Columns from equations table
-- -------------------------------------------

ALTER TABLE equations DROP COLUMN IF EXISTS embedding;

-- -------------------------------------------
-- 5. Remove Vector Columns from figures table
-- -------------------------------------------

ALTER TABLE figures DROP COLUMN IF EXISTS embedding;

-- -------------------------------------------
-- 6. Add chunk_id index for Qdrant lookup
-- -------------------------------------------
-- Ensure fast lookup when fetching content by chunk_ids from Qdrant results

CREATE INDEX IF NOT EXISTS idx_chunks_chunk_id ON chunks(chunk_id);

-- -------------------------------------------
-- 7. Add chunk_type column if not exists
-- -------------------------------------------
-- Ensure chunk_type column exists for v2 schema

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'chunks' AND column_name = 'chunk_type'
    ) THEN
        ALTER TABLE chunks ADD COLUMN chunk_type TEXT DEFAULT 'text';
    END IF;
END $$;

-- -------------------------------------------
-- 8. Update table comments for v2
-- -------------------------------------------

COMMENT ON TABLE chunks IS 'v2: Text chunks with metadata only. Vectors stored in Qdrant.';
COMMENT ON TABLE equations IS 'v2: Equation LaTeX and descriptions. Vectors stored in Qdrant.';
COMMENT ON TABLE figures IS 'v2: Figure metadata and captions. Vectors stored in Qdrant.';

-- -------------------------------------------
-- 9. Verification queries (run manually)
-- -------------------------------------------
-- After applying this migration, verify with:
--
-- SELECT column_name FROM information_schema.columns
-- WHERE table_name = 'chunks' AND column_name LIKE 'embedding%';
-- Expected: 0 rows
--
-- SELECT routine_name FROM information_schema.routines
-- WHERE routine_name LIKE 'match_chunks%';
-- Expected: 0 rows
