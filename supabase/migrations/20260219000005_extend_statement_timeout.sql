-- ===========================================
-- arXiv RAG v1 - Extend Statement Timeout
-- ===========================================
-- Extend statement timeout for embedding UPDATE operations

-- Extend default statement timeout to 30 seconds (from default 8s)
ALTER DATABASE postgres SET statement_timeout = '30s';

-- Comment
COMMENT ON DATABASE postgres IS 'arXiv RAG v1 database with extended statement timeout for embedding operations';
