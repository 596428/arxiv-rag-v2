-- ===========================================
-- arXiv RAG v1 - Search Functions for Hybrid Retrieval
-- ===========================================
-- Session 5: RAG Search API
-- Run in Supabase SQL Editor

-- -------------------------------------------
-- 1. Dense Vector Search (BGE-M3 1024 dims)
-- -------------------------------------------
CREATE OR REPLACE FUNCTION match_chunks_dense(
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
        1 - (c.embedding_dense <=> query_embedding) AS similarity
    FROM chunks c
    WHERE c.embedding_dense IS NOT NULL
    ORDER BY c.embedding_dense <=> query_embedding
    LIMIT match_count;
END;
$$;

-- -------------------------------------------
-- 2. Sparse Vector Search (BM25-style)
-- -------------------------------------------
-- Note: This function computes dot product between query sparse vector
-- and stored sparse vectors in JSONB format
CREATE OR REPLACE FUNCTION match_chunks_sparse(
    query_indices int[],
    query_values float[],
    match_count int DEFAULT 20
)
RETURNS TABLE (
    chunk_id TEXT,
    paper_id TEXT,
    content TEXT,
    section_title TEXT,
    chunk_type TEXT,
    metadata JSONB,
    score float
)
LANGUAGE plpgsql
AS $$
DECLARE
    query_sparse JSONB;
    i int;
BEGIN
    -- Build JSONB from arrays for efficient lookup
    query_sparse := '{}'::jsonb;
    FOR i IN 1..array_length(query_indices, 1) LOOP
        query_sparse := query_sparse || jsonb_build_object(
            query_indices[i]::text,
            query_values[i]
        );
    END LOOP;

    RETURN QUERY
    SELECT
        c.chunk_id,
        c.paper_id,
        c.content,
        c.section_title,
        c.chunk_type,
        c.metadata,
        (
            -- Compute sparse dot product
            SELECT COALESCE(SUM(
                (c.embedding_sparse->>key)::float * (query_sparse->>key)::float
            ), 0)
            FROM jsonb_object_keys(c.embedding_sparse) AS key
            WHERE query_sparse ? key
        ) AS score
    FROM chunks c
    WHERE c.embedding_sparse IS NOT NULL
    AND c.embedding_sparse != '{}'::jsonb
    ORDER BY score DESC
    LIMIT match_count;
END;
$$;

-- -------------------------------------------
-- 3. Hybrid Search (Combined Dense + Sparse with RRF)
-- -------------------------------------------
-- Server-side RRF fusion for better performance
CREATE OR REPLACE FUNCTION match_chunks_hybrid(
    query_embedding vector(1024),
    query_indices int[],
    query_values float[],
    match_count int DEFAULT 10,
    dense_weight float DEFAULT 1.0,
    sparse_weight float DEFAULT 1.0,
    rrf_k int DEFAULT 60
)
RETURNS TABLE (
    chunk_id TEXT,
    paper_id TEXT,
    content TEXT,
    section_title TEXT,
    chunk_type TEXT,
    metadata JSONB,
    dense_score float,
    sparse_score float,
    rrf_score float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    WITH dense_results AS (
        SELECT
            d.chunk_id,
            d.paper_id,
            d.content,
            d.section_title,
            d.chunk_type,
            d.metadata,
            d.similarity as dense_score,
            ROW_NUMBER() OVER (ORDER BY d.similarity DESC) as dense_rank
        FROM match_chunks_dense(query_embedding, match_count * 2) d
    ),
    sparse_results AS (
        SELECT
            s.chunk_id,
            s.paper_id,
            s.content,
            s.section_title,
            s.chunk_type,
            s.metadata,
            s.score as sparse_score,
            ROW_NUMBER() OVER (ORDER BY s.score DESC) as sparse_rank
        FROM match_chunks_sparse(query_indices, query_values, match_count * 2) s
    ),
    combined AS (
        SELECT
            COALESCE(d.chunk_id, s.chunk_id) as chunk_id,
            COALESCE(d.paper_id, s.paper_id) as paper_id,
            COALESCE(d.content, s.content) as content,
            COALESCE(d.section_title, s.section_title) as section_title,
            COALESCE(d.chunk_type, s.chunk_type) as chunk_type,
            COALESCE(d.metadata, s.metadata) as metadata,
            d.dense_score,
            s.sparse_score,
            -- RRF formula: sum(weight / (k + rank))
            COALESCE(dense_weight / (rrf_k + d.dense_rank), 0) +
            COALESCE(sparse_weight / (rrf_k + s.sparse_rank), 0) as rrf_score
        FROM dense_results d
        FULL OUTER JOIN sparse_results s ON d.chunk_id = s.chunk_id
    )
    SELECT
        c.chunk_id,
        c.paper_id,
        c.content,
        c.section_title,
        c.chunk_type,
        c.metadata,
        c.dense_score,
        c.sparse_score,
        c.rrf_score
    FROM combined c
    ORDER BY c.rrf_score DESC
    LIMIT match_count;
END;
$$;

-- -------------------------------------------
-- 4. Get Paper with Chunks
-- -------------------------------------------
CREATE OR REPLACE FUNCTION get_paper_with_chunks(
    p_arxiv_id TEXT
)
RETURNS TABLE (
    arxiv_id TEXT,
    title TEXT,
    authors TEXT[],
    abstract TEXT,
    categories TEXT[],
    published_date DATE,
    citation_count INTEGER,
    chunk_id TEXT,
    content TEXT,
    section_title TEXT,
    chunk_type TEXT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        p.arxiv_id,
        p.title,
        p.authors,
        p.abstract,
        p.categories,
        p.published_date,
        p.citation_count,
        c.chunk_id,
        c.content,
        c.section_title,
        c.chunk_type
    FROM papers p
    LEFT JOIN chunks c ON p.arxiv_id = c.paper_id
    WHERE p.arxiv_id = p_arxiv_id
    ORDER BY (c.metadata->>'chunk_index')::int;
END;
$$;

-- -------------------------------------------
-- 5. Full-text Search (Fallback)
-- -------------------------------------------
CREATE OR REPLACE FUNCTION search_chunks_text(
    search_query TEXT,
    match_count int DEFAULT 20
)
RETURNS TABLE (
    chunk_id TEXT,
    paper_id TEXT,
    content TEXT,
    section_title TEXT,
    rank float
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
        ts_rank(to_tsvector('english', c.content), plainto_tsquery('english', search_query)) as rank
    FROM chunks c
    WHERE to_tsvector('english', c.content) @@ plainto_tsquery('english', search_query)
    ORDER BY rank DESC
    LIMIT match_count;
END;
$$;

-- -------------------------------------------
-- Comments
-- -------------------------------------------
COMMENT ON FUNCTION match_chunks_dense IS 'Dense vector search using BGE-M3 embeddings with cosine similarity';
COMMENT ON FUNCTION match_chunks_sparse IS 'Sparse vector search using BGE-M3 lexical weights (BM25-style)';
COMMENT ON FUNCTION match_chunks_hybrid IS 'Hybrid search combining dense and sparse with RRF fusion';
COMMENT ON FUNCTION search_chunks_text IS 'PostgreSQL full-text search fallback';
