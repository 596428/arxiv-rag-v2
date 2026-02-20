-- ===========================================
-- arXiv RAG v1 - ColBERT MaxSim Search Function
-- ===========================================
-- Implements ColBERT late-interaction scoring:
-- Score = sum(max(cos_sim(q_token, d_tokens)) for q_token in query)

-- -------------------------------------------
-- 1. Helper: Cosine similarity between two vectors
-- -------------------------------------------
CREATE OR REPLACE FUNCTION cosine_similarity(a float[], b float[])
RETURNS float
LANGUAGE plpgsql
IMMUTABLE
AS $$
DECLARE
    dot_product float := 0;
    norm_a float := 0;
    norm_b float := 0;
    i int;
BEGIN
    IF array_length(a, 1) != array_length(b, 1) THEN
        RETURN 0;
    END IF;

    FOR i IN 1..array_length(a, 1) LOOP
        dot_product := dot_product + (a[i] * b[i]);
        norm_a := norm_a + (a[i] * a[i]);
        norm_b := norm_b + (b[i] * b[i]);
    END LOOP;

    IF norm_a = 0 OR norm_b = 0 THEN
        RETURN 0;
    END IF;

    RETURN dot_product / (sqrt(norm_a) * sqrt(norm_b));
END;
$$;

-- -------------------------------------------
-- 2. ColBERT MaxSim Search Function
-- -------------------------------------------
-- Note: This is a simplified implementation for smaller datasets.
-- For production scale, consider a specialized ColBERT index.
CREATE OR REPLACE FUNCTION match_chunks_colbert(
    query_tokens JSONB,  -- [[float, ...], ...] - query token embeddings
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
DECLARE
    q_token JSONB;
    d_token JSONB;
    q_vec float[];
    d_vec float[];
    max_sim float;
    curr_sim float;
    total_score float;
    rec RECORD;
BEGIN
    -- Create temp table to store scores
    CREATE TEMP TABLE IF NOT EXISTS colbert_scores (
        chunk_id TEXT PRIMARY KEY,
        paper_id TEXT,
        content TEXT,
        section_title TEXT,
        chunk_type TEXT,
        metadata JSONB,
        score float
    ) ON COMMIT DROP;

    -- Clear previous results
    DELETE FROM colbert_scores;

    -- Process each chunk with ColBERT embeddings
    FOR rec IN
        SELECT
            c.chunk_id,
            c.paper_id,
            c.content,
            c.section_title,
            c.chunk_type,
            c.metadata,
            c.embedding_colbert
        FROM chunks c
        WHERE c.embedding_colbert IS NOT NULL
        AND c.embedding_colbert->'token_embeddings' IS NOT NULL
    LOOP
        total_score := 0;

        -- For each query token, find max similarity with any doc token
        FOR q_token IN SELECT * FROM jsonb_array_elements(query_tokens)
        LOOP
            -- Convert query token to float array
            SELECT array_agg(val::float)
            INTO q_vec
            FROM jsonb_array_elements_text(q_token) AS val;

            max_sim := -1;

            -- Find max similarity with any document token
            FOR d_token IN SELECT * FROM jsonb_array_elements(rec.embedding_colbert->'token_embeddings')
            LOOP
                -- Convert doc token to float array
                SELECT array_agg(val::float)
                INTO d_vec
                FROM jsonb_array_elements_text(d_token) AS val;

                -- Compute cosine similarity
                curr_sim := cosine_similarity(q_vec, d_vec);

                IF curr_sim > max_sim THEN
                    max_sim := curr_sim;
                END IF;
            END LOOP;

            -- Add max similarity to total score
            IF max_sim > 0 THEN
                total_score := total_score + max_sim;
            END IF;
        END LOOP;

        -- Insert score
        INSERT INTO colbert_scores VALUES (
            rec.chunk_id,
            rec.paper_id,
            rec.content,
            rec.section_title,
            rec.chunk_type,
            rec.metadata,
            total_score
        );
    END LOOP;

    -- Return top results
    RETURN QUERY
    SELECT
        cs.chunk_id,
        cs.paper_id,
        cs.content,
        cs.section_title,
        cs.chunk_type,
        cs.metadata,
        cs.score AS similarity
    FROM colbert_scores cs
    ORDER BY cs.score DESC
    LIMIT match_count;
END;
$$;

-- -------------------------------------------
-- 3. Comments
-- -------------------------------------------
COMMENT ON FUNCTION cosine_similarity IS 'Compute cosine similarity between two float arrays';
COMMENT ON FUNCTION match_chunks_colbert IS 'ColBERT MaxSim search: sum of max similarities between query and document tokens';
