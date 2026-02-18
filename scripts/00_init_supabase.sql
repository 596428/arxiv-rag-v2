-- ===========================================
-- arXiv RAG v1 - Supabase Schema
-- ===========================================
-- Run this script in Supabase SQL Editor
-- https://supabase.com/dashboard/project/wfkectgpoifwbgyjslcl/sql

-- -------------------------------------------
-- 1. Enable Extensions
-- -------------------------------------------
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- -------------------------------------------
-- 2. Papers Table (메타데이터 + 처리 상태)
-- -------------------------------------------
CREATE TABLE IF NOT EXISTS papers (
    id SERIAL PRIMARY KEY,
    arxiv_id TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    authors TEXT[],
    abstract TEXT,
    categories TEXT[],
    published_date DATE,
    citation_count INTEGER DEFAULT 0,
    download_count INTEGER DEFAULT 0,
    pdf_path TEXT,
    latex_path TEXT,
    parse_status TEXT DEFAULT 'pending',  -- pending/parsed/failed
    parse_method TEXT,  -- latex/marker
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- -------------------------------------------
-- 3. Chunks Table (텍스트 청크 + 임베딩)
-- -------------------------------------------
CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    chunk_id TEXT UNIQUE NOT NULL,
    paper_id TEXT NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    section_title TEXT,
    chunk_type TEXT DEFAULT 'text',  -- text, equation, figure, table
    chunk_index INTEGER,
    token_count INTEGER,
    metadata JSONB,

    -- BGE-M3 embeddings
    embedding_dense vector(1024),
    embedding_sparse JSONB,  -- {token_id: weight, ...} Top-128 filtered

    -- OpenAI embedding (comparison)
    embedding_openai vector(3072),

    created_at TIMESTAMP DEFAULT NOW()
);

-- -------------------------------------------
-- 4. Equations Table (수식)
-- -------------------------------------------
CREATE TABLE IF NOT EXISTS equations (
    id SERIAL PRIMARY KEY,
    paper_id TEXT NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
    equation_id TEXT UNIQUE NOT NULL,
    latex TEXT NOT NULL,  -- Source of Truth (원본 보존)
    text_description TEXT,  -- Gemini 생성 설명 (임베딩 대상)
    section_id TEXT,
    context TEXT,  -- 수식 주변 텍스트
    embedding vector(1024),  -- text_description 임베딩
    created_at TIMESTAMP DEFAULT NOW()
);

-- -------------------------------------------
-- 5. Figures Table (이미지/캡션)
-- -------------------------------------------
CREATE TABLE IF NOT EXISTS figures (
    id SERIAL PRIMARY KEY,
    paper_id TEXT NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
    figure_id TEXT UNIQUE NOT NULL,
    image_path TEXT,
    caption TEXT,
    vlm_description TEXT,  -- V2에서 Gemini Vision 사용 예정
    embedding vector(1024),
    created_at TIMESTAMP DEFAULT NOW()
);

-- -------------------------------------------
-- 6. Indexes
-- -------------------------------------------

-- Papers indexes
CREATE INDEX IF NOT EXISTS idx_papers_categories ON papers USING gin (categories);
CREATE INDEX IF NOT EXISTS idx_papers_date ON papers(published_date);
CREATE INDEX IF NOT EXISTS idx_papers_status ON papers(parse_status);
CREATE INDEX IF NOT EXISTS idx_papers_citation ON papers(citation_count DESC);

-- Chunks indexes
CREATE INDEX IF NOT EXISTS idx_chunks_paper ON chunks(paper_id);
CREATE INDEX IF NOT EXISTS idx_chunks_type ON chunks(chunk_type);
CREATE INDEX IF NOT EXISTS idx_chunks_content_trgm ON chunks USING gin (content gin_trgm_ops);

-- Vector indexes (IVFFlat for approximate nearest neighbor)
-- Note: Create after inserting data for better performance
-- Recommended: lists = sqrt(total_rows), e.g., 100 for 10,000 chunks
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks
    USING ivfflat (embedding_dense vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_equations_embedding ON equations
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 50);

-- -------------------------------------------
-- 7. Functions
-- -------------------------------------------

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger for papers table
DROP TRIGGER IF EXISTS update_papers_updated_at ON papers;
CREATE TRIGGER update_papers_updated_at
    BEFORE UPDATE ON papers
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- -------------------------------------------
-- 8. Search Functions
-- -------------------------------------------

-- Dense vector similarity search
CREATE OR REPLACE FUNCTION match_chunks(
    query_embedding vector(1024),
    match_threshold float DEFAULT 0.5,
    match_count int DEFAULT 10
)
RETURNS TABLE (
    chunk_id TEXT,
    paper_id TEXT,
    content TEXT,
    section_title TEXT,
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
        1 - (c.embedding_dense <=> query_embedding) AS similarity
    FROM chunks c
    WHERE c.embedding_dense IS NOT NULL
    AND 1 - (c.embedding_dense <=> query_embedding) > match_threshold
    ORDER BY c.embedding_dense <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Equation search
CREATE OR REPLACE FUNCTION match_equations(
    query_embedding vector(1024),
    match_threshold float DEFAULT 0.5,
    match_count int DEFAULT 10
)
RETURNS TABLE (
    equation_id TEXT,
    paper_id TEXT,
    latex TEXT,
    text_description TEXT,
    similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        e.equation_id,
        e.paper_id,
        e.latex,
        e.text_description,
        1 - (e.embedding <=> query_embedding) AS similarity
    FROM equations e
    WHERE e.embedding IS NOT NULL
    AND 1 - (e.embedding <=> query_embedding) > match_threshold
    ORDER BY e.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- -------------------------------------------
-- 9. Row Level Security (Optional)
-- -------------------------------------------
-- Enable if needed for production
-- ALTER TABLE papers ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE equations ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE figures ENABLE ROW LEVEL SECURITY;

-- -------------------------------------------
-- 10. Comments
-- -------------------------------------------
COMMENT ON TABLE papers IS 'arXiv 논문 메타데이터 및 처리 상태';
COMMENT ON TABLE chunks IS '논문 텍스트 청크 및 벡터 임베딩';
COMMENT ON TABLE equations IS '수식 원본(LaTeX) 및 텍스트 설명';
COMMENT ON TABLE figures IS '논문 그림 및 캡션';

COMMENT ON COLUMN chunks.embedding_sparse IS 'BGE-M3 sparse vector, Top-128 filtered, format: {token_id: weight}';
COMMENT ON COLUMN equations.latex IS 'Source of Truth - 원본 LaTeX 보존';
COMMENT ON COLUMN equations.text_description IS 'Gemini 생성 자연어 설명 (임베딩 대상)';
