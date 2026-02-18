# arXiv RAG Pipeline Blueprint

> 2025년 ML/DL/LLM 관련 arXiv 논문을 수집하여 RAG 시스템에 활용하기 위한 파이프라인 설계

## 목표

- arXiv에서 ML, DL, Transformer, LLM 관련 논문 수집
- 논문 전문(full-text)을 의미 단위로 분할하여 RAG 검색에 활용
- 수식, 이미지, 테이블 등 non-text 데이터도 검색 가능하도록 처리

---

## 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────┐
│                    Phase 1: Collection                       │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐        │
│  │ arXiv API   │ → │ Metadata    │ → │ MongoDB     │        │
│  │ (metadata)  │   │ + PDF URL   │   │ (raw meta)  │        │
│  └─────────────┘   └─────────────┘   └─────────────┘        │
│         ↓                                                    │
│  ┌─────────────┐   ┌─────────────┐                          │
│  │ PDF 다운로드 │ → │ S3/Local    │  ← 원본 보존             │
│  │ LaTeX 소스  │   │ Storage     │                          │
│  └─────────────┘   └─────────────┘                          │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                    Phase 2: Parsing                          │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐        │
│  │ PDF Parser  │ → │ Section     │ → │ Structured  │        │
│  │ (PyMuPDF/   │   │ Extraction  │   │ JSON        │        │
│  │  GROBID)    │   │             │   │             │        │
│  └─────────────┘   └─────────────┘   └─────────────┘        │
│         ↓                                                    │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐        │
│  │ Figure/     │   │ Equation    │   │ Table       │        │
│  │ Image Ext.  │   │ Extraction  │   │ Extraction  │        │
│  └─────────────┘   └─────────────┘   └─────────────┘        │
│         ↓                ↓                 ↓                 │
│  ┌─────────────────────────────────────────────────┐        │
│  │            VLM/LLM 설명 생성 (Gemini)            │        │
│  │   - Figure captioning                           │        │
│  │   - Equation 텍스트 설명                         │        │
│  │   - Table 요약                                   │        │
│  └─────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                   Phase 3: Chunking & Embedding              │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐        │
│  │ Hybrid      │ → │ Metadata    │ → │ Embedding   │        │
│  │ Chunking    │   │ Attachment  │   │ Generation  │        │
│  │ (섹션+문단) │   │ (paper_id,  │   │ (OpenAI/    │        │
│  │             │   │  section)   │   │  Voyage)    │        │
│  └─────────────┘   └─────────────┘   └─────────────┘        │
│                                             ↓                │
│                                    ┌─────────────┐          │
│                                    │ Supabase    │          │
│                                    │ (pgvector)  │          │
│                                    └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Collection (데이터 수집)

### 목적
- arXiv API로 메타데이터 수집
- PDF 및 LaTeX 소스 원본 다운로드
- **원본 보존** - 청킹 전략 변경 시 재크롤링 없이 재처리 가능

### 수집 대상 카테고리

```python
categories = [
    "cs.LG",   # Machine Learning
    "cs.CL",   # Computation and Language (NLP/LLM)
    "cs.AI",   # Artificial Intelligence
    "stat.ML", # Statistics - Machine Learning
]
```

### 재사용 가능한 기존 코드

| 모듈 | 소스 | 재사용 범위 |
|------|------|-------------|
| `ArxivCollector` | `ai-weekly-digest/src/collectors/arxiv_collector.py` | 90% (카테고리 변경만 필요) |
| `ArxivPaper` 데이터클래스 | 동일 파일 | 100% |
| `MongoDBStorage` | `ai-weekly-digest/src/storage/mongodb_client.py` | 70% (스키마 확장 필요) |

### 추가 구현 필요

```python
# 1. PDF 다운로드 기능
def download_pdf(arxiv_id: str, output_dir: Path) -> Path:
    """PDF 파일 다운로드 및 저장"""
    pass

# 2. LaTeX 소스 다운로드 (선택)
def download_latex_source(arxiv_id: str, output_dir: Path) -> Optional[Path]:
    """LaTeX 소스 tarball 다운로드 (있는 경우)"""
    pass

# 3. 확장된 메타데이터 저장
class PaperDocument:
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    categories: list[str]
    published: datetime
    pdf_path: str       # 로컬/S3 경로
    latex_path: str     # LaTeX 소스 경로 (있으면)
    status: str         # collected, parsed, chunked, embedded
```

---

## Phase 2: Parsing (구조 추출)

### 목적
- PDF에서 섹션별 텍스트 추출
- 수식, 이미지, 테이블 분리 추출
- 구조화된 JSON 문서 생성

### PDF 파서 옵션 비교

| 파서 | 장점 | 단점 | 권장 용도 |
|------|------|------|----------|
| **GROBID** | 학술 논문 특화, 섹션/참조 추출 우수 | 설치 복잡 (Java), 서버 필요 | 대규모 처리 |
| **PyMuPDF** | 빠름, 설치 간단 | 구조 추출 약함 | 텍스트 위주 |
| **Marker** | 최신, 수식/테이블 추출 우수 | 상대적으로 느림 | 고품질 필요 시 |
| **Nougat** | Meta AI, 수식 OCR 우수 | GPU 필요, 느림 | 수식 중심 논문 |

### Non-text 데이터 처리 전략

#### 수식 (Equation)

```python
{
    "type": "equation",
    "eq_id": "eq:1",
    "latex": r"E = mc^2",
    "latex_display": r"\begin{equation} E = mc^2 \end{equation}",
    "text_description": "에너지-질량 등가 방정식",  # LLM 생성
    "section": "s3",
    "context": "Section 3.2에서 유도된 핵심 수식"
}
```

- **임베딩 전략**: 텍스트 설명(text_description)만 임베딩 (범용 모델 사용 가능)
- LaTeX 원문은 검색 결과 표시용으로 보존

#### 이미지 (Figure)

```python
{
    "type": "figure",
    "fig_id": "fig:1",
    "image_path": "papers/{arxiv_id}/figures/fig1.png",
    "caption": "Figure 1: Model architecture...",  # 논문에서 추출
    "vlm_description": "...",  # Gemini/GPT-4V로 생성
    "referenced_in": ["s3.1", "s4.2"]
}
```

- **임베딩 전략 옵션**:
  - A: Caption만 임베딩 (빠름, 저렴)
  - B: VLM 설명 임베딩 (더 풍부)
  - C: CLIP 이미지 임베딩 (멀티모달 검색)

#### 테이블 (Table)

```python
{
    "type": "table",
    "tab_id": "tab:1",
    "caption": "Table 1: Model comparison",
    "headers": ["Model", "Accuracy", "F1"],
    "rows": [["GPT-4", "92.3", "0.91"], ...],
    "markdown": "| Model | Accuracy | F1 |\n|...",
    "text_summary": "GPT-4가 정확도 우위..."  # LLM 생성
}
```

### 파싱 출력 스키마

```python
{
    "paper_id": "2401.12345",
    "metadata": {
        "title": "...",
        "authors": [...],
        "abstract": "...",
        "categories": ["cs.CL", "cs.AI"],
        "published": "2024-01-15"
    },

    "sections": [
        {
            "section_id": "s1",
            "title": "Introduction",
            "level": 1,
            "text": "...",
            "subsections": [...]
        }
    ],

    "equations": [...],
    "figures": [...],
    "tables": [...],
    "references": [...]
}
```

---

## Phase 3: Chunking & Embedding

### 청킹 전략 옵션

| 전략 | 설명 | 장단점 |
|------|------|--------|
| **Semantic (섹션 기반)** | Abstract, Intro, Methods 등 섹션 단위 | 구조 보존, 검색 정확도 높음 |
| **Sliding Window** | 고정 토큰 (512~1024) + 20% 오버랩 | 구현 단순, 일관된 크기 |
| **Hybrid (권장)** | 1차 섹션 분리 → 2차 긴 섹션 문단 분할 | 균형 잡힌 접근 |

### Hybrid 청킹 구현 (권장)

```python
def chunk_paper(parsed_doc: dict, max_tokens: int = 512) -> list[dict]:
    chunks = []

    for section in parsed_doc["sections"]:
        section_text = section["text"]

        if count_tokens(section_text) <= max_tokens:
            # 섹션 전체를 하나의 청크로
            chunks.append({
                "chunk_id": f"{parsed_doc['paper_id']}_{section['section_id']}",
                "text": section_text,
                "metadata": {
                    "paper_id": parsed_doc["paper_id"],
                    "section": section["title"],
                    "section_id": section["section_id"]
                }
            })
        else:
            # 문단 단위로 추가 분할
            paragraphs = split_by_paragraphs(section_text)
            for i, para in enumerate(paragraphs):
                chunks.append({
                    "chunk_id": f"{parsed_doc['paper_id']}_{section['section_id']}_p{i}",
                    "text": para,
                    "metadata": {
                        "paper_id": parsed_doc["paper_id"],
                        "section": section["title"],
                        "paragraph_index": i
                    }
                })

    return chunks
```

### 임베딩 모델 옵션

| 모델 | 차원 | 특징 | 비용 |
|------|------|------|------|
| `text-embedding-3-small` | 1536 | OpenAI, 범용 | $0.02/1M tokens |
| `text-embedding-3-large` | 3072 | OpenAI, 고품질 | $0.13/1M tokens |
| `voyage-large-2` | 1536 | 학술 문서 최적화 | 유료 |
| `BAAI/bge-large-en-v1.5` | 1024 | 오픈소스, 무료 | 자체 호스팅 |

---

## 저장소 설계

### 권장 구조: MongoDB + Supabase 이원화

```
MongoDB Atlas (기존)
├── papers (메타데이터)
│   ├── arxiv_id, title, authors, abstract
│   ├── pdf_path, latex_path
│   └── status (collected/parsed/chunked/embedded)
├── parsed_documents (파싱 결과)
│   └── 전체 구조화된 JSON
└── processing_logs (처리 로그)

Supabase (신규 - RAG 전용)
├── chunks (청크 + 벡터)
│   ├── chunk_id, paper_id, text
│   ├── section, metadata (JSONB)
│   └── embedding (vector)
├── equations (수식)
│   └── latex, description, embedding
├── figures (이미지)
│   └── caption, vlm_description, embedding
└── tables (테이블)
    └── markdown, summary, embedding
```

### Supabase 스키마 예시

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Chunks table
CREATE TABLE chunks (
    id SERIAL PRIMARY KEY,
    chunk_id TEXT UNIQUE NOT NULL,
    paper_id TEXT NOT NULL,
    text TEXT NOT NULL,
    section TEXT,
    metadata JSONB,
    embedding vector(1536),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Vector similarity search index
CREATE INDEX ON chunks
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Full-text search
CREATE INDEX ON chunks USING GIN (to_tsvector('english', text));
```

---

## 구현 우선순위

### MVP (Phase 1 + 기본 Phase 2)

1. [ ] PDF 다운로드 기능 추가
2. [ ] 기본 텍스트 추출 (PyMuPDF)
3. [ ] 섹션 기반 청킹
4. [ ] Supabase 연동 + 벡터 저장
5. [ ] 간단한 검색 API

### V2 (고급 파싱)

1. [ ] GROBID 또는 Marker 통합

2. [ ] 수식 추출 + 텍스트 설명 생성
3. [ ] 이미지 추출 + VLM captioning
4. [ ] 테이블 구조화

### V3 (최적화)

1. [ ] 멀티모달 검색 (이미지 + 텍스트)
2. [ ] 하이브리드 검색 (벡터 + 키워드)
3. [ ] 재순위화 (Reranker) 적용
4. [ ] 캐싱 및 성능 최적화

---

## 참고 자료

- [arXiv API 문서](https://arxiv.org/help/api)
- [GROBID](https://github.com/kermitt2/grobid)
- [Marker PDF Parser](https://github.com/VikParuchuri/marker)
- [Supabase Vector](https://supabase.com/docs/guides/ai/vector-columns)
- [pgvector](https://github.com/pgvector/pgvector)

---

*문서 작성: 2026-02-11*
*기반 프로젝트: ai-weekly-digest*
