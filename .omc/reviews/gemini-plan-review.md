Loaded cached credentials.
Hook registry initialized with 0 hook entries
제시해주신 `PLAN.md`는 매우 구체적이고 체계적으로 잘 작성되었습니다. 특히 **Marker**를 이용한 파싱과 **BGE-M3**의 하이브리드 임베딩을 활용한 설계는 최신 RAG 트렌드를 잘 반영하고 있습니다.

다만, 실제 구현에 들어갔을 때 **Blocking Issue**가 되거나 완성도를 떨어뜨릴 수 있는 몇 가지 미정의 사항들을 식별하여 정리해 드립니다.

---

### 1. 누락된 결정사항 (Missing Decisions)

*   **키워드 및 필터링 로직 구체화**: 
    *   "LLM 키워드 필터"라고 명시되어 있으나, 구체적인 키워드 리스트(예: "Large Language Model", "Transformer", "RLHF", "Quantization" 등)가 정의되지 않았습니다. 
    *   arXiv API는 검색 쿼리에 따라 결과가 크게 달라지므로, **우선순위 카테고리(cs.CL, cs.LG 등)와 키워드 조합**을 미리 확정해야 합니다.
*   **논문 선별 알고리즘**: 
    *   2025년 논문은 인용수(Citation count)가 0인 경우가 많습니다. "인기도/인용수 정렬"에서 인용수가 없을 경우의 2차 정렬 기준(예: 저자의 이전 업적, 특정 저널/컨퍼런스 채택 여부 등)이 필요합니다.
*   **UI/UX 계획**: 
    *   포트폴리오용이라면 FastAPI 엔드포인트만 있는 것보다, 사용자 검색 결과를 보여줄 간단한 **Streamlit** 또는 **Gradio** 기반의 인터페이스 포함 여부를 결정해야 합니다.

### 2. 모호한 요구사항 (Ambiguous Requirements)

*   **LaTeX 소스 vs PDF 파싱**: 
    *   플랜에는 "PDF 파서로 Marker 사용"과 "LaTeX 소스 다운로드"가 공존합니다. 일반적으로 LaTeX 소스가 있다면 이를 파싱하는 것이 PDF 파싱보다 훨씬 정확합니다(특히 수식과 테이블). 
    *   Marker를 주력으로 쓰되 LaTeX는 백업용인지, 아니면 두 데이터를 어떻게 조합(예: 수식은 LaTeX에서, 구조는 PDF에서)할 것인지 명확히 해야 합니다.
*   **Sparse 벡터 검색 구현 방식**: 
    *   Supabase(pgvector)는 Dense 벡터에 최적화되어 있습니다. Sparse 벡터(BGE-M3의 특성)를 JSONB에 저장한다고 했는데, 이를 이용한 **Hybrid Search(Dense + Sparse)의 랭킹 로직(예: RRF - Reciprocal Rank Fusion)**을 SQL로 구현할지 애플리케이션 레벨에서 구현할지 정의가 필요합니다.

### 3. 기술적 리스크 (Technical Risks)

*   **Gemini API 비용 및 Rate Limit**: 
    *   500~1000개의 논문에서 수식과 이미지를 모두 Gemini API로 처리할 경우 호출 횟수가 수천 번에 달할 수 있습니다. 무료 티어 사용 시 Rate Limit에 의한 지연, 유료 사용 시 예상 비용에 대한 사전 검토가 필요합니다.
*   **Marker의 리소스 관리**: 
    *   Marker는 GPU 메모리를 많이 소모합니다. 1000개를 연속 파싱할 때 발생할 수 있는 메모리 누수나 좀비 프로세스 방지를 위한 **배치 처리 및 프로세스 격리 전략**이 필요합니다.
*   **데이터 동기화 (MongoDB vs Supabase)**: 
    *   메타데이터는 MongoDB에, 벡터는 Supabase에 분산 저장됩니다. 논문 정보가 업데이트되거나 삭제될 때 **두 DB 간의 일관성을 어떻게 유지할 것인지**가 기술적 부채가 될 수 있습니다. (가능하다면 Supabase/PostgreSQL 하나로 통합하는 것도 검토해 보세요.)

### 4. 의존성 문제 (Dependency Issues)

*   **Semantic Scholar API 제한**: 
    *   무료 계정은 API 호출 속도 제한(100 req/5 min)이 엄격합니다. 1000개의 데이터를 한꺼번에 수집할 때 로직이 멈추지 않도록 **Exponential Backoff** 등의 재시도 전략이 필수입니다.
*   **PDF 품질 이슈**: 
    *   arXiv의 일부 논문은 Marker로도 파싱이 깔끔하지 않은 이미지 위주의 PDF일 수 있습니다. 파싱 실패 시의 예외 처리(Fallback) 로직이 필요합니다.

### 5. 확장성 고려 (Scalability)

*   **청킹 컨텍스트 유지**: 
    *   하이브리드 청킹 시 섹션이 잘릴 때, 해당 청크가 어떤 섹션에 속하는지 뿐만 아니라 **논문 전체의 핵심 주제(Abstract)**를 모든 청크의 메타데이터에 포함시킬지 여부를 결정해야 합니다. (검색 품질에 큰 영향을 미침)
*   **수식 검색 전략**: 
    *   사용자가 수식(LaTeX)으로 검색할 경우를 대비하여, Gemini가 생성한 설명뿐만 아니라 **원본 LaTeX 코드**도 임베딩하거나 검색 가능한 텍스트 필드에 포함시킬지 결정해야 합니다.

---

### **[권장 조치 사항]**
구현을 시작하기 전에 다음 사항을 먼저 확정하시길 추천합니다:
1.  **DB 단일화**: 특별한 이유가 없다면 MongoDB 대신 Supabase(PostgreSQL)에 모든 메타데이터를 저장하여 관리 포인트를 줄이는 것을 권장합니다.
2.  **Hybrid 검색 PoC**: Supabase SQL에서 JSONB Sparse 벡터와 pgvector Dense 벡터를 결합하는 쿼리를 미리 테스트해 보세요.
3.  **우선순위 키워드 리스트 작성**: 수집의 핵심이 되는 키워드 셋을 확정하세요.

이 계획이 확정되면 바로 **Session 1(환경 설정)**부터 진행해도 무방할 만큼 탄탄한 계획입니다. 추가로 궁금한 점이 있으시면 말씀해 주세요.
