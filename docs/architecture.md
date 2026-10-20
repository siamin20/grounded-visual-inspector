# 아키텍처와 흐름도

설계 본문은 [design.md](design.md). 이 문서는 그림만 모은다. GitHub 에서 mermaid 로 렌더된다.

## 1. 전체 구성도

```mermaid
flowchart TB
    Browser["브라우저 (web_chat.html, Vue 3)<br/>품목 선택 · 이미지 업로드 · 승인/반려"] -->|"same-origin"| Proxy["web_demo_server.py (:8600)<br/>/inspect · /chat/stream · /chat/resume 프록시"]
    Proxy --> API
    Curl["curl / eval/run_eval.py"] --> API

    subgraph Server["server.py — FastAPI (:8100)"]
        API["/health · /chat · /chat/stream<br/>/inspect (multipart) · /chat/resume"]
        Lifespan["lifespan 싱글턴<br/>SupervisorDeps 1회 생성"]
    end

    API --> Graph

    subgraph Graph["agent/graph.py — LangGraph StateGraph (v2)"]
        direction TB
        Route["route"] -->|chat| Chat["chat_answer"]
        Route -->|inspection| Plan["plan"] --> Gate["gate"] --> Research
        subgraph Research["research 서브그래프 (compiled)"]
            direction TB
            Obs["Observer<br/>full + 4크롭 병렬"]
            CC["Cross-Checker<br/>좌우반전 · 체크리스트 · 다른 모델"]
            SJ["Spec-Judge<br/>기준서 RAG → 위반 조항"]
            Obs --> SJ
            CC --> SJ
        end
        Research --> Replan["replan_check"]
        Replan -->|부족| Research
        Replan -->|충분| Analysis["analysis<br/>(코드 규칙 합의)"]
        Analysis --> Write["write<br/>보고서 JSON"]
        Write --> HITL["hitl<br/>interrupt()"]
        HITL --> Validate["validate<br/>(코드 규칙 4개)"]
        Validate -->|불합격 ≤2회| Write
        Validate -->|합격| Finalize["finalize"]
    end

    API -. "SUPERVISOR_VERSION=v1" .-> V1["agent/supervisor_v1.py<br/>Before 기준선 B0/B1 (VLM 1회)"]

    Plan -. "find_prompt → inspection_prompt" .-> MCP
    Obs -. "find_tool → inspect_image ×5" .-> MCP
    CC -. "inspect_image (flip, checklist)" .-> MCP
    Write -. "guide://{defect_type}" .-> MCP
    SJ --> VS["SpecIndex (numpy cosine)<br/>data/specs 조항 단위"]

    subgraph MCP["agent/mcp_client.py — 다중 MCP 세션 (stdio subprocess)"]
        subgraph Vision["mcp_server_vision.py (server=vision)"]
            Tool["Tool: inspect_image"]
            Res1["Resource: spec://{item}"]
            Res2["Resource: guide://{defect_type}"]
            Prompt["Prompt: inspection_prompt"]
        end
        subgraph Docs["mcp_server_docs.py (server=docs, decoy)"]
            DocsRes["Resource: docs://api-reference/{section}"]
        end
    end

    Tool --> VLM["agent/vision/backends.py<br/>fake · ollama(qwen2.5vl) · openai(gpt-4o-mini / gpt-4o)"]

    Lifespan --> Deps["SupervisorDeps<br/>fast_llm · quality_llm · mcp · session_store<br/>vector_store · hitl_mode · ablation"]
    Deps --> LLM["텍스트 LLM<br/>fake · ollama · openai"]
    Deps --> Session["session.db (대화 이력)"]
    Graph --> Ckpt["checkpoints.db<br/>LangGraph SqliteSaver (HITL 재개)"]
    Graph --> LS["LangSmith<br/>계층 trace · Dataset 평가"]
```

핵심 원칙 네 가지가 그림에 드러나야 한다.
- VLM 은 MCP tool 뒤에만 있다. 그래프 노드는 VLM 을 직접 모른다.
- research 는 2단계다. Observer 와 Cross-Checker 가 병렬, Spec-Judge 는 Observer 의 관찰문을 받아 뒤에 돈다.
- gate · analysis · validate 는 LLM 을 부르지 않는다.
- Before(v1)와 After(v2)는 같은 서버에서 환경변수 하나로 갈린다.

## 2. 그래프 상태 흐름 (무엇이 어느 노드에서 채워지는가)

```mermaid
flowchart LR
    I["입력<br/>image_path · item · query"] --> R["route<br/>route"]
    R --> P["plan<br/>tasks[observe, cross_check, spec_judge]"]
    P --> G["gate<br/>tasks(검증됨) · gate_removed"]
    G --> O["Observer<br/>observations[5]"]
    G --> C["Cross-Checker<br/>cross_check"]
    O --> S["Spec-Judge<br/>spec_hits[≤3] · judgement"]
    C --> S
    S --> A["analysis<br/>verdict · confidence · defect_type"]
    A --> W["write<br/>report · draft"]
    W --> H["hitl<br/>approval{required, decision}"]
    H --> V["validate<br/>validation_errors · validation_feedback"]
    V --> F["finalize<br/>final (markdown)"]
```

## 3. 시퀀스 — 검사 요청 (정상 흐름, HITL 불필요)

```mermaid
sequenceDiagram
    participant B as 브라우저
    participant S as server.py
    participant G as graph
    participant M as MCP vision 서버
    participant V as VLM
    participant X as SpecIndex

    B->>S: POST /inspect (file, item=screw)
    S->>S: data/uploads/{thread}/{uuid}.png 저장
    S->>G: astream({image_path, item, query})
    G-->>S: RouteDecided(inspection) → SSE
    G->>M: find_prompt → get_prompt(inspection_prompt)
    G-->>S: PlanCreated(3 tasks) · GateApplied
    par Observer
        G->>M: inspect_image ×5 (full, tl, tr, bl, br) with_retry
        M->>V: 관찰 요청 (참조 이미지 1-shot)
        V-->>M: {text, anomaly_seen, regions, features}
        M-->>G: ObservationReady ×5 → SSE
    and Cross-Checker
        G->>M: inspect_image (flip, checklist, 다른 모델)
        M->>V: 체크리스트 재판독
        V-->>M: {anomaly_seen, regions}
        M-->>G: CrossCheckDone → SSE
    end
    G->>X: search(관찰문, item, k=3)
    X-->>G: spec_hits
    G->>G: judge(관찰문, 조항) → judgement
    G-->>S: SpecJudged(clause_ids) → SSE
    G->>G: analysis 합의 규칙 → verdict/confidence
    G-->>S: AnalysisDecided → SSE
    G->>M: guide://{defect_type} (REJECT 일 때)
    G->>G: write → report JSON (AnswerToken 스트리밍)
    G->>G: hitl: required=false → 통과
    G->>G: validate 4규칙
    alt 불합격
        G-->>S: Validation(불합격, errors) → SSE
        G->>G: write 재작성 (≤2회)
    end
    G-->>S: Validation(합격) · Finalize
    S-->>B: FinalAnswer(보고서) · [DONE]
    S->>S: session.db 저장
```

## 4. 시퀀스 — HITL 승인 (interrupt → resume)

```mermaid
sequenceDiagram
    participant B as 브라우저
    participant S as server.py
    participant G as graph (checkpointer)
    participant DB as checkpoints.db

    B->>S: POST /inspect (file, item=bottle)
    S->>G: astream(..., config.thread_id)
    Note over G: … analysis → REJECT, actions=[폐기, 라인정지]
    G->>G: hitl: required=true, HITL_MODE=interactive
    G->>DB: 상태 체크포인트 저장
    G-->>S: __interrupt__ (ApprovalRequired + report)
    S-->>B: ApprovalRequired → data: [PAUSED]
    Note over B: 보고서 요약 카드 + [승인] [반려]
    B->>S: POST /chat/resume {thread_id, decision: approve}
    S->>G: astream(Command(resume={decision}), config.thread_id)
    G->>DB: 체크포인트 복원
    G->>G: hitl 재개 → approval.decision=approve
    G->>G: validate → finalize
    S-->>B: ApprovalResolved · Validation · FinalAnswer · [DONE]
    Note over S,DB: 서버가 재시작돼도 checkpoints.db 가 있으면<br/>같은 thread_id 로 resume 가능 (tests/test_resume_restart.py)
```

## 5. 시퀀스 — Before 기준선 (v1, 평가용)

```mermaid
sequenceDiagram
    participant E as eval/run_eval.py
    participant S as bootstrap (in-process)
    participant V1 as supervisor_v1.run_v1
    participant M as MCP vision 서버

    E->>S: 조건 B0 또는 B1, HITL_MODE=auto_approve
    loop held-out 샘플마다
        E->>V1: run_v1(image_path, item, BASELINE_MODE)
        alt B1
            V1->>M: spec://{item} 전문 읽기
        end
        V1->>M: inspect_image(style=baseline, 참조 이미지, 유형 목록[, 기준서 전문])
        M-->>V1: {verdict, defect_type, observations}
        V1-->>E: report 스키마로 감싼 결과
        E->>E: jsonl 1줄 기록 (latency, tokens, usd, run_id)
    end
    E->>E: report.py → balanced acc · FNR · 유형 F1 · McNemar(B0 vs A)
```

## 6. 평가 데이터 흐름

```mermaid
flowchart LR
    HF["HuggingFace<br/>foersben/mvtec-ad"] -->|"sh/setup_data.sh"| MV["data/mvtec/{item}/test/{label}/*.png<br/>(gitignore)"]
    MV -->|"eval/make_dataset.py<br/>층화 30/70, seed 고정"| DS["eval/dataset.json<br/>192장: dev 52 · heldout 140"]
    DS --> RUN["eval/run_eval.py<br/>--condition B0|B1|A|A-X|A-C|A-V"]
    SPEC["data/specs/*.md<br/>(라벨 미참조로 작성, git 고정)"] --> RUN
    RUN --> RES["eval/results/{condition}_{split}.jsonl"]
    RES --> REP["eval/report.py<br/>지표 표 · McNemar · 품목별 분해"]
    RES --> HC["eval/human_check.py<br/>블라인드 30건 환각 채점"]
    REP --> README["README 결과 표"]
    HC --> README
```

정답 라벨은 폴더명(외부)에서만 온다. 기준서는 판정의 참고 자료이지 채점 기준이 아니다.
