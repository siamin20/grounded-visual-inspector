"""Wave 4/10a — 결정론적 Fake LLM 2종 (fast/quality) + 실측 비용 계산.

실제 LLM 호출 없이도 파이프라인 전체가 결정론적으로 재현되도록, 프롬프트 내용에 따라
정해진 응답을 돌려준다. 각 호출은 LangSmith에 run_type="llm"으로 기록되고, 문자열
길이 기반 토큰 추정치로 비용을 계산해 상태에 누적한다 (Wave 10a의 모델 분리 근거).
"""
import json
import re

from langsmith import traceable


class FakeLLM:
    def __init__(self, name: str, usd_per_1k_tokens: float):
        self.name = name
        self.usd_per_1k = usd_per_1k_tokens

    def _usage(self, prompt: str, completion: str) -> dict:
        tokens = (len(prompt) + len(completion)) // 4 + 1
        return {"model": self.name, "tokens": tokens,
                "usd": round(tokens / 1000 * self.usd_per_1k, 6)}


class FastLLM(FakeLLM):
    """분류·계획 등 짧고 정형적인 단계 전담 (저비용)."""

    @traceable(run_type="llm", name="fast_llm.route")
    def route(self, query: str) -> tuple[dict, dict]:
        """Wave 5 — 쿼리 라우팅. 장애 관련이면 incident 파이프라인, 아니면 chat 직행."""
        incident = bool(re.search(r"장애|진단|다운|지연|급증|incident|티켓", query))
        result = {"route": "incident" if incident else "chat", "needs_analysis": False}
        return result, self._usage(query, str(result))

    @traceable(run_type="llm", name="fast_llm.plan")
    def plan(self, query: str, procedure_prompt: str) -> tuple[dict, dict]:
        """Wave 4 — Plan/Task 분해. MCP Prompt가 준 표준 절차를 따라 계획을 세운다.

        Wave 6 재현: needs_analysis=False라고 스스로 판단해 놓고도 analysis task를
        끼워 넣는 '모델의 자기모순'을 결정론적으로 재현한다 → gate 노드가 걸러낸다.
        """
        systems = re.findall(r"[a-z][a-z0-9-]+-(?:service|\d+)", query) or ["auth-service"]
        if "web-01" not in systems:
            systems.append("web-01")   # 연관 시스템 비교 조회 → Wave 10b 병렬 실행 대상
        tasks = [
            {"id": f"task-{i}", "agent": "research", "kind": "monitoring",
             "target": s, "status": "pending"}
            for i, s in enumerate(dict.fromkeys(systems))
        ]
        tasks.append({"id": "task-analysis", "agent": "analysis",
                      "kind": "pattern-analysis", "status": "pending"})   # 자기모순 task
        result = {"needs_analysis": False, "tasks": tasks}
        return result, self._usage(query + procedure_prompt, str(result))

    @traceable(run_type="llm", name="fast_llm.plan_inspection")
    def plan_inspection(self, item: str, procedure_prompt: str) -> tuple[dict, dict]:
        """제품 검사의 표준 3개 task를 고정 반환한다.

        실모델 백엔드와 동일하게 ``tasks`` 객체를 돌려주되, Fake에서는
        프롬프트 문구에 의해 계획이 흔들리지 않게 한다.
        """
        tasks = [
            {"id": "task-observe", "agent": "research", "kind": "observe",
             "status": "pending"},
            {"id": "task-cross", "agent": "research", "kind": "cross_check",
             "status": "pending"},
            {"id": "task-judge", "agent": "research", "kind": "spec_judge",
             "status": "pending"},
        ]
        result = {"tasks": tasks}
        return result, self._usage(item + procedure_prompt, str(result))

    @traceable(run_type="llm", name="fast_llm.judge")
    def judge(self, observations_text: str, clauses: list[dict],
              item: str) -> tuple[dict, dict]:
        """관찰문에 등장한 기준서 핵심어로 위반 조항을 고른다.

        ``good`` 조항은 합격 조건이므로 결함 위반 후보에서 제외한다. 한국어
        조사를 떼고 짧은 일반어를 제외한 뒤 핵심 토큰의 포함 관계를
        사용해 외부 모델 없이 같은 입력에 같은 결과를 낸다.
        """
        prompt = observations_text + json.dumps(clauses, ensure_ascii=False) + item
        if re.search(r"(?:이상|결함).{0,12}(?:보이지\s*않|없)", observations_text):
            result = {
                "violations": [],
                "defect_type": None,
                "rationale": "관찰문이 눈에 띄는 이상이 없음을 명시합니다.",
            }
            return result, self._usage(prompt, json.dumps(result, ensure_ascii=False))

        observed = set(_normalize_korean(observations_text).split())
        candidates: list[tuple[int, str, str, list[str]]] = []
        for clause in clauses:
            clause_id = clause.get("clause_id")
            defect_type = clause.get("defect_type")
            text = clause.get("text", "")
            if not isinstance(clause_id, str) or defect_type in {None, "good"}:
                continue
            keywords = _clause_keywords(str(text))
            matched = [keyword for keyword in keywords if keyword in observed]
            if not matched:
                continue
            score = sum(_keyword_weight(keyword) for keyword in matched)
            if isinstance(defect_type, str):
                candidates.append((score, clause_id, defect_type, matched))

        violations: list[str] = []
        first_type: str | None = None
        matched_keywords: list[str] = []
        if candidates:
            # 위치·특징어가 가장 많이 겹치는 결함 유형을 먼저 고른 뒤,
            # 같은 유형의 여러 조항은 함께 위반으로 남긴다.
            best = max(candidates, key=lambda candidate: candidate[0])
            first_type = best[2]
            for _score, clause_id, defect_type, matched in candidates:
                if defect_type == first_type:
                    violations.append(clause_id)
                    matched_keywords.extend(matched)

        if violations:
            unique_keywords = list(dict.fromkeys(matched_keywords))
            rationale = (
                f"관찰문의 핵심어({', '.join(unique_keywords[:4])})가 "
                f"기준서 조항과 일치합니다."
            )
        else:
            rationale = "관찰문과 일치하는 결함 조항의 핵심어가 없습니다."
        result = {
            "violations": violations,
            "defect_type": first_type,
            "rationale": rationale,
        }
        return result, self._usage(prompt, json.dumps(result, ensure_ascii=False))

    @traceable(run_type="llm", name="fast_llm.analyze")
    def analyze(self, collected: list[dict]) -> tuple[str, dict]:
        """Wave 6 보완 — 여러 시스템의 monitoring 결과 사이에 공통 패턴이 있는지 비교."""
        mon = [c["data"] for c in collected if c["source"] == "monitoring"]
        degraded = [d["system_name"] for d in mon if d.get("status") != "healthy"]
        if len(degraded) >= 2:
            summary = (f"{', '.join(degraded)} 복수 시스템이 동시에 저하됨 — "
                       f"공유 의존성 등 공통 원인이 의심됨")
        elif degraded:
            summary = f"{degraded[0]}만 단독 저하 — 연관 시스템은 정상, 국소 이슈로 추정"
        else:
            summary = "모니터링된 시스템 모두 healthy — 패턴상 이상 징후 없음"
        return summary, self._usage(str(collected), summary)

    @traceable(run_type="llm", name="fast_llm.chat")
    async def chat(self, query: str, history: list[dict], on_token=None) -> tuple[str, dict]:
        """Wave 5의 direct 경로 + Wave 12 증명용 — 히스토리에서 이름을 기억해낸다.

        fake 백엔드는 실제 네트워크 호출이 없어 토큰 단위로 쪼갤 대상이 없다 —
        ollama_llm.py와 인터페이스(on_token 콜백)만 맞추고 결과를 한 번에 반환한다."""
        for msg in reversed(history):
            m = re.search(r"제 이름은 (\S+?)입니다", msg.get("content", ""))
            if m and re.search(r"이름이 뭐", query):
                reply = f"{m.group(1)}님이라고 말씀하셨어요. 무엇을 도와드릴까요?"
                if on_token is not None:
                    on_token(reply)
                return reply, self._usage(query, reply)
        m = re.search(r"제 이름은 (\S+?)입니다", query)
        if m:
            reply = f"안녕하세요, {m.group(1)}님! 무엇을 도와드릴까요?"
        else:
            reply = "안녕하세요! 무엇을 도와드릴까요?"
        if on_token is not None:
            on_token(reply)
        return reply, self._usage(query, reply)


class QualityLLM(FakeLLM):
    """최종 답변 작성 전담 (고품질·고비용)."""

    @traceable(run_type="llm", name="quality_llm.write")
    async def write(self, query: str, collected: list[dict], attempt: int,
              feedback: str | None, on_token=None) -> tuple[str, dict]:
        """Wave 8 재현: 1차 초안은 의도적으로 저품질(너무 짧음) → validator가 반려하면
        피드백을 반영한 구조화된 2차 초안을 내놓는다.

        fake 백엔드는 실제 네트워크 호출이 없어 토큰 단위로 쪼갤 대상이 없다 —
        ollama_llm.py와 인터페이스(on_token 콜백)만 맞추고 결과를 한 번에 반환한다."""
        if attempt == 1:
            draft = "재시작하면 됩니다."
            if on_token is not None:
                on_token(draft)
            return draft, self._usage(query, draft)

        mon = [c for c in collected if c["source"] == "monitoring"]
        run = [c for c in collected if c["source"] == "runbook"]
        summary = " · ".join(
            f"{c['data']['system_name']} 상태={c['data']['status']}, "
            f"CPU {c['data']['cpu_percent']}% (알림 {len(c['data']['alerts'])}건)"
            for c in mon)
        lines = [f"'{query}' 진단 결과:", f"1) 모니터링: {summary}"]
        if run:
            body_lines = [l.strip() for l in run[0]["data"].strip().splitlines() if l.strip()]
            title = body_lines[0].lstrip("# ")
            steps = [m.group(1) for l in body_lines[1:]
                     if (m := re.match(r"^\d+\.\s+(.*)", l))]
            advice = " ".join(steps[1:3]) or "표준 절차를 따라 조치하세요."
            lines.append(f"2) 표준 절차: '{title}'에 따라 {advice}")
        lines.append("3) 정상화 확인 후 원인 분석과 재발 방지 대책을 문서화하세요.")
        draft = " ".join(lines)
        if on_token is not None:
            on_token(draft)
        return draft, self._usage(query + str(feedback), draft)

    @traceable(run_type="llm", name="quality_llm.write_report")
    async def write_report(self, ctx: dict, attempt: int,
                           feedback: str | None, on_token=None) -> tuple[str, dict]:
        """관찰과 판정 상태만으로 검사 보고서 JSON을 조립한다.

        1차 초안에는 출처 관찰에 없는 문장을 하나 넣어 validate 재작성 경로를
        재현한다. 2차부터는 전달받은 관찰만 인용한다.
        """
        observations = [
            {
                "text": str(observation.get("text", "")),
                "source_call_id": str(observation.get("call_id", "")),
            }
            for observation in ctx.get("observations", [])
            if isinstance(observation, dict) and observation.get("text")
        ]
        if attempt == 1:
            observations.append({
                "text": "우측 상단에 추가 균열이 보입니다.",
                "source_call_id": "hallucinated-observation",
            })

        verdict = (
            ctx.get("verdict")
            if ctx.get("verdict") in {"PASS", "REJECT", "ABSTAIN"}
            else "ABSTAIN"
        )
        defect_type = ctx.get("defect_type") if verdict == "REJECT" else None
        confidence = (
            "low" if verdict == "ABSTAIN"
            else ctx.get("confidence")
            if ctx.get("confidence") in {"high", "low"}
            else "low"
        )
        abstain_reason = (
            ctx.get("abstain_reason")
            if verdict == "ABSTAIN"
            and ctx.get("abstain_reason") in {
                "insufficient_visual_evidence",
                "ungrounded_anomaly",
                "evidence_conflict",
                "validation_failure",
            }
            else "validation_failure" if verdict == "ABSTAIN" else None
        )
        judgement = ctx.get("judgement") if isinstance(ctx.get("judgement"), dict) else {}
        actions = _guide_actions(str(ctx.get("guide_text", "")), verdict)
        if verdict == "PASS":
            summary = "관찰과 기준서 비교 결과 확인된 결함이 없어 통과로 판정합니다."
        elif verdict == "ABSTAIN":
            summary = "근거가 충분하지 않아 판정을 보류하고 재검사를 요청합니다."
        else:
            summary = (
                f"관찰 결과 {defect_type or '유형 미상'} 후보가 기준서 위반으로 "
                f"확인되어 {', '.join(actions)} 조치가 필요합니다."
            )
        report = {
            "verdict": verdict,
            "defect_type": defect_type,
            "confidence": confidence,
            "abstain_reason": abstain_reason,
            "observations": observations,
            "spec_clauses": (
                [
                    clause_id for clause_id in judgement.get("violations", [])
                    if isinstance(clause_id, str)
                ]
                if verdict == "REJECT"
                else []
            ),
            "actions": actions,
            "lot": ctx.get("lot") if isinstance(ctx.get("lot"), dict) else {},
            "summary": summary,
        }
        draft = json.dumps(report, ensure_ascii=False)
        if on_token is not None:
            on_token(draft)
        prompt = json.dumps(ctx, ensure_ascii=False, default=str) + str(feedback)
        return draft, self._usage(prompt, draft)


_JOSA = re.compile(
    r"(?:에서|으로|로서|로써|이라면|라면|이면|면|은|는|이|가|을|를|에|와|과|도|만|의)$"
)
_KEYWORD_STOPWORDS = {
    "관찰", "확인", "보이", "보임", "보이면", "보이지", "않으면", "합격한다",
    "불합격한다", "일부", "부위", "영역", "모양", "형태", "정상", "전체", "주변",
    "표면", "윤곽", "껍질",
    "구별되는", "구별되", "뚜렷하게", "자연스러운", "정도", "한쪽", "반대쪽",
}
_SHORT_KEYWORDS = {"목", "골", "틈", "휘", "색"}
_LOCATION_KEYWORDS = {
    "목", "머리", "윗면", "측면", "상단", "입구", "테두리", "중앙", "구멍", "외곽", "내부",
}


def _normalize_korean(text: str) -> str:
    """구두점과 조사 차이를 줄인 비교용 문자열을 만든다."""
    text = (text.replace("색상 변화", "변색")
            .replace("선형 흔적", "긁힘")
            .replace("인쇄형 흔적", "인쇄 흔적"))
    tokens = []
    for raw in re.findall(r"[0-9A-Za-z_ㄱ-ㅣ가-힣]+", text.lower()):
        token = _JOSA.sub("", raw)
        tokens.append(token)
    return " ".join(tokens)


def _clause_keywords(text: str) -> list[str]:
    """기준서 문장에서 비교 가능한 핵심 토큰을 뽑는다."""
    normalized = _normalize_korean(text)
    return list(dict.fromkeys(
        token for token in normalized.split()
        if (len(token) >= 2 or token in _SHORT_KEYWORDS)
        and token not in _KEYWORD_STOPWORDS
    ))


def _keyword_weight(keyword: str) -> int:
    """결함의 위치를 특정하는 단어에 더 큰 가중치를 준다."""
    return 4 if keyword in _LOCATION_KEYWORDS else 1


def _guide_actions(guide_text: str, verdict: str) -> list[str]:
    """가이드의 ``조치 옵션`` 순서에서 첫 조치를 고른다."""
    if verdict == "PASS":
        return ["통과"]
    if verdict == "ABSTAIN":
        return ["재검사"]
    allowed = {"재작업", "폐기", "라인정지"}
    match = re.search(r"조치\s*옵션\s*:\s*([^\n]+)", guide_text)
    if match:
        actions = [part.strip() for part in match.group(1).split("|")]
        selected = [action for action in actions if action in allowed]
        if selected:
            return [selected[0]]
    return ["재작업"]
