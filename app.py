# app.py
# Streamlit + OpenAI Responses API 기반: "돌다리" (질문 기반 AI 결정 코칭)
# 개선 사항:
# 1) 질문별 text_area key를 분리해서 이전 질문 답변이 다음 질문 입력칸에 남지 않음
# 2) 질문을 한 번에 N개 고정 생성하지 않고, 답변을 반영해 다음 질문을 단계별로 생성(동적 질문)

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore


# -----------------------------
# UI / App Config
# -----------------------------
st.set_page_config(
    page_title="돌다리 - AI 결정 코칭",
    page_icon="🪨",
    layout="wide",
)

APP_TITLE = "🪨 돌다리"
APP_TAGLINE = "결정하기 전에, 돌다리를 두드려보세요"

DEFAULT_MODEL = "gpt-5.2"
DEFAULT_NUM_QUESTIONS = 7


# -----------------------------
# Helpers
# -----------------------------
def safe_strip(x: str) -> str:
    return (x or "").strip()


def get_client(api_key: str) -> OpenAI:
    if OpenAI is None:
        raise RuntimeError("openai 패키지가 설치되어 있지 않습니다. `pip install openai`로 설치해 주세요.")
    return OpenAI(api_key=api_key)


def call_openai_text(client: OpenAI, model: str, system: str, user: str, temperature: float = 0.4) -> str:
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": [{"type": "text", "text": user}]},
        ],
        temperature=temperature,
    )

    if hasattr(resp, "output_text") and resp.output_text:
        return str(resp.output_text)

    # fallback
    try:
        texts: List[str] = []
        for item in getattr(resp, "output", []) or []:
            for c in getattr(item, "content", []) or []:
                if getattr(c, "type", None) == "output_text":
                    texts.append(getattr(c, "text", ""))
        return "\n".join([t for t in texts if t])
    except Exception:
        return ""


def extract_first_json(text: str) -> Optional[Any]:
    if not text:
        return None

    codeblock = re.search(r"```json\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if codeblock:
        candidate = codeblock.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            pass

    start_positions = [(text.find("{"), "{"), (text.find("["), "[")]
    start_positions = [(i, ch) for i, ch in start_positions if i != -1]
    if not start_positions:
        return None

    start_i, start_ch = min(start_positions, key=lambda t: t[0])
    end_ch = "}" if start_ch == "{" else "]"
    end_i = text.rfind(end_ch)
    if end_i == -1 or end_i <= start_i:
        return None

    candidate = text[start_i : end_i + 1].strip()
    try:
        return json.loads(candidate)
    except Exception:
        return None


def answers_as_bullets(answers: Dict[int, str]) -> str:
    lines = []
    for k in sorted(answers.keys()):
        lines.append(f"- Q{k}: {answers[k]}")
    return "\n".join(lines)


# -----------------------------
# Prompt Builders
# -----------------------------
def build_next_question_prompt(
    situation_title: str,
    situation_detail: str,
    user_goal: str,
    total_steps: int,
    questions_so_far: List[str],
    answers_so_far: Dict[int, str],
) -> Tuple[str, str]:
    """
    이전 답변을 반영해 "다음 1개 질문"만 생성
    출력은 반드시 JSON: {"question": "..."}
    """
    system = (
        "너는 '결정 코칭' 전문 코치다. "
        "결론/추천/정답을 제시하지 않는다. "
        "사용자가 스스로 판단하도록 돕는 '다음 질문 1개'만 만든다. "
        "질문은 짧고 명확하게, 유도 질문 금지. "
        "이미 물어본 내용은 반복하지 말고, 사용자의 이전 답변에서 드러난 포인트를 한 단계 더 깊게 탐색하라."
    )

    # 지금까지 Q/A 정리
    qa_lines = []
    for i, q in enumerate(questions_so_far, start=1):
        a = safe_strip(answers_so_far.get(i, ""))
        qa_lines.append(f"Q{i}. {q}\nA{i}. {a}")
    qa_block = "\n\n".join(qa_lines) if qa_lines else "(아직 없음)"

    # 단계 가이드(너무 딱딱하게 고정하지 않고 "가급적" 흐름만 유지)
    user = f"""
[선택 주제]
{safe_strip(situation_title)}

[상황 설명]
{safe_strip(situation_detail)}

[사용자가 얻고 싶은 것]
{safe_strip(user_goal)}

[총 단계 수]
{total_steps}

[지금까지 Q&A]
{qa_block}

다음 조건을 만족하는 '다음 질문 1개'를 만들어라.

- 한 문장, 가능한 짧게(최대 25자 내외 권장)
- 지금까지 답변을 바탕으로 가장 도움이 될 다음 탐색 포인트를 고른다
- 아래 흐름을 "가급적" 따른다(필요하면 건너뛰어도 됨):
  감정/욕구 → 현실 조건 → 가치/우선순위 → 대안 → 리스크/기회비용 → 후회 최소화 기준 → 작은 실험/다음 행동
- 출력은 반드시 아래 JSON만. 다른 텍스트 금지.

{{"question": "..." }}
"""
    return system, user


def build_summary_prompt(
    situation_title: str,
    situation_detail: str,
    user_goal: str,
    questions: List[str],
    answers: Dict[int, str],
) -> Tuple[str, str]:
    system = (
        "너는 '결정 코칭' 전문 코치다. "
        "결론/추천/정답을 제시하지 않는다. "
        "사용자의 답변을 구조화해 '스스로 결정할 수 있게' 정리해준다. "
        "판단을 유도하는 문장(예: ~해야 한다)은 피하고, 선택 기준을 명료화한다."
    )

    qa = []
    for i, q in enumerate(questions, start=1):
        a = safe_strip(answers.get(i, ""))
        qa.append(f"Q{i}. {q}\nA{i}. {a}")
    qa_block = "\n\n".join(qa)

    user = f"""
아래는 사용자의 선택 고민과 질문-답변이다. 이를 바탕으로 '결론/추천 없이' 정리해줘.

[선택 주제]
{safe_strip(situation_title)}

[상황 설명]
{safe_strip(situation_detail)}

[사용자가 얻고 싶은 것]
{safe_strip(user_goal)}

[Q&A]
{qa_block}

아래 형식으로만 출력(마크다운 OK). 반드시 결론/추천 금지.

## 고민의 핵심
- (핵심 쟁점 2~4개)

## 선택 기준 요약
- (사용자에게 중요한 기준 4~7개, 문장 짧게)

## 생각을 정리해주는 코칭 메시지
- (공감 1~2문장)
- (스스로 점검할 질문 2~3개)
- (작은 실험/다음 행동 제안 2~3개: 특정 선택을 추천하지 말고, '검증/탐색' 형태로)
"""
    return system, user


# -----------------------------
# State
# -----------------------------
@dataclass
class SessionState:
    stage: str = "home"  # home | setup | questions | result
    model: str = DEFAULT_MODEL
    total_steps: int = DEFAULT_NUM_QUESTIONS

    situation_title: str = ""
    situation_detail: str = ""
    user_goal: str = ""

    questions: List[str] = None  # type: ignore
    answers: Dict[int, str] = None  # type: ignore
    current_idx: int = 1  # 1-based

    summary_md: str = ""


def init_state():
    if "ss" not in st.session_state:
        st.session_state.ss = SessionState(questions=[], answers={})


def reset_state(keep_settings: bool = True):
    old = st.session_state.ss
    model = old.model
    total_steps = old.total_steps
    st.session_state.ss = SessionState(questions=[], answers={})
    if keep_settings:
        st.session_state.ss.model = model
        st.session_state.ss.total_steps = total_steps

    # 답변 위젯 키들도 같이 정리
    for k in list(st.session_state.keys()):
        if str(k).startswith("answer_"):
            del st.session_state[k]


init_state()
ss: SessionState = st.session_state.ss


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("### 설정")
    api_key = st.text_input(
        "OpenAI API Key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
        help="배포 시 st.secrets 또는 환경변수 OPENAI_API_KEY 사용 권장",
    )
    ss.model = st.text_input("모델", value=ss.model)
    ss.total_steps = st.slider("돌(질문) 개수", min_value=5, max_value=12, value=int(ss.total_steps))

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔄 전체 초기화", use_container_width=True):
            reset_state(keep_settings=True)
            st.rerun()
    with col_b:
        if st.button("🏠 홈으로", use_container_width=True):
            ss.stage = "home"
            ss.current_idx = 1
            ss.summary_md = ""
            st.rerun()

    st.divider()
    st.caption("원칙: 결론/추천 없이 질문으로 사고를 정리합니다.")


# -----------------------------
# Header
# -----------------------------
st.markdown(f"# {APP_TITLE}")
st.caption(APP_TAGLINE)


# -----------------------------
# Home
# -----------------------------
if ss.stage == "home":
    left, right = st.columns([1.2, 1])

    with left:
        st.markdown(
            """
**돌다리**는 중요한 선택 앞에서  
답을 대신 주기보다, **질문을 통해 생각을 정리**하도록 돕는 코칭 앱입니다.

- 한 화면에 **한 질문**
- 답해야 다음 돌로 이동
- 마지막에 **고민의 핵심 / 선택 기준**을 요약 (결론/추천 없음)
"""
        )
        if st.button("🪨 돌다리 건너기 시작", type="primary", use_container_width=True):
            ss.stage = "setup"
            st.rerun()

    with right:
        st.markdown("### 어떻게 진행되나요?")
        for s in ["1) 선택 상황을 적기", "2) 돌(질문)마다 답하기", "3) 요약으로 핵심/기준 정리"]:
            st.write(f"- {s}")


# -----------------------------
# Setup
# -----------------------------
elif ss.stage == "setup":
    st.markdown("## 1) 선택 상황 설정")

    c1, c2 = st.columns([1, 1])

    with c1:
        ss.situation_title = st.text_input(
            "선택 주제(짧게)",
            value=ss.situation_title,
            placeholder="예: 이직을 할지, 현 직장에 남을지",
        )
        ss.user_goal = st.text_input(
            "이번 세션에서 얻고 싶은 것(짧게)",
            value=ss.user_goal,
            placeholder="예: 내 우선순위를 정리하고 기준을 세우고 싶어요",
        )

    with c2:
        ss.situation_detail = st.text_area(
            "상황 설명(조금 더 자세히)",
            value=ss.situation_detail,
            height=160,
            placeholder="예: 현재 조건/제안 조건/걱정되는 점 등",
        )

    can_start = all([safe_strip(ss.situation_title), safe_strip(ss.situation_detail), safe_strip(ss.user_goal)])

    cols = st.columns([1, 1, 2])
    with cols[0]:
        if st.button("⬅️ 돌아가기", use_container_width=True):
            ss.stage = "home"
            st.rerun()

    with cols[1]:
        if st.button("🧱 시작(첫 질문 생성)", type="primary", use_container_width=True, disabled=not can_start):
            if not api_key:
                st.error("사이드바에 OpenAI API Key를 입력해 주세요.")
            else:
                # 초기화
                ss.questions = []
                ss.answers = {}
                ss.current_idx = 1
                ss.summary_md = ""
                # 위젯 키 초기화
                for k in list(st.session_state.keys()):
                    if str(k).startswith("answer_"):
                        del st.session_state[k]

                with st.spinner("첫 질문을 만드는 중..."):
                    client = get_client(api_key)
                    system, user = build_next_question_prompt(
                        ss.situation_title,
                        ss.situation_detail,
                        ss.user_goal,
                        int(ss.total_steps),
                        questions_so_far=[],
                        answers_so_far={},
                    )
                    raw = call_openai_text(client, ss.model, system, user, temperature=0.3)
                    parsed = extract_first_json(raw)
                    q = None
                    if isinstance(parsed, dict):
                        q = safe_strip(str(parsed.get("question", "")))
                    if not q:
                        q = "지금 가장 크게 흔들리는 감정은?"

                    ss.questions = [q]
                    ss.stage = "questions"
                    st.rerun()


# -----------------------------
# Questions
# -----------------------------
elif ss.stage == "questions":
    total_steps = int(ss.total_steps)
    idx = int(ss.current_idx)
    idx = max(1, min(idx, max(1, len(ss.questions))))

    st.markdown("## 2) 질문에 답하며 건너기")
    st.progress(min(idx, total_steps) / total_steps)
    st.caption(f"돌 {idx} / {total_steps}  ·  질문 하나 = 돌 하나")

    q = ss.questions[idx - 1]
    st.markdown(f"### 🪨 {q}")

    # ✅ 핵심 수정: 질문별로 key 분리
    widget_key = f"answer_{idx}"
    # 해당 질문의 저장된 답변이 있으면 초기값으로 동기화(처음만)
    if widget_key not in st.session_state:
        st.session_state[widget_key] = ss.answers.get(idx, "")

    answer = st.text_area(
        "내 답변",
        key=widget_key,
        height=140,
        placeholder="떠오르는 대로 적어도 괜찮아요.",
    )

    # 항상 세션 답변 저장
    ss.answers[idx] = answer

    nav1, nav2, nav3, nav4 = st.columns([1, 1, 2, 2])

    with nav1:
        if st.button("⬅️ 이전", use_container_width=True, disabled=(idx == 1)):
            ss.current_idx = idx - 1
            st.rerun()

    with nav2:
        must_answer = len(safe_strip(answer)) == 0

        # 다음 질문으로
        if idx < total_steps:
            if st.button("다음 ➡️", type="primary", use_container_width=True, disabled=must_answer):
                # 아직 다음 질문이 생성되지 않았다면(처음 도달) -> 동적으로 생성
                if idx == len(ss.questions):
                    if not api_key:
                        st.error("사이드바에 OpenAI API Key를 입력해 주세요.")
                    else:
                        with st.spinner("다음 질문을 만드는 중... (이전 답변 반영)"):
                            client = get_client(api_key)
                            system, user = build_next_question_prompt(
                                ss.situation_title,
                                ss.situation_detail,
                                ss.user_goal,
                                total_steps,
                                questions_so_far=ss.questions,
                                answers_so_far=ss.answers,
                            )
                            raw = call_openai_text(client, ss.model, system, user, temperature=0.35)
                            parsed = extract_first_json(raw)
                            next_q = None
                            if isinstance(parsed, dict):
                                next_q = safe_strip(str(parsed.get("question", "")))

                            if not next_q:
                                # 안전한 폴백
                                fallback = [
                                    "결정에 영향을 주는 현실 조건은?",
                                    "가장 중요한 가치/우선순위는?",
                                    "가능한 선택지들을 넓게 적어보면?",
                                    "각 선택의 리스크/기회비용은?",
                                    "어떤 기준이면 후회를 줄일까?",
                                    "작게 실험해볼 다음 행동은?",
                                ]
                                next_q = fallback[min(idx - 1, len(fallback) - 1)]

                            ss.questions.append(next_q)

                ss.current_idx = idx + 1

                # ✅ 다음 질문칸이 이전 답변으로 안 채워지도록
                # 새 질문으로 이동하는 순간 그 질문 key가 없으면 빈 값으로 시작하도록 유지
                # (위에서 key 없을 때만 answers.get(idx,"")를 넣으므로, 새 질문은 자동으로 빈 값)
                st.rerun()
        else:
            # 마지막 단계: 요약 생성
            if st.button("✅ 건너기 완료", type="primary", use_container_width=True, disabled=must_answer):
                if not api_key:
                    st.error("사이드바에 OpenAI API Key를 입력해 주세요.")
                else:
                    with st.spinner("생각을 정리하는 중... (요약 생성)"):
                        client = get_client(api_key)
                        system, user = build_summary_prompt(
                            ss.situation_title,
                            ss.situation_detail,
                            ss.user_goal,
                            ss.questions,
                            ss.answers,
                        )
                        summary = call_openai_text(client, ss.model, system, user, temperature=0.4)
                        ss.summary_md = summary.strip()
                        ss.stage = "result"
                        st.rerun()

    with nav3:
        with st.expander("📌 지금까지 답변 보기"):
            st.markdown(answers_as_bullets(ss.answers) or "- (아직 없음)")

    with nav4:
        if st.button("🧨 세션 초기화", use_container_width=True):
            reset_state(keep_settings=True)
            ss.stage = "home"
            st.rerun()


# -----------------------------
# Result
# -----------------------------
elif ss.stage == "result":
    st.markdown("## 3) 결과: 생각 정리")

    col1, col2, col3 = st.columns(3)

    md = ss.summary_md or ""
    parts = re.split(r"^##\s+", md, flags=re.MULTILINE)
    sections: Dict[str, str] = {}
    for p in parts:
        p = p.strip()
        if not p:
            continue
        lines = p.splitlines()
        title = lines[0].strip()
        body = "\n".join(lines[1:]).strip()
        sections[title] = body

    def render_card(container, title: str, body: str):
        with container:
            st.markdown(
                f"""
<div style="padding:16px;border:1px solid rgba(49,51,63,0.2);border-radius:16px;">
<h4 style="margin:0 0 8px 0;">{title}</h4>
</div>
""",
                unsafe_allow_html=True,
            )
            st.markdown(body if body else "_(내용이 비어있어요)_")

    render_card(col1, "고민의 핵심", sections.get("고민의 핵심", ""))
    render_card(col2, "선택 기준 요약", sections.get("선택 기준 요약", ""))
    render_card(col3, "코칭 메시지", sections.get("생각을 정리해주는 코칭 메시지", ""))

    st.divider()

    with st.expander("🧾 전체 요약(원문) 보기"):
        st.markdown(ss.summary_md or "_(요약이 없습니다)_")

    with st.expander("🪨 질문/답변 전체 보기"):
        for i, q in enumerate(ss.questions, start=1):
            a = ss.answers.get(i, "")
            st.markdown(f"**Q{i}. {q}**")
            st.write(a if a else "(무응답)")
            st.write("---")

    export_text = (
        f"[선택 주제]\n{ss.situation_title}\n\n"
        f"[상황 설명]\n{ss.situation_detail}\n\n"
        f"[사용자가 얻고 싶은 것]\n{ss.user_goal}\n\n"
        f"[Q&A]\n{answers_as_bullets(ss.answers)}\n\n"
        f"[요약]\n{ss.summary_md}\n"
    )
    st.download_button(
        "⬇️ 결과 텍스트 다운로드",
        data=export_text.encode("utf-8"),
        file_name="돌다리_결정코칭_결과.txt",
        mime="text/plain",
        use_container_width=True,
    )

    cta1, cta2 = st.columns([1, 1])
    with cta1:
        if st.button("🔁 같은 주제로 다시(처음부터)", use_container_width=True):
            ss.stage = "setup"
            ss.questions = []
            ss.answers = {}
            ss.current_idx = 1
            ss.summary_md = ""
            for k in list(st.session_state.keys()):
                if str(k).startswith("answer_"):
                    del st.session_state[k]
            st.rerun()
    with cta2:
        if st.button("🏠 홈으로", use_container_width=True):
            ss.stage = "home"
            ss.current_idx = 1
            ss.summary_md = ""
            st.rerun()


st.caption("© 돌다리 — 결론 대신 질문으로 생각을 정리합니다.")

