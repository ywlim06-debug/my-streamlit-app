# app.py
# ─────────────────────────────────────────────────────────────
# 🪨 돌멩이 결정 코치 (Pebble Decision Coach) — Clean Tone Version
#
# Fix: 마지막 질문이 이전 질문과 반복되는 현상 해결
# - 프롬프트에 "이전 질문과 중복 금지" + 이전 질문 목록 제공
# - 중복 감지 시 자동 재생성 1회(랜덤 nonce 추가)
# - 그래도 중복이면 코치별 '안전 마지막 질문'으로 대체
#
# 필요 패키지:
#   pip install streamlit openai
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import base64
import json
import random
import textwrap
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore


# =========================
# Page Config
# =========================
st.set_page_config(page_title="돌멩이 결정 코치", page_icon="🪨", layout="wide")

MODEL_PRIMARY = "gpt-5-mini"
MODEL_FALLBACK = "gpt-4o-mini"

TOPIC_CATEGORIES = [
    ("🎓 학업/진로", "학업, 전공 선택, 진로 방향, 취업/이직, 목표 설정"),
    ("💼 커리어/일", "업무 선택, 프로젝트, 협업, 리더십, 커리어 성장"),
    ("💖 관계", "친구/연인/가족, 갈등, 소통, 거리두기, 선택의 기준"),
    ("💰 돈/소비", "예산, 소비 습관, 투자/저축, 큰 구매 결정"),
    ("🧠 마음/삶", "불안/번아웃, 가치관, 인생 방향, 루틴/균형"),
    ("📦 기타", "정리되지 않은 고민, 일상 선택, 기타"),
]

DECISION_TYPES = [
    "A vs B 선택(둘 중 하나)",
    "여러 옵션 중 선택",
    "해야 할지 말지(Yes/No)",
    "언제/어떻게 할지(전략/시점)",
    "갈등 해결/대화 방향",
]

COACHES = [
    {
        "id": "logic",
        "name": "🔎 논리 코치",
        "tagline": "정보를 구조화해서 결정을 명료하게 돕습니다",
        "style": "논리적/간결/프레임워크 중심",
        "method": [
            "핵심 쟁점·제약조건 정의",
            "옵션/기준/가중치 정리",
            "장단점·리스크·가정 검증",
            "결론 + 선택 근거",
        ],
        "prompt_hint": "MECE, 의사결정 기준표, 리스크/가정 검증 질문",
    },
    {
        "id": "value",
        "name": "💗 가치/감정 코치",
        "tagline": "감정과 가치관을 명료화해 ‘나다운 선택’을 돕습니다",
        "style": "공감/가치관/감정 명료화",
        "method": [
            "감정/두려움/기대 분해",
            "진짜 원하는 것(가치) 발굴",
            "후회 최소화 관점(미래의 나) 질문",
            "나답게 선택하는 문장 만들기",
        ],
        "prompt_hint": "감정 라벨링, 가치 우선순위, 후회 테스트",
    },
    {
        "id": "action",
        "name": "⚔️ 실행 코치",
        "tagline": "결정을 실행 가능한 행동으로 바꿉니다",
        "style": "구체적/실행/작은 실험",
        "method": [
            "7일 안에 할 수 있는 실험 설계",
            "최소 행동(15분) + 체크리스트",
            "장애물/대응계획(If-Then)",
            "실행 후 리뷰 질문",
        ],
        "prompt_hint": "작은 실험, 일정/루틴, 장애물 대응",
    },
]

STEPS = ["주제 선택", "고민 정리(1)", "고민 정리(2)", "기준·옵션", "최종 정리"]


# =========================
# Pebble (Rock) UI: SVG → base64 HTML img
# =========================
def _pebble_svg(fill: str, shine: str, stroke: str = "#3a3a3a") -> str:
    return f"""
<svg width="160" height="120" viewBox="0 0 160 120" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="pebble">
  <defs>
    <radialGradient id="g" cx="35%" cy="25%" r="80%">
      <stop offset="0%" stop-color="{shine}" stop-opacity="0.95"/>
      <stop offset="55%" stop-color="{fill}" stop-opacity="1"/>
      <stop offset="100%" stop-color="{fill}" stop-opacity="1"/>
    </radialGradient>
  </defs>
  <path d="M28 80
           C14 60, 20 30, 50 20
           C66 14, 96 12, 114 24
           C142 44, 150 68, 130 88
           C112 108, 54 110, 28 80 Z"
        fill="url(#g)" stroke="{stroke}" stroke-width="2" />
  <path d="M54 30 C68 22, 82 22, 94 30"
        fill="none" stroke="{shine}" stroke-width="7" stroke-linecap="round" opacity="0.55"/>
</svg>
""".strip()


def pebble_svg_b64(progress_0_to_1: float, inactive: bool = False) -> str:
    p = max(0.0, min(1.0, float(progress_0_to_1)))
    if inactive:
        fill = "#2f3136"
        shine = "#6b6f7a"
    else:
        fill = "#5f6672" if p < 0.25 else "#707888" if p < 0.5 else "#8892a6" if p < 0.75 else "#a6b2c8"
        shine = "#aab8ff" if p < 0.25 else "#c8d3ff" if p < 0.5 else "#e3e8ff" if p < 0.75 else "#ffffff"
    svg = _pebble_svg(fill=fill, shine=shine)
    return base64.b64encode(svg.encode("utf-8")).decode("ascii")


def render_pebble_row(step_idx: int, total: int) -> None:
    cols = st.columns(total)
    for i in range(total):
        active = i <= step_idx
        p = (i + 1) / total
        b64 = pebble_svg_b64(p, inactive=not active)
        html = f"""
        <div style="text-align:center;">
          <img src="data:image/svg+xml;base64,{b64}" style="width:100%; max-width:150px;"/>
          <div style="font-size:12px; margin-top:4px; opacity:{1.0 if active else 0.55};">
            {STEPS[i]}
          </div>
        </div>
        """
        cols[i].markdown(html, unsafe_allow_html=True)


def render_hero_pebble(progress: float, label: str) -> None:
    b64 = pebble_svg_b64(progress, inactive=False)
    html = f"""
    <div style="text-align:center;">
      <img src="data:image/svg+xml;base64,{b64}" style="width:100%; max-width:240px;"/>
      <div style="margin-top:6px; font-size:14px;">
        {label}
      </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# =========================
# OpenAI Helpers
# =========================
def get_api_key() -> str:
    try:
        k = st.secrets.get("OPENAI_API_KEY", "")  # type: ignore
        if k:
            return str(k).strip()
    except Exception:
        pass
    return str(st.session_state.get("openai_api_key_input", "")).strip()


def get_client(api_key: str) -> "OpenAI":
    if OpenAI is None:
        raise RuntimeError("openai 패키지가 설치되어 있지 않습니다. `pip install openai`를 실행하세요.")
    return OpenAI(api_key=api_key)


def call_openai_text(system: str, user: str, temperature: float = 0.7) -> Tuple[Optional[str], Optional[str], List[str]]:
    debug: List[str] = []
    api_key = get_api_key()
    if not api_key:
        return None, "OpenAI API Key가 필요합니다. Secrets에 OPENAI_API_KEY를 넣거나 사이드바에 입력하세요.", debug

    try:
        client = get_client(api_key)
    except Exception as e:
        return None, str(e), debug

    if hasattr(client, "responses"):
        for model in [MODEL_PRIMARY, MODEL_FALLBACK]:
            try:
                debug.append(f"Responses API / model={model}")
                resp = client.responses.create(
                    model=model,
                    input=[
                        {"role": "system", "content": [{"type": "text", "text": system}]},
                        {"role": "user", "content": [{"type": "text", "text": user}]},
                    ],
                    temperature=temperature,
                )
                if getattr(resp, "output_text", None):
                    return str(resp.output_text).strip(), None, debug

                out_texts: List[str] = []
                for item in getattr(resp, "output", []) or []:
                    for c in getattr(item, "content", []) or []:
                        if getattr(c, "type", None) == "output_text":
                            out_texts.append(getattr(c, "text", ""))
                text = "\n".join([t for t in out_texts if t]).strip()
                if text:
                    return text, None, debug
                raise RuntimeError("응답 텍스트 추출 실패")
            except Exception as e:
                debug.append(f"Responses failed: {type(e).__name__}: {e}")

    for model in [MODEL_PRIMARY, MODEL_FALLBACK]:
        try:
            debug.append(f"Chat Completions / model={model}")
            cc = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=temperature,
            )
            text = ""
            if cc.choices:
                text = (cc.choices[0].message.content or "").strip()
            if text:
                return text, None, debug
            raise RuntimeError("빈 응답")
        except Exception as e:
            debug.append(f"Chat failed: {type(e).__name__}: {e}")

    return None, "OpenAI 호출에 실패했습니다. 아래 디버그 로그를 확인하세요.", debug


# =========================
# State
# =========================
def init_state() -> None:
    if "step" not in st.session_state:
        st.session_state.step = 0
    if "category" not in st.session_state:
        st.session_state.category = TOPIC_CATEGORIES[0][0]
    if "decision_type" not in st.session_state:
        st.session_state.decision_type = DECISION_TYPES[0]
    if "coach_id" not in st.session_state:
        st.session_state.coach_id = COACHES[0]["id"]

    if "answers" not in st.session_state:
        st.session_state.answers = []
    if "generated_questions" not in st.session_state:
        st.session_state.generated_questions = {}

    if "final_report" not in st.session_state:
        st.session_state.final_report = None
    if "debug_log" not in st.session_state:
        st.session_state.debug_log = []
    if "openai_api_key_input" not in st.session_state:
        st.session_state.openai_api_key_input = ""


def coach_by_id(coach_id: str) -> Dict[str, Any]:
    for c in COACHES:
        if c["id"] == coach_id:
            return c
    return COACHES[0]


def add_answer(q: str, a: str) -> None:
    st.session_state.answers.append({"q": q, "a": a, "ts": datetime.now().isoformat(timespec="seconds")})


def reset_flow() -> None:
    st.session_state.step = 0
    st.session_state.answers = []
    st.session_state.generated_questions = {}
    st.session_state.final_report = None
    st.session_state.debug_log = []


# =========================
# Prompting
# =========================
def system_prompt_for(coach: Dict[str, Any]) -> str:
    if coach["id"] == "logic":
        return (
            "당신은 '논리 코치'입니다. 목표는 사용자의 고민을 의사결정 문제로 구조화하는 것입니다.\n"
            "- 쟁점/옵션/기준/제약/가정/리스크를 분리해서 다루세요.\n"
            "- 질문은 짧고, 답변을 표로 만들기 쉬운 형태로 구성하세요.\n"
        )
    if coach["id"] == "value":
        return (
            "당신은 '가치/감정 코치'입니다. 목표는 감정과 가치관을 명료화해 사용자가 '나다운 선택'을 하도록 돕는 것입니다.\n"
            "- 감정 라벨링 + 그 감정의 근원(욕구/두려움)을 탐색하세요.\n"
            "- 가치(중요한 것)를 3개로 좁히고, 후회 최소화 관점 질문을 포함하세요.\n"
        )
    return (
        "당신은 '실행 코치'입니다. 목표는 결정을 실행 가능한 실험과 다음 행동으로 바꾸는 것입니다.\n"
        "- 7일 안에 할 수 있는 작은 실험 1~2개를 설계하게 하세요.\n"
        "- 장애물과 If-Then 대응을 구체화하세요.\n"
    )


def build_context_block() -> str:
    cat = st.session_state.category
    dtype = st.session_state.decision_type
    answers = st.session_state.answers

    hist = ""
    for i, qa in enumerate(answers[-6:], start=1):
        hist += f"{i}) Q: {qa['q']}\n   A: {qa['a']}\n"

    return textwrap.dedent(f"""
    [고민 카테고리]
    {cat}

    [결정 유형]
    {dtype}

    [지금까지의 Q/A (최근 6개)]
    {hist if hist.strip() else "(아직 없음)"}
    """).strip()


def previous_questions_text() -> str:
    # 이미 생성된 질문(캐시)을 단계 순서대로 나열
    items = []
    for k in sorted(st.session_state.generated_questions.keys()):
        items.append(f"- (step {k}) {st.session_state.generated_questions[k]}")
    return "\n".join(items) if items else "(없음)"


def question_instruction(step_idx: int, coach: Dict[str, Any]) -> str:
    if step_idx == 1:
        return "사용자의 고민을 한 문단으로 '상황' 중심으로 설명하게 만드는 질문 1개를 작성하세요."
    if step_idx == 2:
        return "사용자의 '원하는 결과/두려운 결과/가장 중요한 제약'을 드러내는 질문 1개를 작성하세요."
    if step_idx == 3:
        if coach["id"] == "logic":
            return "옵션 2~4개 + 평가 기준 3개를 뽑게 하는 질문 1개를 작성하세요. 답은 표로 만들기 좋게."
        if coach["id"] == "value":
            return "가치 우선순위 상위 3개 + 후회 테스트(1년/5년)를 하게 하는 질문 1개를 작성하세요."
        return "이번 주에 할 수 있는 '작은 실험'을 고르게 하는 질문 1개를 작성하세요. (예: 15분 행동/하루 체크)"
    # step 4
    if coach["id"] == "logic":
        return "결정 전 마지막 검증 질문 1개(가정/리스크/대안)를 작성하세요."
    if coach["id"] == "value":
        return "결정 문장을 한 줄로 만들게 하는 질문 1개(‘나는 ___를 위해 ___을 선택한다’)를 작성하세요."
    return "실행 약속을 고정하는 질문 1개(언제/어디서/무엇을/막히면 어떻게)를 작성하세요."


def normalize_question(s: str) -> str:
    return " ".join((s or "").strip().split())


def fallback_last_question(coach_id: str) -> str:
    if coach_id == "logic":
        return "이 결정을 내리기 전에 확인해야 할 가장 큰 가정 1개와, 그 가정이 틀렸을 때의 대안(플랜B)은 무엇인가요?"
    if coach_id == "value":
        return "‘나는 ___를 위해 ___을 선택한다’ 문장을 완성해보면, 빈칸에는 무엇이 들어가나요?"
    return "이번 주 안에 실행할 첫 행동을 ‘언제/어디서/몇 분/무엇을’ 한 문장으로 적어보면 어떻게 되나요?"


def generate_next_question(step_idx: int) -> Tuple[Optional[str], Optional[str], List[str]]:
    coach = coach_by_id(st.session_state.coach_id)
    system = system_prompt_for(coach)

    prev_qs = previous_questions_text()
    last_q = st.session_state.generated_questions.get(step_idx - 1, "")

    def _prompt(nonce: int) -> str:
        return textwrap.dedent(f"""
        당신은 사용자의 생각을 정리하기 위한 '단 하나의 질문'을 만듭니다.

        규칙:
        - 질문은 1개만 출력 (설명/머리말 금지)
        - 한국어
        - 이전 질문과 동일하거나 매우 유사한 질문은 금지
        - 질문의 초점/관점이 이전 질문과 겹치지 않게 구성
        - 사용자가 답하기 쉽게 예시(괄호 1줄) 허용
        - 금지: 아래 "이전 질문 목록"에 있는 문장을 그대로/유사하게 반복

        [이전 질문 목록]
        {prev_qs}

        [직전 질문(참고)]
        {last_q if last_q else "(없음)"}

        {build_context_block()}

        [이번 단계 목적]
        {question_instruction(step_idx, coach)}

        (nonce={nonce})  # 재생성 시 중복 방지용

        이제 질문 1개만 출력하세요.
        """).strip()

    # 1차 생성
    q1, err, dbg = call_openai_text(system=system, user=_prompt(nonce=random.randint(1000, 9999)), temperature=0.75)
    if not q1:
        return None, err, dbg

    q1n = normalize_question(q1)
    lastn = normalize_question(last_q)

    # 중복/유사(간단판) 감지: 동일 문자열 or 직전 질문이 포함되는 경우
    is_dup = (q1n == lastn) or (lastn and (q1n in lastn or lastn in q1n))
    if not is_dup:
        return q1.strip(), None, dbg

    # 2차 재생성(더 강하게)
    dbg.append("Detected duplicate with previous question. Regenerating once with stronger constraints.")
    q2, err2, dbg2 = call_openai_text(system=system, user=_prompt(nonce=random.randint(10000, 99999)), temperature=0.85)
    dbg.extend(dbg2)
    if q2:
        q2n = normalize_question(q2)
        is_dup2 = (q2n == lastn) or (lastn and (q2n in lastn or lastn in q2n))
        if not is_dup2:
            return q2.strip(), None, dbg

    # 그래도 실패하면 안전 질문으로 대체
    dbg.append("Still duplicated after retry. Using deterministic fallback question.")
    return fallback_last_question(coach["id"]), None, dbg


def generate_final_report() -> Tuple[Optional[str], Optional[str], List[str]]:
    coach = coach_by_id(st.session_state.coach_id)
    system = system_prompt_for(coach)

    if coach["id"] == "logic":
        format_spec = """
출력 형식:
## 한 줄 결론
- 결론: ...

## 의사결정 구조
- 쟁점:
- 옵션(2~4):
- 평가 기준(3~5):
- 제약/가정:
- 리스크/대응:

## 추천안 (근거)
- 추천: ...
- 이유(3줄):
- 보완책(리스크 줄이기):

## 다음 행동(24시간 내)
- ...
"""
    elif coach["id"] == "value":
        format_spec = """
출력 형식:
## 지금의 마음 요약
- 감정(3개): ...
- 진짜 욕구/두려움: ...

## 나의 기준(가치)
- 상위 3가지: ...
- 내려놓을 수 있는 것 1가지: ...

## 나다운 선택 문장
- “나는 ___를 위해 ___을 선택한다.”

## 후회 최소화 체크
- 1년 뒤의 나: ...
- 5년 뒤의 나: ...

## 내일의 작은 약속
- ...
"""
    else:
        format_spec = """
출력 형식:
## 결정을 행동으로 바꾸기
- 이번 주 핵심 목표(1개): ...

## 7일 실험(1~2개)
- 실험1: (15분 단위로 쪼개서)
- 실험2(선택): ...

## If-Then 대응표
- 만약 ___이면 → ___한다 (3개)

## 오늘(24시간 내) 체크리스트
- [ ] ...
- [ ] ...
- [ ] ...

## 리뷰 질문(실험 후)
- ...
"""

    user = textwrap.dedent(f"""
아래 Q/A를 바탕으로, 코치 역할에 맞는 '최종 정리 리포트'를 작성하세요.

규칙:
- 한국어
- 선택을 강요하지 말고, 근거와 다음 스텝을 명확히
- 불확실한 부분은 '추가 확인 질문' 1개를 마지막에 제안
- 길이: 500~900자

{build_context_block()}

{format_spec}

마지막 줄:
- 추가 확인 질문: ...
""").strip()

    return call_openai_text(system=system, user=user, temperature=0.65)


# =========================
# UI
# =========================
init_state()

with st.sidebar:
    st.header("설정")
    st.text_input("OpenAI API Key (Secrets 우선)", type="password", key="openai_api_key_input")

    st.divider()
    st.subheader("고민 범위")
    st.selectbox("고민 카테고리", [x[0] for x in TOPIC_CATEGORIES], key="category")
    st.selectbox("결정 유형", DECISION_TYPES, key="decision_type")

    st.divider()
    st.subheader("코치 선택")
    coach_labels = [f"{c['name']} — {c['tagline']}" for c in COACHES]
    current_idx = next((i for i, c in enumerate(COACHES) if c["id"] == st.session_state.coach_id), 0)
    picked = st.radio("코치", coach_labels, index=current_idx)
    st.session_state.coach_id = COACHES[coach_labels.index(picked)]["id"]

    coach = coach_by_id(st.session_state.coach_id)
    with st.expander("코치 진행 방식"):
        st.markdown(f"**{coach['name']}**  \n_{coach['style']}_")
        for m in coach["method"]:
            st.write(f"- {m}")
        st.caption(f"특징: {coach['prompt_hint']}")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("새 고민 시작", use_container_width=True):
            reset_flow()
            st.rerun()
    with c2:
        disabled_next = st.session_state.step >= (len(STEPS) - 1)
        if st.button("다음 단계", use_container_width=True, disabled=disabled_next):
            st.session_state.step += 1
            st.rerun()

st.title("🪨 돌멩이 결정 코치")
st.caption("질문에 답하며 생각을 정리합니다. 단계가 진행될수록 시각적으로도 진행도가 표시됩니다.")

render_pebble_row(st.session_state.step, len(STEPS))

progress = st.session_state.step / (len(STEPS) - 1)
with st.columns([1, 2, 1])[1]:
    render_hero_pebble(progress, f"현재 단계: {STEPS[st.session_state.step]} · 진행도: {int(progress*100)}%")

st.divider()

coach = coach_by_id(st.session_state.coach_id)

if st.session_state.step == 0:
    st.subheader("1) 먼저 고민을 구체화합니다")
    cat_desc = next((d for n, d in TOPIC_CATEGORIES if n == st.session_state.category), "")
    st.info(f"**카테고리:** {st.session_state.category}\n\n{cat_desc}")
    st.success("사이드바에서 ‘다음 단계’를 눌러 질문을 시작하세요.")

else:
    step_idx = st.session_state.step

    if step_idx not in st.session_state.generated_questions:
        q, err, dbg = generate_next_question(step_idx)
        st.session_state.debug_log = dbg
        if q:
            st.session_state.generated_questions[step_idx] = q
        else:
            st.error(err or "질문 생성 실패")
            with st.expander("디버그 로그"):
                st.write(dbg)
            st.stop()

    question = st.session_state.generated_questions[step_idx]

    with st.container(border=True):
        st.markdown(f"### 질문 {step_idx} (코치: {coach['name']})")
        st.markdown(f"**Q. {question}**")

    with st.form(f"answer_form_{step_idx}", clear_on_submit=True):
        hint = ""
        if st.session_state.answers:
            last_a = st.session_state.answers[-1]["a"]
            hint = f"이전 답 요약: {last_a[:90]}{'…' if len(last_a) > 90 else ''}"
        answer = st.text_area("답변", placeholder=hint or "여기에 답변을 입력하세요", height=140)
        submitted = st.form_submit_button("답변 저장하고 진행", use_container_width=True)

    if submitted:
        if not answer.strip():
            st.warning("답변이 비어 있습니다. 한 줄만 입력해도 진행 가능합니다.")
        else:
            add_answer(question, answer.strip())
            st.success("저장되었습니다.")
            if st.session_state.step < len(STEPS) - 1:
                st.session_state.step += 1
            st.rerun()

    st.subheader("답변 기록")
    if not st.session_state.answers:
        st.caption("아직 답변이 없습니다.")
    else:
        for i, qa in enumerate(st.session_state.answers, start=1):
            with st.expander(f"Q{i}. {qa['q']}"):
                st.write(qa["a"])
                st.caption(qa["ts"])

    if st.session_state.step == len(STEPS) - 1:
        st.divider()
        st.subheader("최종 정리 리포트")

        gen = st.button("최종 리포트 생성", type="primary", use_container_width=True)
        if gen:
            with st.spinner("리포트를 생성하는 중..."):
                report, err, dbg = generate_final_report()
                st.session_state.debug_log = dbg
                if report:
                    st.session_state.final_report = report
                else:
                    st.session_state.final_report = None
                    st.error(err or "리포트 생성 실패")

        if st.session_state.final_report:
            st.success("리포트가 생성되었습니다.")
            st.markdown(st.session_state.final_report)

            st.markdown("공유용 요약(JSON)")
            share = {
                "category": st.session_state.category,
                "decision_type": st.session_state.decision_type,
                "coach": coach["name"],
                "questions": st.session_state.generated_questions,
                "answers": st.session_state.answers,
                "final_report": st.session_state.final_report,
            }
            st.code(json.dumps(share, ensure_ascii=False, indent=2), language="json")

    with st.expander("디버그 로그(문제 발생 시 확인)"):
        st.write(st.session_state.debug_log)

st.divider()
with st.expander("Streamlit Cloud 배포 체크리스트"):
    st.markdown(
        """
- Secrets 설정: Settings → Secrets에 `OPENAI_API_KEY = "sk-..."` 추가
- requirements.txt:
  - streamlit
  - openai
- 모델 권한 문제가 있으면 자동으로 gpt-4o-mini로 재시도합니다.
"""
    )
