# app.py
# ─────────────────────────────────────────────────────────────
# 🪨 돌멩이 결정 코치 (Pebble Decision Coach) — Streamlit Cloud 안정 버전
#
# ✅ 수정 포인트 (이번 에러 해결)
# - SVG 바이트를 st.image()에 직접 넣으면 PIL이 열려다 실패할 수 있음
# - 따라서 SVG는 base64로 인코딩해서 <img src="data:image/svg+xml;base64,..."> 로 렌더링
#
# ✅ 기능
# - 고민 범위 좁히기(카테고리/결정유형)
# - 코치 3종 컨셉 강하게 구분(논리/가치·감정/실행)
# - 질문 단계부터 돌멩이 UI 적극 활용(단계 진행, 반짝/광택)
# - 이전 답변 기억하고 다음 질문에 반영
# - 최종 리포트 코치별 형식 다르게 출력
# - OpenAI: st.secrets["OPENAI_API_KEY"] 우선, 없으면 사이드바 입력
# - Responses API → ChatCompletions fallback, 모델 fallback
#
# 필요 패키지:
#   pip install streamlit openai
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import base64
import json
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
        "tagline": "정보를 구조화해서 결정을 '명료'하게 만드는 코치핑",
        "style": "논리적/간결/프레임워크 중심",
        "method": [
            "핵심 쟁점·제약조건 정의",
            "옵션/기준/가중치 정리",
            "장단점·리스크·가정 검증",
            "결론 + 선택 근거",
        ],
        "prompt_hint": "MECE, 의사결정 기준표, 리스크/가정 검증 질문을 잘 씀",
    },
    {
        "id": "value",
        "name": "💗 가치/감정 코치",
        "tagline": "내가 '왜 흔들리는지'를 찾아 기준을 세워주는 코치핑",
        "style": "공감/가치관/감정 명료화",
        "method": [
            "감정/두려움/기대 분해",
            "진짜 원하는 것(가치) 발굴",
            "후회 최소화 관점(미래의 나) 질문",
            "나답게 선택하는 문장 만들기",
        ],
        "prompt_hint": "감정 라벨링, 가치 우선순위, 후회 테스트 질문을 잘 씀",
    },
    {
        "id": "action",
        "name": "⚔️ 실행 코치",
        "tagline": "결정을 '행동'으로 바꾸는 코치핑 (실험·다음 스텝)",
        "style": "구체적/실행/작은 실험",
        "method": [
            "오늘~7일 안에 할 수 있는 실험 설계",
            "최소 행동(15분) + 체크리스트",
            "장애물/대응계획(If-Then)",
            "실행 후 리뷰 질문",
        ],
        "prompt_hint": "작은 실험, 일정/루틴, 장애물 대응을 매우 구체화",
    },
]

STEPS = ["주제 선택", "고민 정리(1)", "고민 정리(2)", "기준·옵션", "최종 정리"]


# =========================
# Pebble (Rock) UI: SVG → base64 HTML img
# =========================
def _pebble_svg(fill: str, shine: str, stroke: str = "#3a3a3a") -> str:
    # 저작권 이슈 없는 오리지널 SVG 도형
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
        # 진행될수록 밝아지기
        fill = "#5f6672" if p < 0.25 else "#707888" if p < 0.5 else "#8892a6" if p < 0.75 else "#a6b2c8"
        shine = "#aab8ff" if p < 0.25 else "#c8d3ff" if p < 0.5 else "#e3e8ff" if p < 0.75 else "#ffffff"

    svg = _pebble_svg(fill=fill, shine=shine)
    b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return b64


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
# OpenAI Helpers (robust + debug)
# =========================
def get_api_key() -> str:
    # Streamlit Cloud: secrets 우선
    try:
        k = st.secrets.get("OPENAI_API_KEY", "")  # type: ignore
        if k:
            return str(k).strip()
    except Exception:
        pass
    return str(st.session_state.get("openai_api_key_input", "")).strip()


def get_client(api_key: str) -> "OpenAI":
    if OpenAI is None:
        raise RuntimeError("openai 패키지가 설치되어 있지 않아요. `pip install openai` 해주세요.")
    return OpenAI(api_key=api_key)


def call_openai_text(system: str, user: str, temperature: float = 0.7) -> Tuple[Optional[str], Optional[str], List[str]]:
    debug: List[str] = []
    api_key = get_api_key()
    if not api_key:
        return None, "OpenAI API Key가 필요해요. Secrets에 OPENAI_API_KEY를 넣거나 사이드바에 입력해 주세요.", debug

    try:
        client = get_client(api_key)
    except Exception as e:
        return None, str(e), debug

    # 1) Responses API
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

    # 2) Chat Completions fallback
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

    return None, "OpenAI 호출에 실패했어요. 아래 디버그 로그를 확인해 주세요.", debug


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
        st.session_state.answers = []  # [{"q","a","ts"}]
    if "generated_questions" not in st.session_state:
        st.session_state.generated_questions = {}  # step->question

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
# Prompting (Coach differentiation)
# =========================
def system_prompt_for(coach: Dict[str, Any]) -> str:
    if coach["id"] == "logic":
        return (
            "너는 '논리 코치'야. 목표는 사용자의 고민을 의사결정 문제로 구조화하는 것.\n"
            "- 반드시: 쟁점/옵션/기준/제약/가정/리스크를 분리해서 다루기\n"
            "- 질문은 짧고, 답변을 표/목록으로 만들기 쉬운 형태로 묻기\n"
            "- 감정 공감은 짧게만, 구조화가 최우선\n"
            "- 말투는 깔끔하고 단호하지만 공격적이지 않게\n"
        )
    if coach["id"] == "value":
        return (
            "너는 '가치/감정 코치'야. 목표는 감정과 가치관을 명료화해서 '나다운 선택'을 돕는 것.\n"
            "- 반드시: 감정 라벨링 + 그 감정의 근원(욕구/두려움)을 탐색\n"
            "- 가치(중요한 것)를 3개로 좁히고, 후회 최소화 관점 질문 포함\n"
            "- 말투는 따뜻하고 공감적. 사용자가 스스로 말로 정리하게 유도\n"
        )
    return (
        "너는 '실행 코치'야. 목표는 결정을 실행 가능한 실험과 다음 행동으로 바꾸는 것.\n"
        "- 반드시: 7일 안에 할 수 있는 작은 실험 1~2개 설계\n"
        "- 장애물(시간/돈/심리)과 If-Then 대응을 묻기\n"
        "- 말투는 에너지 있고 구체적. 체크리스트/일정 표현을 잘 쓰기\n"
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


def question_instruction(step_idx: int, coach: Dict[str, Any]) -> str:
    if step_idx == 1:
        return "사용자의 고민을 한 문단으로 '상황' 중심으로 설명하게 만드는 1개의 질문을 만들어."
    if step_idx == 2:
        return "사용자의 '원하는 결과/두려운 결과/가장 중요한 제약'을 드러내는 1개의 질문을 만들어."
    if step_idx == 3:
        if coach["id"] == "logic":
            return "옵션을 2~4개로 나누고 평가 기준 3개를 뽑게 하는 질문 1개를 만들어. 답은 표로 만들기 좋게."
        if coach["id"] == "value":
            return "가치 우선순위(상위 3개)와 후회 테스트(1년 뒤/5년 뒤)를 하게 하는 질문 1개를 만들어."
        return "이번 주에 할 수 있는 '작은 실험'을 고르게 하는 질문 1개를 만들어. (예: 15분 행동/하루 체크)"
    if coach["id"] == "logic":
        return "결정을 내리기 전 마지막 확인 질문 1개(가정 검증/리스크 대비)를 만들어."
    if coach["id"] == "value":
        return "결정 문장을 한 줄로 만들게 하는 질문 1개(‘나는 ___를 위해 ___을 선택한다’)를 만들어."
    return "실행 약속을 구체화하는 질문 1개(언제/어디서/무엇을/막히면 어떻게)를 만들어."


def generate_next_question(step_idx: int) -> Tuple[Optional[str], Optional[str], List[str]]:
    coach = coach_by_id(st.session_state.coach_id)
    system = system_prompt_for(coach)

    user = textwrap.dedent(f"""
    너는 사용자의 생각을 정리하기 위한 '단 하나의 질문'을 만든다.

    규칙:
    - 질문은 1개만 출력 (설명 금지)
    - 질문은 한국어
    - 너무 광범위하지 않게, 지금 단계 목적에 맞게 구체적으로
    - 사용자가 답하기 쉽게 예시(괄호 1줄)는 허용

    {build_context_block()}

    [이번 단계 목적]
    {question_instruction(step_idx, coach)}

    이제 질문 1개만 출력해.
    """).strip()

    return call_openai_text(system=system, user=user, temperature=0.7)


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
- 버릴 수 있는 것 1가지: ...

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
아래 Q/A를 바탕으로, 코치 역할에 맞는 '최종 정리 리포트'를 작성해.

규칙:
- 한국어
- 사용자에게 선택을 강요하지 말고, 근거와 다음 스텝을 명확히
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
    st.header("🪨 돌멩이 설정")
    st.text_input("OpenAI API Key (Secrets 우선)", type="password", key="openai_api_key_input")

    st.divider()
    st.subheader("🧭 고민 범위 좁히기")
    st.selectbox("고민 카테고리", [x[0] for x in TOPIC_CATEGORIES], key="category")
    st.selectbox("결정 유형", DECISION_TYPES, key="decision_type")

    st.divider()
    st.subheader("🧑‍🏫 결정 코치 선택")
    coach_labels = [f"{c['name']} — {c['tagline']}" for c in COACHES]
    current_idx = next((i for i, c in enumerate(COACHES) if c["id"] == st.session_state.coach_id), 0)
    picked = st.radio("코치", coach_labels, index=current_idx)
    st.session_state.coach_id = COACHES[coach_labels.index(picked)]["id"]

    coach = coach_by_id(st.session_state.coach_id)
    with st.expander("코치가 어떻게 도와주핑?"):
        st.markdown(f"**{coach['name']}**  \n_{coach['style']}_")
        st.markdown("**진행 방식**")
        for m in coach["method"]:
            st.write(f"- {m}")
        st.caption(f"특징: {coach['prompt_hint']}")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔄 새 고민 시작", use_container_width=True):
            reset_flow()
            st.rerun()
    with c2:
        disabled_next = st.session_state.step >= (len(STEPS) - 1)
        if st.button("🪨 다음 단계", use_container_width=True, disabled=disabled_next):
            st.session_state.step += 1
            st.rerun()

st.title("🪨 돌멩이 결정 코치")
st.caption("질문을 따라가면 고민이 정리되고, 돌멩이가 반짝일수록 결론이 또렷해져요 ✨")

# Step pebbles row (NO PIL)
render_pebble_row(st.session_state.step, len(STEPS))

# Hero pebble (NO PIL)
progress = st.session_state.step / (len(STEPS) - 1)
st.columns([1, 2, 1])[1].container()
with st.columns([1, 2, 1])[1]:
    render_hero_pebble(progress, f"현재 단계: **{STEPS[st.session_state.step]}** · 진행도: **{int(progress*100)}%**")

st.divider()

coach = coach_by_id(st.session_state.coach_id)

# Step 0
if st.session_state.step == 0:
    st.subheader("🧭 먼저 고민을 '작게' 만들기")
    cat_desc = next((d for n, d in TOPIC_CATEGORIES if n == st.session_state.category), "")
    st.info(f"**카테고리:** {st.session_state.category}\n\n{cat_desc}")
    st.success("좋아요! 사이드바에서 **‘다음 단계’**를 눌러 질문을 시작해봐요 🪨")

else:
    step_idx = st.session_state.step

    # Generate question (cached)
    if step_idx not in st.session_state.generated_questions:
        q, err, dbg = generate_next_question(step_idx)
        st.session_state.debug_log = dbg
        if q:
            st.session_state.generated_questions[step_idx] = q
        else:
            st.error(err or "질문 생성 실패")
            with st.expander("🔧 디버그 로그"):
                st.write(dbg)
            st.stop()

    question = st.session_state.generated_questions[step_idx]

    with st.container(border=True):
        st.markdown(f"### 🪨 질문 {step_idx} (코치: {coach['name']})")
        st.markdown(f"**Q. {question}**")
        st.caption("짧게 적어도 괜찮아요. 핵심만 적어도 돌멩이가 다듬어져요 🪨")

    # Answer form (clear_on_submit=True)
    with st.form(f"answer_form_{step_idx}", clear_on_submit=True):
        hint = ""
        if st.session_state.answers:
            last_a = st.session_state.answers[-1]["a"]
            hint = f"이전 답 참고: {last_a[:90]}{'…' if len(last_a) > 90 else ''}"
        answer = st.text_area("📝 내 답변", placeholder=hint or "예) 상황/원하는 결과/제약을 적어줘요", height=140)
        submitted = st.form_submit_button("✅ 답변 저장하고 다음으로", use_container_width=True)

    if submitted:
        if not answer.strip():
            st.warning("답변이 비어있어요. 한 줄만 적어도 괜찮아요핑!")
        else:
            add_answer(question, answer.strip())
            st.success("저장 완료! 돌멩이가 더 반짝였어요 ✨")
            if st.session_state.step < len(STEPS) - 1:
                st.session_state.step += 1
            st.rerun()

    st.subheader("📚 지금까지의 답변(기억하고 있어요)")
    if not st.session_state.answers:
        st.caption("아직 답변이 없어요.")
    else:
        for i, qa in enumerate(st.session_state.answers, start=1):
            with st.expander(f"🪨 Q{i}. {qa['q']}"):
                st.write(qa["a"])
                st.caption(qa["ts"])

    # Final step report
    if st.session_state.step == len(STEPS) - 1:
        st.divider()
        st.subheader("🧾 최종 정리 리포트(돌멩이 윤내기)")

        gen = st.button("✨ 최종 리포트 생성", type="primary", use_container_width=True)
        if gen:
            with st.spinner("돌멩이에 윤을 내는 중…(리포트 생성)"):
                report, err, dbg = generate_final_report()
                st.session_state.debug_log = dbg
                if report:
                    st.session_state.final_report = report
                else:
                    st.session_state.final_report = None
                    st.error(err or "리포트 생성 실패")

        if st.session_state.final_report:
            st.success("완료! 돌멩이가 반짝반짝 윤이 났어요 ✨")
            st.markdown(st.session_state.final_report)

            st.markdown("### 📌 공유용 요약(JSON)")
            share = {
                "category": st.session_state.category,
                "decision_type": st.session_state.decision_type,
                "coach": coach["name"],
                "answers": st.session_state.answers,
                "final_report": st.session_state.final_report,
            }
            st.code(json.dumps(share, ensure_ascii=False, indent=2), language="json")

    with st.expander("🔧 디버그 로그 (문제 생기면 복사해서 보내줘요)"):
        st.write(st.session_state.debug_log)

st.divider()
with st.expander("✅ Streamlit Cloud 배포 체크리스트"):
    st.markdown(
        """
- **Secrets 설정**: Streamlit Cloud → Settings → Secrets에 아래 추가  
  - `OPENAI_API_KEY = "sk-..."`

- **requirements.txt** 예시
  - `streamlit`
  - `openai`

- 모델 권한 문제면 앱이 **gpt-4o-mini로 자동 재시도**해요.
"""
    )

