# app.py
# ─────────────────────────────────────────────────────────────
# 🪨 돌멩이 결정 코치 (Pebble Decision Coach)
#
# 요청 반영:
# - 질문 개수 설정(2~10)
# - 질문 완료 후 "레포트 페이지"로 이동(session_state 라우팅)
# - 질문 중복 방지(유사하면 1회 재생성 + fallback)
# - 실행 코치 진행 방식 강화:
#   1) 우선순위 정하기(가치/효과/난이도 기준)
#   2) 계획 질문 추가: 년 → 달 → 주(목표를 쪼개는 질문)
#   3) If-Then, 장애물 대응, 7일 실험, 체크리스트
# - Streamlit Cloud: st.secrets["OPENAI_API_KEY"] 우선
# - SVG는 base64 HTML로 렌더링(PIL 오류 방지)
#
# 필요 패키지:
#   pip install streamlit openai
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import base64
import json
import random
import re
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
        "prompt_hint": "MECE, 기준표, 리스크/가정 검증",
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
        "tagline": "결정을 실행 가능한 행동·계획으로 바꿉니다",
        "style": "구체적/우선순위/계획(년→달→주)/작은 실험",
        "method": [
            "우선순위 정하기: 효과/중요도/난이도 기준으로 1~3개 선정",
            "큰 목표를 계획으로 쪼개기: 년 → 달 → 주 단위로 구체화",
            "7일 실험 1~2개 설계(15~30분 단위로 시작)",
            "장애물/대응계획(If-Then) 정리",
            "실행 후 리뷰 질문(무엇이 작동/방해했는가)",
        ],
        "prompt_hint": "우선순위, 로드맵(년→달→주), 7일 실험, If-Then",
    },
]


# =========================
# Pebble UI (SVG -> base64 HTML)
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
        fill, shine = "#2f3136", "#6b6f7a"
    else:
        fill = "#5f6672" if p < 0.25 else "#707888" if p < 0.5 else "#8892a6" if p < 0.75 else "#a6b2c8"
        shine = "#aab8ff" if p < 0.25 else "#c8d3ff" if p < 0.5 else "#e3e8ff" if p < 0.75 else "#ffffff"
    svg = _pebble_svg(fill=fill, shine=shine)
    return base64.b64encode(svg.encode("utf-8")).decode("ascii")


def render_pebble_row(current_idx: int, total: int, labels: List[str]) -> None:
    cols = st.columns(total)
    for i in range(total):
        active = i <= current_idx
        p = (i + 1) / total
        b64 = pebble_svg_b64(p, inactive=not active)
        html = f"""
        <div style="text-align:center;">
          <img src="data:image/svg+xml;base64,{b64}" style="width:100%; max-width:140px;"/>
          <div style="font-size:12px; margin-top:4px; opacity:{1.0 if active else 0.55};">
            {labels[i]}
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
# OpenAI helpers
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
# State + routing
# =========================
def init_state() -> None:
    if "page" not in st.session_state:
        st.session_state.page = "setup"  # setup | questions | report

    if "category" not in st.session_state:
        st.session_state.category = TOPIC_CATEGORIES[0][0]
    if "decision_type" not in st.session_state:
        st.session_state.decision_type = DECISION_TYPES[0]
    if "coach_id" not in st.session_state:
        st.session_state.coach_id = COACHES[0]["id"]

    if "num_questions" not in st.session_state:
        st.session_state.num_questions = 5
    if "q_index" not in st.session_state:
        st.session_state.q_index = 0

    if "questions" not in st.session_state:
        st.session_state.questions = []
    if "answers" not in st.session_state:
        st.session_state.answers = []

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


def reset_flow(to_page: str = "setup") -> None:
    st.session_state.page = to_page
    st.session_state.q_index = 0
    st.session_state.questions = []
    st.session_state.answers = []
    st.session_state.final_report = None
    st.session_state.debug_log = []


def add_answer(q: str, a: str) -> None:
    st.session_state.answers.append({"q": q, "a": a, "ts": datetime.now().isoformat(timespec="seconds")})


# =========================
# Similarity + Question generation
# =========================
def system_prompt_for(coach: Dict[str, Any]) -> str:
    if coach["id"] == "logic":
        return (
            "당신은 '논리 코치'입니다. 목표는 사용자의 고민을 의사결정 문제로 구조화하는 것입니다.\n"
            "- 쟁점/옵션/기준/제약/가정/리스크를 분리해서 다루세요.\n"
            "- 질문은 짧고, 답변을 표/목록으로 만들기 쉬운 형태로 구성하세요.\n"
        )
    if coach["id"] == "value":
        return (
            "당신은 '가치/감정 코치'입니다. 목표는 감정과 가치관을 명료화해 사용자가 '나다운 선택'을 하도록 돕는 것입니다.\n"
            "- 감정 라벨링 + 그 감정의 근원(욕구/두려움)을 탐색하세요.\n"
            "- 가치(중요한 것)를 3개로 좁히고, 후회 최소화 관점 질문을 포함하세요.\n"
        )
    return (
        "당신은 '실행 코치'입니다. 목표는 결정을 실행 가능한 계획과 행동으로 바꾸는 것입니다.\n"
        "- 반드시 우선순위를 정하게 하세요(효과/중요도/난이도 기준).\n"
        "- 큰 목표를 년→달→주 단위로 쪼개 구체화하게 하세요.\n"
        "- 7일 실험 1~2개와 If-Then(장애물 대응)을 포함하세요.\n"
        "- 질문은 짧고, 바로 실행할 수 있는 답을 끌어내는 형태로 구성하세요.\n"
    )


def build_context_block() -> str:
    cat = st.session_state.category
    dtype = st.session_state.decision_type
    hist = ""
    for i, qa in enumerate(st.session_state.answers[-6:], start=1):
        hist += f"{i}) Q: {qa['q']}\n   A: {qa['a']}\n"
    return textwrap.dedent(f"""
    [고민 카테고리]
    {cat}

    [결정 유형]
    {dtype}

    [지금까지의 Q/A (최근 6개)]
    {hist if hist.strip() else "(아직 없음)"}
    """).strip()


def normalize(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def token_overlap(a: str, b: str) -> float:
    def toks(s: str) -> set:
        s = re.sub(r"[^\w가-힣 ]", " ", s)
        s = re.sub(r"\s+", " ", s).strip().lower()
        return set([t for t in s.split(" ") if len(t) >= 2])

    ta, tb = toks(a), toks(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    denom = max(1, min(len(ta), len(tb)))
    return inter / denom


def is_similar(a: str, b: str) -> bool:
    a0, b0 = normalize(a), normalize(b)
    if not a0 or not b0:
        return False
    if a0 == b0:
        return True
    if a0 in b0 or b0 in a0:
        return True
    return token_overlap(a0, b0) >= 0.75


def instruction_for_question(i: int, n: int, coach_id: str) -> str:
    """
    질문을 n개로 늘려도 각 질문이 역할이 겹치지 않도록 설계.
    실행 코치는 우선순위 + 년→달→주 계획 질문이 중간에 반드시 나오도록 구성.
    """
    # 공통 시작 2개
    if i == 0:
        return "상황을 구체적으로 파악하는 질문 1개를 작성하세요."
    if i == 1:
        return "원하는 결과와 피하고 싶은 결과를 분리해 드러내는 질문 1개를 작성하세요."

    # 실행 코치 전용: 우선순위/로드맵 질문을 강제 배치
    if coach_id == "action":
        # i=2: 우선순위
        if i == 2:
            return "해야 할 것(또는 옵션)들을 3~6개로 나열하고, 효과/중요도/난이도로 우선순위 1~3개를 고르게 하는 질문 1개를 작성하세요."
        # i=3: 년→달→주 로드맵
        if i == 3 and n >= 5:
            return "우선순위 1개를 기준으로 목표를 년→달→주로 쪼개 계획을 세우게 하는 질문 1개를 작성하세요."
        # 중반(마지막 2개 이전): 7일 실험/첫 행동
        if i < n - 2:
            return "이번 주에 할 수 있는 7일 실험(1개)과 시작 행동(15~30분)을 구체화하게 하는 질문 1개를 작성하세요."
        # 마지막 2개: 장애물 If-Then, 실행 약속
        if i == n - 2:
            return "장애물 3가지를 예상하고 If-Then(만약~이면→~한다) 대응을 만들게 하는 질문 1개를 작성하세요."
        return "실행 약속을 한 문장으로 고정하게 하는 질문 1개(언제/어디서/몇 분/무엇을) 작성하세요."

    # 논리/가치 코치: 기존 흐름
    if i == 2 and n >= 4:
        return "제약(시간/돈/관계/규칙)과 바꿀 수 없는 조건을 명확히 하는 질문 1개를 작성하세요."

    if i < n - 2:
        if coach_id == "logic":
            return "옵션을 나누고 평가 기준(3~5)을 설정하게 하는 질문 1개를 작성하세요. (표로 정리 가능하게)"
        return "가치 우선순위(상위 3개)와 감정/욕구/두려움을 드러내는 질문 1개를 작성하세요."

    if i == n - 2:
        if coach_id == "logic":
            return "가정/리스크를 검증하고 플랜B를 떠올리게 하는 질문 1개를 작성하세요."
        return "후회 최소화 관점(1년/5년 후)을 점검하는 질문 1개를 작성하세요."

    if coach_id == "logic":
        return "결정을 내리기 위한 최종 확인 질문 1개(가정/리스크/대안 중 하나에 초점)를 작성하세요."
    return "결정 문장을 한 줄로 완성하게 하는 질문 1개를 작성하세요. (나는 ___를 위해 ___을 선택한다)"


def fallback_question(coach_id: str, i: int, n: int) -> str:
    if i == 0:
        return "지금 고민 상황을 한 문단으로 설명해 주세요. (무슨 일이 있었고, 무엇을 결정해야 하나요?)"
    if i == 1:
        return "이 결정에서 얻고 싶은 최선의 결과 1가지와 피하고 싶은 최악의 결과 1가지는 무엇인가요?"

    if coach_id == "action":
        if i == 2:
            return "해야 할 일(또는 옵션)을 3~6개 적고, 그중 가장 효과가 큰 1~3개를 우선순위로 고르면 무엇인가요?"
        if i == 3 and n >= 5:
            return "우선순위 1개를 ‘1년 목표 → 이번 달 목표 → 이번 주 할 일’로 쪼개면 각각 무엇인가요?"
        if i == n - 2:
            return "이번 주 실행을 방해할 장애물 3가지를 적고, 각각에 대해 ‘만약 ~이면 → ~한다’로 대응을 만들어보면요?"
        if i == n - 1:
            return "이번 주 첫 행동을 ‘언제/어디서/몇 분/무엇을’ 한 문장으로 적어 주세요."
        return "이번 주에 할 7일 실험 1개와, 오늘 15~30분 안에 할 시작 행동은 무엇인가요?"

    # logic/value 공통 fallback
    if i == 2 and n >= 4:
        return "시간/돈/관계/규칙 측면에서 바꿀 수 없는 제약 2가지는 무엇인가요?"
    if i == n - 1:
        if coach_id == "logic":
            return "이 결정을 내리기 전에 확인해야 할 가장 큰 가정 1개와, 그 가정이 틀렸을 때의 대안(플랜B)은 무엇인가요?"
        return "‘나는 ___를 위해 ___을 선택한다’ 문장을 완성하면, 빈칸에 무엇이 들어가나요?"
    if coach_id == "logic":
        return "선택 기준 3개를 정해보면 무엇인가요? (예: 성장/비용/리스크)"
    return "이 고민에서 가장 중요한 가치 3개는 무엇인가요? (예: 안정/성장/관계)"


def generate_question(i: int, n: int) -> Tuple[str, Optional[str], List[str]]:
    coach = coach_by_id(st.session_state.coach_id)
    system = system_prompt_for(coach)
    prev_qs = st.session_state.questions[:]

    def prompt(nonce: int) -> str:
        prev_txt = "\n".join([f"- {q}" for q in prev_qs]) if prev_qs else "(없음)"
        return textwrap.dedent(f"""
        당신은 사용자의 생각을 정리하기 위한 질문을 1개 생성합니다.

        규칙:
        - 질문 1개만 출력 (설명/머리말/번호 금지)
        - 한국어
        - 이전 질문과 동일하거나 매우 유사한 질문 금지
        - 질문의 초점/관점이 이전 질문들과 겹치지 않게 구성
        - 사용자가 답하기 쉽게 예시(괄호 1줄) 허용

        [이전 질문 목록]
        {prev_txt}

        {build_context_block()}

        [이번 질문의 목적]
        {instruction_for_question(i, n, coach["id"])}

        (nonce={nonce})

        이제 질문 1개만 출력하세요.
        """).strip()

    q1, err, dbg = call_openai_text(system=system, user=prompt(random.randint(1000, 9999)), temperature=0.75)
    if not q1:
        return fallback_question(coach["id"], i, n), err, dbg

    q1 = normalize(q1)
    if not any(is_similar(q1, pq) for pq in prev_qs):
        return q1, None, dbg

    dbg.append("Similar question detected. Regenerating once.")
    q2, err2, dbg2 = call_openai_text(system=system, user=prompt(random.randint(10000, 99999)), temperature=0.85)
    dbg.extend(dbg2)
    if q2:
        q2 = normalize(q2)
        if not any(is_similar(q2, pq) for pq in prev_qs):
            return q2, None, dbg

    dbg.append("Still similar after retry. Using fallback question.")
    return fallback_question(coach["id"], i, n), None, dbg


def ensure_question(index: int, total: int) -> None:
    while len(st.session_state.questions) <= index:
        i = len(st.session_state.questions)
        q, err, dbg = generate_question(i, total)
        st.session_state.debug_log = dbg
        st.session_state.questions.append(q)


# =========================
# Final report
# =========================
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
        # 실행 코치는 년→달→주 계획을 레포트에서도 강하게
        format_spec = """
출력 형식:
## 우선순위(Top 1~3)
- 1) ...
- 2) ...
- 3) ...

## 목표 로드맵(년 → 달 → 주)
- 1년 목표(1개):
- 이번 달 목표(1개):
- 이번 주 계획(3~5개):

## 7일 실험(1~2개)
- 실험1: (시작 행동 15~30분 포함)
- 실험2(선택):

## If-Then 대응표
- 만약 ___이면 → ___한다 (3개)

## 오늘(24시간 내) 체크리스트
- [ ] ...
- [ ] ...

## 리뷰 질문
- ...
"""

    qa_text = ""
    for i, qa in enumerate(st.session_state.answers, start=1):
        qa_text += f"{i}) Q: {qa['q']}\n   A: {qa['a']}\n"

    user = textwrap.dedent(f"""
아래 Q/A를 바탕으로, 코치 역할에 맞는 '최종 정리 레포트'를 작성하세요.

규칙:
- 한국어
- 선택을 강요하지 말고, 근거와 다음 스텝을 명확히
- 불확실한 부분은 '추가 확인 질문' 1개를 마지막에 제안
- 길이: 700~1200자

[설정]
- 카테고리: {st.session_state.category}
- 결정 유형: {st.session_state.decision_type}
- 코치: {coach["name"]}

[Q/A]
{qa_text if qa_text.strip() else "(없음)"}

{format_spec}

마지막 줄:
- 추가 확인 질문: ...
""").strip()

    return call_openai_text(system=system, user=user, temperature=0.65)


# =========================
# Main UI
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
    cur = next((i for i, c in enumerate(COACHES) if c["id"] == st.session_state.coach_id), 0)
    picked = st.radio("코치", coach_labels, index=cur)
    st.session_state.coach_id = COACHES[coach_labels.index(picked)]["id"]

    coach = coach_by_id(st.session_state.coach_id)
    with st.expander("코치 진행 방식"):
        st.markdown(f"**{coach['name']}**  \n_{coach['style']}_")
        for m in coach["method"]:
            st.write(f"- {m}")
        st.caption(f"특징: {coach['prompt_hint']}")

    st.divider()
    st.subheader("질문 개수")
    st.session_state.num_questions = st.slider("질문 개수(2~10)", 2, 10, int(st.session_state.num_questions))

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("처음부터", use_container_width=True):
            reset_flow("setup")
            st.rerun()
    with c2:
        if st.session_state.page == "setup":
            if st.button("질문 시작", type="primary", use_container_width=True):
                reset_flow("questions")
                st.rerun()
        elif st.session_state.page == "questions":
            done = len(st.session_state.answers) >= int(st.session_state.num_questions)
            if st.button("최종 레포트로", use_container_width=True, disabled=not done):
                st.session_state.page = "report"
                st.rerun()
        else:
            if st.button("질문 페이지로", use_container_width=True):
                st.session_state.page = "questions"
                st.rerun()

# Progress bar labels
nq = int(st.session_state.num_questions)
progress_labels = ["설정"] + [f"Q{i}" for i in range(1, nq + 1)] + ["레포트"]

if st.session_state.page == "setup":
    current_progress = 0
elif st.session_state.page == "questions":
    current_progress = 1 + int(st.session_state.q_index)
else:
    current_progress = 1 + nq

render_pebble_row(current_progress, len(progress_labels), progress_labels)

progress = current_progress / max(1, (len(progress_labels) - 1))
with st.columns([1, 2, 1])[1]:
    render_hero_pebble(progress, f"진행도: {int(progress*100)}%")

st.divider()

# Pages
coach = coach_by_id(st.session_state.coach_id)

if st.session_state.page == "setup":
    st.title("🪨 돌멩이 결정 코치")
    st.caption("질문 개수를 설정하고 시작하세요. 질문이 끝나면 자동으로 레포트 페이지로 이동합니다.")

    cat_desc = next((d for n, d in TOPIC_CATEGORIES if n == st.session_state.category), "")
    st.info(f"**카테고리:** {st.session_state.category}\n\n{cat_desc}")
    st.write(f"**결정 유형:** {st.session_state.decision_type}")
    st.write(f"**선택한 코치:** {coach['name']}")
    st.write(f"**질문 개수:** {nq}개")
    st.success("사이드바에서 ‘질문 시작’을 누르면 질문 페이지로 이동합니다.")

elif st.session_state.page == "questions":
    st.title("질문")
    st.caption("답변을 저장하면 다음 질문으로 이동합니다.")

    q_idx = int(st.session_state.q_index)
    if q_idx >= nq:
        st.session_state.q_index = nq - 1
        q_idx = nq - 1

    ensure_question(q_idx, nq)
    current_q = st.session_state.questions[q_idx]

    st.subheader(f"Q{q_idx + 1} / {nq}")
    with st.container(border=True):
        st.markdown(f"**{current_q}**")

    with st.form(f"answer_form_{q_idx}", clear_on_submit=True):
        hint = ""
        if st.session_state.answers:
            last_a = st.session_state.answers[-1]["a"]
            hint = f"이전 답 요약: {last_a[:90]}{'…' if len(last_a) > 90 else ''}"
        answer = st.text_area("답변", placeholder=hint or "여기에 답변을 입력하세요", height=140)
        submitted = st.form_submit_button("답변 저장", use_container_width=True)

    if submitted:
        if not answer.strip():
            st.warning("답변이 비어 있습니다. 한 줄만 입력해도 진행 가능합니다.")
        else:
            add_answer(current_q, answer.strip())

            if len(st.session_state.answers) >= nq:
                st.session_state.page = "report"
                st.session_state.q_index = nq - 1
            else:
                st.session_state.q_index += 1
            st.rerun()

    with st.expander("답변 기록 보기"):
        if not st.session_state.answers:
            st.caption("아직 답변이 없습니다.")
        else:
            for i, qa in enumerate(st.session_state.answers, start=1):
                st.markdown(f"**Q{i}. {qa['q']}**")
                st.write(qa["a"])
                st.caption(qa["ts"])
                st.divider()

    with st.expander("디버그 로그(문제 발생 시 확인)"):
        st.write(st.session_state.debug_log)

else:
    st.title("최종 정리 레포트")
    st.caption("질문과 답변을 바탕으로 최종 정리를 생성합니다.")

    st.info(
        f"- **카테고리:** {st.session_state.category}\n"
        f"- **결정 유형:** {st.session_state.decision_type}\n"
        f"- **코치:** {coach['name']}\n"
        f"- **질문 개수:** {nq}개"
    )

    if len(st.session_state.answers) < nq:
        st.warning("아직 모든 질문이 완료되지 않았습니다. 질문 페이지로 돌아가 답변을 완료하세요.")
        if st.button("질문 페이지로 이동", type="primary"):
            st.session_state.page = "questions"
            st.rerun()
        st.stop()

    colA, colB = st.columns([1, 1])
    with colA:
        gen = st.button("레포트 생성/새로고침", type="primary", use_container_width=True)
    with colB:
        if st.button("새 고민 시작", use_container_width=True):
            reset_flow("setup")
            st.rerun()

    if gen or (st.session_state.final_report is None):
        with st.spinner("레포트를 생성하는 중..."):
            report, err, dbg = generate_final_report()
            st.session_state.debug_log = dbg
            if report:
                st.session_state.final_report = report
            else:
                st.session_state.final_report = None
                st.error(err or "레포트 생성 실패")

    if st.session_state.final_report:
        st.success("레포트가 준비되었습니다.")
        st.markdown(st.session_state.final_report)

        st.subheader("공유용 요약(JSON)")
        share = {
            "category": st.session_state.category,
            "decision_type": st.session_state.decision_type,
            "coach": coach["name"],
            "num_questions": nq,
            "qa": st.session_state.answers,
            "final_report": st.session_state.final_report,
        }
        st.code(json.dumps(share, ensure_ascii=False, indent=2), language="json")

    with st.expander("Q/A 전체 보기"):
        for i, qa in enumerate(st.session_state.answers, start=1):
            st.markdown(f"**Q{i}. {qa['q']}**")
            st.write(qa["a"])
            st.caption(qa["ts"])
            st.divider()

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
