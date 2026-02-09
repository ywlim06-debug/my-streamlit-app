# app.py
# ─────────────────────────────────────────────────────────────
# 🪨 돌멩이 결정 코치 (Pebble Decision Coach) — Personalized Report + Visualization
#
# 추가 반영:
# - 최종 리포트를 "구조화 JSON"으로 생성 → Streamlit에서 시각화(우선순위/로드맵/주간 플랜)
# - 특히 실행 코치(action) 선택 시:
#   * 우선순위 Top1~3 표
#   * 년→달→주 로드맵 표
#   * 이번 주 계획(요일별) 테이블(간단 캘린더 느낌)
#   * If-Then 대응표 + 체크리스트
# - 논리/가치 코치는 JSON을 텍스트 중심으로 보여주되, 핵심 항목을 카드/표 형태로 정리
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
# Config
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
# Pebble UI (no PIL)
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
# OpenAI
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

    if "final_report_text" not in st.session_state:
        st.session_state.final_report_text = None
    if "final_report_json" not in st.session_state:
        st.session_state.final_report_json = None

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
    st.session_state.final_report_text = None
    st.session_state.final_report_json = None
    st.session_state.debug_log = []


def add_answer(q: str, a: str) -> None:
    st.session_state.answers.append({"q": q, "a": a, "ts": datetime.now().isoformat(timespec="seconds")})


# =========================
# Similarity + Question gen
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


def build_context_block() -> str:
    hist = ""
    for i, qa in enumerate(st.session_state.answers[-6:], start=1):
        hist += f"{i}) Q: {qa['q']}\n   A: {qa['a']}\n"
    return textwrap.dedent(f"""
    [고민 카테고리] {st.session_state.category}
    [결정 유형] {st.session_state.decision_type}

    [최근 Q/A]
    {hist if hist.strip() else "(없음)"}
    """).strip()


def instruction_for_question(i: int, n: int, coach_id: str) -> str:
    if i == 0:
        return "상황을 구체적으로 파악하는 질문 1개"
    if i == 1:
        return "원하는 결과 vs 피하고 싶은 결과를 분리하는 질문 1개"

    if coach_id == "action":
        if i == 2:
            return "해야 할 일/옵션 3~6개를 적고 효과/중요도/난이도로 우선순위 1~3개를 고르게 하는 질문 1개"
        if i == 3 and n >= 5:
            return "우선순위 1개를 기준으로 목표를 년→달→주로 쪼개 계획을 세우게 하는 질문 1개"
        if i < n - 2:
            return "이번 주 7일 실험 1개 + 오늘 시작 행동(15~30분)을 구체화하는 질문 1개"
        if i == n - 2:
            return "장애물 3개와 If-Then 대응을 만들게 하는 질문 1개"
        return "실행 약속을 한 문장(언제/어디서/몇 분/무엇을)으로 고정하는 질문 1개"

    if i == 2 and n >= 4:
        return "제약(시간/돈/관계/규칙)을 명확히 하는 질문 1개"

    if i < n - 2:
        if coach_id == "logic":
            return "옵션 2~4개와 평가 기준 3~5개를 뽑게 하는 질문 1개(표로 정리 가능)"
        return "가치 우선순위 3개와 감정/욕구/두려움을 드러내는 질문 1개"

    if i == n - 2:
        if coach_id == "logic":
            return "가정/리스크 검증 + 플랜B를 떠올리게 하는 질문 1개"
        return "후회 최소화 관점(1년/5년)을 점검하는 질문 1개"

    if coach_id == "logic":
        return "최종 확인 질문 1개(가정/리스크/대안 중 1개에 초점)"
    return "결정 문장 한 줄을 완성하게 하는 질문 1개(나는 ___를 위해 ___을 선택한다)"


def fallback_question(coach_id: str, i: int, n: int) -> str:
    if i == 0:
        return "지금 고민 상황을 한 문단으로 설명해 주세요. (무슨 일이 있었고, 무엇을 결정해야 하나요?)"
    if i == 1:
        return "이 결정에서 얻고 싶은 최선의 결과 1가지와 피하고 싶은 최악의 결과 1가지는 무엇인가요?"

    if coach_id == "action":
        if i == 2:
            return "해야 할 일(또는 옵션)을 3~6개 적고, 그중 효과가 큰 1~3개를 우선순위로 고르면 무엇인가요?"
        if i == 3 and n >= 5:
            return "우선순위 1개를 ‘1년 목표 → 이번 달 목표 → 이번 주 할 일’로 쪼개면 각각 무엇인가요?"
        if i == n - 2:
            return "이번 주 실행을 방해할 장애물 3가지를 적고, 각각 ‘만약 ~이면 → ~한다’로 대응을 만들어보면요?"
        if i == n - 1:
            return "이번 주 첫 행동을 ‘언제/어디서/몇 분/무엇을’ 한 문장으로 적어 주세요."
        return "이번 주 7일 실험 1개와, 오늘 15~30분 안에 할 시작 행동은 무엇인가요?"

    if i == 2 and n >= 4:
        return "시간/돈/관계/규칙 측면에서 바꿀 수 없는 제약 2가지는 무엇인가요?"

    if i == n - 1:
        if coach_id == "logic":
            return "확인해야 할 가장 큰 가정 1개와, 그 가정이 틀렸을 때의 플랜B는 무엇인가요?"
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
        질문 1개만 생성하세요(설명/번호 금지).

        규칙:
        - 한국어
        - 이전 질문과 동일/유사 금지
        - 관점이 겹치지 않게
        - 괄호 예시 1줄 허용

        [이전 질문 목록]
        {prev_txt}

        {build_context_block()}

        [이번 질문 목적]
        {instruction_for_question(i, n, coach["id"])}

        (nonce={nonce})
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

    dbg.append("Still similar after retry. Using fallback.")
    return fallback_question(coach["id"], i, n), None, dbg


def ensure_question(index: int, total: int) -> None:
    while len(st.session_state.questions) <= index:
        i = len(st.session_state.questions)
        q, err, dbg = generate_question(i, total)
        st.session_state.debug_log = dbg
        st.session_state.questions.append(q)


# =========================
# Final report JSON generation + rendering
# =========================
def json_schema_hint(coach_id: str) -> str:
    # 모델이 반환할 JSON 형태를 강하게 고정(시각화에 쓰기 위함)
    if coach_id == "action":
        return textwrap.dedent("""
        반드시 JSON만 출력하세요. (코드블록 금지)

        JSON 스키마:
        {
          "one_line_summary": "string",
          "priorities": [
            {"item":"string","reason":"string","impact":1-5,"difficulty":1-5}
          ],
          "roadmap": {
            "year_goal": "string",
            "month_goal": "string",
            "week_plan": ["string","string","string"]
          },
          "weekly_calendar": {
            "Mon": ["task","task"],
            "Tue": [],
            "Wed": [],
            "Thu": [],
            "Fri": [],
            "Sat": [],
            "Sun": []
          },
          "experiments": [
            {"name":"string","steps":["string","string"],"start_action":"string"}
          ],
          "if_then": [
            {"if":"string","then":"string"}
          ],
          "today_checklist": ["string","string"],
          "review_questions": ["string","string"],
          "extra_check_question": "string"
        }
        """).strip()

    if coach_id == "logic":
        return textwrap.dedent("""
        반드시 JSON만 출력하세요. (코드블록 금지)

        JSON 스키마:
        {
          "one_line_conclusion": "string",
          "issue": "string",
          "options": ["string","string"],
          "criteria": ["string","string","string"],
          "constraints_assumptions": ["string"],
          "risks_mitigations": [{"risk":"string","mitigation":"string"}],
          "recommendation": {"pick":"string","reasons":["string","string","string"],"next_24h":["string"]},
          "extra_check_question":"string"
        }
        """).strip()

    return textwrap.dedent("""
    반드시 JSON만 출력하세요. (코드블록 금지)

    JSON 스키마:
    {
      "now_feelings": {"emotions":["string","string"],"core_need_or_fear":"string"},
      "top_values": ["string","string","string"],
      "decision_sentence": "string",
      "regret_check": {"one_year":"string","five_years":"string"},
      "tomorrow_promise": ["string","string"],
      "extra_check_question":"string"
    }
    """).strip()


def safe_json_parse(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    t = text.strip()

    # 만약 모델이 실수로 앞뒤 텍스트를 붙였을 경우를 대비해 JSON만 추출
    # 가장 바깥 { ... } 범위를 찾는 단순 방식
    if not t.startswith("{"):
        m = re.search(r"\{.*\}", t, flags=re.S)
        if m:
            t = m.group(0).strip()

    try:
        return json.loads(t)
    except Exception:
        return None


def generate_report_json() -> Tuple[Optional[Dict[str, Any]], Optional[str], List[str], Optional[str]]:
    coach = coach_by_id(st.session_state.coach_id)
    system = system_prompt_for(coach)

    qa_text = ""
    for i, qa in enumerate(st.session_state.answers, start=1):
        qa_text += f"{i}) Q: {qa['q']}\n   A: {qa['a']}\n"

    user = textwrap.dedent(f"""
    사용자의 답변을 바탕으로 최종 리포트를 작성하세요.

    목표:
    - 사용자의 답변을 반영해 맞춤형 계획/정리 제공
    - 실행 코치라면 반드시 계획을 시각화 가능한 구조(우선순위/로드맵/주간 캘린더)로 만들기
    - 논리/가치 코치도 구조화된 핵심을 담기

    규칙:
    - {json_schema_hint(coach["id"])}

    [설정]
    - 카테고리: {st.session_state.category}
    - 결정 유형: {st.session_state.decision_type}
    - 코치: {coach["name"]}

    [Q/A]
    {qa_text}
    """).strip()

    text, err, dbg = call_openai_text(system=system, user=user, temperature=0.4)
    if not text:
        return None, err, dbg, None

    data = safe_json_parse(text)
    if data is None:
        # 파싱 실패 시: 텍스트를 그대로 보관
        return None, "리포트 JSON 파싱에 실패했습니다. (모델이 JSON만 출력하지 않았을 수 있어요)", dbg, text

    return data, None, dbg, text


def render_action_report(data: Dict[str, Any]) -> None:
    st.subheader("한 줄 요약")
    st.write(data.get("one_line_summary", ""))

    st.subheader("우선순위 Top")
    pr = data.get("priorities", []) or []
    if pr:
        rows = []
        for p in pr:
            rows.append(
                {
                    "항목": p.get("item", ""),
                    "이유": p.get("reason", ""),
                    "임팩트(1~5)": p.get("impact", ""),
                    "난이도(1~5)": p.get("difficulty", ""),
                }
            )
        st.dataframe(rows, use_container_width=True)
        # 임팩트/난이도 바 차트(간단)
        chart_rows = [{"label": r["항목"], "impact": r["임팩트(1~5)"], "difficulty": r["난이도(1~5)"]} for r in rows]
        st.bar_chart(chart_rows, x="label", y=["impact", "difficulty"])
    else:
        st.caption("우선순위 데이터가 없습니다.")

    st.subheader("로드맵 (년 → 달 → 주)")
    roadmap = data.get("roadmap", {}) or {}
    c1, c2, c3 = st.columns(3)
    c1.metric("1년 목표", roadmap.get("year_goal", "-"))
    c2.metric("이번 달 목표", roadmap.get("month_goal", "-"))
    c3.metric("이번 주 핵심", "3~5개")

    week_plan = roadmap.get("week_plan", []) or []
    if week_plan:
        st.write("이번 주 계획:")
        for w in week_plan:
            st.write(f"- {w}")

    st.subheader("이번 주 캘린더(간단)")
    cal = data.get("weekly_calendar", {}) or {}
    # 요일 순서 고정
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    table = []
    for d in days:
        table.append({"Day": d, "Tasks": "\n".join(cal.get(d, []) or [])})
    st.dataframe(table, use_container_width=True, hide_index=True)

    st.subheader("7일 실험")
    exps = data.get("experiments", []) or []
    if exps:
        for i, e in enumerate(exps, start=1):
            with st.container(border=True):
                st.markdown(f"**실험 {i}: {e.get('name','')}**")
                st.write("Steps:")
                for s in e.get("steps", []) or []:
                    st.write(f"- {s}")
                st.write(f"시작 행동: {e.get('start_action','')}")
    else:
        st.caption("실험 데이터가 없습니다.")

    st.subheader("If-Then 대응표")
    it = data.get("if_then", []) or []
    if it:
        st.dataframe([{"If": x.get("if", ""), "Then": x.get("then", "")} for x in it], use_container_width=True, hide_index=True)
    else:
        st.caption("If-Then 데이터가 없습니다.")

    st.subheader("오늘 체크리스트(24시간)")
    for t in data.get("today_checklist", []) or []:
        st.checkbox(t, value=False)

    st.subheader("리뷰 질문")
    for q in data.get("review_questions", []) or []:
        st.write(f"- {q}")

    st.divider()
    st.write(f"추가 확인 질문: **{data.get('extra_check_question','')}**")


def render_logic_report(data: Dict[str, Any]) -> None:
    st.subheader("한 줄 결론")
    st.write(data.get("one_line_conclusion", ""))

    st.subheader("의사결정 구조")
    st.write(f"쟁점: {data.get('issue','')}")
    c1, c2 = st.columns(2)
    with c1:
        st.write("옵션")
        for x in data.get("options", []) or []:
            st.write(f"- {x}")
    with c2:
        st.write("기준")
        for x in data.get("criteria", []) or []:
            st.write(f"- {x}")

    st.subheader("제약/가정")
    for x in data.get("constraints_assumptions", []) or []:
        st.write(f"- {x}")

    st.subheader("리스크/대응")
    rms = data.get("risks_mitigations", []) or []
    if rms:
        st.dataframe([{"Risk": r.get("risk", ""), "Mitigation": r.get("mitigation", "")} for r in rms], use_container_width=True, hide_index=True)

    st.subheader("추천안")
    rec = data.get("recommendation", {}) or {}
    st.write(f"추천: **{rec.get('pick','')}**")
    st.write("이유:")
    for x in rec.get("reasons", []) or []:
        st.write(f"- {x}")
    st.write("다음 24시간:")
    for x in rec.get("next_24h", []) or []:
        st.write(f"- {x}")

    st.divider()
    st.write(f"추가 확인 질문: **{data.get('extra_check_question','')}**")


def render_value_report(data: Dict[str, Any]) -> None:
    st.subheader("지금의 마음")
    nf = data.get("now_feelings", {}) or {}
    st.write("감정:")
    for x in nf.get("emotions", []) or []:
        st.write(f"- {x}")
    st.write(f"핵심 욕구/두려움: {nf.get('core_need_or_fear','')}")

    st.subheader("가치 우선순위 Top3")
    for x in data.get("top_values", []) or []:
        st.write(f"- {x}")

    st.subheader("나다운 선택 문장")
    st.write(data.get("decision_sentence", ""))

    st.subheader("후회 최소화 체크")
    rc = data.get("regret_check", {}) or {}
    st.write(f"- 1년 뒤의 나: {rc.get('one_year','')}")
    st.write(f"- 5년 뒤의 나: {rc.get('five_years','')}")

    st.subheader("내일의 작은 약속")
    for x in data.get("tomorrow_promise", []) or []:
        st.write(f"- {x}")

    st.divider()
    st.write(f"추가 확인 질문: **{data.get('extra_check_question','')}**")


# =========================
# App UI
# =========================
init_state()

# Sidebar
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

# Progress
nq = int(st.session_state.num_questions)
labels = ["설정"] + [f"Q{i}" for i in range(1, nq + 1)] + ["레포트"]
if st.session_state.page == "setup":
    idx = 0
elif st.session_state.page == "questions":
    idx = 1 + int(st.session_state.q_index)
else:
    idx = 1 + nq

render_pebble_row(idx, len(labels), labels)
progress = idx / max(1, (len(labels) - 1))
with st.columns([1, 2, 1])[1]:
    render_hero_pebble(progress, f"진행도: {int(progress*100)}%")

st.divider()

# Page: setup
coach = coach_by_id(st.session_state.coach_id)
if st.session_state.page == "setup":
    st.title("🪨 돌멩이 결정 코치")
    st.caption("질문에 답하면 자동으로 다음 질문으로 이동하고, 끝나면 맞춤형 최종 리포트를 시각화해서 보여줍니다.")

    cat_desc = next((d for n, d in TOPIC_CATEGORIES if n == st.session_state.category), "")
    st.info(f"**카테고리:** {st.session_state.category}\n\n{cat_desc}")
    st.write(f"**결정 유형:** {st.session_state.decision_type}")
    st.write(f"**코치:** {coach['name']}")
    st.write(f"**질문 개수:** {nq}개")
    st.success("사이드바에서 ‘질문 시작’을 누르세요.")

# Page: questions
elif st.session_state.page == "questions":
    st.title("질문")
    st.caption("답변을 저장하면 다음 질문으로 이동합니다. 모든 질문을 끝내면 레포트 페이지로 넘어갑니다.")

    q_idx = int(st.session_state.q_index)
    if q_idx >= nq:
        st.session_state.q_index = nq - 1
        q_idx = nq - 1

    ensure_question(q_idx, nq)
    q = st.session_state.questions[q_idx]

    st.subheader(f"Q{q_idx + 1} / {nq}")
    with st.container(border=True):
        st.markdown(f"**{q}**")

    with st.form(f"answer_form_{q_idx}", clear_on_submit=True):
        hint = ""
        if st.session_state.answers:
            last_a = st.session_state.answers[-1]["a"]
            hint = f"이전 답 요약: {last_a[:90]}{'…' if len(last_a) > 90 else ''}"
        ans = st.text_area("답변", placeholder=hint or "여기에 답변을 입력하세요", height=140)
        submitted = st.form_submit_button("답변 저장", use_container_width=True)

    if submitted:
        if not ans.strip():
            st.warning("답변이 비어 있습니다. 한 줄만 입력해도 진행 가능합니다.")
        else:
            add_answer(q, ans.strip())
            if len(st.session_state.answers) >= nq:
                st.session_state.page = "report"
                st.session_state.q_index = nq - 1
            else:
                st.session_state.q_index += 1
            st.rerun()

    with st.expander("답변 기록 보기"):
        for i, qa in enumerate(st.session_state.answers, start=1):
            st.markdown(f"**Q{i}. {qa['q']}**")
            st.write(qa["a"])
            st.caption(qa["ts"])
            st.divider()

    with st.expander("디버그 로그"):
        st.write(st.session_state.debug_log)

# Page: report
else:
    st.title("최종 리포트 (맞춤형)")
    st.caption("사용자의 답변을 바탕으로 구조화된 리포트를 만들고, 그 데이터를 시각화해 보여줍니다.")

    if len(st.session_state.answers) < nq:
        st.warning("아직 모든 질문이 완료되지 않았습니다. 질문 페이지로 돌아가 답변을 완료하세요.")
        if st.button("질문 페이지로 이동", type="primary"):
            st.session_state.page = "questions"
            st.rerun()
        st.stop()

    colA, colB = st.columns([1, 1])
    with colA:
        gen = st.button("리포트 생성/새로고침", type="primary", use_container_width=True)
    with colB:
        if st.button("새 고민 시작", use_container_width=True):
            reset_flow("setup")
            st.rerun()

    if gen or (st.session_state.final_report_json is None and st.session_state.final_report_text is None):
        with st.spinner("맞춤형 리포트를 생성하는 중..."):
            data, err, dbg, raw_text = generate_report_json()
            st.session_state.debug_log = dbg
            if data is not None:
                st.session_state.final_report_json = data
                st.session_state.final_report_text = None
            else:
                # 파싱 실패 시 raw_text를 보여주기(디버깅)
                st.session_state.final_report_json = None
                st.session_state.final_report_text = raw_text
                st.error(err or "리포트 생성 실패")

    coach_id = coach_by_id(st.session_state.coach_id)["id"]

    if st.session_state.final_report_json:
        data = st.session_state.final_report_json
        st.success("맞춤형 리포트가 준비되었습니다.")

        if coach_id == "action":
            render_action_report(data)
        elif coach_id == "logic":
            render_logic_report(data)
        else:
            render_value_report(data)

        st.subheader("공유용(JSON)")
        st.code(json.dumps(data, ensure_ascii=False, indent=2), language="json")

    elif st.session_state.final_report_text:
        st.warning("JSON 파싱 실패로 텍스트 리포트 원문을 표시합니다(모델 출력이 JSON만이 아닐 수 있음).")
        st.code(st.session_state.final_report_text, language="text")

    with st.expander("Q/A 전체 보기"):
        for i, qa in enumerate(st.session_state.answers, start=1):
            st.markdown(f"**Q{i}. {qa['q']}**")
            st.write(qa["a"])
            st.caption(qa["ts"])
            st.divider()

    with st.expander("디버그 로그"):
        st.write(st.session_state.debug_log)

st.divider()
with st.expander("배포 체크리스트 (Streamlit Cloud)"):
    st.markdown(
        """
- Secrets 설정: Settings → Secrets에 `OPENAI_API_KEY = "sk-..."` 추가
- requirements.txt:
  - streamlit
  - openai
"""
    )

