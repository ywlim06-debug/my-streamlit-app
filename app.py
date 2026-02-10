# app.py
# ─────────────────────────────────────────────────────────────
# 🪨 돌멩이 AI 결정 코칭 (Pebble Decision Coach)
#
# 계획서 준수:
# - 정답/결론/추천 제공 금지
# - 한 화면에 한 질문씩
# - 이전 답변 반영 동적 질문 생성
# - 마지막: 고민의 핵심 / 선택 기준 / 코칭 메시지(추천 금지)
#
# 추가 기능:
# - 질문 개수 설정(2~10)
# - 질문 완료 후 레포트 페이지로 이동
# - 질문 중복 방지(유사하면 재생성 + fallback)
# - 실행 코치: 우선순위 + 계획(년→달→주) + 장애물 If-Then 질문
# - 돌다리 진행 UI: 돌 위를 사람이(🚶) 건너감
#   - 사람 아이콘 크게(40px)
#   - 방향 반대(좌측 바라봄)
# - PIL 미사용 (SVG base64 HTML 렌더)
#
# 필요:
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
st.set_page_config(page_title="돌멩이 AI 결정 코칭", page_icon="🪨", layout="wide")

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
        "name": "🔎 구조 코치",
        "tagline": "정보를 구조화하는 질문으로 정리를 돕습니다",
        "style": "프레임워크/기준 분해/명료화",
        "method": [
            "상황·제약·옵션을 분리해서 적게 하기",
            "선택 기준(3~5)을 뽑아 우선순위를 확인하기",
            "가정/불확실성을 드러내 추가 질문 찾기",
        ],
        "prompt_hint": "MECE, 기준 목록, 불확실성 질문",
    },
    {
        "id": "value",
        "name": "💗 가치 코치",
        "tagline": "감정과 가치관을 드러내는 질문으로 정리를 돕습니다",
        "style": "공감/가치 우선순위/후회 최소화 질문",
        "method": [
            "감정 라벨링(지금 느끼는 것) → 핵심 욕구 찾기",
            "가치 Top3 도출(내게 중요한 것)",
            "후회 최소화 관점 질문으로 기준 정리",
        ],
        "prompt_hint": "감정 라벨링, 가치 Top3, 미래의 나 질문",
    },
    {
        "id": "action",
        "name": "⚔️ 실행 코치",
        "tagline": "실행을 돕는 질문으로 계획을 ‘정리’합니다(추천 금지)",
        "style": "우선순위/계획 쪼개기(년→달→주)/장애물 질문",
        "method": [
            "우선순위 정하기: 효과/중요도/난이도 기준으로 Top1~3 ‘정리’",
            "사용자 목표를 년→달→주로 쪼개 ‘사용자가 말한 계획’을 구조화",
            "장애물/If-Then을 ‘질문’으로 명료화",
        ],
        "prompt_hint": "우선순위, 로드맵(년→달→주), 장애물 질문",
    },
]


# =========================
# Pebble SVG (no PIL)
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


# =========================
# Pebble bridge with walker
# =========================
def render_pebble_bridge(current_idx: int, total: int, labels: List[str]) -> None:
    total = max(2, int(total))
    current_idx = max(0, min(int(current_idx), total - 1))

    left_pct = ((current_idx + 0.5) / total) * 100.0

    pebble_imgs = []
    for i in range(total):
        active = i <= current_idx
        p = (i + 1) / total
        b64 = pebble_svg_b64(p, inactive=not active)
        pebble_imgs.append(b64)

    html = """
<style>
.pebble-bridge-wrap{
  position: relative;
  width: 100%;
  margin: 6px 0 2px 0;
  padding: 16px 4px 0 4px;
}
.pebble-row{
  display: flex;
  gap: 10px;
  align-items: flex-end;
  justify-content: space-between;
}
.pebble-cell{
  flex: 1;
  min-width: 0;
  text-align: center;
}
.pebble-img{
  width: 100%;
  max-width: 120px;
  height: auto;
  display: inline-block;
}
.pebble-label{
  font-size: 12px;
  margin-top: 4px;
  opacity: 0.85;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

/* 사람(🚶) 크게 + 방향 반대로 */
.walker{
  position: absolute;
  top: -10px;
  left: VAR_LEFT%;
  transform: translateX(-50%) scaleX(-1);
  font-size: 40px;
  line-height: 1;
  transition: left 520ms cubic-bezier(.2,.9,.2,1);
  filter: drop-shadow(0px 2px 2px rgba(0,0,0,0.25));
  animation: bob 800ms ease-in-out infinite;
  user-select: none;
}

@keyframes bob{
  0%{ transform: translateX(-50%) translateY(0px) scaleX(-1); }
  50%{ transform: translateX(-50%) translateY(-3px) scaleX(-1); }
  100%{ transform: translateX(-50%) translateY(0px) scaleX(-1); }
}
</style>
<div class="pebble-bridge-wrap">
  <div class="walker">🚶</div>
  <div class="pebble-row">
    VAR_PEBBLES
  </div>
</div>
""".strip()

    pebble_cells = []
    for i in range(total):
        opacity = "1.0" if i <= current_idx else "0.55"
        cell = f"""
<div class="pebble-cell" style="opacity:{opacity}">
  <img class="pebble-img" src="data:image/svg+xml;base64,{pebble_imgs[i]}" />
  <div class="pebble-label">{labels[i] if i < len(labels) else ""}</div>
</div>
""".strip()
        pebble_cells.append(cell)

    html = html.replace("VAR_LEFT", f"{left_pct:.3f}")
    html = html.replace("VAR_PEBBLES", "\n".join(pebble_cells))

    st.markdown(html, unsafe_allow_html=True)


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


def call_openai_text(system: str, user: str, temperature: float = 0.6) -> Tuple[Optional[str], Optional[str], List[str]]:
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

    if "situation" not in st.session_state:
        st.session_state.situation = ""
    if "goal" not in st.session_state:
        st.session_state.goal = ""
    if "options" not in st.session_state:
        st.session_state.options = ""

    if "num_questions" not in st.session_state:
        st.session_state.num_questions = 5
    if "q_index" not in st.session_state:
        st.session_state.q_index = 0

    if "questions" not in st.session_state:
        st.session_state.questions = []
    if "answers" not in st.session_state:
        st.session_state.answers = []

    if "final_report_json" not in st.session_state:
        st.session_state.final_report_json = None
    if "final_report_raw" not in st.session_state:
        st.session_state.final_report_raw = None

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
    st.session_state.final_report_json = None
    st.session_state.final_report_raw = None
    st.session_state.debug_log = []


def add_answer(q: str, a: str) -> None:
    st.session_state.answers.append({"q": q, "a": a, "ts": datetime.now().isoformat(timespec="seconds")})


# =========================
# Question generation
# =========================
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


def system_prompt_for_questions(coach: Dict[str, Any]) -> str:
    base = (
        "당신은 'AI 결정 코칭 앱'의 질문 생성기입니다.\n"
        "정답/해결책/추천을 주지 말고, 사용자가 스스로 생각을 정리하도록 질문만 던지세요.\n"
        "금지: 결론, 추천, 선택 강요, 판단문(예: A가 낫다).\n"
        "출력: 질문 1개만. (설명/번호/머리말 금지)\n"
    )
    if coach["id"] == "logic":
        return base + "스타일: 구조화/기준 분해/명료화 질문.\n"
    if coach["id"] == "value":
        return base + "스타일: 감정/가치/후회 최소화 관점 질문.\n"
    return base + "스타일: 우선순위/계획(년→달→주)/장애물 질문. 단, 지시가 아니라 질문으로만 유도.\n"


def build_context_block() -> str:
    hist = ""
    for i, qa in enumerate(st.session_state.answers[-6:], start=1):
        hist += f"{i}) Q: {qa['q']}\n   A: {qa['a']}\n"

    opts = [o.strip() for o in (st.session_state.options or "").split(",") if o.strip()]
    opts_txt = "\n".join([f"- {o}" for o in opts]) if opts else "(미입력)"

    return textwrap.dedent(f"""
    [세션 시작 정보]
    - 카테고리: {st.session_state.category}
    - 결정 유형: {st.session_state.decision_type}
    - 상황 설명: {st.session_state.situation or "(미입력)"}
    - 원하는 목표: {st.session_state.goal or "(미입력)"}
    - 고려 옵션(있다면): {opts_txt}

    [최근 Q/A]
    {hist if hist.strip() else "(아직 없음)"}
    """).strip()


def instruction_for_question(i: int, n: int, coach_id: str) -> str:
    if i == 0:
        return "상황의 핵심을 더 구체화하는 질문 1개"
    if i == 1:
        return "원하는 목표를 측정 가능한 형태로 정리하게 하는 질문 1개"

    if coach_id == "action":
        if i == 2:
            return "옵션/해야 할 일 3~6개를 펼치고 Top1~3 우선순위를 정리하게 하는 질문(효과/중요도/난이도 기준을 질문으로 제시)"
        if i == 3 and n >= 5:
            return "Top1을 ‘1년→이번 달→이번 주’로 쪼개 사용자가 계획을 적게 만드는 질문 1개"
        if i < n - 2:
            return "이번 주 계획을 더 구체화(무엇을/얼마나/언제)하는 질문 1개"
        if i == n - 2:
            return "예상 장애물과 If-Then 대응을 스스로 쓰게 하는 질문 1개"
        return "마지막으로 내 기준을 한 문장으로 정리하는 질문 1개(추천 금지)"

    if coach_id == "logic":
        if i == 2 and n >= 4:
            return "선택 기준(3~5)을 뽑게 하는 질문 1개"
        if i < n - 2:
            return "옵션/정보/제약을 더 분리해 명료화하는 질문 1개"
        if i == n - 2:
            return "불확실한 가정/추가로 확인할 정보 1~2개를 드러내는 질문 1개"
        return "마지막으로 선택 기준 우선순위를 정리하게 하는 질문 1개(추천 금지)"

    if i == 2 and n >= 4:
        return "지금 감정(2~3개)과 그 감정의 이유를 말하게 하는 질문 1개"
    if i < n - 2:
        return "가치 Top3(내게 중요한 것)와 내려놓을 것 1개를 정리하게 하는 질문 1개"
    if i == n - 2:
        return "후회 최소화 관점(1년/5년 후)을 점검하게 하는 질문 1개"
    return "마지막으로 ‘내 기준’을 한 문장으로 정리하게 하는 질문 1개(추천 금지)"


def fallback_question(coach_id: str, i: int, n: int) -> str:
    if i == 0:
        return "지금 고민 상황에서 ‘가장 핵심적인 쟁점’은 무엇인가요? (한 문장으로)"
    if i == 1:
        return "이번 결정으로 얻고 싶은 목표를 ‘측정 가능하게’ 바꾸면 어떻게 표현할 수 있나요? (언제까지/어느 정도)"

    if coach_id == "action":
        if i == 2:
            return "옵션/해야 할 일 3~6개를 적고, 효과/중요도/난이도를 생각했을 때 Top3는 무엇인가요?"
        if i == 3 and n >= 5:
            return "Top1을 기준으로 ‘1년 목표 → 이번 달 목표 → 이번 주 계획(3개)’을 적어보면 무엇인가요?"
        if i == n - 2:
            return "이번 주 계획을 방해할 장애물 3가지를 적고, 각각 ‘만약 ~이면 → ~한다’로 대응을 써볼까요?"
        return "이 선택에서 내가 가장 중요하게 보는 기준을 한 문장으로 적어보면 무엇인가요?"

    if coach_id == "logic":
        if i == 2 and n >= 4:
            return "이 선택을 평가할 기준 3~5개를 적어보면 무엇인가요?"
        if i == n - 2:
            return "지금 결정을 어렵게 만드는 ‘불확실한 정보/가정’은 무엇인가요?"
        return "내 기준(우선순위)을 1~3위로 정리하면 무엇인가요?"

    if i == 2 and n >= 4:
        return "지금 감정을 2~3개 단어로 적고, 각 감정이 생긴 이유를 한 줄씩 써볼까요?"
    if i == n - 2:
        return "1년/5년 뒤의 내가 지금의 나에게 뭐라고 말해줄 것 같나요?"
    return "이 고민에서 가장 중요한 가치 Top3는 무엇인가요?"


def generate_question(i: int, n: int) -> Tuple[str, Optional[str], List[str]]:
    coach = coach_by_id(st.session_state.coach_id)
    system = system_prompt_for_questions(coach)
    prev_qs = st.session_state.questions[:]

    def prompt(nonce: int) -> str:
        prev_txt = "\n".join([f"- {q}" for q in prev_qs]) if prev_qs else "(없음)"
        return textwrap.dedent(f"""
        [이전 질문 목록]
        {prev_txt}

        {build_context_block()}

        [이번 질문 목적]
        {instruction_for_question(i, n, coach["id"])}

        추가 규칙:
        - 결론/추천/정답 금지
        - 질문 1개만 출력

        (nonce={nonce})
        """).strip()

    q1, err, dbg = call_openai_text(system=system, user=prompt(random.randint(1000, 9999)), temperature=0.7)
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
# Final report JSON (추천 금지)
# =========================
def report_schema_hint(coach_id: str) -> str:
    base = """
반드시 JSON만 출력하세요(코드블록/설명 금지).
반드시 '추천/결론/정답'을 내리지 마세요.
대신 사용자가 말한 내용을 "요약/정리/기준화"하고, 마지막에 '다음에 스스로에게 던질 1개 질문'을 포함하세요.
"""
    if coach_id == "action":
        return textwrap.dedent(
            base
            + """
JSON 스키마:
{
  "summary": {
    "core_issue": "string",
    "goal": "string",
    "constraints": ["string"],
    "options_mentioned": ["string"]
  },
  "criteria": [
    {"name":"string","priority":1-5,"why":"string"}
  ],
  "plan_visualization": {
    "year": "string",
    "month": "string",
    "week": ["string","string","string"]
  },
  "weekly_table": {
    "Mon": ["string"], "Tue": ["string"], "Wed": ["string"], "Thu": ["string"],
    "Fri": ["string"], "Sat": ["string"], "Sun": ["string"]
  },
  "coaching_message": ["string","string"],
  "next_self_question": "string"
}
"""
        ).strip()

    if coach_id == "logic":
        return textwrap.dedent(
            base
            + """
JSON 스키마:
{
  "summary": {
    "core_issue":"string",
    "goal":"string",
    "constraints":["string"],
    "options_mentioned":["string"]
  },
  "criteria": [{"name":"string","priority":1-5,"why":"string"}],
  "key_points": {"uncertainties":["string"], "tradeoffs":["string"]},
  "coaching_message":["string","string"],
  "next_self_question":"string"
}
"""
        ).strip()

    return textwrap.dedent(
        base
        + """
JSON 스키마:
{
  "summary": {
    "core_issue":"string",
    "goal":"string",
    "constraints":["string"],
    "options_mentioned":["string"]
  },
  "criteria": [{"name":"string","priority":1-5,"why":"string"}],
  "emotions_values": {"emotions":["string","string"], "top_values":["string","string","string"]},
  "coaching_message":["string","string"],
  "next_self_question":"string"
}
"""
    ).strip()


def safe_json_parse(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    t = text.strip()
    if not t.startswith("{"):
        m = re.search(r"\{.*\}", t, flags=re.S)
        if m:
            t = m.group(0).strip()
    try:
        return json.loads(t)
    except Exception:
        return None


def system_prompt_for_report() -> str:
    return (
        "당신은 'AI 결정 코칭 앱'의 최종 요약 생성기입니다.\n"
        "절대 정답/결론/추천을 제시하지 마세요.\n"
        "사용자의 답변을 기반으로 '고민의 핵심, 선택 기준, 코칭 메시지'를 정리해 주세요.\n"
        "출력은 반드시 JSON만.\n"
    )


def generate_final_report_json() -> Tuple[Optional[Dict[str, Any]], Optional[str], List[str], Optional[str]]:
    coach = coach_by_id(st.session_state.coach_id)
    system = system_prompt_for_report()

    qa_text = ""
    for i, qa in enumerate(st.session_state.answers, start=1):
        qa_text += f"{i}) Q: {qa['q']}\n   A: {qa['a']}\n"

    opts = [o.strip() for o in (st.session_state.options or "").split(",") if o.strip()]

    user = textwrap.dedent(f"""
{report_schema_hint(coach["id"])}

[세션 시작 정보]
- 카테고리: {st.session_state.category}
- 결정 유형: {st.session_state.decision_type}
- 상황 설명: {st.session_state.situation}
- 원하는 목표: {st.session_state.goal}
- 옵션(있다면): {opts if opts else "(없음)"}

[Q/A]
{qa_text}

중요:
- 추천/결론/정답 금지
- 사용자가 말한 계획/의도/기준을 "정리"만 하기
- 사용자가 계획을 거의 말하지 않았다면 plan_visualization/weekly_table은 과장하지 말고, 말한 범위에서만 작성
""").strip()

    text, err, dbg = call_openai_text(system=system, user=user, temperature=0.35)
    if not text:
        return None, err, dbg, None

    data = safe_json_parse(text)
    if data is None:
        return None, "리포트 JSON 파싱 실패(모델이 JSON만 출력하지 않았을 수 있음)", dbg, text

    return data, None, dbg, text


# =========================
# Render report
# =========================
def render_summary_block(data: Dict[str, Any]) -> None:
    s = data.get("summary", {}) or {}
    st.subheader("고민의 핵심 요약")
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"**핵심 고민:** {s.get('core_issue','')}")
        st.write(f"**목표:** {s.get('goal','')}")
    with c2:
        cons = s.get("constraints", []) or []
        opts = s.get("options_mentioned", []) or []
        st.write("**제약/조건:**")
        if cons:
            for x in cons:
                st.write(f"- {x}")
        else:
            st.caption("제약이 명확히 언급되지 않았어요.")
        st.write("**언급된 옵션:**")
        if opts:
            for x in opts:
                st.write(f"- {x}")
        else:
            st.caption("옵션이 명확히 언급되지 않았어요.")


def render_criteria(data: Dict[str, Any]) -> None:
    st.subheader("선택 기준 정리(우선순위 포함)")
    crit = data.get("criteria", []) or []
    if not crit:
        st.caption("선택 기준이 충분히 드러나지 않았어요.")
        return
    rows = []
    for c in crit:
        rows.append(
            {
                "기준": c.get("name", ""),
                "우선순위(1~5)": c.get("priority", ""),
                "왜 중요한가": c.get("why", ""),
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)


def render_action_visualization(data: Dict[str, Any]) -> None:
    st.subheader("계획 정리(년 → 달 → 주) — 사용자 답변 기반")
    pv = data.get("plan_visualization", {}) or {}
    c1, c2, c3 = st.columns(3)
    c1.metric("년", pv.get("year", "") or "-")
    c2.metric("달", pv.get("month", "") or "-")
    c3.metric("주(핵심 3개)", " ")

    week = pv.get("week", []) or []
    if week:
        for x in week:
            st.write(f"- {x}")
    else:
        st.caption("사용자 답변에서 주 단위 계획이 충분히 드러나지 않았어요.")

    st.subheader("주간 테이블(정리용)")
    cal = data.get("weekly_table", {}) or {}
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    table = [{"Day": d, "Tasks": "\n".join(cal.get(d, []) or [])} for d in days]
    st.dataframe(table, use_container_width=True, hide_index=True)


def render_key_points_logic(data: Dict[str, Any]) -> None:
    kp = data.get("key_points", {}) or {}
    st.subheader("정리 포인트")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**불확실한 부분(추가 확인이 필요한 것)**")
        for x in kp.get("uncertainties", []) or []:
            st.write(f"- {x}")
    with c2:
        st.write("**트레이드오프(얻는 것 vs 잃는 것)**")
        for x in kp.get("tradeoffs", []) or []:
            st.write(f"- {x}")


def render_emotions_values(data: Dict[str, Any]) -> None:
    ev = data.get("emotions_values", {}) or {}
    st.subheader("감정/가치 정리")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**감정**")
        for x in ev.get("emotions", []) or []:
            st.write(f"- {x}")
    with c2:
        st.write("**가치 Top3**")
        for x in ev.get("top_values", []) or []:
            st.write(f"- {x}")


def render_coaching_message(data: Dict[str, Any]) -> None:
    st.subheader("코칭 메시지(정답/추천 없이 정리)")
    msgs = data.get("coaching_message", []) or []
    for m in msgs:
        st.write(f"- {m}")


def render_next_question(data: Dict[str, Any]) -> None:
    st.subheader("다음에 스스로에게 던질 질문(1개)")
    st.write(f"**{data.get('next_self_question','')}**")


# =========================
# App UI
# =========================
init_state()

with st.sidebar:
    st.header("설정")
    st.text_input("OpenAI API Key (Secrets 우선)", type="password", key="openai_api_key_input")

    st.divider()
    st.subheader("상황 설정(세션 시작)")
    st.selectbox("카테고리", [x[0] for x in TOPIC_CATEGORIES], key="category")
    st.selectbox("결정 유형", DECISION_TYPES, key="decision_type")
    st.text_area("상황 설명", key="situation", height=90, placeholder="무슨 일이 있었고 무엇을 결정해야 하나요?")
    st.text_input("원하는 목표", key="goal", placeholder="이 결정에서 얻고 싶은 결과(가능하면 측정 가능하게)")
    st.text_input("옵션(쉼표로 구분, 선택)", key="options", placeholder="예: A, B, C")

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
            if st.button("최종 결과로", use_container_width=True, disabled=not done):
                st.session_state.page = "report"
                st.rerun()
        else:
            if st.button("질문 페이지로", use_container_width=True):
                st.session_state.page = "questions"
                st.rerun()

# Progress (돌다리 + 사람)
nq = int(st.session_state.num_questions)
labels = ["설정"] + [f"Q{i}" for i in range(1, nq + 1)] + ["요약"]
if st.session_state.page == "setup":
    idx = 0
elif st.session_state.page == "questions":
    idx = 1 + int(st.session_state.q_index)
else:
    idx = 1 + nq

render_pebble_bridge(idx, len(labels), labels)

progress = idx / max(1, (len(labels) - 1))
with st.columns([1, 2, 1])[1]:
    render_hero_pebble(progress, f"진행도: {int(progress*100)}%")

st.divider()

coach = coach_by_id(st.session_state.coach_id)

if st.session_state.page == "setup":
    st.title("🪨 AI 결정 코칭")
    st.caption("정답을 주기보다, 질문으로 생각을 정리하도록 돕습니다. 한 화면에 한 질문씩 진행됩니다.")
    st.info("사이드바에서 상황을 입력하고 ‘질문 시작’을 누르세요.")

    with st.container(border=True):
        st.subheader("현재 설정 미리보기")
        st.write(f"- 카테고리: {st.session_state.category}")
        st.write(f"- 결정 유형: {st.session_state.decision_type}")
        st.write(f"- 코치: {coach['name']}")
        st.write(f"- 질문 개수: {nq}")
        st.write(f"- 상황 설명: {st.session_state.situation or '(미입력)'}")
        st.write(f"- 목표: {st.session_state.goal or '(미입력)'}")
        st.write(f"- 옵션: {st.session_state.options or '(미입력)'}")

elif st.session_state.page == "questions":
    st.title("질문")
    st.caption("한 화면에 한 질문. 답변을 저장하면 다음 질문으로 이동합니다.")

    q_idx = int(st.session_state.q_index)
    q_idx = max(0, min(q_idx, nq - 1))

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
        ans = st.text_area("답변", placeholder=hint or "여기에 답변을 입력하세요", height=150)
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

    with st.expander("답변 기록"):
        for i, qa in enumerate(st.session_state.answers, start=1):
            st.markdown(f"**Q{i}. {qa['q']}**")
            st.write(qa["a"])
            st.caption(qa["ts"])
            st.divider()

    with st.expander("디버그 로그"):
        st.write(st.session_state.debug_log)

else:
    st.title("최종 정리")
    st.caption("정답/추천 없이, 고민의 핵심과 기준을 정리해 줍니다(사용자 답변 기반).")

    if len(st.session_state.answers) < nq:
        st.warning("아직 모든 질문이 완료되지 않았습니다. 질문 페이지로 돌아가 답변을 완료하세요.")
        if st.button("질문 페이지로 이동", type="primary"):
            st.session_state.page = "questions"
            st.rerun()
        st.stop()

    colA, colB = st.columns([1, 1])
    with colA:
        gen = st.button("정리 생성/새로고침", type="primary", use_container_width=True)
    with colB:
        if st.button("새 세션 시작", use_container_width=True):
            reset_flow("setup")
            st.rerun()

    if gen or (st.session_state.final_report_json is None and st.session_state.final_report_raw is None):
        with st.spinner("최종 정리를 생성하는 중..."):
            data, err, dbg, raw = generate_final_report_json()
            st.session_state.debug_log = dbg
            if data is not None:
                st.session_state.final_report_json = data
                st.session_state.final_report_raw = raw
            else:
                st.session_state.final_report_json = None
                st.session_state.final_report_raw = raw
                st.error(err or "정리 생성 실패")

    data = st.session_state.final_report_json
    if data:
        st.success("최종 정리가 준비되었습니다.")
        render_summary_block(data)
        render_criteria(data)

        if coach["id"] == "action":
            render_action_visualization(data)
        elif coach["id"] == "logic":
            render_key_points_logic(data)
        else:
            render_emotions_values(data)

        render_coaching_message(data)
        render_next_question(data)

        st.subheader("공유용(JSON)")
        st.code(json.dumps(data, ensure_ascii=False, indent=2), language="json")

    elif st.session_state.final_report_raw:
        st.warning("JSON 파싱 실패로 원문을 표시합니다.")
        st.code(st.session_state.final_report_raw, language="text")

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
