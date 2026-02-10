# app.py
# ─────────────────────────────────────────────────────────────
# 🪨 돌멩이 AI 결정 코칭 (Pebble Decision Coach)
#
# 원칙(유지):
# - 정답/결론/추천 제공 금지 (강제)
# - 한 화면에 한 질문씩
# - 이전 답변 반영 동적 질문 생성
# - 마지막: 고민의 핵심 / 선택 기준 / 코칭 메시지(거울 비추기, 추천 금지)
#
# 이번 고도화 반영:
# 1) 질문 생성 고도화
#   - Logic Cross-Check(답변 간 논리 충돌 감지 → 충돌을 짚는 질문 우선 생성)
#   - Probing(답변 10자 미만이면 1회 구체화 질문)
#   - Action Coach 강화: If-Then 트리거 + Pre-mortem 질문 포함
#
# 2) UI/UX
#   - 이전으로(Back) 버튼: q_index 감소 + answers 마지막 답변 제거(+probe 상태 정리)
#   - 결정 유형별 템플릿: 결정 유형 선택 시 상황 설명에 가이드 자동 입력(상황이 비어있거나 가이드 문구일 때)
#   - 리포트 진입 시 st.balloons()
#
# 3) 리포트/공유
#   - 의사결정 매트릭스(st.data_editor): 옵션 x 기준 점수(1~5)
#   - 내면의 목소리(Mirroring): 답변 키워드/감정어 빈도 시각 요약
#   - 저장: 클립보드 복사 버튼 + 타임스탬프 파일명 .txt 다운로드
#   - 결정 유효기간: 오늘로부터 7일 뒤까지
#
# 필요:
#   pip install streamlit openai pandas
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import base64
import json
import random
import re
import textwrap
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
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

DECISION_TEMPLATES: Dict[str, str] = {
    "A vs B 선택(둘 중 하나)": textwrap.dedent(
        """\
        [가이드] A vs B 선택 정리
        1) A와 B를 한 문장으로 정의해보세요(무엇이 다른가?)
        2) A의 장점/단점, B의 장점/단점을 각각 2~3개씩 적어보세요
        3) '내게 중요한 기준' 3개를 적고, 각 기준에서 A/B가 어떻게 다른지 써보세요
        4) 최악의 경우(리스크)와 감당 가능한 정도를 적어보세요
        """
    ).strip(),
    "여러 옵션 중 선택": textwrap.dedent(
        """\
        [가이드] 여러 옵션 비교
        1) 후보 옵션을 모두 나열해보세요(최소 3개)
        2) 비교 기준 3~5개(비용/시간/성장/스트레스/관계 등)를 적어보세요
        3) 각 옵션이 기준별로 어떤 느낌인지(좋음/보통/나쁨)부터 대략 적어보세요
        4) '지금의 나'에게 중요한 것과 '1년 뒤의 나'에게 중요한 것을 구분해보세요
        """
    ).strip(),
    "해야 할지 말지(Yes/No)": textwrap.dedent(
        """\
        [가이드] 해야 할지 말지(Yes/No)
        1) '한다'의 의미를 구체화(무엇을/얼마나/어떤 수준으로?)
        2) 한다면 얻는 것/잃는 것, 안 한다면 얻는 것/잃는 것을 각각 적어보세요
        3) 결정이 미뤄질 때 생기는 비용(불안/기회/관계 등)을 적어보세요
        4) 지금 당장 필요한 추가 정보 1~2개가 무엇인지 적어보세요
        """
    ).strip(),
    "언제/어떻게 할지(전략/시점)": textwrap.dedent(
        """\
        [가이드] 전략/시점 결정
        1) 성공의 정의를 한 문장으로(측정 가능하게)
        2) 선택 가능한 시나리오 2~3개(빠르게/천천히/부분 적용 등) 나열
        3) 각 시나리오의 리스크와 완충장치(보험)를 적어보세요
        4) '시작 트리거(If) → 행동(Then)' 형태로 실행 조건을 설계해보세요
        """
    ).strip(),
    "갈등 해결/대화 방향": textwrap.dedent(
        """\
        [가이드] 갈등/대화 방향 정리
        1) 지금 갈등의 쟁점을 '사실/해석/감정/요구'로 나눠 적어보세요
        2) 내가 원하는 변화(요구) 1~2개를 구체적으로 적어보세요
        3) 상대가 중요하게 여길 만한 것을 추측해 적어보세요(확정 아님)
        4) 대화에서 지키고 싶은 원칙(톤/타이밍/한계선)을 적어보세요
        """
    ).strip(),
}

COACHES = [
    {
        "id": "logic",
        "name": "🔎 구조 코치",
        "tagline": "정보를 구조화하고, 가정을 흔드는 질문으로 정리를 돕습니다",
        "style": "MECE/기준/가정 깨기(역발상)",
        "method": [
            "상황·제약·옵션을 분리해서 적게 하기",
            "선택 기준(3~5)을 뽑아 우선순위를 확인하기",
            "‘내가 당연하다고 믿는 가정’을 반대로 뒤집어 보기",
        ],
        "prompt_hint": "MECE, 기준 목록, 역발상(가정 깨기)",
    },
    {
        "id": "value",
        "name": "💗 가치 코치",
        "tagline": "감정과 가치를 분리해, 후회가 적은 기준을 찾게 돕습니다",
        "style": "감정 라벨링/가치 분리/후회 최소화",
        "method": [
            "감정 라벨링(지금 느끼는 것) → 이유",
            "그 감정이 ‘일시적 편안함’인지 ‘장기 가치’인지 분리",
            "가치 Top3 도출 + 후회 최소화 질문",
        ],
        "prompt_hint": "감정-가치 분리, 가치 Top3, 미래의 나 질문",
    },
    {
        "id": "action",
        "name": "⚔️ 실행 코치",
        "tagline": "계획을 ‘정리’하고, 오늘 5분 Quick Win까지 스스로 찾게 돕습니다(추천 금지)",
        "style": "우선순위/If-Then/프리모템/Quick Win",
        "method": [
            "우선순위 정하기: 효과/중요도/난이도 기준으로 Top1~3 정리",
            "실행을 ‘언제’가 아니라 ‘If-Then 트리거’로 설계",
            "실패를 미리 가정(프리모템)해 방해 요인을 드러내기",
            "마지막에 ‘오늘 5분 안에 할 수 있는 가장 작은 행동’ 도출",
        ],
        "prompt_hint": "우선순위, If-Then, 프리모템, Quick Win",
    },
]

# Probing 기준: "10자 미만"이면 1회 추가 질문 (요구사항 반영)
MIN_ANSWER_CHARS = 10
SHORT_ANSWER_PATTERNS = [
    r"^모르겠",
    r"^잘\s*모르",
    r"^그냥",
    r"^없어",
    r"^모름$",
    r"^ㄴㄴ$",
    r"^몰라$",
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
    st.markdown(
        f"""
<div style="text-align:center;">
  <img src="data:image/svg+xml;base64,{b64}" style="width:100%; max-width:240px;"/>
  <div style="margin-top:6px; font-size:14px;">{label}</div>
</div>
""",
        unsafe_allow_html=True,
    )


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

    # Responses API 우선
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

    # Chat Completions fallback
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

    # main question index (0..n-1)
    if "q_index" not in st.session_state:
        st.session_state.q_index = 0

    # 질문 목록은 "main 질문"만 저장
    if "questions" not in st.session_state:
        st.session_state.questions = []

    # answers: {"q":..., "a":..., "ts":..., "kind":"main"|"probe", "main_index":int}
    if "answers" not in st.session_state:
        st.session_state.answers = []

    # probe 모드
    if "probe_active" not in st.session_state:
        st.session_state.probe_active = False
    if "probe_question" not in st.session_state:
        st.session_state.probe_question = ""
    if "probe_for_index" not in st.session_state:
        st.session_state.probe_for_index = None  # type: ignore

    # logic cross-check(충돌 질문) 생성 여부: main_index별 1회
    if "crosscheck_used_for" not in st.session_state:
        st.session_state.crosscheck_used_for = set()  # type: ignore

    if "final_report_json" not in st.session_state:
        st.session_state.final_report_json = None
    if "final_report_raw" not in st.session_state:
        st.session_state.final_report_raw = None

    # 리포트 진입 효과
    if "report_just_entered" not in st.session_state:
        st.session_state.report_just_entered = False

    # 의사결정 매트릭스 상태
    if "decision_matrix_df" not in st.session_state:
        st.session_state.decision_matrix_df = None

    if "debug_log" not in st.session_state:
        st.session_state.debug_log = []
    if "openai_api_key_input" not in st.session_state:
        st.session_state.openai_api_key_input = ""

    # 결정 유형 템플릿 자동 입력을 위한 last 기억
    if "last_decision_type" not in st.session_state:
        st.session_state.last_decision_type = st.session_state.decision_type


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
    st.session_state.probe_active = False
    st.session_state.probe_question = ""
    st.session_state.probe_for_index = None
    st.session_state.crosscheck_used_for = set()
    st.session_state.final_report_json = None
    st.session_state.final_report_raw = None
    st.session_state.decision_matrix_df = None
    st.session_state.report_just_entered = False
    st.session_state.debug_log = []


def add_answer(q: str, a: str, kind: str, main_index: int) -> None:
    st.session_state.answers.append(
        {"q": q, "a": a, "ts": datetime.now().isoformat(timespec="seconds"), "kind": kind, "main_index": main_index}
    )


def main_answer_count() -> int:
    return sum(1 for x in st.session_state.answers if x.get("kind") == "main")


# =========================
# Helpers
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


def is_too_short_answer(ans: str) -> bool:
    a = (ans or "").strip()
    if len(a) < MIN_ANSWER_CHARS:
        return True
    for pat in SHORT_ANSWER_PATTERNS:
        if re.search(pat, a):
            return True
    return False


def parse_options() -> List[str]:
    return [o.strip() for o in (st.session_state.options or "").split(",") if o.strip()]


# =========================
# Question generation
# =========================
def system_prompt_for_questions(coach: Dict[str, Any]) -> str:
    base = (
        "당신은 'AI 결정 코칭 앱'의 질문 생성기입니다.\n"
        "정답/해결책/추천을 주지 말고, 사용자가 스스로 생각을 정리하도록 질문만 던지세요.\n"
        "금지: 결론, 추천, 선택 강요, 판단문(예: A가 낫다), 지시문(해야 한다/하자).\n"
        "출력: 질문 1개만. (설명/번호/머리말 금지)\n"
    )
    if coach["id"] == "logic":
        return base + "스타일: 구조화/기준/역발상(가정 깨기) 질문.\n"
    if coach["id"] == "value":
        return base + "스타일: 감정 라벨링 + 감정/가치 분리 + 후회 최소화 질문.\n"
    return base + "스타일: If-Then 트리거/프리모템/우선순위/Quick Win을 모두 질문으로만 유도.\n"


def build_context_block() -> str:
    # 최근 main/probe 포함 최대 6개
    hist = ""
    tail = st.session_state.answers[-6:]
    for i, qa in enumerate(tail, start=1):
        tag = "PROBE" if qa.get("kind") == "probe" else "MAIN"
        hist += f"{i}) ({tag}) Q: {qa['q']}\n   A: {qa['a']}\n"

    opts = parse_options()
    opts_txt = "\n".join([f"- {o}" for o in opts]) if opts else "(미입력)"

    return textwrap.dedent(
        f"""
        [세션 시작 정보]
        - 카테고리: {st.session_state.category}
        - 결정 유형: {st.session_state.decision_type}
        - 상황 설명: {st.session_state.situation or "(미입력)"}
        - 원하는 목표: {st.session_state.goal or "(미입력)"}
        - 고려 옵션(있다면): {opts_txt}

        [최근 Q/A]
        {hist if hist.strip() else "(아직 없음)"}
        """
    ).strip()


def probing_instruction(last_q: str, last_a: str) -> str:
    return textwrap.dedent(
        f"""
        사용자의 답변이 너무 짧거나 모호합니다.
        아래의 직전 질문과 답변을 바탕으로, 사용자가 구체화할 수 있도록 딱 1개의 추가 질문(Probe)을 만들어 주세요.

        - 직전 질문: {last_q}
        - 직전 답변: {last_a}

        요구사항:
        - '구체화'를 돕는 질문(예: 예시/상황/기준/이유/범위/기간/우선순위 중 하나를 더 묻기)
        - 판단/추천/지시 금지
        - 질문 1개만 출력
        """
    ).strip()


def crosscheck_system_prompt() -> str:
    return (
        "당신은 'AI 결정 코칭 앱'의 논리 충돌 감지기입니다.\n"
        "사용자의 이전 답변들 사이에 '우선순위/기준/목표'가 서로 충돌하는지 점검하세요.\n"
        "중요: 추천/결론/정답/지시를 절대 하지 마세요. 질문을 만들 때도 강요/판단 금지.\n"
        "출력은 반드시 JSON만. (설명/코드블록 금지)\n"
    )


def crosscheck_user_prompt(current_main_index: int) -> str:
    # main 답변만 대상으로 최근 6개 정도로 교차 검증
    mains = [x for x in st.session_state.answers if x.get("kind") == "main"]
    tail = mains[-6:]
    qa = ""
    for i, x in enumerate(tail, start=1):
        qa += f"{i}) Q: {x['q']}\n   A: {x['a']}\n"

    return textwrap.dedent(
        f"""
        아래는 사용자 답변 일부입니다. 답변들 사이에 논리적 충돌(기준/우선순위의 상충)이 있는지 판단하세요.
        충돌이 있다면, 그 충돌을 사용자가 스스로 '정리'하도록 돕는 질문 1개를 제안하세요.
        충돌이 없다면 has_conflict=false로 두세요.

        [답변들]
        {qa if qa.strip() else "(답변 없음)"}

        [출력 JSON 스키마]
        {{
          "has_conflict": true/false,
          "conflict_summary": "string (없으면 빈 문자열)",
          "question": "string (has_conflict=true일 때만, 질문 1개)"
        }}

        추가 규칙:
        - question은 질문 1개만(물음표 포함 권장)
        - 판단/추천/지시/선택 강요 금지
        - 같은 내용을 반복하지 않게 간결하게
        - current_main_index={current_main_index}
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


def try_logic_crosscheck_question(main_index: int) -> Tuple[Optional[str], List[str]]:
    """
    main 질문 생성 직전에 1회:
    - 이전 main 답변들 간 충돌이 있으면, 그 충돌을 짚는 질문을 우선 반환
    - main_index별 1회만 사용
    """
    dbg: List[str] = []
    if main_index in st.session_state.crosscheck_used_for:
        return None, dbg

    # 답변이 충분히 쌓였을 때만(최소 2개 main)
    mains = [x for x in st.session_state.answers if x.get("kind") == "main"]
    if len(mains) < 2:
        return None, dbg

    system = crosscheck_system_prompt()
    user = crosscheck_user_prompt(main_index)
    txt, err, d = call_openai_text(system=system, user=user, temperature=0.2)
    dbg.extend(d)
    if not txt:
        if err:
            dbg.append(f"Crosscheck error: {err}")
        return None, dbg

    data = safe_json_parse(txt)
    if not data:
        dbg.append("Crosscheck JSON parse failed.")
        return None, dbg

    has_conflict = bool(data.get("has_conflict", False))
    q = normalize(str(data.get("question", "") or ""))

    if has_conflict and q:
        st.session_state.crosscheck_used_for.add(main_index)
        dbg.append("Crosscheck conflict detected -> using conflict question.")
        return q, dbg

    st.session_state.crosscheck_used_for.add(main_index)
    dbg.append("Crosscheck: no conflict (or no question).")
    return None, dbg


def instruction_for_question(i: int, n: int, coach_id: str) -> str:
    """
    고정 로직:
    - logic: 가정 깨기(역발상) 고정 질문 포함
    - value: 감정 후 '감정 vs 가치 분리' 고정
    - action: If-Then 트리거 + Pre-mortem 포함 + 마지막 Quick Win
    """
    if i == 0:
        return "상황의 핵심을 더 구체화하는 질문 1개"
    if i == 1:
        return "원하는 목표를 측정 가능한 형태로 정리하게 하는 질문 1개"

    # 실행 코치
    if coach_id == "action":
        if i == n - 1:
            return (
                "‘지금 앱을 끄고 나서 5분 안에 실행할 수 있는 가장 작은 행동’을 "
                "스스로 적게 만드는 질문 1개(Quick Win, 추천 금지)"
            )

        # n 크기에 따라 구성
        if i == 2:
            return "옵션/해야 할 일 3~6개를 펼치고 Top1~3 우선순위를 정리하게 하는 질문(효과/중요도/난이도 기준을 질문으로 제시)"
        if n >= 6 and i == 3:
            return "Top1을 ‘1년→이번 달→이번 주’로 쪼개 사용자가 계획을 적게 만드는 질문 1개(지시 금지)"
        if (n == 5 and i == 3) or (n >= 6 and i == n - 2):
            return (
                "프리모템(Pre-mortem) + If-Then 트리거 설계 질문 1개. "
                "예: ‘2주 뒤 실패했다고 가정하면, 가장 그럴듯한 원인 3가지는?’ "
                "그리고 각 원인에 대해 ‘만약 (If) ~ 상황이면 → (Then) ~ 행동’으로 대응을 적게 하기"
            )
        # 중간 단계: If-Then 트리거를 더 명확히
        if n >= 6 and i == 4:
            return "실행을 ‘언제’가 아니라 ‘If(어떤 상황) → Then(어떤 행동)’으로 설계하게 하는 질문 1개(트리거 2~3개)"
        return "다음 행동을 더 구체화(무엇을/얼마나/어떤 조건에서)하는 질문 1개"

    # 구조 코치
    if coach_id == "logic":
        if n >= 5 and i == n - 2:
            return (
                "역발상/반대 상황 가정 질문 1개. "
                "예: ‘만약 당신이 세운 기준이 완전히 틀렸다면 어떤 상황이 벌어질까요?’ "
                "또는 ‘가장 가능성이 낮다고 생각하는 옵션이 유리해지는 시나리오는?’"
            )
        if i == 2:
            return "선택 기준(3~5)을 뽑게 하는 질문 1개"
        if i == n - 1:
            return "마지막으로 선택 기준의 우선순위를 1~3위로 정리하게 하는 질문 1개(추천 금지)"
        if i == n - 2 and n < 5:
            return "불확실한 가정/추가로 확인할 정보 1~2개를 드러내는 질문 1개"
        return "옵션/정보/제약을 더 분리해 명료화하는 질문 1개"

    # 가치 코치
    if coach_id == "value":
        if i == 2:
            return "지금 감정(2~3개)과 그 감정의 이유를 말하게 하는 질문 1개"
        if i == 3 and n >= 5:
            return (
                "감정과 가치의 분리 질문 1개. "
                "예: ‘지금의 불안이 핵심 가치를 침해해서 생긴 건가요, 아니면 낯선 변화에 대한 본능적 거부감인가요?’"
            )
        if i == n - 2:
            return "후회 최소화 관점(1년/5년 후)을 점검하게 하는 질문 1개"
        if i == n - 1:
            return "마지막으로 ‘내 기준’을 한 문장으로 정리하게 하는 질문 1개(추천 금지)"
        return "가치 Top3(내게 중요한 것)와 내려놓을 것 1개를 정리하게 하는 질문 1개"

    return "사용자가 스스로 정리하도록 돕는 질문 1개"


def fallback_question(coach_id: str, i: int, n: int) -> str:
    if i == 0:
        return "지금 고민에서 ‘가장 핵심적인 쟁점’은 무엇인가요? (한 문장)"
    if i == 1:
        return "이번 결정으로 얻고 싶은 목표를 ‘측정 가능하게’ 바꾸면 어떻게 표현할 수 있나요? (언제까지/어느 정도)"

    if coach_id == "action":
        if i == n - 1:
            return "앱을 끄고 나서 5분 안에 할 수 있는 ‘가장 작은 행동’은 무엇인가요?"
        if i == 2:
            return "옵션/해야 할 일 3~6개를 적고, 효과/중요도/난이도를 고려했을 때 Top3는 무엇인가요?"
        if (n == 5 and i == 3) or (n >= 6 and i == n - 2):
            return "2주 뒤 실패했다고 가정하면, 가장 그럴듯한 원인 3가지는 무엇이고 각각 ‘만약 ~이면 → ~한다’로 대응을 적어볼 수 있나요?"
        if n >= 6 and i == 3:
            return "Top1을 ‘1년 목표 → 이번 달 목표 → 이번 주 계획(3개)’로 적어보면 무엇인가요?"
        if n >= 6 and i == 4:
            return "실행을 ‘언제’가 아니라 ‘만약(If) ~ 상황이면 → 그러면(Then) ~ 행동’으로 트리거 2~3개를 만들어보면 무엇인가요?"
        return "다음 행동을 더 구체화하면(무엇을/얼마나/어떤 조건에서) 어떻게 적을 수 있나요?"

    if coach_id == "logic":
        if n >= 5 and i == n - 2:
            return "만약 당신이 세운 기준이 완전히 틀렸다면 어떤 상황이 벌어질까요?"
        if i == 2:
            return "이 선택을 평가할 기준 3~5개를 적어보면 무엇인가요?"
        if i == n - 1:
            return "선택 기준의 우선순위를 1~3위로 정리하면 무엇인가요?"
        if i == n - 2 and n < 5:
            return "지금 결정을 어렵게 만드는 ‘불확실한 가정/정보’는 무엇인가요?"
        return "옵션/제약/정보를 분리해서 지금 부족한 정보는 무엇인지 적어볼까요?"

    # value
    if i == 2:
        return "지금 감정을 2~3개 단어로 적고, 각 감정이 생긴 이유를 한 줄씩 써볼까요?"
    if i == 3 and n >= 5:
        return "그 감정은 ‘지금 당장의 편안함’ 때문인가요, ‘미래의 나를 위한 가치’ 때문인가요?"
    if i == n - 2:
        return "1년/5년 뒤의 내가 지금의 나에게 뭐라고 말해줄 것 같나요?"
    return "이 고민에서 가장 중요한 가치 Top3는 무엇인가요?"


def generate_question(i: int, n: int) -> Tuple[str, Optional[str], List[str]]:
    coach = coach_by_id(st.session_state.coach_id)
    system = system_prompt_for_questions(coach)
    prev_qs = st.session_state.questions[:]

    # 1) Logic Cross-Check: 충돌 질문 우선 생성(가능하면)
    cross_q, cross_dbg = try_logic_crosscheck_question(i)
    if cross_q:
        # 중복 방지 체크는 아래 공통 로직에서 처리(너무 비슷하면 fallback)
        if not any(is_similar(cross_q, pq) for pq in prev_qs):
            return cross_q, None, cross_dbg
        cross_dbg.append("Crosscheck question was similar to previous. Falling back to normal generation.")
        # 이어서 일반 생성 진행
    dbg_acc: List[str] = cross_dbg[:]

    def prompt(nonce: int) -> str:
        prev_txt = "\n".join([f"- {q}" for q in prev_qs]) if prev_qs else "(없음)"
        return textwrap.dedent(
            f"""
            [이전 질문 목록]
            {prev_txt}

            {build_context_block()}

            [이번 질문 목적]
            {instruction_for_question(i, n, coach["id"])}

            규칙:
            - 결론/추천/정답/지시 금지
            - 질문 1개만 출력
            - 이전 질문과 너무 비슷하면 피하기

            (nonce={nonce})
            """
        ).strip()

    q1, err, dbg = call_openai_text(system=system, user=prompt(random.randint(1000, 9999)), temperature=0.7)
    dbg_acc.extend(dbg)
    if not q1:
        return fallback_question(coach["id"], i, n), err, dbg_acc

    q1 = normalize(q1)
    if not any(is_similar(q1, pq) for pq in prev_qs):
        return q1, None, dbg_acc

    dbg_acc.append("Similar question detected. Regenerating once.")
    q2, err2, dbg2 = call_openai_text(system=system, user=prompt(random.randint(10000, 99999)), temperature=0.85)
    dbg_acc.extend(dbg2)
    if q2:
        q2 = normalize(q2)
        if not any(is_similar(q2, pq) for pq in prev_qs):
            return q2, None, dbg_acc

    dbg_acc.append("Still similar after retry. Using fallback.")
    return fallback_question(coach["id"], i, n), err2, dbg_acc


def ensure_question(index: int, total: int) -> None:
    while len(st.session_state.questions) <= index:
        i = len(st.session_state.questions)
        q, err, dbg = generate_question(i, total)
        st.session_state.debug_log = dbg
        st.session_state.questions.append(q)


def generate_probe_question(last_q: str, last_a: str) -> Tuple[str, Optional[str], List[str]]:
    coach = coach_by_id(st.session_state.coach_id)
    system = system_prompt_for_questions(coach)
    user = probing_instruction(last_q, last_a)
    q, err, dbg = call_openai_text(system=system, user=user, temperature=0.6)
    if not q:
        return "방금 답변에서 ‘예시 1개’만 들어서 조금 더 자세히 설명해줄 수 있을까요?", err, dbg
    return normalize(q), None, dbg


# =========================
# Final report (Mirroring only)
# =========================
FORBIDDEN_RECOMMEND_PATTERNS = [
    r"추천",
    r"~?하는 것이 좋",
    r"해야 합니다",
    r"하시길",
    r"하는 게 낫",
    r"A를 선택",
    r"B를 선택",
    r"정답",
    r"결론",
]


def contains_forbidden_recommendation(text: str) -> bool:
    t = text or ""
    for pat in FORBIDDEN_RECOMMEND_PATTERNS:
        if re.search(pat, t):
            return True
    return False


def report_schema_hint(coach_id: str) -> str:
    base = """
반드시 JSON만 출력하세요(코드블록/설명 금지).
절대 추천/결론/정답/지시를 하지 마세요.
coaching_message는 반드시 "거울 비추기(Mirroring)" 화법만 사용하세요.
- 예: "당신은 ___를 중요하게 생각하는 것으로 보입니다"
- 예: "당신의 답변에서 ___와 ___ 사이의 긴장이 드러납니다"
금지 표현: "추천", "좋겠습니다", "해야 합니다", "하자", "정답", "결론", "A를 선택".
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


def system_prompt_for_report() -> str:
    return (
        "당신은 'AI 결정 코칭 앱'의 최종 요약 생성기입니다.\n"
        "절대 추천/결론/정답/지시를 제시하지 마세요.\n"
        "오직 사용자의 답변을 바탕으로 핵심/기준/긴장/불확실성을 정리(거울 비추기)하세요.\n"
        "coaching_message는 반드시 거울 비추기 문장만(‘당신은 ~로 보입니다’).\n"
        "출력은 반드시 JSON만.\n"
    )


def build_qa_text_for_report() -> str:
    qa_text = ""
    for i, qa in enumerate(st.session_state.answers, start=1):
        tag = "PROBE" if qa.get("kind") == "probe" else "MAIN"
        qa_text += f"{i}) ({tag}) Q: {qa['q']}\n   A: {qa['a']}\n"
    return qa_text


def generate_final_report_json() -> Tuple[Optional[Dict[str, Any]], Optional[str], List[str], Optional[str]]:
    coach = coach_by_id(st.session_state.coach_id)
    system = system_prompt_for_report()

    qa_text = build_qa_text_for_report()
    opts = parse_options()

    user = textwrap.dedent(
        f"""
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
- 추천/결론/정답/지시 금지
- coaching_message는 거울 비추기만
- 사용자가 말하지 않은 계획을 ‘지어내지’ 마세요
"""
    ).strip()

    text, err, dbg = call_openai_text(system=system, user=user, temperature=0.25)
    if not text:
        return None, err, dbg, None

    data = safe_json_parse(text)
    if data is None:
        return None, "리포트 JSON 파싱 실패(모델이 JSON만 출력하지 않았을 수 있음)", dbg, text

    combined = json.dumps(data, ensure_ascii=False)
    if contains_forbidden_recommendation(combined):
        dbg.append("Forbidden recommendation-like phrasing detected. Regenerating once with stricter warning.")
        stricter_user = user + "\n\n[경고] 이전 출력에 추천/지시 표현이 포함되었습니다. 절대 포함하지 말고 거울 비추기 문장만 사용하세요."
        text2, err2, dbg2 = call_openai_text(system=system, user=stricter_user, temperature=0.1)
        dbg.extend(dbg2)
        if text2:
            data2 = safe_json_parse(text2)
            if data2 is not None and not contains_forbidden_recommendation(json.dumps(data2, ensure_ascii=False)):
                return data2, None, dbg, text2
        dbg.append("Regeneration did not fully remove forbidden phrasing.")
        return data, None, dbg, text

    return data, None, dbg, text


# =========================
# Report rendering
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


def render_criteria(data: Dict[str, Any]) -> List[str]:
    st.subheader("선택 기준 정리(우선순위 포함)")
    crit = data.get("criteria", []) or []
    if not crit:
        st.caption("선택 기준이 충분히 드러나지 않았어요.")
        return []
    rows = []
    names: List[str] = []
    for c in crit:
        nm = str(c.get("name", "") or "").strip()
        if nm:
            names.append(nm)
        rows.append({"기준": nm, "우선순위(1~5)": c.get("priority", ""), "왜 중요한가": c.get("why", "")})
    st.dataframe(rows, use_container_width=True, hide_index=True)
    return names


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
    st.subheader("코칭 메시지(거울 비추기)")
    msgs = data.get("coaching_message", []) or []
    for m in msgs:
        st.write(f"- {m}")


def render_next_question(data: Dict[str, Any]) -> None:
    st.subheader("다음에 스스로에게 던질 질문(1개)")
    st.write(f"**{data.get('next_self_question','')}**")


# =========================
# Mirroring 분석(키워드/감정어)
# =========================
STOPWORDS = {
    "그냥",
    "너무",
    "진짜",
    "근데",
    "그리고",
    "그래서",
    "하지만",
    "제가",
    "저는",
    "나는",
    "내가",
    "이게",
    "그게",
    "저",
    "것",
    "수",
    "좀",
    "약간",
    "때문",
    "때문에",
    "같아요",
    "같은",
    "하는",
    "해야",
    "하고",
    "있는",
    "있다",
    "없다",
    "없어요",
    "모르겠",
    "모르겠어요",
}

EMOTION_WORDS = [
    "불안",
    "두려움",
    "걱정",
    "긴장",
    "답답",
    "후회",
    "죄책감",
    "부담",
    "스트레스",
    "우울",
    "짜증",
    "화",
    "분노",
    "설렘",
    "기대",
    "안도",
    "편안",
    "행복",
    "의욕",
    "지침",
    "번아웃",
]


def analyze_mirroring_from_answers() -> Tuple[pd.DataFrame, pd.DataFrame]:
    text = " ".join([str(x.get("a", "")) for x in st.session_state.answers if x.get("a")])
    clean = re.sub(r"[^\w가-힣 ]", " ", text)
    clean = re.sub(r"\s+", " ", clean).strip().lower()

    toks = [t for t in clean.split(" ") if len(t) >= 2 and t not in STOPWORDS]
    # 키워드 빈도
    freq: Dict[str, int] = {}
    for t in toks:
        freq[t] = freq.get(t, 0) + 1
    kw = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10]
    kw_df = pd.DataFrame(kw, columns=["키워드", "빈도"])

    # 감정어 빈도(부분 포함)
    emo_freq: Dict[str, int] = {}
    for ew in EMOTION_WORDS:
        c = len(re.findall(re.escape(ew), text))
        if c > 0:
            emo_freq[ew] = c
    emo = sorted(emo_freq.items(), key=lambda x: x[1], reverse=True)[:10]
    emo_df = pd.DataFrame(emo, columns=["감정어", "빈도"])
    return kw_df, emo_df


def render_mirroring_visual() -> None:
    st.subheader("내면의 목소리(Mirroring) — 답변에서 많이 등장한 표현")
    kw_df, emo_df = analyze_mirroring_from_answers()

    c1, c2 = st.columns(2)
    with c1:
        st.write("**자주 등장한 키워드(Top 10)**")
        if len(kw_df) == 0:
            st.caption("키워드가 충분히 잡히지 않았어요.")
        else:
            st.dataframe(kw_df, use_container_width=True, hide_index=True)
            st.bar_chart(kw_df.set_index("키워드")["빈도"])
    with c2:
        st.write("**감정어(Top 10)**")
        if len(emo_df) == 0:
            st.caption("뚜렷한 감정어가 많이 등장하지 않았어요.")
        else:
            st.dataframe(emo_df, use_container_width=True, hide_index=True)
            st.bar_chart(emo_df.set_index("감정어")["빈도"])

    st.caption("이 결과는 ‘정답’이 아니라, 당신의 답변에 나타난 반복 표현을 요약한 거울입니다.")


# =========================
# 의사결정 매트릭스
# =========================
def build_decision_matrix(options: List[str], criteria_names: List[str]) -> pd.DataFrame:
    if not options:
        options = ["옵션 1", "옵션 2"]
    if not criteria_names:
        criteria_names = ["기준 1", "기준 2", "기준 3"]

    cols = ["옵션"] + criteria_names + ["메모"]
    rows = []
    for opt in options:
        r = {"옵션": opt, "메모": ""}
        for c in criteria_names:
            r[c] = 3
        rows.append(r)
    return pd.DataFrame(rows, columns=cols)


def render_decision_matrix(criteria_names: List[str], data: Dict[str, Any]) -> None:
    st.subheader("의사결정 매트릭스(직접 점수 매기기)")
    st.caption("각 옵션이 ‘내 기준’에서 어느 정도인지 1~5점으로 적어보세요. 점수 자체는 결론이 아니라, 생각을 꺼내는 도구예요.")

    # 옵션 소스: 사용자 입력 options → 없으면 리포트의 options_mentioned → 그래도 없으면 기본
    user_opts = parse_options()
    report_opts = (data.get("summary", {}) or {}).get("options_mentioned", []) or []
    opts = user_opts or [str(x) for x in report_opts if str(x).strip()] or ["옵션 1", "옵션 2"]

    if st.session_state.decision_matrix_df is None:
        st.session_state.decision_matrix_df = build_decision_matrix(opts, criteria_names)

    df: pd.DataFrame = st.session_state.decision_matrix_df

    # 옵션 변경에 대응(간단 동기화)
    existing_opts = [str(x) for x in df["옵션"].tolist()] if "옵션" in df.columns else []
    if set(existing_opts) != set(opts):
        st.session_state.decision_matrix_df = build_decision_matrix(opts, criteria_names)
        df = st.session_state.decision_matrix_df

    # 기준 변경에 대응(간단 동기화)
    desired_cols = ["옵션"] + (criteria_names or []) + ["메모"]
    if list(df.columns) != desired_cols:
        st.session_state.decision_matrix_df = build_decision_matrix(opts, criteria_names)
        df = st.session_state.decision_matrix_df

    col_cfg: Dict[str, Any] = {}
    for c in criteria_names:
        col_cfg[c] = st.column_config.NumberColumn(c, min_value=1, max_value=5, step=1, format="%d")

    edited = st.data_editor(
        df,
        use_container_width=True,
        hide_index=True,
        column_config=col_cfg,
        num_rows="fixed",
    )
    st.session_state.decision_matrix_df = edited

    # 합계(참고용)
    if criteria_names:
        try:
            totals = edited[criteria_names].sum(axis=1)
            show = edited.copy()
            show["총점(참고)"] = totals
            st.write("**총점(참고용)**")
            st.dataframe(show[["옵션", "총점(참고)"]], use_container_width=True, hide_index=True)
            st.caption("총점은 ‘결론’이 아니라, 어떤 옵션이 어떤 기준에서 강/약한지 다시 보게 하는 참고치예요.")
        except Exception:
            pass


# =========================
# 공유/저장(클립보드 + 다운로드)
# =========================
def render_copy_to_clipboard_button(text: str, button_label: str = "클립보드에 복사") -> None:
    # Streamlit 기본에 복사 기능이 없어 JS로 구현
    # (HTTPS/브라우저 정책에 따라 동작하지 않을 수 있음)
    safe = text.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")
    html = f"""
    <div style="display:flex; gap:8px; align-items:center;">
      <button
        onclick="navigator.clipboard.writeText(`{safe}`).then(()=>{{const el=document.getElementById('cpmsg'); el.innerText='복사됨'; setTimeout(()=>el.innerText='',1200);}});"
        style="padding:8px 12px; border-radius:10px; border:1px solid #444; background:#111; color:#fff; cursor:pointer;">
        {button_label}
      </button>
      <span id="cpmsg" style="font-size:12px; opacity:0.8;"></span>
    </div>
    """
    st.components.v1.html(html, height=55)


def build_report_text_for_export(data: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("🪨 돌멩이 AI 결정 코칭 — 최종 정리(거울 비추기)")
    lines.append(f"- 생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("[세션 정보]")
    lines.append(f"- 카테고리: {st.session_state.category}")
    lines.append(f"- 결정 유형: {st.session_state.decision_type}")
    lines.append(f"- 상황 설명: {st.session_state.situation}")
    lines.append(f"- 목표: {st.session_state.goal}")
    lines.append(f"- 옵션: {st.session_state.options or '(없음)'}")
    lines.append("")
    lines.append("[리포트 JSON]")
    lines.append(json.dumps(data, ensure_ascii=False, indent=2))
    lines.append("")
    lines.append("[Q/A]")
    for i, qa in enumerate(st.session_state.answers, start=1):
        tag = "PROBE" if qa.get("kind") == "probe" else "MAIN"
        lines.append(f"{i}. ({tag}) Q: {qa['q']}")
        lines.append(f"   A: {qa['a']}")
        lines.append(f"   ts: {qa['ts']}")
        lines.append("")
    return "\n".join(lines).strip()


# =========================
# Back 버튼 로직
# =========================
def handle_back() -> None:
    """
    요구사항:
    - 질문 화면에서 q_index를 줄이고 answers에서 마지막 답변을 제거하여 이전 질문으로 돌아가기
    - probe 상태가 꼬이지 않도록 정리
    """
    if not st.session_state.answers:
        st.session_state.q_index = max(0, int(st.session_state.q_index) - 1)
        st.session_state.probe_active = False
        st.session_state.probe_question = ""
        st.session_state.probe_for_index = None
        return

    last = st.session_state.answers.pop()

    # probe는 같은 main_index에 대한 추가 질문이므로, 뒤로 가기 시 probe 모드 해제하고 해당 main으로 유지
    if last.get("kind") == "probe":
        st.session_state.probe_active = False
        st.session_state.probe_question = ""
        st.session_state.probe_for_index = None
        # q_index는 last의 main_index로 맞춤(대개 현재)
        st.session_state.q_index = int(last.get("main_index", st.session_state.q_index))
        return

    # main 답변을 되돌리면, 해당 질문으로 돌아가야 함
    mi = int(last.get("main_index", 0))
    st.session_state.probe_active = False
    st.session_state.probe_question = ""
    st.session_state.probe_for_index = None
    st.session_state.q_index = max(0, mi)


# =========================
# Sidebar: 결정 유형 템플릿 자동 입력
# =========================
def on_decision_type_change() -> None:
    new_dt = st.session_state.decision_type
    prev_dt = st.session_state.last_decision_type
    st.session_state.last_decision_type = new_dt

    template = DECISION_TEMPLATES.get(new_dt, "")
    if not template:
        return

    cur = (st.session_state.situation or "").strip()
    # "자동 입력"이지만, 사용자가 이미 작성한 내용을 덮어쓰지 않도록:
    # - 비어있거나
    # - 기존이 [가이드]로 시작하면(템플릿 상태) 덮어쓰기
    if (not cur) or cur.startswith("[가이드]"):
        st.session_state.situation = template


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

    st.selectbox("결정 유형", DECISION_TYPES, key="decision_type", on_change=on_decision_type_change)

    st.text_area("상황 설명", key="situation", height=120, placeholder="무슨 일이 있었고 무엇을 결정해야 하나요?")
    st.text_input("원하는 목표", key="goal", placeholder="이 결정에서 얻고 싶은 결과(가능하면 측정 가능하게)")
    st.text_input("옵션(쉼표로 구분, 선택)", key="options", placeholder="예: A, B, C")

    with st.expander("결정 유형 가이드 다시 넣기"):
        st.caption("상황 설명이 비어있거나 [가이드] 텍스트라면, 결정 유형에 맞춘 템플릿이 자동으로 들어갑니다.")
        if st.button("가이드 삽입/갱신", use_container_width=True):
            tmpl = DECISION_TEMPLATES.get(st.session_state.decision_type, "")
            if tmpl:
                st.session_state.situation = tmpl
                st.rerun()

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
            done = main_answer_count() >= int(st.session_state.num_questions)
            if st.button("최종 결과로", use_container_width=True, disabled=not done):
                st.session_state.page = "report"
                st.session_state.report_just_entered = True
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
    render_hero_pebble(progress, f"진행도: {int(progress * 100)}%")

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
    st.caption("한 화면에 한 질문. 답변을 저장하면 다음으로 진행합니다. (답변이 10자 미만이면 1회 구체화 질문을 합니다)")

    q_idx = int(st.session_state.q_index)
    q_idx = max(0, min(q_idx, nq - 1))

    # 메인 질문 준비
    ensure_question(q_idx, nq)
    main_q = st.session_state.questions[q_idx]

    # 현재 표시할 질문: probe가 활성화면 probe, 아니면 main
    if st.session_state.probe_active and st.session_state.probe_for_index == q_idx:
        show_q = st.session_state.probe_question
        kind = "probe"
        badge = "추가 질문(구체화)"
    else:
        show_q = main_q
        kind = "main"
        badge = "메인 질문"

    # 상단 컨트롤: Back
    top_c1, top_c2, top_c3 = st.columns([1, 2, 1])
    with top_c1:
        if st.button("⬅️ 이전으로", use_container_width=True, disabled=(q_idx == 0 and not st.session_state.answers)):
            handle_back()
            st.rerun()
    with top_c3:
        st.caption(f"메인 답변: {main_answer_count()} / {nq}")

    st.subheader(f"Q{q_idx + 1} / {nq}  ·  {badge}")
    with st.container(border=True):
        st.markdown(f"**{show_q}**")

    with st.form(f"answer_form_{q_idx}_{kind}", clear_on_submit=True):
        hint = ""
        if st.session_state.answers:
            last_a = st.session_state.answers[-1]["a"]
            hint = f"이전 답 요약: {last_a[:90]}{'…' if len(last_a) > 90 else ''}"
        ans = st.text_area("답변", placeholder=hint or "여기에 답변을 입력하세요", height=150)
        submitted = st.form_submit_button("답변 저장", use_container_width=True)

    if submitted:
        a = (ans or "").strip()
        if not a:
            st.warning("답변이 비어 있습니다. 한 줄만 입력해도 진행 가능합니다.")
        else:
            add_answer(show_q, a, kind=kind, main_index=q_idx)

            if kind == "probe":
                # probe 종료 → 다음 main으로
                st.session_state.probe_active = False
                st.session_state.probe_question = ""
                st.session_state.probe_for_index = None
                st.session_state.q_index = min(q_idx + 1, nq - 1)

            else:
                # main 답변이 너무 짧으면 probe 생성(1회) + 같은 단계 유지
                if is_too_short_answer(a):
                    pq, err, dbg = generate_probe_question(show_q, a)
                    st.session_state.debug_log = dbg
                    st.session_state.probe_active = True
                    st.session_state.probe_question = pq
                    st.session_state.probe_for_index = q_idx
                else:
                    # 정상 진행
                    if main_answer_count() >= nq:
                        st.session_state.page = "report"
                        st.session_state.report_just_entered = True
                        st.session_state.q_index = nq - 1
                    else:
                        st.session_state.q_index = min(q_idx + 1, nq - 1)

            st.rerun()

    with st.expander("답변 기록"):
        grouped: Dict[int, List[Dict[str, Any]]] = {}
        for qa in st.session_state.answers:
            grouped.setdefault(int(qa.get("main_index", 0)), []).append(qa)

        for mi in sorted(grouped.keys()):
            st.markdown(f"### Q{mi + 1}")
            for qa in grouped[mi]:
                tag = "PROBE" if qa.get("kind") == "probe" else "MAIN"
                st.markdown(f"**({tag}) {qa['q']}**")
                st.write(qa["a"])
                st.caption(qa["ts"])
                st.divider()

    with st.expander("디버그 로그"):
        st.write(st.session_state.debug_log)

else:
    st.title("최종 정리")
    st.caption("추천/정답 없이, 고민의 핵심과 기준을 ‘거울 비추기’ 방식으로 정리합니다.")

    if st.session_state.report_just_entered:
        st.balloons()
        st.session_state.report_just_entered = False

    if main_answer_count() < nq:
        st.warning("아직 모든 메인 질문이 완료되지 않았습니다. 질문 페이지로 돌아가 답변을 완료하세요.")
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
        criteria_names = render_criteria(data)

        # 의사결정 매트릭스(옵션x기준)
        render_decision_matrix(criteria_names, data)

        # 코치별 섹션
        if coach["id"] == "action":
            render_action_visualization(data)
        elif coach["id"] == "logic":
            render_key_points_logic(data)
        else:
            render_emotions_values(data)

        # 내면의 목소리(Mirroring 시각화)
        render_mirroring_visual()

        render_coaching_message(data)
        render_next_question(data)

        st.subheader("공유/저장")
        export_text = build_report_text_for_export(data)
        render_copy_to_clipboard_button(export_text, "리포트 텍스트 복사")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            "리포트 .txt 다운로드",
            data=export_text.encode("utf-8"),
            file_name=f"pebble_decision_report_{ts}.txt",
            mime="text/plain",
            use_container_width=True,
        )

        st.subheader("공유용(JSON)")
        st.code(json.dumps(data, ensure_ascii=False, indent=2), language="json")

        if contains_forbidden_recommendation(json.dumps(data, ensure_ascii=False)):
            st.warning("리포트에 추천/지시처럼 보이는 표현이 섞였을 수 있어요. 필요하면 ‘정리 생성/새로고침’을 눌러 보세요.")

        # 결정 유효기간 문구
        valid_until = (datetime.now().date() + timedelta(days=7)).strftime("%Y-%m-%d")
        st.divider()
        st.caption(f"이 정리는 **{valid_until}**까지 유효합니다.")

    elif st.session_state.final_report_raw:
        st.warning("JSON 파싱 실패로 원문을 표시합니다.")
        st.code(st.session_state.final_report_raw, language="text")

    with st.expander("Q/A 전체 보기"):
        for i, qa in enumerate(st.session_state.answers, start=1):
            tag = "PROBE" if qa.get("kind") == "probe" else "MAIN"
            st.markdown(f"**{i}. ({tag}) {qa['q']}**")
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
  - pandas
"""
    )

