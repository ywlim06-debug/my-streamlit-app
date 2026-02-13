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
# 유지 기능:
# - Logic Cross-Check(답변 간 충돌 감지 → 충돌을 짚는 질문 우선 생성)
# - Probing(답변 10자 미만이면 1회 구체화 질문)
# - "잘 모르겠어요" 등 난감 답변 시: 질문 재프레이밍/대체 질문 1회 생성(상황 반영)
# - Action Coach 강화: If-Then 트리거 + Pre-mortem 질문 포함
# - Back 버튼
# - 결정 유형별 템플릿(2단계에서 상황설명 가이드 삽입 버튼)
# - 리포트: 의사결정 매트릭스(st.data_editor), Mirroring 시각화, 복사/다운로드, 유효기간, balloons
#
# 이번 반영(추가 기능 + 개선):
# 1) “모순/긴장 지도” 시각화(리포트)
# 2) “정보 부족 체크리스트”(질문 형태 1~3개, 리포트)
# 3) “세션 템플릿 저장/불러오기”(프리셋)
# 4) “감정 변화 트래킹(셀프 체크)” (질문 시작 전/리포트에서)
# 5) “다음 세션 연결 질문”(리포트 next_self_question 답변 → 새 세션 시작)
# 6) “프라이버시 모드”(답변 기록 숨기기 + 내보내기 마스킹 + 일부 화면 가림)
# 7) crosscheck_used_for: set() → list 저장(세션 직렬화 안정)
# 8) 난감 답변 트리거 정교화(“난감 키워드 + 정보 부족”일 때만)
# 9) JSON 파싱 robustness 강화(후보 여러 개 추출 후 첫 성공)
# 10) 금칙어 탐지 정밀도 개선(“추천” 단독 제거, 문장 패턴 중심)
# 11) 온보딩/리포트: LLM 실패 시 규칙 기반 fallback JSON 생성(UX 안정)
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

# Probing 기준: 10자 미만이면 1회 추가 질문
MIN_ANSWER_CHARS = 10

# 난감 키워드(“모르겠/감이 안 와” 등)
CONFUSED_ANSWER_PATTERNS = [
    r"모르겠",
    r"잘\s*모르",
    r"감이\s*안",
    r"생각이\s*안",
    r"어렵",
]

# “짧은/회피” 답변 패턴(기존 probing)
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
def coach_by_id(coach_id: str) -> Dict[str, Any]:
    for c in COACHES:
        if c["id"] == coach_id:
            return c
    return COACHES[0]


def init_state() -> None:
    if "page" not in st.session_state:
        st.session_state.page = "landing"

    if "user_problem" not in st.session_state:
        st.session_state.user_problem = ""

    if "category" not in st.session_state:
        st.session_state.category = TOPIC_CATEGORIES[0][0]
    if "decision_type" not in st.session_state:
        st.session_state.decision_type = DECISION_TYPES[0]
    if "coach_id" not in st.session_state:
        st.session_state.coach_id = COACHES[0]["id"]
    if "goal" not in st.session_state:
        st.session_state.goal = ""
    if "options" not in st.session_state:
        st.session_state.options = ""
    if "situation" not in st.session_state:
        st.session_state.situation = ""

    if "num_questions" not in st.session_state:
        st.session_state.num_questions = 5

    if "q_index" not in st.session_state:
        st.session_state.q_index = 0
    if "questions" not in st.session_state:
        st.session_state.questions = []
    if "answers" not in st.session_state:
        st.session_state.answers = []

    # probe 상태 + 종류(짧음 probe vs 재프레이밍)
    if "probe_active" not in st.session_state:
        st.session_state.probe_active = False
    if "probe_question" not in st.session_state:
        st.session_state.probe_question = ""
    if "probe_for_index" not in st.session_state:
        st.session_state.probe_for_index = None  # type: ignore
    if "probe_mode" not in st.session_state:
        st.session_state.probe_mode = ""  # "short" | "reframe" | ""

    # ★ 세션 직렬화 안정: set() 대신 list 저장
    if "crosscheck_used_for" not in st.session_state:
        st.session_state.crosscheck_used_for = []  # list[int]

    if "final_report_json" not in st.session_state:
        st.session_state.final_report_json = None
    if "final_report_raw" not in st.session_state:
        st.session_state.final_report_raw = None
    if "report_just_entered" not in st.session_state:
        st.session_state.report_just_entered = False

    if "decision_matrix_df" not in st.session_state:
        st.session_state.decision_matrix_df = None

    # 온보딩 추천 상태
    if "onboarding_reco" not in st.session_state:
        st.session_state.onboarding_reco = None
    if "onboarding_raw" not in st.session_state:
        st.session_state.onboarding_raw = None
    if "onboarding_applied" not in st.session_state:
        st.session_state.onboarding_applied = False

    # 프리셋(세션 템플릿)
    if "saved_templates" not in st.session_state:
        st.session_state.saved_templates = []  # list[dict]

    # 감정 트래킹(셀프 체크)
    if "emotion_pre" not in st.session_state:
        st.session_state.emotion_pre = None
    if "emotion_post" not in st.session_state:
        st.session_state.emotion_post = None

    # 프라이버시 모드
    if "privacy_mode" not in st.session_state:
        st.session_state.privacy_mode = False
    if "hide_history" not in st.session_state:
        st.session_state.hide_history = False
    if "mask_export" not in st.session_state:
        st.session_state.mask_export = True

    if "debug_log" not in st.session_state:
        st.session_state.debug_log = []
    if "openai_api_key_input" not in st.session_state:
        st.session_state.openai_api_key_input = ""


def reset_flow(to_page: str = "landing", keep_problem: bool = False) -> None:
    """
    keep_problem=True면 user_problem은 유지하고 나머지 흐름을 초기화(유실 방지 옵션)
    """
    st.session_state.page = to_page

    if not keep_problem:
        st.session_state.user_problem = ""

    # 온보딩 추천 상태 초기화
    st.session_state.onboarding_reco = None
    st.session_state.onboarding_raw = None
    st.session_state.onboarding_applied = False

    # setup details
    st.session_state.category = TOPIC_CATEGORIES[0][0]
    st.session_state.decision_type = DECISION_TYPES[0]
    st.session_state.coach_id = COACHES[0]["id"]
    st.session_state.goal = ""
    st.session_state.options = ""
    st.session_state.situation = (st.session_state.user_problem or "").strip()

    st.session_state.num_questions = int(st.session_state.get("num_questions", 5))

    # q flow
    st.session_state.q_index = 0
    st.session_state.questions = []
    st.session_state.answers = []
    st.session_state.probe_active = False
    st.session_state.probe_question = ""
    st.session_state.probe_for_index = None
    st.session_state.probe_mode = ""
    st.session_state.crosscheck_used_for = []  # list로 초기화

    # report
    st.session_state.final_report_json = None
    st.session_state.final_report_raw = None
    st.session_state.decision_matrix_df = None
    st.session_state.report_just_entered = False

    # emotion
    st.session_state.emotion_pre = None
    st.session_state.emotion_post = None

    st.session_state.debug_log = []


def add_answer(q: str, a: str, kind: str, main_index: int, subkind: str = "") -> None:
    st.session_state.answers.append(
        {
            "q": q,
            "a": a,
            "ts": datetime.now().isoformat(timespec="seconds"),
            "kind": kind,  # "main" | "probe"
            "subkind": subkind,  # "short" | "reframe" | ""
            "main_index": main_index,
        }
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


def _has_meaningful_content(ans: str) -> bool:
    """
    난감 키워드가 있어도, 실제로는 꽤 많은 구체 정보가 담긴 경우가 있음.
    - ‘난감 키워드 + 정보 부족’일 때만 재프레이밍을 트리거하기 위한 보조 함수.
    """
    a = normalize(ans)
    if not a:
        return False

    # 숫자/기간/고유명/옵션(A,B 등) 같은 “정보성” 신호가 있으면 의미 있는 내용으로 간주
    signals = 0
    if re.search(r"\d", a):
        signals += 1
    if re.search(r"(이번\s*주|다음\s*주|이번\s*달|올해|내년|오늘|내일|어제|주말)", a):
        signals += 1
    if re.search(r"(A|B|C)\s*(안|을|를)?", a):
        signals += 1
    if len(a) >= 35:
        signals += 1
    # 쉼표/줄바꿈 등 나열 구조도 정보성 신호
    if a.count(",") >= 2:
        signals += 1

    return signals >= 2


def is_confused_answer(ans: str) -> bool:
    """
    개선: “난감 키워드”가 있더라도 정보가 충분하면 재프레이밍을 강제하지 않음.
    즉, (난감 키워드) AND (짧거나 정보 부족)일 때만 True.
    """
    a = (ans or "").strip()
    if not a:
        return False

    has_confused_kw = any(re.search(pat, a) for pat in CONFUSED_ANSWER_PATTERNS)
    if not has_confused_kw:
        return False

    # 짧거나 정보가 부족할 때만 난감 처리
    if is_too_short_answer(a):
        return True
    if not _has_meaningful_content(a):
        return True
    return False


def parse_options() -> List[str]:
    return [o.strip() for o in (st.session_state.options or "").split(",") if o.strip()]


def mask_text_for_privacy(text: str) -> str:
    """
    프라이버시 모드 내보내기 마스킹:
    - 이메일, 전화/숫자열(길게), URL 비슷한 것, 날짜/시간 일부 마스킹
    - 완벽한 익명화가 아니라 “공유 위험 낮추기” 목적
    """
    t = text or ""
    t = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[이메일]", t)
    t = re.sub(r"(https?://\S+)", "[링크]", t)
    # 길게 이어진 숫자(계좌/전화 등 가능)
    t = re.sub(r"\b\d{6,}\b", "[숫자]", t)
    # 날짜 형태 일부
    t = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", "[날짜]", t)
    # 시간 형태 일부
    t = re.sub(r"\b\d{1,2}:\d{2}(:\d{2})?\b", "[시간]", t)
    return t


# =========================
# JSON parsing robustness
# =========================
def extract_json_candidates(text: str) -> List[str]:
    """
    중괄호 균형 스캔으로 JSON 후보들을 추출.
    - 모델이 JSON 앞뒤로 설명을 붙이거나
    - 중괄호 블록이 여러 개인 경우를 견딤
    """
    if not text:
        return []
    s = text.strip()

    candidates: List[str] = []
    stack = 0
    start = None

    for i, ch in enumerate(s):
        if ch == "{":
            if stack == 0:
                start = i
            stack += 1
        elif ch == "}":
            if stack > 0:
                stack -= 1
                if stack == 0 and start is not None:
                    block = s[start : i + 1].strip()
                    if len(block) >= 2:
                        candidates.append(block)
                    start = None

    # 가장 큰 블록을 우선(대개 최종 JSON)
    candidates = sorted(set(candidates), key=len, reverse=True)
    return candidates


def safe_json_parse(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    raw = text.strip()

    # 1) 통으로 파싱 시도
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 2) 후보 블록들 중 첫 성공 사용
    for cand in extract_json_candidates(raw):
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue

    return None


# =========================
# Onboarding: AI 분석/추천(2단계)
# =========================
def system_prompt_for_onboarding() -> str:
    return (
        "당신은 'AI 결정 코칭 앱'의 온보딩 분석기입니다.\n"
        "사용자의 고민 텍스트를 읽고, 아래 항목을 '추천'하되, 결론/정답/지시를 하지 마세요.\n"
        "추천은 '분류/초안 제안' 수준이며 사용자가 수정할 수 있습니다.\n"
        "출력은 반드시 JSON만(설명/코드블록 금지).\n"
    )


def user_prompt_for_onboarding(problem_text: str) -> str:
    cats = [c[0] for c in TOPIC_CATEGORIES]
    coaches = [{"id": c["id"], "name": c["name"], "tagline": c["tagline"]} for c in COACHES]
    dtypes = DECISION_TYPES

    return textwrap.dedent(
        f"""
        [사용자 고민]
        {problem_text}

        [가능한 카테고리]
        {cats}

        [가능한 결정 유형]
        {dtypes}

        [가능한 코치]
        {coaches}

        아래 JSON 스키마로만 출력하세요:
        {{
          "recommended_category": "string (cats 중 하나)",
          "recommended_decision_type": "string (dtypes 중 하나)",
          "recommended_coach_id": "string (logic|value|action)",
          "coach_reason": "string (짧게, 왜 이 코치가 맞는지)",
          "goal_draft": "string (사용자가 얻고 싶어 할 법한 '원하는 목표' 초안, 지시/추천 금지 표현)",
          "options_hint": "string (옵션이 있을 수도 있음을 상기시키는 짧은 질문형 힌트. 없다면 빈 문자열 가능)"
        }}

        규칙:
        - 결론/정답/지시/강요 금지
        - goal_draft는 '초안'으로만 제시
        """
    ).strip()


def onboarding_fallback(problem_text: str) -> Dict[str, Any]:
    # 아주 보수적인 규칙 기반 초안
    txt = normalize(problem_text)
    cat = "📦 기타"
    if any(k in txt for k in ["취업", "이직", "진로", "전공", "학업", "대학원"]):
        cat = "🎓 학업/진로"
    elif any(k in txt for k in ["프로젝트", "업무", "팀", "회사", "리더", "성과", "커리어"]):
        cat = "💼 커리어/일"
    elif any(k in txt for k in ["연인", "친구", "가족", "갈등", "관계", "대화"]):
        cat = "💖 관계"
    elif any(k in txt for k in ["돈", "예산", "소비", "저축", "투자", "구매"]):
        cat = "💰 돈/소비"
    elif any(k in txt for k in ["불안", "번아웃", "우울", "스트레스", "마음", "삶"]):
        cat = "🧠 마음/삶"

    dtype = "해야 할지 말지(Yes/No)" if any(k in txt for k in ["할까", "말까", "해야", "그만", "시작"]) else "여러 옵션 중 선택"
    coach_id = "logic"
    if any(k in txt for k in ["불안", "후회", "감정", "마음", "관계"]):
        coach_id = "value"
    if any(k in txt for k in ["계획", "실행", "루틴", "습관", "일정", "공부법"]):
        coach_id = "action"

    return {
        "recommended_category": cat,
        "recommended_decision_type": dtype,
        "recommended_coach_id": coach_id,
        "coach_reason": "사용자가 말한 고민에서 ‘정리/기준/감정/실행’ 중 무엇이 두드러지는지에 맞춘 초안입니다.",
        "goal_draft": "지금 고민에서 내가 중요하게 여기는 기준과 감당 가능한 리스크를 더 선명하게 적어보고 싶다(초안).",
        "options_hint": "지금 떠오르는 선택지/가능성(있다면)을 쉼표로 2~4개만 적어볼 수 있을까요?",
    }


def generate_onboarding_recommendation(problem_text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str], List[str], Optional[str]]:
    system = system_prompt_for_onboarding()
    user = user_prompt_for_onboarding(problem_text)
    txt, err, dbg = call_openai_text(system=system, user=user, temperature=0.2)
    if not txt:
        # fallback
        fb = onboarding_fallback(problem_text)
        dbg.append("Onboarding fallback used (no model output).")
        return fb, err, dbg, None

    data = safe_json_parse(txt)
    if not data:
        fb = onboarding_fallback(problem_text)
        dbg.append("Onboarding fallback used (JSON parse fail).")
        return fb, "온보딩 추천 JSON 파싱 실패(대체 초안을 표시합니다)", dbg, txt

    return data, None, dbg, txt


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


def reframe_instruction(last_q: str, last_a: str) -> str:
    return textwrap.dedent(
        f"""
        사용자가 질문에 대해 "잘 모르겠어요/감이 안 와요/어려워요" 같은 반응을 보였습니다.
        아래 정보를 참고해, 사용자의 상황에 맞게 질문을 더 쉽게 풀어 쓰거나(재프레이밍),
        또는 더 답하기 쉬운 대체 질문 1개를 만들어 주세요.

        [사용자 상황 설명]
        {st.session_state.situation or "(미입력)"}

        [직전 질문]
        {last_q}

        [사용자 답변(난감 표현 포함)]
        {last_a}

        요구사항:
        - 질문 1개만 출력
        - 정답/추천/지시/판단 금지
        - “먼저 A를 하세요” 같은 단계 지시 금지
        - 답하기 쉬운 형태로:
          예) 선택지를 제공(둘 중 무엇에 더 가까운지), 예시 요구, 범위 좁히기(이번 주/오늘), 기준 1개만 묻기 등
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
        - question은 질문 1개만
        - 판단/추천/지시/선택 강요 금지
        - current_main_index={current_main_index}
        """
    ).strip()


def try_logic_crosscheck_question(main_index: int) -> Tuple[Optional[str], List[str]]:
    dbg: List[str] = []
    used_set = set(int(x) for x in (st.session_state.crosscheck_used_for or []))

    if main_index in used_set:
        return None, dbg

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

    # list로 저장
    used_set.add(main_index)
    st.session_state.crosscheck_used_for = sorted(list(used_set))

    if has_conflict and q:
        dbg.append("Crosscheck conflict detected -> using conflict question.")
        return q, dbg

    dbg.append("Crosscheck: no conflict (or no question).")
    return None, dbg


def instruction_for_question(i: int, n: int, coach_id: str) -> str:
    if i == 0:
        return "상황의 핵심을 더 구체화하는 질문 1개"
    if i == 1:
        return "원하는 목표를 측정 가능한 형태로 정리하게 하는 질문 1개"

    if coach_id == "action":
        if i == n - 1:
            return "‘지금 앱을 끄고 나서 5분 안에 실행할 수 있는 가장 작은 행동’을 스스로 적게 만드는 질문 1개(Quick Win, 추천 금지)"
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
        if n >= 6 and i == 4:
            return "실행을 ‘언제’가 아니라 ‘If(어떤 상황) → Then(어떤 행동)’으로 설계하게 하는 질문 1개(트리거 2~3개)"
        return "다음 행동을 더 구체화(무엇을/얼마나/어떤 조건에서)하는 질문 1개"

    if coach_id == "logic":
        if n >= 5 and i == n - 2:
            return "역발상/반대 상황 가정 질문 1개."
        if i == 2:
            return "선택 기준(3~5)을 뽑게 하는 질문 1개"
        if i == n - 1:
            return "마지막으로 선택 기준의 우선순위를 1~3위로 정리하게 하는 질문 1개(추천 금지)"
        if i == n - 2 and n < 5:
            return "불확실한 가정/추가로 확인할 정보 1~2개를 드러내는 질문 1개"
        return "옵션/정보/제약을 더 분리해 명료화하는 질문 1개"

    if coach_id == "value":
        if i == 2:
            return "지금 감정(2~3개)과 그 감정의 이유를 말하게 하는 질문 1개"
        if i == 3 and n >= 5:
            return "감정과 가치의 분리 질문 1개."
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
            return "실행을 ‘만약(If) ~ 상황이면 → 그러면(Then) ~ 행동’으로 트리거 2~3개를 만들어보면 무엇인가요?"
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

    cross_q, cross_dbg = try_logic_crosscheck_question(i)
    if cross_q and not any(is_similar(cross_q, pq) for pq in prev_qs):
        return cross_q, None, cross_dbg

    dbg_acc: List[str] = cross_dbg[:]

    def prompt(nonce: int) -> str:
        prev_txt = "\n".join([f"- {q}" for q in prev_qs[-6:]]) if prev_qs else "(없음)"
        return textwrap.dedent(
            f"""
            [최근 질문 목록]
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


def generate_reframe_question(last_q: str, last_a: str) -> Tuple[str, Optional[str], List[str]]:
    coach = coach_by_id(st.session_state.coach_id)
    system = system_prompt_for_questions(coach)
    user = reframe_instruction(last_q, last_a)
    q, err, dbg = call_openai_text(system=system, user=user, temperature=0.55)
    if not q:
        return "이 질문이 어렵다면, ‘이번 상황에서 가장 신경 쓰이는 한 가지’만 고르면 무엇인가요?", err, dbg
    return normalize(q), None, dbg


# =========================
# Report generation + rendering
# =========================
# 금칙어(추천/지시) 탐지 정밀도 개선: “추천” 단독 제거, 문장 패턴 중심
FORBIDDEN_RECOMMEND_PATTERNS = [
    r"추천합니다",
    r"추천드",
    r"~?하는 것이 좋",
    r"~?하는 게 좋",
    r"~?하시면 좋",
    r"해야 합니다",
    r"하시길",
    r"하는 게 낫",
    r"정답(은|:)",
    r"결론(은|:)",
    r"\bA를\s*선택",
    r"\bB를\s*선택",
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
금지 표현: "추천합니다", "좋겠습니다", "해야 합니다", "하자", "정답", "결론", "A를 선택".
"""

    common_extra = """
추가 필드(원칙 유지):
- "info_check_questions": ["string", ...]  # ‘추가로 확인하면’ 결정을 가볍게 하는 질문 1~3개(질문 형태만)
"""

    if coach_id == "action":
        return textwrap.dedent(
            base
            + common_extra
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
  "info_check_questions": ["string"],
  "coaching_message": ["string","string"],
  "next_self_question": "string"
}
"""
        ).strip()

    if coach_id == "logic":
        return textwrap.dedent(
            base
            + common_extra
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
  "info_check_questions": ["string"],
  "coaching_message":["string","string"],
  "next_self_question":"string"
}
"""
        ).strip()

    return textwrap.dedent(
        base
        + common_extra
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
  "info_check_questions": ["string"],
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
        "coaching_message는 반드시 거울 비추기 문장만.\n"
        "출력은 반드시 JSON만.\n"
    )


def build_qa_text_for_report() -> str:
    qa_text = ""
    for i, qa in enumerate(st.session_state.answers, start=1):
        tag = "PROBE" if qa.get("kind") == "probe" else "MAIN"
        qa_text += f"{i}) ({tag}) Q: {qa['q']}\n   A: {qa['a']}\n"
    return qa_text


def fallback_report_json() -> Dict[str, Any]:
    coach = coach_by_id(st.session_state.coach_id)
    opts = parse_options()
    base = {
        "summary": {
            "core_issue": normalize(st.session_state.situation)[:180] or "핵심 고민이 요약되지 않았습니다.",
            "goal": normalize(st.session_state.goal)[:180] or "목표가 명확히 적히지 않았습니다.",
            "constraints": [],
            "options_mentioned": opts or [],
        },
        "criteria": [],
        "info_check_questions": [
            "이 결정을 더 가볍게 만들기 위해, 지금 ‘확인되지 않은 사실/가정’은 무엇인가요?",
            "최악의 경우를 상상했을 때, 실제로 감당 가능한 비용/손실의 범위는 어느 정도인가요?",
        ],
        "coaching_message": [
            "지금은 ‘정리’가 필요하다는 느낌과, 동시에 ‘확신이 부족하다’는 느낌이 함께 있는 상태처럼 보입니다.",
            "당신에게 중요한 기준이 무엇인지가 선명해질수록, 선택이 덜 무겁게 느껴질 가능성이 있어요.",
        ],
        "next_self_question": "내가 지금 가장 놓치기 싫은 기준 1개는 무엇이고, 왜 그것이 중요한가요?",
    }
    if coach["id"] == "logic":
        base["key_points"] = {"uncertainties": [], "tradeoffs": []}
    elif coach["id"] == "action":
        base["plan_visualization"] = {"year": "", "month": "", "week": []}
        base["weekly_table"] = {"Mon": [], "Tue": [], "Wed": [], "Thu": [], "Fri": [], "Sat": [], "Sun": []}
    else:
        base["emotions_values"] = {"emotions": [], "top_values": []}
    return base


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
- info_check_questions는 질문 형태로 1~3개만
"""
    ).strip()

    text, err, dbg = call_openai_text(system=system, user=user, temperature=0.25)
    if not text:
        fb = fallback_report_json()
        dbg.append("Report fallback used (no model output).")
        return fb, err, dbg, None

    data = safe_json_parse(text)
    if data is None:
        fb = fallback_report_json()
        dbg.append("Report fallback used (JSON parse fail).")
        return fb, "리포트 JSON 파싱 실패(대체 정리를 표시합니다)", dbg, text

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


def render_info_check_questions(data: Dict[str, Any]) -> None:
    st.subheader("정보 부족 체크리스트(질문 형태)")
    qs = data.get("info_check_questions", []) or []
    qs = [str(x).strip() for x in qs if str(x).strip()]
    if not qs:
        st.caption("추가로 확인할 질문이 충분히 드러나지 않았어요.")
        return
    for q in qs[:3]:
        # 질문 형태 유지(추천/지시 금지)
        st.write(f"- {q}")


# ---- “모순/긴장 지도” 시각화(원칙 유지) ----
TENSION_AXES = [
    ("안정", "성장"),
    ("자유", "안정"),
    ("돈", "시간"),
    ("속도", "완성도"),
    ("성과", "건강"),
    ("관계", "경계"),
    ("도전", "안정"),
    ("단기", "장기"),
]


def _collect_tension_signals(data: Dict[str, Any]) -> Dict[str, str]:
    """
    리포트 기반으로 ‘신호 텍스트’를 모아 간단 매칭용 텍스트로 반환
    """
    s = data.get("summary", {}) or {}
    crit = data.get("criteria", []) or []
    crit_text = " ".join([str(c.get("name", "")) + " " + str(c.get("why", "")) for c in crit if isinstance(c, dict)])
    core = str(s.get("core_issue", "") or "")
    goal = str(s.get("goal", "") or "")

    extras = ""
    if "key_points" in data:
        kp = data.get("key_points", {}) or {}
        extras += " " + " ".join(kp.get("uncertainties", []) or [])
        extras += " " + " ".join(kp.get("tradeoffs", []) or [])
    if "emotions_values" in data:
        ev = data.get("emotions_values", {}) or {}
        extras += " " + " ".join(ev.get("emotions", []) or [])
        extras += " " + " ".join(ev.get("top_values", []) or [])

    blob = normalize(" ".join([core, goal, crit_text, extras]))
    return {"blob": blob, "core": core, "goal": goal, "crit": crit_text, "extras": extras}


def render_tension_map(data: Dict[str, Any]) -> None:
    st.subheader("모순/긴장 지도(관찰용)")
    st.caption("결론을 내기 위한 게 아니라, ‘내 안의 기준들이 어디에서 서로 당기는지’를 한 번 더 보려는 지도예요.")

    sig = _collect_tension_signals(data)
    blob = sig["blob"]

    # 1) 축 감지
    found_axes = []
    for a, b in TENSION_AXES:
        if (a in blob) and (b in blob):
            found_axes.append((a, b))

    # 2) 기준 Top3 + 불확실/트레이드오프 + 감정어(미러링에서 가져오는 건 별도이므로 여기선 리포트만)
    crit = data.get("criteria", []) or []
    crit_sorted = []
    for c in crit:
        try:
            p = int(c.get("priority", 999))
        except Exception:
            p = 999
        crit_sorted.append((p, str(c.get("name", "") or "").strip()))
    crit_sorted = [x for x in sorted(crit_sorted, key=lambda x: x[0]) if x[1]]
    top3 = [x[1] for x in crit_sorted[:3]]

    uncertainties = []
    tradeoffs = []
    if "key_points" in data:
        kp = data.get("key_points", {}) or {}
        uncertainties = [str(x).strip() for x in (kp.get("uncertainties", []) or []) if str(x).strip()]
        tradeoffs = [str(x).strip() for x in (kp.get("tradeoffs", []) or []) if str(x).strip()]

    emotions = []
    if "emotions_values" in data:
        ev = data.get("emotions_values", {}) or {}
        emotions = [str(x).strip() for x in (ev.get("emotions", []) or []) if str(x).strip()]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.write("**기준 Top3(우선순위 기준)**")
        if top3:
            for x in top3:
                st.write(f"- {x}")
        else:
            st.caption("기준 Top3가 충분히 드러나지 않았어요.")
    with c2:
        st.write("**불확실/리스크 신호**")
        if uncertainties:
            for x in uncertainties[:4]:
                st.write(f"- {x}")
        else:
            st.caption("불확실 신호가 충분히 드러나지 않았어요.")
    with c3:
        st.write("**감정 신호(리포트 기반)**")
        if emotions:
            for x in emotions[:4]:
                st.write(f"- {x}")
        else:
            st.caption("감정 신호가 충분히 드러나지 않았어요.")

    st.write("**긴장 축(텍스트 매칭 기반)**")
    if found_axes:
        for a, b in found_axes:
            st.write(f"- {a} ↔ {b}")
    else:
        st.caption("명확한 ‘긴장 축’이 자동으로 잡히지 않았어요. (기준/불확실/감정에서 키워드가 다르게 표현됐을 수 있어요.)")

    if tradeoffs:
        st.write("**사용자가 말한 트레이드오프(리포트 기반)**")
        for x in tradeoffs[:5]:
            st.write(f"- {x}")


def render_coaching_message(data: Dict[str, Any]) -> None:
    st.subheader("코칭 메시지(거울 비추기)")
    msgs = data.get("coaching_message", []) or []
    for m in msgs:
        st.write(f"- {m}")


def render_next_question(data: Dict[str, Any]) -> None:
    st.subheader("다음에 스스로에게 던질 질문(1개)")
    st.write(f"**{data.get('next_self_question','')}**")


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
    freq: Dict[str, int] = {}
    for t in toks:
        freq[t] = freq.get(t, 0) + 1
    kw = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10]
    kw_df = pd.DataFrame(kw, columns=["키워드", "빈도"])

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
    st.caption("각 옵션이 ‘내 기준’에서 어느 정도인지 1~5점으로 적어보세요. 점수는 결론이 아니라 생각을 꺼내는 도구예요.")

    user_opts = parse_options()
    report_opts = (data.get("summary", {}) or {}).get("options_mentioned", []) or []
    opts = user_opts or [str(x) for x in report_opts if str(x).strip()] or ["옵션 1", "옵션 2"]

    if st.session_state.decision_matrix_df is None:
        st.session_state.decision_matrix_df = build_decision_matrix(opts, criteria_names)

    df: pd.DataFrame = st.session_state.decision_matrix_df

    existing_opts = [str(x) for x in df["옵션"].tolist()] if "옵션" in df.columns else []
    if set(existing_opts) != set(opts):
        st.session_state.decision_matrix_df = build_decision_matrix(opts, criteria_names)
        df = st.session_state.decision_matrix_df

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


def render_copy_to_clipboard_button(text: str, button_label: str = "클립보드에 복사") -> None:
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
    if st.session_state.emotion_pre is not None or st.session_state.emotion_post is not None:
        lines.append(f"- 감정 강도(시작/끝): {st.session_state.emotion_pre} → {st.session_state.emotion_post}")
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
    if not st.session_state.answers:
        st.session_state.q_index = max(0, int(st.session_state.q_index) - 1)
        st.session_state.probe_active = False
        st.session_state.probe_question = ""
        st.session_state.probe_for_index = None
        st.session_state.probe_mode = ""
        return

    last = st.session_state.answers.pop()

    if last.get("kind") == "probe":
        st.session_state.probe_active = False
        st.session_state.probe_question = ""
        st.session_state.probe_for_index = None
        st.session_state.probe_mode = ""
        st.session_state.q_index = int(last.get("main_index", st.session_state.q_index))
        return

    mi = int(last.get("main_index", 0))
    st.session_state.probe_active = False
    st.session_state.probe_question = ""
    st.session_state.probe_for_index = None
    st.session_state.probe_mode = ""
    st.session_state.q_index = max(0, mi)


# =========================
# Sidebar (보조 기능 + 프라이버시 + 프리셋)
# =========================
init_state()

with st.sidebar:
    st.header("보조 메뉴")
    st.text_input("OpenAI API Key (Secrets 우선)", type="password", key="openai_api_key_input")

    st.divider()

    st.subheader("프라이버시 모드")
    st.toggle("프라이버시 모드", key="privacy_mode")
    if st.session_state.privacy_mode:
        st.toggle("답변 기록 숨기기", key="hide_history")
        st.toggle("내보내기 마스킹(권장)", key="mask_export")
        st.caption("프라이버시 모드는 ‘표시/공유 위험’을 낮추는 옵션입니다(완전 익명화는 아님).")
    else:
        st.session_state.hide_history = False

    st.divider()
    st.subheader("세션 템플릿(프리셋)")
    st.caption("카테고리/결정유형/코치/질문개수를 저장해 다음에 빠르게 시작할 수 있어요.")

    with st.expander("프리셋 저장/불러오기"):
        tpl_name = st.text_input("프리셋 이름", placeholder="예: 커리어 결정(구조 코치)")
        colx, coly = st.columns(2)
        with colx:
            if st.button("현재 설정 저장", use_container_width=True):
                if tpl_name.strip():
                    st.session_state.saved_templates.append(
                        {
                            "name": tpl_name.strip(),
                            "category": st.session_state.category,
                            "decision_type": st.session_state.decision_type,
                            "coach_id": st.session_state.coach_id,
                            "num_questions": int(st.session_state.num_questions),
                            "saved_at": datetime.now().isoformat(timespec="seconds"),
                        }
                    )
                    st.success("저장했어요.")
                else:
                    st.warning("프리셋 이름을 입력해 주세요.")
        with coly:
            if st.button("프리셋 전체 내보내기(JSON)", use_container_width=True):
                pass  # 아래 download_button로 대체

        if st.session_state.saved_templates:
            names = [t["name"] for t in st.session_state.saved_templates]
            picked = st.selectbox("불러올 프리셋", names, index=0)
            col1, col2 = st.columns(2)
            with col1:
                if st.button("불러오기(현재 설정에 적용)", use_container_width=True):
                    t = next((x for x in st.session_state.saved_templates if x["name"] == picked), None)
                    if t:
                        st.session_state.category = t["category"]
                        st.session_state.decision_type = t["decision_type"]
                        st.session_state.coach_id = t["coach_id"]
                        st.session_state.num_questions = int(t["num_questions"])
                        st.success("적용했어요.")
            with col2:
                if st.button("삭제", use_container_width=True):
                    st.session_state.saved_templates = [x for x in st.session_state.saved_templates if x["name"] != picked]
                    st.success("삭제했어요.")
                    st.rerun()

            st.download_button(
                "프리셋 JSON 다운로드",
                data=json.dumps(st.session_state.saved_templates, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name="pebble_templates.json",
                mime="application/json",
                use_container_width=True,
            )

            up = st.file_uploader("프리셋 JSON 불러오기", type=["json"])
            if up is not None:
                try:
                    loaded = json.loads(up.read().decode("utf-8"))
                    if isinstance(loaded, list):
                        # 단순 병합(동명 중복은 뒤에 추가)
                        for item in loaded:
                            if isinstance(item, dict) and "name" in item:
                                st.session_state.saved_templates.append(item)
                        st.success("불러왔어요.")
                        st.rerun()
                    else:
                        st.warning("형식이 올바르지 않습니다(리스트 JSON이어야 해요).")
                except Exception:
                    st.warning("JSON을 읽는 데 실패했어요.")
        else:
            st.caption("저장된 프리셋이 아직 없어요.")

    st.divider()
    if st.button("처음부터 다시 하기", use_container_width=True):
        reset_flow("landing", keep_problem=False)
        st.rerun()

    if st.session_state.page in ("setup_details", "questions", "report"):
        if st.button("고민만 유지하고 다시 설정", use_container_width=True):
            reset_flow("landing", keep_problem=True)
            st.rerun()

    st.divider()
    with st.expander("디버그 로그"):
        st.write(st.session_state.debug_log)

    st.caption("사이드바는 보조 기능만 제공합니다.")


# =========================
# Progress Bar indexing
# =========================
nq = int(st.session_state.num_questions)
labels = ["고민", "설정"] + [f"Q{i}" for i in range(1, nq + 1)] + ["요약"]

if st.session_state.page == "landing":
    idx = 0
elif st.session_state.page == "setup_details":
    idx = 1
elif st.session_state.page == "questions":
    idx = 2 + int(st.session_state.q_index)
else:
    idx = 2 + nq

render_pebble_bridge(idx, len(labels), labels)
progress = idx / max(1, (len(labels) - 1))
with st.columns([1, 2, 1])[1]:
    render_hero_pebble(progress, f"진행도: {int(progress * 100)}%")

st.divider()


# =========================
# Pages
# =========================
def render_landing() -> None:
    st.title("🪨 돌멩이 AI 결정 코칭")
    st.caption("정답을 주기보다, 질문으로 생각을 정리하도록 돕습니다.")

    cols = st.columns([1, 3, 1])
    with cols[1]:
        st.subheader("1단계 · 고민 작성")
        st.caption("지금 고민 중인 상황을 자유롭게 적어주세요.")

        with st.container(border=True):
            if st.session_state.privacy_mode:
                st.caption("프라이버시 모드: 입력은 동일하지만 화면 공유 시 주의가 덜 되도록 일부 표시를 줄입니다.")
            st.text_area(
                "고민 내용",
                key="user_problem",
                height=220,
                placeholder="예: 이직 제안을 받았는데 안정성과 성장 사이에서 고민돼요. 지금 팀도 좋지만…",
                label_visibility="collapsed",
            )

        c1, c2 = st.columns([2, 1])
        with c1:
            st.session_state.num_questions = st.slider("질문 개수(2~10)", 2, 10, int(st.session_state.num_questions))
        with c2:
            if st.button("다음 단계로", type="primary", use_container_width=True):
                txt = (st.session_state.user_problem or "").strip()
                if not txt:
                    st.warning("고민 내용을 먼저 한 줄이라도 적어주세요.")
                else:
                    if not (st.session_state.situation or "").strip():
                        st.session_state.situation = txt
                    st.session_state.page = "setup_details"
                    st.rerun()


def render_setup_details() -> None:
    st.title("2단계 · AI 분석 및 추천")
    st.caption("아래 값들은 ‘추천/초안’입니다. 마음에 들지 않으면 직접 바꿔도 괜찮아요.")

    problem_text = (st.session_state.user_problem or "").strip()

    # 자동 1회 생성: onboarding_reco가 None이면 생성
    auto_generate = st.session_state.onboarding_reco is None and bool(problem_text)
    if auto_generate:
        with st.spinner("AI가 고민을 읽고 추천을 만드는 중..."):
            reco, err, dbg, raw = generate_onboarding_recommendation(problem_text)
            st.session_state.debug_log = dbg
            st.session_state.onboarding_reco = reco
            st.session_state.onboarding_raw = raw
            if err:
                st.warning(err)

    top = st.columns([2, 1])
    with top[0]:
        st.subheader("내가 적은 고민")
    with top[1]:
        if st.button("추천 다시 생성", use_container_width=True):
            with st.spinner("추천을 다시 생성하는 중..."):
                reco, err, dbg, raw = generate_onboarding_recommendation(problem_text)
                st.session_state.debug_log = dbg
                st.session_state.onboarding_reco = reco
                st.session_state.onboarding_raw = raw
                st.session_state.onboarding_applied = False  # 다시 적용 가능하게
                if err:
                    st.warning(err)
            st.rerun()

    with st.container(border=True):
        if st.session_state.privacy_mode:
            st.write("프라이버시 모드: (표시 숨김) — 이 영역은 화면 공유 시 민감할 수 있어요.")
        else:
            st.write(problem_text)

    reco = st.session_state.onboarding_reco or {}

    # 추천값 반영(초기 1회만, 사용자 수정 보호)
    if reco and not st.session_state.onboarding_applied:
        rec_cat = reco.get("recommended_category", "")
        if rec_cat in [c[0] for c in TOPIC_CATEGORIES]:
            st.session_state.category = rec_cat

        rec_dt = reco.get("recommended_decision_type", "")
        if rec_dt in DECISION_TYPES:
            st.session_state.decision_type = rec_dt

        rec_coach = reco.get("recommended_coach_id", "")
        if rec_coach in [c["id"] for c in COACHES]:
            st.session_state.coach_id = rec_coach

        goal_draft = str(reco.get("goal_draft", "") or "").strip()
        if goal_draft and not (st.session_state.goal or "").strip():
            st.session_state.goal = goal_draft

        if not (st.session_state.situation or "").strip():
            st.session_state.situation = problem_text

        st.session_state.onboarding_applied = True

    st.divider()
    st.subheader("추천값 확인/수정")

    c1, c2 = st.columns(2)
    with c1:
        st.selectbox("카테고리", [x[0] for x in TOPIC_CATEGORIES], key="category")
        st.selectbox("결정 유형", DECISION_TYPES, key="decision_type")
        st.text_input("원하는 목표(초안)", key="goal", placeholder="예: 내가 중요하게 여기는 기준을 선명하게 만들고 싶다")
        st.text_input("옵션(쉼표로 구분, 선택)", key="options", placeholder="예: A, B, C")
        st.slider("질문 개수(2~10)", 2, 10, int(st.session_state.num_questions), key="num_questions")
    with c2:
        coach_labels = [f"{c['name']} — {c['tagline']}" for c in COACHES]
        cur = next((i for i, c in enumerate(COACHES) if c["id"] == st.session_state.coach_id), 0)
        picked = st.radio("코치 선택", coach_labels, index=cur)
        st.session_state.coach_id = COACHES[coach_labels.index(picked)]["id"]
        coach = coach_by_id(st.session_state.coach_id)

        reason = str(reco.get("coach_reason", "") or "").strip()
        if reason:
            st.info(f"**AI가 이 코치를 추천한 이유(참고):** {reason}")

        with st.expander("코치 진행 방식"):
            st.markdown(f"**{coach['name']}**  \n_{coach['style']}_")
            for m in coach["method"]:
                st.write(f"- {m}")
            st.caption(f"특징: {coach['prompt_hint']}")

    st.subheader("상황 설명(편집 가능)")
    st.caption("기본값은 1단계에서 적은 고민입니다. 필요하면 다듬어주세요.")
    st.text_area("상황 설명", key="situation", height=180)

    with st.expander("결정 유형 가이드(템플릿)"):
        st.caption("필요하면 아래 가이드를 상황 설명에 삽입할 수 있어요.")
        tmpl = DECISION_TEMPLATES.get(st.session_state.decision_type, "")
        if tmpl:
            st.code(tmpl, language="text")
            if st.button("가이드 삽입(상황 설명에 추가)", use_container_width=True):
                cur_txt = (st.session_state.situation or "").strip()
                st.session_state.situation = (cur_txt + "\n\n" + tmpl).strip() if cur_txt else tmpl
                st.rerun()

    st.divider()
    b1, b2, b3 = st.columns([1, 1, 1])
    with b1:
        if st.button("⬅️ 이전 단계", use_container_width=True):
            st.session_state.page = "landing"
            st.rerun()
    with b2:
        if st.button("추천 원문(JSON) 보기", use_container_width=True):
            if st.session_state.onboarding_raw:
                st.code(st.session_state.onboarding_raw, language="json")
            else:
                st.caption("추천 원문이 아직 없습니다.")
    with b3:
        if st.button("코칭 시작하기(실행하기)", type="primary", use_container_width=True):
            st.session_state.q_index = 0
            st.session_state.questions = []
            st.session_state.answers = []
            st.session_state.probe_active = False
            st.session_state.probe_question = ""
            st.session_state.probe_for_index = None
            st.session_state.probe_mode = ""
            st.session_state.crosscheck_used_for = []
            st.session_state.final_report_json = None
            st.session_state.final_report_raw = None
            st.session_state.decision_matrix_df = None
            st.session_state.page = "questions"
            st.rerun()


def render_questions() -> None:
    st.title("질문")
    st.caption("한 화면에 한 질문. 답변이 10자 미만이면 구체화 질문, ‘난감(정보 부족)’이면 재프레이밍 질문을 1회 제공합니다.")

    nq = int(st.session_state.num_questions)
    q_idx = int(st.session_state.q_index)
    q_idx = max(0, min(q_idx, nq - 1))

    # 감정 트래킹: 질문 시작 전에 한 번만(첫 질문에서만)
    if q_idx == 0 and st.session_state.emotion_pre is None:
        st.subheader("시작 전 셀프 체크(1초)")
        st.caption("지금 마음의 무게/불편함/긴장 정도를 1~5로 찍어주세요(정답 없음).")
        st.session_state.emotion_pre = st.slider("현재 감정 강도", 1, 5, 3, key="emotion_pre_slider")
        st.divider()

    ensure_question(q_idx, nq)
    main_q = st.session_state.questions[q_idx]

    if st.session_state.probe_active and st.session_state.probe_for_index == q_idx:
        show_q = st.session_state.probe_question
        kind = "probe"
        badge = "도움 질문(재프레이밍)" if st.session_state.probe_mode == "reframe" else "추가 질문(구체화)"
    else:
        show_q = main_q
        kind = "main"
        badge = "메인 질문"

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
        ans = st.text_area("답변", placeholder="여기에 답변을 입력하세요", height=150)
        submitted = st.form_submit_button("답변 저장", use_container_width=True)

    if submitted:
        a = (ans or "").strip()
        if not a:
            st.warning("답변이 비어 있습니다. 한 줄만 입력해도 진행 가능합니다.")
        else:
            if kind == "probe":
                add_answer(show_q, a, kind="probe", main_index=q_idx, subkind=st.session_state.probe_mode or "")
                # probe 끝 -> 다음 main으로
                st.session_state.probe_active = False
                st.session_state.probe_question = ""
                st.session_state.probe_for_index = None
                st.session_state.probe_mode = ""
                st.session_state.q_index = min(q_idx + 1, nq - 1)
                st.rerun()

            # main answer 저장
            add_answer(show_q, a, kind="main", main_index=q_idx, subkind="")

            # 1) 난감(키워드+정보부족) 답변이면: 재프레이밍 질문 1회 제공(다음 단계로 안 넘어감)
            if is_confused_answer(a):
                rq, err, dbg = generate_reframe_question(show_q, a)
                st.session_state.debug_log = dbg
                st.session_state.probe_active = True
                st.session_state.probe_question = rq
                st.session_state.probe_for_index = q_idx
                st.session_state.probe_mode = "reframe"
                st.rerun()

            # 2) 짧은 답변이면: 구체화 질문 1회 제공
            if is_too_short_answer(a):
                pq, err, dbg = generate_probe_question(show_q, a)
                st.session_state.debug_log = dbg
                st.session_state.probe_active = True
                st.session_state.probe_question = pq
                st.session_state.probe_for_index = q_idx
                st.session_state.probe_mode = "short"
                st.rerun()

            # 3) 정상 진행
            if main_answer_count() >= nq:
                st.session_state.page = "report"
                st.session_state.report_just_entered = True
                st.session_state.q_index = nq - 1
            else:
                st.session_state.q_index = min(q_idx + 1, nq - 1)
            st.rerun()

    # 프라이버시 모드: 답변 기록 숨기기 토글 지원
    if not (st.session_state.privacy_mode and st.session_state.hide_history):
        with st.expander("답변 기록"):
            grouped: Dict[int, List[Dict[str, Any]]] = {}
            for qa in st.session_state.answers:
                grouped.setdefault(int(qa.get("main_index", 0)), []).append(qa)

            for mi in sorted(grouped.keys()):
                st.markdown(f"### Q{mi + 1}")
                for qa in grouped[mi]:
                    tag = "PROBE" if qa.get("kind") == "probe" else "MAIN"
                    sub = qa.get("subkind", "")
                    tag2 = f"{tag}:{sub}" if sub else tag
                    st.markdown(f"**({tag2}) {qa['q']}**")
                    st.write(qa["a"])
                    st.caption(qa["ts"])
                    st.divider()
    else:
        st.caption("프라이버시 모드: 답변 기록이 숨김 처리되었습니다.")


def render_emotion_delta_block() -> None:
    st.subheader("감정 변화(셀프 체크)")
    pre = st.session_state.emotion_pre
    post = st.session_state.emotion_post
    if pre is None:
        st.caption("시작 전 감정 강도가 기록되지 않았어요.")
        return

    if post is None:
        st.caption("끝난 뒤 감정 강도를 아직 기록하지 않았어요.")
        return

    delta = int(post) - int(pre)
    c1, c2, c3 = st.columns(3)
    c1.metric("시작", str(pre))
    c2.metric("끝", str(post))
    c3.metric("변화(끝-시작)", f"{delta:+d}")
    st.caption("이 값은 ‘좋고 나쁨’이 아니라, 정리 전/후의 체감 변화를 관찰하기 위한 기록이에요.")


def render_report() -> None:
    coach = coach_by_id(st.session_state.coach_id)
    nq = int(st.session_state.num_questions)

    st.title("최종 정리")
    st.caption("추천/정답 없이, 고민의 핵심과 기준을 ‘거울 비추기’ 방식으로 정리합니다.")

    if st.session_state.report_just_entered:
        st.balloons()
        st.session_state.report_just_entered = False

    # 감정 트래킹: 리포트에서 post 기록
    st.subheader("끝난 뒤 셀프 체크(1초)")
    st.caption("정리를 마친 지금의 감정 강도를 1~5로 찍어주세요(정답 없음).")
    st.session_state.emotion_post = st.slider("현재 감정 강도", 1, 5, 3, key="emotion_post_slider")
    st.divider()

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
            reset_flow("landing", keep_problem=False)
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

        render_emotion_delta_block()
        render_summary_block(data)

        # 1) 기준
        criteria_names = render_criteria(data)

        # 2) 매트릭스
        render_decision_matrix(criteria_names, data)

        # 3) 코치별 블록
        if coach["id"] == "action":
            render_action_visualization(data)
        elif coach["id"] == "logic":
            render_key_points_logic(data)
        else:
            render_emotions_values(data)

        # 4) 모순/긴장 지도(신규)
        render_tension_map(data)

        # 5) 정보 부족 체크리스트(신규)
        render_info_check_questions(data)

        # 6) mirroring 시각화(기존)
        render_mirroring_visual()

        # 7) 코칭 메시지 + 다음 질문
        render_coaching_message(data)
        render_next_question(data)

        # 8) 다음 세션 연결(신규)
        st.subheader("다음 세션으로 연결하기")
        st.caption("아래 질문에 답한 내용을 ‘다음 세션의 시작 고민’으로 삼을 수 있어요(추천 아님).")
        nsq = str(data.get("next_self_question", "") or "").strip()
        if nsq:
            st.write(f"**질문:** {nsq}")
        next_seed = st.text_area("내 답변(다음 세션 시작용)", height=120, placeholder="예: 내가 놓치기 싫은 기준은 …")
        colx, coly = st.columns([1, 1])
        with colx:
            if st.button("이 답변으로 새 세션 시작", use_container_width=True):
                if next_seed.strip():
                    st.session_state.user_problem = next_seed.strip()
                    reset_flow("setup_details", keep_problem=True)
                    st.session_state.page = "setup_details"
                    st.rerun()
                else:
                    st.warning("답변을 한 줄이라도 입력해 주세요.")
        with coly:
            if st.button("그냥 랜딩으로", use_container_width=True):
                reset_flow("landing", keep_problem=False)
                st.rerun()

        # 9) 공유/저장
        st.subheader("공유/저장")
        export_text = build_report_text_for_export(data)

        if st.session_state.privacy_mode and st.session_state.mask_export:
            export_text = mask_text_for_privacy(export_text)

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
        json_text = json.dumps(data, ensure_ascii=False, indent=2)
        if st.session_state.privacy_mode and st.session_state.mask_export:
            json_text = mask_text_for_privacy(json_text)
        st.code(json_text, language="json")

        if contains_forbidden_recommendation(json.dumps(data, ensure_ascii=False)):
            st.warning("리포트에 추천/지시처럼 보이는 표현이 섞였을 수 있어요. 필요하면 ‘정리 생성/새로고침’을 눌러 보세요.")

        valid_until = (datetime.now().date() + timedelta(days=7)).strftime("%Y-%m-%d")
        st.divider()
        st.caption(f"이 정리는 **{valid_until}**까지 유효합니다.")

    elif st.session_state.final_report_raw:
        st.warning("JSON 파싱 실패로 원문을 표시합니다.")
        st.code(st.session_state.final_report_raw, language="text")


# =========================
# Router
# =========================
if st.session_state.page == "landing":
    render_landing()
elif st.session_state.page == "setup_details":
    render_setup_details()
elif st.session_state.page == "questions":
    render_questions()
else:
    render_report()

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

