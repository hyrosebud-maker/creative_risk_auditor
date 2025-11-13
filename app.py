# -*- coding: utf-8 -*-
# app.py - Creative Risk Auditor (v1.9)
# 변경점 (v1.9):
# - "Risk" 정의를 '논란/문제 소지'로 한정. 마케팅 성과/효율(CTR·전환·매출·브랜딩 등) 평가는 금지.
# - LLM 프롬프트 강화: 효과성/효율성 언급 금지, 오직 Risk(논란/법/윤리/규정/차별/문화·종교 감수성/환경/오해소지)만.
# - 추가 안전장치: 모델 응답에서 성과/효율성 관련 문구를 자동 필터링(sanitize)하여 UI 표시.

import os, re, json, base64, math, html
from typing import Optional, List, Tuple
import streamlit as st

# Gemini SDK
from google import genai
from google.genai import types

# ========== 0) API KEY ==========
def _parse_env_file(path: str) -> dict:
    out = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#") or "=" not in s:
                    continue
                k, v = s.split("=", 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
    except FileNotFoundError:
        pass
    return out

def load_api_key() -> Optional[str]:
    if hasattr(st, "secrets"):
        v = st.secrets.get("GEMINI_API_KEY", None)
        if v:
            return v
    v = os.environ.get("GEMINI_API_KEY")
    if v:
        return v
    envmap = _parse_env_file(".env")
    v = envmap.get("GEMINI_API_KEY")
    if v:
        os.environ["GEMINI_API_KEY"] = v
        return v
    return None

API_KEY = load_api_key()
if not API_KEY:
    st.error("❌ GEMINI_API_KEY가 없습니다. .env 또는 환경변수/Streamlit secrets에 설정하세요.")
    st.stop()

# ========== 1) Gemini ==========
@st.cache_resource(show_spinner=False)
def get_client(api_key: str):
    return genai.Client(api_key=api_key)

client = get_client(API_KEY)

def _gen_config():
    return types.GenerateContentConfig(
        response_modalities=["TEXT"],
        response_mime_type="application/json",
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )

def call_gemini_text(prompt: str, model: str) -> str:
    try:
        cfg = _gen_config()
        resp = client.models.generate_content(model=model, contents=prompt, config=cfg)
        return (getattr(resp, "text", "") or "").strip()
    except Exception as e:
        return f"Gemini Error: {e}"

def call_gemini_mm(prompt: str, image_parts: List[types.Part], model: str) -> str:
    try:
        cfg = _gen_config()
        parts = [types.Part.from_text(text=prompt)] + (image_parts or [])
        resp = client.models.generate_content(model=model, contents=parts, config=cfg)
        return (getattr(resp, "text", "") or "").strip()
    except Exception as e:
        return f"Gemini Error: {e}"

def parse_json_or_fail(raw: str, fail_title: str) -> dict:
    try:
        s = raw.find("{")
        e = raw.rfind("}")
        data = json.loads(raw[s : e + 1]) if s != -1 and e != -1 and e > s else None
    except Exception:
        data = None
    if not data:
        st.error(f"{fail_title} — LLM JSON 파싱 실패")
        with st.expander("LLM 원문 보기"):
            st.code(raw)
        st.stop()
    return data

# ========== 2) Upload/Util ==========
def to_image_part(up) -> Optional[types.Part]:
    if not up:
        return None
    try:
        data = up.read()
        up.seek(0)
        mime = up.type or "application/octet-stream"
        return types.Part.from_bytes(data=data, mime_type=mime)
    except Exception:
        return None

def uploaded_to_data_uri(up) -> Optional[str]:
    if not up:
        return None
    try:
        data = up.read()
        up.seek(0)
        mime = up.type or "image/png"
        b64 = base64.b64encode(data).decode("utf-8")
        return f"data:{mime};base64,{b64}"
    except Exception:
        return None

def esc(s: str) -> str:
    s = str(s or "")
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def attr_esc(s: str) -> str:
    return esc(s).replace('"', "&quot;").replace("'", "&#39;")

CIRCLED_RANGE = r"[\u2460-\u2473\u24F5-\u24FE\u2776-\u277F]"

def strip_circled(text: str) -> str:
    if not text:
        return ""
    t = re.sub(CIRCLED_RANGE, "", str(text))
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t

# ===== 성과/효율 언급 제거 필터 =====
PERF_KEYWORDS = [
    "전환",
    "전환율",
    "컨버전",
    "conversion",
    "CVR",
    "구매율",
    "매출",
    "Revenue",
    "ROAS",
    "CPA",
    "CAC",
    "클릭",
    "클릭률",
    "CTR",
    "도달",
    "노출수",
    "impression",
    "reach",
    "브랜딩",
    "브랜드 리프트",
    "성과",
    "퍼포먼스",
    "효율",
    "효과",
    "전략적",
    "성장률",
    "KPI",
    "트래픽",
    "세션",
    "리텐션",
    "재방문",
]

def _looks_performance(line: str) -> bool:
    low = (line or "").lower()
    for kw in PERF_KEYWORDS:
        if kw.lower() in low:
            return True
    return False

def sanitize_lines(lines: List[str]) -> List[str]:
    # 성과/효율 관련 문장을 제거하고, 모두 제거되면 Risk 관점의 안전 코멘트 추가
    outs = []
    for x in lines or []:
        t = strip_circled(x)
        if not t:
            continue
        if _looks_performance(t):  # 성과/효율 언급 제거
            continue
        outs.append(t)
    if not outs:
        outs = [
            "해당 항목은 성과·효율과 무관하게, 현재 기준에서 뚜렷한 논란·문제 소지가 확인되지 않습니다."
        ]
    return outs

# ========== 3) Prompts (안전도: 높을수록 안전, 'Risk'만 평가) ==========
TEXT_RISK_PROMPT = """
당신은 글로벌 마케팅 거버넌스 'Risk' 심사관이다.
여기서 'Risk'란 **논란이나 큰 문제가 될 수 있는 요소**를 뜻한다.
예: 법적·규정 위반 가능성, 윤리/차별/혐오, 정치·종교·문화 감수성 침해, 환경/지속가능성 침해, 잘못된 주장/오해 유발 등.

⚠️ 금지: 클릭/전환/CTR/매출/브랜딩 효과 등 **마케팅의 성과·효율성**에 대한 언급·평가·제안은 절대 포함하지 마라.
수정 제안 역시 오직 **Risk 완화/제거**를 위한 조치로만 제시한다(효과성 최적화 제안 금지).

입력 텍스트의 **안전도**를 정치·문화·환경·사회 4축으로 각 0~25점(높을수록 안전) 평가하라.
각 축: score(0~25), why(25점이어도 Risk 관점 코멘트), edits(완화/제거 조치), checks(필요 점검).
JSON ONLY:
{
  "country":"",
  "core_dimensions":[
    {"name":"Political","score":0,"why":[""],"edits":[""],"checks":[""]},
    {"name":"Cultural","score":0,"why":[""],"edits":[""],"checks":[""]},
    {"name":"Environmental","score":0,"why":[""],"edits":[""],"checks":[""]},
    {"name":"Social","score":0,"why":[""],"edits":[""],"checks":[""]}
  ],
  "text_feedback":{"flags":[{"span":"","issues":[""],"edits":[""]}]}
}
주의: 번호/원형숫자 기호는 넣지 말라. 성과/효율 관련 언급 금지.
"""

IMAGE_RISK_PROMPT = """
당신은 글로벌 마케팅 거버넌스 'Risk' 심사관이다.
'Risk'는 **논란이나 큰 문제가 될 수 있는 요소**로 한정한다(법/윤리/차별/정치·종교·문화 감수성/환경/오해 소지).
⚠️ 금지: 클릭/전환/CTR/매출/브랜딩 효과 등 **마케팅 성과·효율성** 언급·평가·제안.

업로드된 Key Visual의 **안전도**를 정치·문화·환경·사회 4축으로 각 0~25점(높을수록 안전) 평가하라.
각 축: score/why/edits/checks. 각 이미지 index(1부터) notes와 **Risk가 존재하는 영역만** 핫스팟(0~1 좌표) 제공.
핫스팟에는 가능하면 severity(매우 위험/위험/주의)를 포함하라. edits는 **Risk 완화/제거 조치**로만 작성.

JSON ONLY:
{
  "country":"",
  "core_dimensions":[
    {"name":"Political","score":0,"why":[""],"edits":[""],"checks":[""]},
    {"name":"Cultural","score":0,"why":[""],"edits":[""],"checks":[""]},
    {"name":"Environmental","score":0,"why":[""],"edits":[""],"checks":[""]},
    {"name":"Social","score":0,"why":[""],"edits":[""],"checks":[""]}
  ],
  "image_feedback":[
    {"index":1,"notes":"","hotspots":[
      {"shape":"circle","cx":0.65,"cy":0.42,"r":0.08,"label":"","severity":"매우 위험","risks":[""],"suggested_edits":[""]}
    ]}
  ]
}
주의: 번호/원형숫자 기호는 넣지 말라. 성과/효율 관련 언급 금지.
"""

# ========== 4) Styles ==========
CARD_CSS = """
<style>
.block-container {max-width: 1400px !important;}
.section-sep{border:0;border-top:1px solid #e5e7eb;margin:18px 0}
.card{border:0;border-radius:0;padding:0;margin:6px 0 14px 0;}
.card h4{margin:0 0 10px 0; padding:0; background:transparent;}
.subcard{border:1px solid #e5e7eb;border-radius:12px;padding:12px;background:#fff;margin:10px 0}
.score-text{font-weight:900;font-size:26px}

.legend{display:flex;gap:8px;flex-wrap:wrap;margin-top:8px}
.legend .pill{border-radius:999px;padding:2px 8px;font-size:12px;color:#fff}

/* 상태칩 + 점수 */
.risk-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:12px;margin-top:6px}
.risk-tile{border:1px solid #e2e8f0;border-radius:12px;background:#fff;padding:12px}
.risk-tile h5{margin:0 0 8px 0;font-size:14px}
.status-line{display:flex;align-items:center;gap:10px;margin-bottom:6px}
.status-chip{
  display:inline-block; min-width:108px; text-align:center;
  border-radius:10px; padding:4px 8px; color:#fff; font-weight:800; font-size:13px;
}
.score-small{font-size:12px; color:#6b7280; font-weight:700}

/* Key Visual overlay */
.kv-wrap{position:relative;width:100%}
.kv-img{width:100%;height:auto;border-radius:8px;border:1px solid #e5e7eb;display:block}
.kv-svg{position:absolute;left:0;top:0;width:100%;height:100%;pointer-events:auto}
.kv-badge{position:absolute; right:10px; top:10px; background:rgba(17,24,39,.8); color:#fff;
  font-size:12px; padding:4px 8px; border-radius:999px; z-index:4;}
/* Glow & fill (alpha=0.20) */
.kv-hot{stroke:#FF1F1F; stroke-width:3; fill:rgba(255,31,31,var(--alpha,0.20)); filter:url(#kv-glow); cursor:pointer}
.kv-hot.warn{stroke:#F59E0B; fill:rgba(245,158,11,var(--alpha,0.20))}
.kv-hot.caution{stroke:#D97706; fill:rgba(217,119,6,var(--alpha,0.20))}
.kv-hot:hover{stroke-width:4}
.kv-tip{
  position:absolute; left:50%; top:100%; transform:translate(-50%, 10px);
  background:#111827; color:#fff; font-size:12px; padding:6px 8px; border-radius:6px;
  white-space:normal; max-width:260px; display:none; z-index:3;
}

/* 캡션 하이라이트 */
.caption-strong{font-size:18px; font-weight:900}
.caption-flag{
  color:#FF1F1F;
  font-weight:900;
  background:rgba(255,31,31,.08);
  padding:0 2px;
  border-radius:4px;
}

/* 상단 결과 배너 */
.decision-banner{border-radius:14px; padding:14px 16px; color:#fff; font-weight:800; margin:6px 0 16px 0;}
</style>
"""

# ========== 5) Levels/Colors ==========
PALETTE = {
    "매우 안전": "#16A34A",
    "안전": "#65A30D",
    "주의": "#D97706",
    "위험": "#F59E0B",
    "매우 위험": "#FF1F1F",  # vivid red
}
LEVELS = [
    (21, 25, "매우 안전"),
    (16, 20, "안전"),
    (11, 15, "주의"),
    (6, 10, "위험"),
    (0, 5, "매우 위험"),
]

def level_of(score: int) -> str:
    s = max(0, min(25, int(score)))
    for lo, hi, name in LEVELS:
        if lo <= s <= hi:
            return name
    return "—"

def level_color(score: int) -> str:
    return PALETTE.get(level_of(score), "#6B7280")

def severity_rank(level: str) -> int:
    order = {"매우 안전": 0, "안전": 1, "주의": 2, "위험": 3, "매우 위험": 4}
    return order.get(level, 0)

def overall_from_text_image(text_dims: List[dict], image_dims: List[dict]) -> dict:
    def min_dim(dims):
        if not dims:
            return ("", 25, "—")
        m = min(dims, key=lambda d: int(d.get("score", 0)))
        return (
            m.get("name", ""),
            int(m.get("score", 0)),
            level_of(int(m.get("score", 0))),
        )

    t_axis, t_score, _ = min_dim(text_dims)
    i_axis, i_score, _ = min_dim(image_dims)
    if t_score <= i_score:
        worst_src, worst_axis, worst_score = "텍스트", t_axis, t_score
    else:
        worst_src, worst_axis, worst_score = "이미지", i_axis, i_score
    lvl = level_of(worst_score)
    if lvl == "매우 위험":
        bg, emoji, summary = (
            PALETTE["매우 위험"],
            "🛑",
            f"{worst_axis} 측면에서 ({worst_src} 내) 매우 큰 리스크가 있습니다.",
        )
    elif lvl == "위험":
        bg, emoji, summary = (
            PALETTE["위험"],
            "⚠️",
            f"{worst_axis} 측면에서 ({worst_src} 내) 유의미한 리스크가 있습니다.",
        )
    elif lvl == "주의":
        bg, emoji, summary = (
            PALETTE["주의"],
            "⚠️",
            f"{worst_axis} 측면에서 ({worst_src} 내) 주의 신호가 있습니다.",
        )
    elif lvl == "안전":
        bg, emoji, summary = (
            PALETTE["안전"],
            "✅",
            "전반적으로 안전 수준입니다. 최소 안전 점수 16점 이상.",
        )
    else:
        bg, emoji, summary = (
            PALETTE["매우 안전"],
            "✅",
            "전반적으로 매우 안전합니다. 모든 축이 21점 이상.",
        )
    return {
        "level": lvl,
        "worst_axis": worst_axis,
        "worst_src": worst_src,
        "worst_score": worst_score,
        "bg": bg,
        "emoji": emoji,
        "summary": summary,
    }

# ========== 6) Hotspot helpers ==========
def _bbox(h: dict) -> Tuple[float, float, float, float]:
    if (h.get("shape") or "circle").lower() == "rect":
        x = float(h.get("x", 0))
        y = float(h.get("y", 0))
        w = float(h.get("w", 0))
        hgt = float(h.get("h", 0))
        return (x, y, x + w, y + hgt)
    cx = float(h.get("cx", 0.5))
    cy = float(h.get("cy", 0.5))
    r = float(h.get("r", 0.1))
    return (cx - r, cy - r, cx + r, cy + r)

def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

def _iou(b1, b2):
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = _area(b1) + _area(b2) - inter
    return inter / union if union > 0 else 0.0

def _centerdist(b1, b2):
    c1 = ((b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2)
    c2 = ((b2[0] + b2[2]) / 2, (b2[1] + b2[3]) / 2)
    return math.hypot(c1[0] - c2[0], c1[1] - c2[1])

def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    out["risks"] = [*{*(out.get("risks") or []), *(b.get("risks") or [])}]
    out["suggested_edits"] = [
        *{*(out.get("suggested_edits") or [])},
        *(b.get("suggested_edits") or []),
    ]
    if not out.get("label") and b.get("label"):
        out["label"] = b["label"]
    if not out.get("severity") and b.get("severity"):
        out["severity"] = b["severity"]
    return out

def dedupe_hotspots(hotspots: list) -> list:
    hs = [h for h in hotspots or [] if isinstance(h, dict)]
    hs_sorted = sorted(hs, key=lambda h: _area(_bbox(h)), reverse=True)
    kept = []
    for h in hs_sorted:
        b = _bbox(h)
        merged = False
        for i, k in enumerate(kept):
            bk = _bbox(k)
            if _iou(b, bk) > 0.55 or _centerdist(b, bk) < 0.12:
                kept[i] = _merge(k, h)
                merged = True
                break
        if not merged:
            hh = dict(h)
            for key in ["x", "y", "w", "h", "cx", "cy", "r"]:
                if key in hh:
                    try:
                        v = float(hh[key])
                        hh[key] = max(0.0, min(1.0, v))
                    except Exception:
                        pass
            kept.append(hh)
    return kept[:12]

def _color_class_from_severity(h: dict) -> str:
    sev = (h.get("severity") or "").strip()
    if sev in ("위험", "Risk"):
        return "warn"
    if sev in ("주의", "Caution"):
        return "caution"
    return ""  # 기본: 매우 위험(빨강)

def make_kv_overlay_html(img_src: str, hotspots: list, alpha: float = 0.20) -> str:
    """이미지 위에 SVG 오버레이를 얹는 HTML 반환 (숫자 배지 없음, 글로우+마스크). alpha=0.20 고정"""
    alpha = max(0.05, min(0.9, float(alpha)))
    hs = hotspots or []
    shapes = []
    for h in hs:
        shape = (h.get("shape") or "circle").lower()
        label = strip_circled(h.get("label") or "")
        klass = _color_class_from_severity(h)
        if shape == "rect":
            x = float(h.get("x", 0)) * 1000
            y = float(h.get("y", 0)) * 1000
            w = float(h.get("w", 0)) * 1000
            ht = float(h.get("h", 0)) * 1000
            shapes.append(
                f'<rect class="kv-hot {klass}" x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{ht:.1f}"><title>{attr_esc(label)}</title></rect>'
            )
        else:
            cx = float(h.get("cx", 0.5)) * 1000
            cy = float(h.get("cy", 0.5)) * 1000
            r = float(h.get("r", 0.08)) * 1000
            shapes.append(
                f'<circle class="kv-hot {klass}" cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}"><title>{attr_esc(label)}</title></circle>'
            )
    svg = (
        f'<svg class="kv-svg" viewBox="0 0 1000 1000" preserveAspectRatio="none" style="--alpha:{alpha}">'
        "<defs>"
        '<filter id="kv-glow" x="-50%" y="-50%" width="200%" height="200%">'
        '<feGaussianBlur stdDeviation="6" result="coloredBlur"/>'
        "<feMerge><feMergeNode in=\"coloredBlur\"/><feMergeNode in=\"SourceGraphic\"/></feMerge>"
        "</filter>"
        "</defs>"
        f'{"".join(shapes)}'
        "</svg>"
    )
    return (
        '<div class="kv-wrap">'
        f'<img src="{img_src}" class="kv-img"/>'
        f"{svg}"
        '<div class="kv-badge">Risk Overlay</div>'
        "</div>"
    )

# ========== 7) UI ==========
st.set_page_config(page_title="Creative Risk Auditor", page_icon="⚠️", layout="wide")
st.markdown(CARD_CSS, unsafe_allow_html=True)
st.title("⚠️ Creative Risk Auditor")
st.caption(
    "※ 각 축 25점 만점(높을수록 안전). 최종 판정은 ‘최악 축(가장 낮은 점수)’ 기준으로 결정됨. (성과/효율 평가는 하지 않습니다)"
)

# 고정 모델 적용 및 선택 옵션 제거
model = "gemini-2.5-flash"

country = st.text_input(
    "대상 국가/지역", value="인도", placeholder="예: 대한민국, 미국-캘리포니아, 사우디아라비아 …"
)
sector = st.text_input(
    "산업/카테고리(선택)", value="OLED TV", placeholder="예: 소비자가전, 식품/음료, 금융 등"
)
copy_txt = st.text_area(
    "카피라이트(캡션) 입력",
    value="OLED TV의 놀라운 색 재현율을 경험하세요!",
    placeholder="카피/캡션/해시태그/문구를 입력",
    height=140,
)
imgs = st.file_uploader(
    "Key Visual 업로드 (최대 3장)", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True
)

# sample01.png 자동 첨부 + 썸네일 미리보기
try:
    from io import BytesIO

    default_imgs = []
    if os.path.exists("sample01.png"):
        with open("sample01.png", "rb") as _f:
            _b = _f.read()
        _bio = BytesIO(_b)
        _bio.name = "sample01.png"
        _bio.type = "image/png"
        default_imgs = [_bio]
    imgs = list(imgs) + default_imgs if imgs else default_imgs
    if imgs:
        st.caption("미리보기")
        _cols = st.columns(min(5, len(imgs)))
        for i, _up in enumerate(imgs):
            try:
                _up.seek(0)
                _data = _up.read()
                _up.seek(0)
                with _cols[i % len(_cols)]:
                    st.image(_data, caption=getattr(_up, "name", "image"), width=120)
            except Exception:
                pass
except Exception:
    pass

go = st.button("Risk 분석", type="primary")

def legend_html():
    return (
        "<div class='legend'>"
        f"<span class='pill' style='background:{PALETTE['매우 안전']}'>매우 안전 (21~25)</span>"
        f"<span class='pill' style='background:{PALETTE['안전']}'>안전 (16~20)</span>"
        f"<span class='pill' style='background:{PALETTE['주의']}'>주의 (11~15)</span>"
        f"<span class='pill' style='background:{PALETTE['위험']}'>위험 (6~10)</span>"
        f"<span class='pill' style='background:{PALETTE['매우 위험']}'>매우 위험 (0~5)</span>"
        "</div>"
    )

def status_chip_html(score: int) -> str:
    lvl = level_of(score)
    col = level_color(score)
    return (
        f"<span class='status-chip' style='background:{col}'>{esc(lvl)}</span> "
        f"<span class='score-small'>{score}/25</span>"
    )

# --- Caption highlight helpers ---
def _extract_spans_from_flags(flags: List[dict]) -> List[str]:
    spans = []
    for f in flags or []:
        s = (f.get("span") or "").strip()
        if s:
            spans.append(s)
        for iss in f.get("issues") or []:
            for m in re.findall(r"[“”\"']([^“”\"']+)[“”\"']", iss):
                t = m.strip()
                if t:
                    spans.append(t)
    spans = [s for s in {s for s in spans} if len(s) >= 2]
    return sorted(spans, key=len, reverse=True)

def _find_all_ranges(text: str, needle: str) -> List[tuple]:
    ranges = []
    if not needle:
        return ranges
    pattern = re.compile(re.escape(needle), re.IGNORECASE)
    for m in pattern.finditer(text):
        ranges.append((m.start(), m.end()))
    return ranges

def _merge_ranges(ranges: List[tuple]) -> List[tuple]:
    if not ranges:
        return []
    ranges.sort()
    merged = [ranges[0]]
    for s, e in ranges[1:]:
        ls, le = merged[-1]
        if s <= le:
            merged[-1] = (ls, max(le, e))
        else:
            merged.append((s, e))
    return merged

def highlight_caption(text: str, flags: List[dict]) -> str:
    original = text or ""
    spans = _extract_spans_from_flags(flags)
    all_ranges = []
    for sp in spans:
        all_ranges += _find_all_ranges(original, sp)
    all_ranges = _merge_ranges(all_ranges)

    if not all_ranges:
        return f"<div class='caption-strong'>{html.escape(original)}</div>"

    parts = []
    last = 0
    for s, e in all_ranges:
        if last < s:
            parts.append(html.escape(original[last:s]))
        parts.append(f"<span class='caption-flag'>{html.escape(original[s:e])}</span>")
        last = e
    if last < len(original):
        parts.append(html.escape(original[last:]))

    return f"<div class='caption-strong'>{''.join(parts)}</div>"

# ========== 8) Run ==========
if go:
    if not (copy_txt or imgs):
        st.warning("텍스트 또는 이미지를 최소 1개 이상 제공하세요.")
        st.stop()
    if not country:
        st.warning("대상 국가/지역을 입력하세요.")
        st.stop()

    # 이미지 준비
    image_parts, data_uris = [], []
    if imgs:
        for up in imgs[:3]:
            p = to_image_part(up)
            if p:
                image_parts.append(p)
            data_uris.append(uploaded_to_data_uri(up))

    # 텍스트 Risk 평가
    text_ctx = (
        f"[국가/지역]\n{country}\n"
        f"[산업/카테고리]\n{sector or '(미지정)'}\n"
        f"[텍스트]\n{copy_txt.strip() or '(제공 없음)'}"
    )
    with st.spinner("카피라이트(캡션) Risk 평가 중…"):
        text_raw = call_gemini_text(TEXT_RISK_PROMPT + "\n\n" + text_ctx, model=model)
        text_risk = parse_json_or_fail(text_raw, "텍스트 Risk 평가")

    # 이미지 Risk 평가
    if image_parts:
        img_ctx = (
            f"[국가/지역]\n{country}\n"
            f"[산업/카테고리]\n{sector or '(미지정)'}\n"
            "[이미지] 업로드 순서 기준 1부터."
        )
        with st.spinner("Key Visual Risk 평가 중…"):
            image_raw = call_gemini_mm(IMAGE_RISK_PROMPT + "\n\n" + img_ctx, image_parts, model=model)
            image_risk = parse_json_or_fail(image_raw, "이미지 Risk 평가")
    else:
        image_risk = {
            "country": country,
            "core_dimensions": [
                {
                    "name": "Political",
                    "score": 25,
                    "why": ["이미지 미제공 — 해당 축에서 뚜렷한 논란·문제 소지가 확인되지 않습니다."],
                    "edits": ["유지 권장"],
                    "checks": ["—"],
                },
                {
                    "name": "Cultural",
                    "score": 25,
                    "why": ["이미지 미제공 — 해당 축에서 뚜렷한 논란·문제 소지가 확인되지 않습니다."],
                    "edits": ["유지 권장"],
                    "checks": ["—"],
                },
                {
                    "name": "Environmental",
                    "score": 25,
                    "why": ["이미지 미제공 — 해당 축에서 뚜렷한 논란·문제 소지가 확인되지 않습니다."],
                    "edits": ["유지 권장"],
                    "checks": ["—"],
                },
                {
                    "name": "Social",
                    "score": 25,
                    "why": ["이미지 미제공 — 해당 축에서 뚜렷한 논란·문제 소지가 확인되지 않습니다."],
                    "edits": ["유지 권장"],
                    "checks": ["—"],
                },
            ],
            "image_feedback": [],
        }

    # 성과/효율 언급 제거(후처리)
    def _sanitize_dim_items(dims: List[dict]) -> List[dict]:
        out = []
        for d in dims or []:
            dd = dict(d)
            dd["why"] = sanitize_lines(d.get("why") or [])
            dd["edits"] = sanitize_lines(d.get("edits") or [])
            dd["checks"] = sanitize_lines(d.get("checks") or [])
            out.append(dd)
        return out

    text_risk["core_dimensions"] = _sanitize_dim_items(text_risk.get("core_dimensions") or [])
    image_risk["core_dimensions"] = _sanitize_dim_items(image_risk.get("core_dimensions") or [])

    tfb = text_risk.get("text_feedback") or {}
    flags = []
    for f in tfb.get("flags") or []:
        ff = dict(f)
        ff["issues"] = sanitize_lines(f.get("issues") or [])
        ff["edits"] = sanitize_lines(f.get("edits") or [])
        flags.append(ff)
    text_risk["text_feedback"] = {"flags": flags}

    # 종합 결과
    core_t = text_risk.get("core_dimensions") or []
    core_i = image_risk.get("core_dimensions") or []
    overall = overall_from_text_image(core_t, core_i)

    st.markdown(
        f"<div class='subcard' style='background:{overall['bg']}; color:#fff;'>"
        f"<span class='score-text'>{overall['emoji']} 결과: {esc(overall['level'])}</span>"
        f"<br><b>{esc(overall['summary'])}</b>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Key Visual 평가 결과
    st.markdown("<div class='card'><h4>Key Visual 평가 결과</h4>", unsafe_allow_html=True)
    st.markdown(
        "<div class='note-muted'>Key Visual 내 Risk가 존재하는 영역을 표시합니다.</div>",
        unsafe_allow_html=True,
    )
    imgs_feedback = image_risk.get("image_feedback") or []
    if imgs_feedback:
        for it in imgs_feedback[:3]:
            idx = int(it.get("index", 1))
            notes = strip_circled((it.get("notes", "") or "").strip())
            hotspots_all = dedupe_hotspots(it.get("hotspots") or [])
            hotspots = [h for h in hotspots_all if any((h.get("risks") or []))]
            img_src = None
            if imgs and 1 <= idx <= len(imgs):
                img_src = uploaded_to_data_uri(imgs[idx - 1])
            if img_src and hotspots:
                html_overlay = make_kv_overlay_html(img_src, hotspots, alpha=0.20)
                st.markdown(f"<div class='subcard'>{html_overlay}</div>", unsafe_allow_html=True)
                if notes:
                    st.markdown(f"<div class='anno'><b>{esc(notes)}</b></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Key Visual 세부 평가 내용
    st.markdown("<div class='card'><h4>Key Visual 세부 평가 내용</h4>", unsafe_allow_html=True)
    st.markdown(legend_html(), unsafe_allow_html=True)
    order = ["Political", "Cultural", "Environmental", "Social"]
    imap = {d.get("name"): d for d in (image_risk.get("core_dimensions") or [])}
    tiles = []
    for name in order:
        d = imap.get(
            name,
            {
                "name": name,
                "score": 25,
                "why": [f"{name} 축: 현재 기준에서 뚜렷한 논란·문제 소지가 확인되지 않습니다."],
                "edits": ["유지 권장"],
                "checks": ["—"],
            },
        )
        score = int(d.get("score", 25))
        why = sanitize_lines(d.get("why") or [])
        edits = sanitize_lines(d.get("edits") or [])
        chip = status_chip_html(score)
        why_bold = [f"<b>{esc(why[0])}</b>"] + [esc(x) for x in why[1:2]] + [esc(x) for x in why[2:]]
        edits_bold = [f"<b>{esc(edits[0])}</b>"] + [esc(x) for x in edits[1:2]] + [esc(x) for x in edits[2:]]
        inner = (
            f"<div class='risk-tile'><h5>{esc(name)}</h5>"
            f"<div class='status-line'>{chip}</div>"
            "<div class='anno'><b>위험 요소</b><ul>"
            + "".join([f"<li>{x}</li>" for x in why_bold[:3]])
            + "</ul></div>"
            "<div class='anno'><b>수정 제안(리스크 완화)</b><ul>"
            + "".join([f"<li>{x}</li>" for x in edits_bold[:3]])
            + "</ul></div>"
            "</div>"
        )
        tiles.append(inner)
    st.markdown("<div class='risk-grid'>" + "".join(tiles) + "</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # 구분선/여백
    st.write("\n\n")
    st.markdown("<hr class='section-sep'/>", unsafe_allow_html=True)

    # 카피라이트(캡션) 입력 원문
    st.markdown("<div class='card'><h4>카피라이트(캡션) 입력 원문</h4>", unsafe_allow_html=True)
    tflags = (text_risk.get("text_feedback") or {}).get("flags") or []
    st.markdown(
        f"<div class='subcard'>{highlight_caption(copy_txt or '(입력 없음)', tflags)}</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # 카피라이트(캡션) 세부 평가 내용
    st.markdown("<div class='card'><h4>카피라이트(캡션) 세부 평가 내용</h4>", unsafe_allow_html=True)
    st.markdown(legend_html(), unsafe_allow_html=True)
    tmap = {d.get("name"): d for d in (text_risk.get("core_dimensions") or [])}
    tiles = []
    for name in order:
        d = tmap.get(
            name,
            {
                "name": name,
                "score": 25,
                "why": [f"{name} 축: 현재 기준에서 뚜렷한 논란·문제 소지가 확인되지 않습니다."],
                "edits": ["유지 권장"],
                "checks": ["—"],
            },
        )
        score = int(d.get("score", 25))
        why = sanitize_lines(d.get("why") or [])
        edits = sanitize_lines(d.get("edits") or [])
        chip = status_chip_html(score)
        why_bold = [f"<b>{esc(why[0])}</b>"] + [esc(x) for x in why[1:2]] + [esc(x) for x in why[2:]]
        edits_bold = [f"<b>{esc(edits[0])}</b>"] + [esc(x) for x in edits[1:2]] + [esc(x) for x in edits[2:]]
        inner = (
            f"<div class='risk-tile'><h5>{esc(name)}</h5>"
            f"<div class='status-line'>{chip}</div>"
            "<div class='anno'><b>위험 요소</b><ul>"
            + "".join([f"<li>{x}</li>" for x in why_bold[:3]])
            + "</ul></div>"
            "<div class='anno'><b>수정 제안(리스크 완화)</b><ul>"
            + "".join([f"<li>{x}</li>" for x in edits_bold[:3]])
            + "</ul></div>"
            "</div>"
        )
        tiles.append(inner)
    st.markdown("<div class='risk-grid'>" + "".join(tiles) + "</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # 다운로드
    out = {"text_risk": text_risk, "image_risk": image_risk, "overall": overall}
    st.download_button(
        "JSON 결과 다운로드",
        data=json.dumps(out, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="creative_risk_result.json",
        mime="application/json",
    )
    st.success("✅ 분석 완료")
