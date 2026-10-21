"""VOCAI — 고객의 목소리 → 고칠 곳 (Streamlit 화면). CSS 주입은 이 파일 한 곳에서만 한다."""

import html
import os
import time
from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer

import config
import data_utils
import stage1
import stage2

st.set_page_config(page_title="VOCAI", page_icon="🔎", layout="wide")

CSS = """
<style>
.block-container {padding-top: 4.5rem;}  /* Streamlit 상단 바에 헤더가 가리지 않게 */
.vocai-header {background:#FFFFFF; border:1px solid #E3E6EF; border-radius:12px; padding:14px 20px; margin-bottom:10px;}
.vocai-title {font-size:1.5rem; font-weight:800; color:#1428A0;}
.vocai-badge {display:inline-block; color:#FFFFFF; border-radius:6px; padding:1px 8px; font-size:0.8rem; font-weight:700;}
[class*="st-key-card"] {background:#FFFFFF; border:1px solid #E3E6EF; border-radius:12px; padding:14px 16px;}
/* 요약 카드: 모두 같은 높이, 글자가 길면 한 줄 말줄임(…) */
[data-testid="stMetric"] {background:#FFFFFF; border:1px solid #E3E6EF; border-radius:12px; padding:12px 16px;
  height:124px; box-sizing:border-box; overflow:hidden;}
[data-testid="stMetric"] * {min-width:0;}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p,
[data-testid="stMetricValue"], [data-testid="stMetricValue"] > div,
[data-testid="stMetricDelta"], [data-testid="stMetricDelta"] > div {
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
[data-testid="stMetricValue"] {font-size:1.9rem;}
/* 근거 표: 흰 배경, 긴 내용은 줄바꿈 그대로 (마지막 수정자는 한 줄 요약 · 범위별은 아래 접힌 상자) */
[class*="st-key-card_evidence"] table, [class*="st-key-card_evidence"] th, [class*="st-key-card_evidence"] td {background:#FFFFFF;}
[class*="st-key-card_evidence"] td {white-space:pre-line;}
/* 티켓 카드 마지막 줄: [처리 완료](담당 카드만, 오른쪽 끝) — 버튼 없는 카드도 같은 높이의 빈 줄 */
[class*="st-key-tact_"] {min-height:40px;}
/* 현재 개발자의 티켓 카드: primaryColor 테두리 */
[class*="st-key-card_mine"] {border:2px solid #1428A0 !important;}
/* 티켓 카드 문제점: 최대 2줄, 넘치면 … (카드 높이 고정) */
[class*="st-key-tprob_"] p {display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; margin:0;}
[class*="st-key-tprob_"] {min-height:3.2em;}  /* 한 줄이어도 두 줄 자리 → 모든 카드의 마지막 줄(처리 완료)이 카드 바닥에 맞춰진다 */
[class*="st-key-tline_"] p {white-space:nowrap; overflow:hidden; text-overflow:ellipsis; margin:0;}
/* 사이드바 내 티켓: 본문 티켓 카드를 줄인 압축 카드 (일반 배치 · 오른쪽 위 작은 '선택' 버튼). 높이는 내용에 맞게 */
[class*="st-key-sbcards"] {gap:6px;}
[class*="st-key-sbcard_"] {background:#FFFFFF; border:1px solid #E3E6EF; border-radius:8px;
  padding:10px 10px; gap:4px; text-align:left; height:auto; overflow:visible;}
[class*="st-key-sbcard_sel_"] {border-left:3px solid #1428A0;}
/* Streamlit은 문단 아래 여백 1rem을 마크다운 칸의 -1rem 여백으로 상쇄한다. 문단 여백을 0으로 둔 곳은
   그 -1rem도 0으로 돌려야 칸 높이가 찌그러져(약 4px) 다음 줄과 겹치지 않는다 */
[class*="st-key-sbcard_"] [data-testid="stMarkdownContainer"], [class*="st-key-sbcard_"] [data-testid="stCaptionContainer"],
[class*="st-key-tprob_"] [data-testid="stMarkdownContainer"],
[class*="st-key-tline_"] [data-testid="stMarkdownContainer"], [class*="st-key-repvoc_"] [data-testid="stMarkdownContainer"],
[class*="st-key-lastauth_"] [data-testid="stMarkdownContainer"] {margin-bottom:0 !important;}
[class*="st-key-sbcard_"] p {font-size:0.85rem; line-height:1.4; margin:0; text-align:left;}
[class*="st-key-sbcard_"] [data-testid="stCaptionContainer"] p {font-size:0.75rem;}
[class*="st-key-sbprob_"] p {display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden;}
[class*="st-key-sbpick_"] button {min-height:24px; height:24px; padding:0 8px; font-size:0.75rem; line-height:1;}
/* 대표 VOC · 마지막 수정자 상세: expander 제목과 같은 본문 크기, 항목 사이 4px */
[class*="st-key-repvoc_"] p, [class*="st-key-repvoc_"] li, [class*="st-key-lastauth_"] p, [class*="st-key-lastauth_"] li {font-size:14px; line-height:1.4;}
[class*="st-key-repvoc_"] ul, [class*="st-key-lastauth_"] ul {margin:0; padding-left:1.2rem;}
[class*="st-key-repvoc_"] li, [class*="st-key-lastauth_"] li {margin:0 0 4px 0; padding:0;}
</style>
"""
# 앱 전체 글꼴은 config.APP_FONT 하나 (st.html 폴더 후보 카드도 같은 상수). 코드 뷰(pre)만 고정폭.
FONT_CSS = (f"<style>html, body, .stApp, .vocai-header {{font-family: {config.APP_FONT};}} "
            ".vocai-code pre {font-family: ui-monospace, Consolas, 'Courier New', monospace;}</style>")
st.markdown(CSS + FONT_CSS, unsafe_allow_html=True)

VIEW_RAW = "원본 VOC"
VIEW_CLUSTER = "유형 분석 결과"
VIEW_OPTIONS = [VIEW_RAW, VIEW_CLUSTER]
PERIOD_RECENT, PERIOD_CUSTOM, PERIOD_ALL = "최근 2주", "직접 선택", "전체 기간"
PERIOD_MODES = [PERIOD_RECENT, PERIOD_CUSTOM, PERIOD_ALL]

# ════════════════════════════ 캐시 ════════════════════════════

@st.cache_data
def cached_vocs() -> pd.DataFrame:
    return data_utils.load_vocs()


@st.cache_data
def cached_kpis(start: date, end: date) -> dict:
    df = cached_vocs()
    ps, pe = data_utils.prev_period(start, end)
    return data_utils.compute_kpis(data_utils.filter_period(df, start, end), data_utils.filter_period(df, ps, pe))


@st.cache_data(show_spinner=False)
def cached_voc_vectors(voc_ids: tuple[str, ...]):
    """VOC 임베딩. outputs/에도 캐시되어 투영 방식을 바꿔도 다시 부르지 않는다."""
    df = cached_vocs().set_index("voc_id").loc[list(voc_ids)].reset_index()
    return stage1.embed_vocs(df)


@st.cache_data(show_spinner=False)
def cached_voc_map(voc_ids: tuple[str, ...]):
    """(임베딩, t-SNE 3D 좌표). 좌표는 outputs/voc_map_*.json에도 캐시 (같은 기간이면 다시 계산 안 함)."""
    vectors = cached_voc_vectors(voc_ids)
    return vectors, stage1.cached_projection(voc_ids, vectors)


@st.cache_data(show_spinner=False)
def cached_embed_clusters(voc_ids: tuple[str, ...]):
    """임베딩 클러스터링 결과 (VOC별 이름, {k: 실루엣}). outputs/voc_map_*.json에도 캐시."""
    vectors = cached_voc_vectors(voc_ids)
    texts = cached_vocs().set_index("voc_id").loc[list(voc_ids), "text"].tolist()
    return stage1.cached_embedding_clusters(voc_ids, vectors, texts)


@st.cache_data
def cached_users() -> list[dict]:
    return data_utils.load_users()


@st.cache_data(show_spinner=False)
def cached_rep_vocs(voc_ids: tuple[str, ...]) -> list[str]:
    """유형의 대표 VOC: 임베딩(의미 공간과 같은 캐시) 중심에 가까운 순 REP_VOC_MAX건.
    API 키가 없으면 유형 목록에서 고르게 뽑는다."""
    texts = cached_vocs().set_index("voc_id").reindex(list(voc_ids))["text"].dropna()
    if texts.empty:
        return []
    if os.getenv("OPENAI_API_KEY"):
        vectors = cached_voc_vectors(tuple(texts.index))
        return [texts.iloc[i] for i in stage1.nearest_to_center(vectors, config.REP_VOC_MAX)]
    step = max(len(texts) // config.REP_VOC_MAX, 1)
    return texts.iloc[::step].head(config.REP_VOC_MAX).tolist()


@st.cache_resource
def cached_llm():
    return stage1.get_llm()


@st.cache_resource
def cached_vectorstore():
    return stage2.get_vectorstore()


@st.cache_resource
def cached_graph():
    return stage2.build_stage2_graph()


# ════════════════════════════ 상태 ════════════════════════════

ss = st.session_state
ss.setdefault("stage1", {})          # {(start, end): {"clusters", "assignments", "secs"}}
# 발행된 티켓은 outputs/tickets.json이 기준 — 매 실행마다 읽어 개발자를 바꾸거나 새로고침해도 남고, 번호도 이어진다
ss["tickets"], ss["next_ticket_no"] = stage1.load_tickets()
# 화면(티켓 카드 · 사이드바 · 2️⃣·3️⃣ 티켓 선택 · 대표 VOC)에는 처리 완료되지 않은 티켓만. 완료 기록은 파일에 남는다
OPEN_TICKETS = stage1.open_tickets(ss["tickets"])
ss.setdefault("stage2", {})          # {ticket_id: 최종 state} — outputs/stage2/{ticket_id}.json에도 저장
# 코드 찾기 대기 상태(폴더 확인에서 멈춤): {ticket_id: {"thread_id", "payload", "state"}} — 티켓별로 따로 둔다
ss.setdefault("stage2_wait", {})
ss.setdefault("stage2_run_no", {})   # {ticket_id: 실행 번호} — thread_id = f"{ticket_id}:{번호}"
ss.setdefault("stage2_notice", {})   # {ticket_id: 안내 문구} — 수정 라인 찾기 실패·0건 (2️⃣ 탭에 한 번 표시)
ss.setdefault("current_ticket", None)
ss.setdefault("editor_nonce", 0)        # 발행 후 유형 표(data_editor)를 체크 없이 새로 그리기 위한 번호
ss.setdefault("voc_view", VIEW_RAW)
ss.setdefault("period_mode", PERIOD_RECENT)   # 처음 열면 '최근 2주'(오늘 포함)
ss.setdefault("custom_dates", (config.ANALYSIS_START, config.ANALYSIS_END))  # 1️⃣ 탭 보기 전환 (segmented_control key)


def badge(priority: str) -> str:
    """고정 라벨만 들어가는 HTML 배지 (LLM 텍스트 아님)."""
    color = config.PRIORITY_COLOR.get(priority, "#9AA0A6")
    return f'<span class="vocai-badge" style="background:{color}">{priority}</span>'


def has_api_key() -> bool:
    if os.getenv("OPENAI_API_KEY"):
        return True
    st.error("`.env`에 OPENAI_API_KEY가 없습니다.")
    return False


def daily_bar_chart(daily: pd.DataFrame, title: str, weekly: bool = False) -> None:
    """일별(주별) 누적 막대. 범례는 차트 위 가로, 날짜는 MM/DD, 아래 여백을 줘 눈금이 잘리지 않게 한다."""
    order = daily.groupby("group")["count"].sum().sort_values(ascending=False).index.tolist()
    tail = [g for g in (config.OTHER_LABEL, config.UNASSIGNED_LABEL) if g in order]  # '기타' → '미분류'는 항상 마지막 · 회색
    order = [g for g in order if g not in tail] + tail
    palette = px.colors.qualitative.Bold
    colors = {g: palette[i % len(palette)] for i, g in enumerate(order)}
    colors[config.OTHER_LABEL] = config.OTHER_COLOR
    colors[config.UNASSIGNED_LABEL] = config.UNASSIGNED_COLOR
    with st.container(key="card_chart"):
        st.markdown(f"**{title}**")
        fig = px.bar(daily, x="date", y="count", color="group", labels={"count": "건수", "group": ""},
                     category_orders={"group": order}, color_discrete_map=colors)
        fig.update_layout(barmode="stack", height=380, margin=dict(l=10, r=10, t=40, b=90),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title=None),
                          plot_bgcolor="#FFFFFF")
        # x축 제목 "날짜"는 눈금(10/03) 아래에 간격(standoff)을 두고, 범례는 차트 위라 겹치지 않는다
        fig.update_xaxes(title=dict(text="날짜", standoff=20), tickformat="%m/%d",
                         dtick=86_400_000 * (7 if weekly else 1), tickangle=0)
        st.plotly_chart(fig, width="stretch")


def render_folder_cards(cards, repo: str, branch: str, voc_type: str) -> None:
    """폴더 후보: 제목 줄 · 범례 · 폴더 카드 그리드(한 줄 3장). HTML은 st.html로, 모든 값은 html.escape.
    글꼴은 본문과 같은 config.APP_FONT (고정폭 없음)."""
    esc = html.escape
    pill = "display:block; border-radius:8px; padding:3px 10px; margin:4px 0; font-size:13px;"
    title = (f'<div style="font-weight:700; font-size:15px; margin-bottom:4px;">'
             f'{esc(repo)} / {esc(branch)} · {esc(voc_type)} 기준</div>')
    legend_items = []
    for status in config.FILE_STATUS_ORDER:
        style = config.FILE_STATUS_STYLE[status]
        mark = "┅" if status == "감점" else "■"
        sw = (f'<span style="display:inline-block; width:14px; height:14px; border-radius:3px; vertical-align:-2px; '
              f'{style}"></span>') if mark == "■" else f'<span style="color:#D93025; font-weight:700;">{mark}</span>'
        legend_items.append(f"{sw} {esc(status)}")
    legend = f'<div style="font-size:13px; color:#3C4043; margin-bottom:10px;">{" · ".join(legend_items)}</div>'
    if not cards:
        st.html(f'<div style="font-family:{esc(config.APP_FONT)};">{title}{legend}'
                '<div style="color:#5F6368; font-size:13px;">보여 줄 폴더가 없습니다.</div></div>')
        return
    boxes = []
    for card in cards:
        pills = []
        for item in card.items:
            label = esc(item.name) + (" ✕" if item.status == "감점" else "")
            tip = f' title="{esc(item.note)}"' if item.note else f' title="{esc(item.file)}"'
            pills.append(f'<span style="{pill}{config.FILE_STATUS_STYLE[item.status]}"{tip}>{label}</span>')
        boxes.append(
            '<div style="background:#FFFFFF; border:1px solid #E3E6EF; border-radius:12px; padding:12px 14px;">'
            f'<div style="font-weight:700; margin-bottom:6px;">📁 {esc(card.path)}</div>'
            f'{"".join(pills)}</div>')
    grid = (f'<div style="display:grid; grid-template-columns:repeat({config.FOLDER_CARD_COLUMNS}, minmax(0, 1fr)); '
            f'gap:12px;">{"".join(boxes)}</div>')
    st.html(f'<div style="font-family:{esc(config.APP_FONT)};">{title}{legend}{grid}</div>')


def render_ticket_cards() -> None:
    """발행된 티켓(모든 개발자에게 전체 표시): 카드 한 줄 3장, 우선순위 순, 높이 모두 같음(TICKET_CARD_HEIGHT).
    항목: ID · 우선순위 배지 / 문제점(최대 2줄) / 건수·담당 부서·담당자 / 발행일→예상 완료일.
    현재 개발자의 티켓은 글자 없이 테두리만 primaryColor로 강조. 티켓이 없으면 숨긴다."""
    tickets = stage1.sort_tickets(OPEN_TICKETS.values())
    if not tickets:
        return
    st.subheader("발행된 티켓")
    n = config.TICKET_CARDS_PER_ROW
    for i in range(0, len(tickets), n):
        cols = st.columns(n)
        for col, t in zip(cols, tickets[i:i + n]):
            mine = t.assignee_id == me["id"]
            card_key = f"card_mine_{t.ticket_id}" if mine else f"card_ticket_{t.ticket_id}"
            # 흰 배경 · 테두리 · 모서리 12px (CSS), 높이 고정 — 내 티켓은 primaryColor 테두리
            with col.container(border=True, height=config.TICKET_CARD_HEIGHT, key=card_key):
                c_id, c_badge = st.columns([2, 1], vertical_alignment="center")
                c_id.markdown(f"**{t.ticket_id}**")
                c_badge.markdown(f'<div style="text-align:right">{badge(t.priority)}</div>', unsafe_allow_html=True)
                with st.container(key=f"tprob_{t.ticket_id}"):   # 문제점: 최대 2줄, 넘치면 … (CSS line-clamp)
                    st.markdown(f"**{t.problem}**")
                with st.container(key=f"tline_{t.ticket_id}"):   # 한 줄 고정
                    st.markdown(f"{t.count:,}건 · {t.owner_part} · {t.assignee}")
                st.caption(f"발행일 {date.fromisoformat(t.issued_on):%m/%d} → "
                           f"예상 완료일 {date.fromisoformat(t.due_date):%m/%d}")
                # 마지막 줄: [발행 취소] 왼쪽 끝 · [처리 완료] 오른쪽 끝 (현재 개발자가 담당자인 카드에만, 기본 크기).
                # distribute = 두 버튼을 양 끝으로 벌린다. 다른 카드는 같은 높이의 빈 줄
                actions = st.container(horizontal=True, horizontal_alignment="distribute",
                                       key=f"tact_{t.ticket_id}")
                if mine:
                    if actions.button("발행 취소", type="secondary", key=f"cancel_card_{t.ticket_id}"):
                        confirm_cancel(t.ticket_id)
                    if actions.button("처리 완료", type="secondary", key=f"done_card_{t.ticket_id}"):
                        confirm_complete(t.ticket_id)


@st.dialog("처리 완료")
def confirm_complete(ticket_id: str) -> None:
    """처리 완료 확인 창. [완료]하면 status "완료" · completed_at을 tickets.json에 저장하고 화면에서 뺀다 (되돌리기 없음).
    커밋을 감지해 자동으로 완료하지 않는다 — 수정 여부는 담당 개발자가 판단한다."""
    st.write(f"{ticket_id}을 처리 완료할까요? 완료한 티켓은 화면에서 사라집니다.")
    row = st.container(horizontal=True, gap="small")  # [완료] [취소] 바로 옆에
    if row.button("완료", type="primary", key="dlg_done"):
        stage1.complete_ticket(ticket_id)
        if ss.get("current_ticket") == ticket_id:
            ss["current_ticket"] = None  # 다음 실행에서 남은 내 티켓 중 첫 번째로
        st.rerun()
    if row.button("취소", key="dlg_cancel"):
        st.rerun()


@st.dialog("발행 취소")
def confirm_cancel(ticket_id: str) -> None:
    """발행 취소(담당 해제) 확인 창. [발행 취소]하면 status "취소" · canceled_at 저장 + 코드 찾기 결과 삭제 →
    티켓은 화면에서 사라지고 유형은 유형 표로(처리 전) 돌아가 누구나 다시 발행할 수 있다 (되돌리기 없음)."""
    st.write(f"{ticket_id} 발행을 취소할까요?")
    row = st.container(horizontal=True, gap="small")  # [발행 취소] [닫기] 바로 옆에
    if row.button("발행 취소", type="primary", key="dlg_cancel_ticket"):
        stage1.cancel_ticket(ticket_id)
        ss["stage2"].pop(ticket_id, None)  # 세션에 남은 코드 찾기 결과도 버린다 (파일은 cancel_ticket이 지움)
        ss["stage2_wait"].pop(ticket_id, None)
        if ss.get("current_ticket") == ticket_id:
            ss["current_ticket"] = None
        ss["editor_nonce"] += 1  # 유형 표를 새로 그려 돌아온 유형이 체크 없이 보이게
        st.rerun()
    if row.button("닫기", key="dlg_close"):
        st.rerun()


def shorten(text, n: int = config.CELL_MAX_CHARS):
    """n자를 넘으면 앞 n자 + '…'. 글자가 아니면 그대로."""
    if not isinstance(text, str) or len(text) <= n:
        return text
    return text[:n] + "…"


def show_table(df: pd.DataFrame, key: str, **kwargs) -> None:
    """모든 st.dataframe 표가 쓰는 규칙: 화면용 복사본만 CELL_MAX_CHARS로 잘라 '…'(원본 df는 그대로).
    행 선택(체크박스)은 어느 표에도 켜지 않는다 — 체크박스가 있는 표는 클러스터 표(st.data_editor, 티켓 발행) 하나뿐."""
    view = df.copy()
    for c in view.columns:
        if view[c].map(lambda v: isinstance(v, str) and len(v) > config.CELL_MAX_CHARS).any():
            view[c] = view[c].map(shorten)
    st.dataframe(view, hide_index=True, width="stretch", key=key, **kwargs)


def md_escape(text: str) -> str:
    """고객 VOC 원문을 마크다운으로 해석하지 않고 그대로 보이게."""
    for ch in "\\`*_{}[]<>()#+-.!|~":
        text = text.replace(ch, "\\" + ch)
    return text


def render_representative_vocs() -> None:
    """대표 VOC 보기: 현재 개발자가 담당한 티켓마다 접힌 expander 하나 (다른 개발자의 티켓은 없음).
    펼치면 그 유형의 대표 VOC를 유형 중심에 가까운 순으로 최대 REP_VOC_MAX건, 원문 그대로."""
    st.subheader("대표 VOC 보기")
    if not my_tickets:
        st.info("티켓을 발행하면 해당 유형의 대표 VOC를 볼 수 있습니다")
        return
    for t in my_tickets:
        with st.expander(f"{t.ticket_id} · {t.problem}", expanded=False):
            with st.container(key=f"repvoc_{t.ticket_id}"):  # 3건을 마크다운 목록 하나로 (간격 좁게)
                st.markdown("\n".join(f"- {md_escape(v)}" for v in cached_rep_vocs(tuple(t.voc_ids))))


def stage2_result_ok(result: dict) -> bool:
    """세션의 코드 찾기 결과가 지금 모델로 만든 것인지. 코드가 바뀌면 Streamlit이 모듈을 다시 읽어
    예전 클래스 객체(예: repo 없는 Impact)가 남을 수 있다."""
    try:
        return (all(isinstance(i, stage2.Impact) and hasattr(i, "repo") for i in result.get("impacts", []))
                and all(isinstance(r, stage2.LineRec) for r in result.get("line_recs", []))
                and all(isinstance(c, stage2.RepoCandidate) for c in result.get("repo_candidates", []))
                and all(isinstance(f, stage2.FolderScore) for f in result.get("folder_scores", [])))
    except Exception:  # noqa: BLE001
        return False


def hover_text(text: str, width: int = config.HOVER_LINE_CHARS, limit: int = config.HOVER_MAX_CHARS) -> str:
    """Plotly 툴팁용: limit자를 넘으면 …, width 글자마다 <br> (HTML 이스케이프)."""
    if len(text) > limit:
        text = text[:limit] + "…"
    return "<br>".join(html.escape(text[i:i + width]) for i in range(0, len(text), width))


def render_voc_map(cur: pd.DataFrame, clusters) -> None:
    """VOC 유사도: 임베딩 3D 의미 공간 + 선택한 VOC의 유사 VOC. 색은 LLM 또는 임베딩 클러스터, 위치는 임베딩."""
    st.subheader("VOC 유사도")
    st.caption(f"위치는 VOC 문장 임베딩({config.VOC_EMBEDDING_MODEL})을 t-SNE로 3D 투영한 것입니다. "
               "가까울수록 고객 표현이 비슷합니다. 색은 유형 또는 임베딩만으로 묶은 분류(KMeans)입니다.")
    c_on, c_color = st.columns([1, 3], vertical_alignment="center")
    if not c_on.toggle("의미 공간", key="voc_map_on"):
        return
    color_by = c_color.radio("색 기준", ["유형", "임베딩 분류"], horizontal=True,
                             key="voc_map_color", label_visibility="collapsed")
    if not has_api_key():
        return

    if len(cur) > config.SIMILARITY_MAX_POINTS:  # 대용량 대비: seed 고정 무작위 표본
        st.caption(f"{len(cur):,}건 중 {config.SIMILARITY_MAX_POINTS:,}건 표본")
        cur = cur.sample(n=config.SIMILARITY_MAX_POINTS, random_state=config.PROJECTION_SEED)
    cur = cur.sort_values("voc_id").reset_index(drop=True)
    voc_ids = tuple(cur["voc_id"])
    with st.spinner("VOC 임베딩 · t-SNE 투영 · 임베딩 분류 중…"):
        vectors, coords = cached_voc_map(voc_ids)
        emb_labels, sil_by_k = cached_embed_clusters(voc_ids)
    cur = cur.assign(x=coords[:, 0], y=coords[:, 1], z=coords[:, 2], emb=emb_labels)
    labels = cur["cluster"].tolist()

    palette = px.colors.qualitative.Bold
    if color_by == "유형":
        color_col = "cluster"
        order = [c.label for c in clusters] + [config.OTHER_LABEL]
    else:
        color_col = "emb"
        order = sorted(set(emb_labels), key=lambda s: int(s.split(" ·")[0][1:]))
        palette = px.colors.qualitative.Alphabet if len(order) > len(palette) else palette
    colors = {lab: palette[i % len(palette)] for i, lab in enumerate(order)}
    colors[config.OTHER_LABEL] = "#BDC1C6"

    pick = st.selectbox(
        "기준 VOC", range(len(cur)),
        format_func=lambda i: f"{cur.at[i, 'voc_id']} · [{labels[i]}] {cur.at[i, 'text'][:50]}",
        key="voc_map_pick",
    )
    neighbors = stage1.similar_vocs(vectors, pick, config.SIMILAR_TOP_K)

    fig = go.Figure()
    for lab in order:
        part = cur[cur[color_col] == lab]
        if part.empty:
            continue
        fig.add_trace(go.Scatter3d(
            x=part["x"], y=part["y"], z=part["z"], mode="markers", name=html.escape(lab),
            marker=dict(size=config.MAP_POINT_SIZE, color=colors[lab], opacity=0.8, line=dict(width=0)),
            # 툴팁 두 줄만: 유형 · 원문 (<extra></extra>로 trace 이름 숨김)
            customdata=[f"유형: {html.escape(a)}<br>원문: {hover_text(t)}" for a, t in zip(part["cluster"], part["text"])],
            hovertemplate="%{customdata}<extra></extra>",
        ))
    # 기준 VOC → 유사 VOC 연결선
    p = cur.loc[pick]
    lx, ly, lz = [], [], []
    for j, _ in neighbors:
        q = cur.loc[j]
        lx += [p["x"], q["x"], None]
        ly += [p["y"], q["y"], None]
        lz += [p["z"], q["z"], None]
    fig.add_trace(go.Scatter3d(x=lx, y=ly, z=lz, mode="lines", name="유사 연결",
                               line=dict(color="#1428A0", width=2), hoverinfo="skip"))
    nb = cur.loc[[j for j, _ in neighbors]]
    fig.add_trace(go.Scatter3d(
        x=nb["x"], y=nb["y"], z=nb["z"], mode="markers", name=f"유사 VOC Top {len(neighbors)}",
        marker=dict(size=config.MAP_HIGHLIGHT_SIZE, symbol="circle-open", color="#1428A0", line=dict(width=2)),
        customdata=[f"유형: {html.escape(a)}<br>원문: {hover_text(t)}" for a, t in zip(nb["cluster"], nb["text"])],
        hovertemplate="%{customdata}<extra></extra>",
    ))
    fig.add_trace(go.Scatter3d(
        x=[p["x"]], y=[p["y"]], z=[p["z"]], mode="markers", name="기준 VOC",
        marker=dict(size=config.MAP_HIGHLIGHT_SIZE, symbol="diamond", color="#202124", line=dict(color="#FFFFFF", width=1)),
        customdata=[f"유형: {html.escape(p['cluster'])}<br>원문: {hover_text(p['text'])}"],
        hovertemplate="%{customdata}<extra></extra>",
    ))
    axis = dict(showticklabels=False, title="", showbackground=True,
                backgroundcolor="#F8F9FC", gridcolor="#E3E6EF", zeroline=False)
    fig.update_layout(
        height=640, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="#FFFFFF",
        scene=dict(xaxis=axis, yaxis=axis, zaxis=axis, aspectmode="cube"),
        legend=dict(orientation="h", y=-0.02, itemsizing="constant"),
    )
    with st.container(key="card_voc_map"):
        st.plotly_chart(fig, width="stretch")

    st.markdown(f"**{p['voc_id']}** 와 가장 비슷한 VOC")
    # 화면에는 유형만 (같은 LLM 클러스터 여부 · 임베딩 클러스터 열은 표시하지 않음 — 계산은 그대로)
    near = data_utils.to_display_vocs(pd.DataFrame([{
        "순위": r, "voc_id": cur.at[j, "voc_id"], "유사도": s, "유형": labels[j], "text": cur.at[j, "text"],
    } for r, (j, s) in enumerate(neighbors, 1)]))
    show_table(near, key="tbl_near", column_config={
        "유사도": st.column_config.ProgressColumn(format="%.3f", min_value=0.0, max_value=1.0)})

    render_cluster_comparison(vectors, labels, emb_labels, sil_by_k)


def render_cluster_comparison(vectors, llm_labels: list[str], emb_labels: list[str], sil_by_k: dict) -> None:
    """LLM 분류 vs 임베딩 분류: 클러스터 수 · ARI(두 분류의 일치도) · 임베딩 분류 k별 실루엣."""
    from sklearn.metrics import adjusted_rand_score

    st.markdown("#### LLM 분류 vs 임베딩 분류")
    c1, c2 = st.columns(2)
    c1.metric("유형 수 (LLM · 임베딩)", f"{len(set(llm_labels))} · {len(set(emb_labels))}")
    c2.metric("ARI (LLM ↔ 임베딩)", f"{adjusted_rand_score(llm_labels, emb_labels):.3f}",
              help="두 분류가 VOC를 같은 묶음으로 나눈 정도 (1 = 완전히 같음, 0 = 우연 수준)")

    if sil_by_k:
        with st.expander(f"실루엣 — 임베딩 분류 k = {config.MIN_CLUSTERS}~{config.MAX_CLUSTERS}"):
            sil = pd.DataFrame({"k": list(sil_by_k), "실루엣": list(sil_by_k.values())})
            fig_k = px.line(sil, x="k", y="실루엣", markers=True)
            fig_k.update_traces(name="임베딩 (k별)", showlegend=True)
            best = sil.loc[sil["실루엣"].idxmax()]
            fig_k.add_scatter(x=[best["k"]], y=[best["실루엣"]], mode="markers", name=f"임베딩 선택 k={int(best['k'])}",
                              marker=dict(size=12, color="#D93025"))
            fig_k.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="#FFFFFF")
            st.plotly_chart(fig_k, width="stretch")


# ════════════════════════════ 헤더 ════════════════════════════

df_all = cached_vocs()
min_d, max_d = df_all["date"].min(), df_all["date"].max()

st.markdown('<div class="vocai-header"><span class="vocai-title">VOCAI</span></div>', unsafe_allow_html=True)

# ════════════════════════════ 사이드바: 현재 개발자 · 내 티켓 ════════════════════════════
# 실제 로그인 대신 '현재 개발자'를 고른다 (사용자는 모두 개발자, 발행자 역할 없음)

USERS = cached_users()
USER_BY_ID = {u["id"]: u for u in USERS}
ss.setdefault("current_user_id", USERS[0]["id"])


def user_label(u: dict) -> str:
    """현재 개발자 표시 형식 — 화면 어디서나 "김서준 (센서 파트)"."""
    return f"{u['name']} ({u['part']})"


def to_raw_view() -> None:
    """기간을 바꾸면 보기를 '원본 VOC'로 (위젯이 그려진 뒤엔 키를 못 바꾸므로 콜백에서). 분석 결과·캐시는 그대로."""
    ss["voc_view"] = VIEW_RAW


def to_raw_view_if_period_changed() -> None:
    if ss.get("period_mode") in (PERIOD_CUSTOM, PERIOD_ALL):
        to_raw_view()


def pick_ticket(ticket_id: str) -> None:
    ss["current_ticket"] = ticket_id


with st.sidebar:
    user_labels = {user_label(u): u["id"] for u in USERS}
    # key로 선택 상태를 고정한다 (index로 그리면 바꾼 직후 다시 바꿀 때 다른 위젯으로 인식돼 선택이 무시된다)
    ss.setdefault("current_user_label", user_label(USER_BY_ID[ss["current_user_id"]]))
    picked_user = st.selectbox("현재 개발자", list(user_labels), key="current_user_label")
    ss["current_user_id"] = user_labels[picked_user]
    me = USER_BY_ID[ss["current_user_id"]]

    st.markdown("**내 티켓**")
    my_tickets = [t for t in stage1.sort_tickets(OPEN_TICKETS.values()) if t.assignee_id == me["id"]]
    # 현재 티켓은 항상 내 티켓 중 하나 (개발자를 바꾸면 그 사람의 첫 티켓)
    if ss["current_ticket"] not in {t.ticket_id for t in my_tickets}:
        ss["current_ticket"] = my_tickets[0].ticket_id if my_tickets else None
    if not my_tickets:
        st.caption("맡은 티켓이 없습니다")
    with st.container(key="sbcards"):
        for t in my_tickets:
            due = date.fromisoformat(t.due_date)
            left = (due - date.today()).days
            selected = t.ticket_id == ss["current_ticket"]
            # 선택된 카드는 왼쪽 primaryColor 세로 줄 (CSS st-key-sbcard_sel_)
            with st.container(key=f"sbcard_{'sel_' if selected else ''}{t.ticket_id}"):
                # 1줄: ID + 우선순위 배지(왼쪽) · 오른쪽 위 작은 '선택' 버튼 → 누르면 현재 티켓 (2️⃣·3️⃣ 탭에 반영)
                c_id, c_pick = st.columns([3, 1.4], vertical_alignment="center")
                c_id.markdown(f"**{t.ticket_id}** {badge(t.priority)}", unsafe_allow_html=True)
                c_pick.button("현재" if selected else "선택", key=f"sbpick_{t.ticket_id}", disabled=selected,
                              on_click=pick_ticket, args=(t.ticket_id,))
                with st.container(key=f"sbprob_{t.ticket_id}"):   # 문제점 최대 2줄, 넘치면 …
                    st.markdown(t.problem)
                st.caption(":red[기한 지남]" if left < 0 else f"~{due:%m/%d} · D-{left}")

TAB_LABELS = ["1️⃣ VOC 대시보드", "2️⃣ 의심 위치", "3️⃣ 수정 라인"]
# 수정 라인 탐색이 끝나면 3️⃣ 탭으로 — st.tabs(key=…)의 선택 탭은 위젯을 그리기 전에만 바꿀 수 있어서
# 탐색 직후에는 표시만 남기고 st.rerun(), 다음 실행에서 탭을 그리기 전에 바꾼다
if ss.pop("goto_lines_tab", False):
    ss["active_tab"] = TAB_LABELS[2]
tab1, tab2, tab3 = st.tabs(TAB_LABELS, key="active_tab", on_change="rerun")  # key로 선택 탭을 코드에서 바꾼다

# ════════════════════════════ 탭 1: VOC 대시보드 ════════════════════════════

with tab1:
    # 기간 방식: 최근 2주(기본, 오늘 포함) · 직접 선택 · 전체 기간(조회 전용)
    # 세 경우 모두 같은 줄 구성: 날짜 위젯은 항상 같은 자리(직접 선택일 때만 활성) → 방식을 바꿔도 화면이 움직이지 않는다
    # 왼쪽부터 붙인 한 줄 (화면이 넓어져도 같은 자리): [기간 방식] [날짜 칸 고정 폭 DATE_INPUT_WIDTH]
    period_row = st.container(horizontal=True, gap="small", vertical_alignment="bottom")
    mode = period_row.segmented_control("기간", PERIOD_MODES, key="period_mode", required=True,
                                        on_change=to_raw_view_if_period_changed)
    today = date.today()
    date_max = max(max_d, today)
    if mode == PERIOD_CUSTOM:
        picked = period_row.date_input("날짜", key="custom_dates", min_value=min_d, max_value=date_max,
                                       format="YYYY/MM/DD", width=config.DATE_INPUT_WIDTH, on_change=to_raw_view)
        if isinstance(picked, (tuple, list)) and len(picked) == 2:
            start, end = picked
            ss["last_custom"] = (start, end)
        else:  # 끝 날짜를 아직 안 골랐으면 직전 선택 유지
            start, end = ss.get("last_custom", (config.ANALYSIS_START, config.ANALYSIS_END))
    else:
        if mode == PERIOD_ALL:
            start, end = min_d, max_d
        else:  # 최근 2주: 데이터 마지막 날짜가 아니라 오늘 기준
            start, end = today - timedelta(days=config.RECENT_PERIOD_DAYS), today
        period_row.date_input("날짜", value=(start, end), min_value=min(min_d, start), max_value=max(date_max, end),
                              format="YYYY/MM/DD", width=config.DATE_INPUT_WIDTH, disabled=True,
                              key=f"fixed_dates_{mode}")
    ss["period"] = (start, end)
    key = (start, end)
    full_period = mode == PERIOD_ALL

    cur = data_utils.filter_period(df_all, start, end)
    ps, pe = data_utils.prev_period(start, end)
    n_for_cluster = len(cur) + (0 if full_period else len(data_utils.filter_period(df_all, ps, pe)))
    if full_period:
        block_reason = "전체 기간은 조회만 가능합니다. 분석하려면 기간을 줄여 주세요"
    elif cur.empty:
        block_reason = "선택한 기간에 VOC가 없습니다."
    elif n_for_cluster > config.CLUSTER_MAX_VOCS:
        block_reason = (f"분석은 최대 {config.CLUSTER_MAX_VOCS:,}건까지 가능합니다 (현재 {n_for_cluster:,}건). "
                        "기간을 줄여 주세요")
    else:
        block_reason = ""

    # 보기 전환 줄: 왼쪽 보기 전환 · 오른쪽 분석 버튼 (누르면 보기도 '유형 분석 결과'로 — 콜백은 위젯보다 먼저 실행)
    c_view, c_btn = st.columns([4, 1], vertical_alignment="center")
    run_clicked = c_btn.button("유형 분석", type="primary", width="stretch", disabled=bool(block_reason),
                               help=block_reason or None, on_click=lambda: ss.update(voc_view=VIEW_CLUSTER))
    view = c_view.segmented_control("보기", VIEW_OPTIONS, key="voc_view", required=True,
                                    label_visibility="collapsed")
    if block_reason and not cur.empty:
        st.caption(block_reason)

    if run_clicked and has_api_key():
        with st.spinner("VOC를 유형으로 묶는 중… (같은 기간은 캐시 사용)"):
            cached_llm()
            t0 = time.perf_counter()
            clusters, assignments = stage1.run_stage1(start, end, df_all)
            ss["stage1"][key] = {"clusters": clusters, "assignments": assignments,
                                 "secs": round(time.perf_counter() - t0, 1)}

    s1 = None if full_period else ss["stage1"].get(key)  # 분석 결과는 세션에 남아 보기를 바꿔도 LLM을 다시 부르지 않는다
    weekly = (end - start).days + 1 > config.WEEKLY_CHART_MIN_DAYS
    chart_title = "주별 VOC 건수" if weekly else "일별 VOC 건수"

    if cur.empty:
        st.info("선택한 기간에 VOC가 없습니다. 기간을 바꿔 보세요.")
    else:
        kpi = cached_kpis(start, end)
        period_txt = f"{start}~{end}"
        # 용도별 목록 (같은 목록을 같이 쓰지 않는다): period_vocs = 선택 기간 VOC 전체(거르지 않음) → 총 VOC ·
        # open_clusters(처리 전 + 진행 중) → VOC 유형·긴급 · completed_voc_ids → 처리 완료. 유형 표는 아래 table_clusters
        period_vocs = cur
        cards = stage1.summary_cards(period_vocs["voc_id"], s1["clusters"] if s1 else None, ss["tickets"], period_txt)
        # 카드 순서: 총 VOC · 처리 완료(총 VOC 바로 옆) · VOC 유형 · 긴급
        k1, k4, k2, k3 = st.columns(4)
        if full_period:  # 비교할 이전 기간이 없다
            k1.metric("총 VOC", f"{cards['total']:,}건", help=f"전체 기간 {start:%Y/%m/%d}~{end:%Y/%m/%d}")
        else:
            k1.metric("총 VOC", f"{cards['total']:,}건",
                      delta=f"{kpi['change_pct']:+.1f}%" if kpi["change_pct"] is not None else None,
                      delta_color="inverse",
                      help=f"이전 기간({ps:%m/%d}~{pe:%m/%d}) 대비 · 이전 기간 {kpi['prev_total']:,}건 · "
                           "처리 전 + 진행 중 + 처리 완료 모두 포함")
        if cards["types"] is not None:
            k2.metric("VOC 유형", f"{cards['types']}개", help="처리 완료되지 않은 유형 수 (처리 전 + 진행 중)")
            k3.metric("긴급 유형", f"{cards['urgent_types']}개",
                      help="처리 완료되지 않은 유형(처리 전 + 진행 중) 중 우선순위가 '긴급'인 유형 수")
        else:
            k2.metric("VOC 유형", "–", help="유형 분석 후 표시")
            k3.metric("긴급 유형", "–", help="유형 분석 후 표시")
        # 처리 완료: 티켓 voc_ids로 계산하므로 분석 전에도 표시. 처리율은 증감이 아니라서 화살표·색 없음
        k4.metric("처리 완료", f"{cards['done_count']:,}건", delta=f"처리율 {cards['done_rate']:.1f}%",
                  delta_color="off", delta_arrow="off",
                  help=f"처리 완료한 티켓에 묶인 VOC 건수 ({cards['done_types']}개 유형) · 처리율 = 처리 완료 ÷ 총 VOC")

        if view == VIEW_RAW:
            daily_bar_chart(data_utils.daily_counts(cur, "channel", weekly=weekly), chart_title, weekly=weekly)
            st.subheader("VOC 원문")
            raw = data_utils.voc_table(cur)  # 최신순
            if len(raw) > config.TABLE_MAX_ROWS:  # 대용량 대비: 최신 N행만 표에 넘긴다
                st.caption(f"최신 {config.TABLE_MAX_ROWS:,}건 표시 (전체 {len(raw):,}건)")
                raw = raw.head(config.TABLE_MAX_ROWS)
            show_table(raw, key="tbl_raw",
                       height=(config.RAW_TABLE_ROWS + 1) * config.TABLE_ROW_PX + 3)  # 헤더 + 10줄
        elif full_period:
            st.info("전체 기간은 조회만 가능합니다. 분석하려면 기간을 줄여 주세요")
        elif not s1:
            st.info("유형 분석을 누르면 이 기간의 VOC 유형과 우선순위가 여기에 나옵니다.")
        else:
            all_clusters = sorted(s1["clusters"], key=lambda c: -c.priority_score)  # 처리 시급도 높은 순 (의미 공간용 전체)

            # 1. 일별 VOC 건수 차트 (유형별 누적)
            cur = cur.assign(cluster=cur["voc_id"].map(s1["assignments"]).fillna(config.OTHER_LABEL))
            # 차트에서는 LLM 미분류('기타')를 '미분류'로 따로 두고, 상위 N개 밖 유형만 '기타'로 묶는다
            unassigned = cur["cluster"].replace(config.OTHER_LABEL, config.UNASSIGNED_LABEL)
            chart_df = cur.assign(cluster=data_utils.top_groups(unassigned, config.CHART_TOP_TYPES, config.OTHER_LABEL,
                                                                keep=(config.UNASSIGNED_LABEL,)))
            daily_bar_chart(data_utils.daily_counts(chart_df, "cluster", weekly=weekly), chart_title, weekly=weekly)
            st.caption(f"· 기타 = 상위 {config.CHART_TOP_TYPES}개 밖 유형의 합계 · "
                       f"{config.UNASSIGNED_LABEL} = 어느 유형에도 맞지 않은 VOC (티켓 대상 아님)")

            # 2. VOC 유형별 처리 우선순위 표 — 제목 줄 오른쪽 [티켓 발행] (버튼은 표의 체크 결과를 본 뒤 채운다)
            c_title, c_issue = st.columns([4, 1], vertical_alignment="center")
            c_title.subheader("VOC 유형별 처리 우선순위")
            st.caption("체크한 유형만 티켓 발행")
            # 처리 전 유형만 — 진행 중(티켓 카드에만 보임) · 처리 완료 유형은 표에서 빠진다
            clusters = stage1.table_clusters(s1["clusters"], ss["tickets"], period_txt)
            # 열: 우선순위·처리 시급도를 건수보다 앞에, 그 뒤는 시급도 계산식 순서(건수 → 증가율(추세) → 심각도)
            columns = ["발행", "우선순위", "유형", "처리 시급도", "이번 기간", "이전 기간", "추세", "심각도", "카테고리 · 제품"]
            table = pd.DataFrame([{
                "발행": False,
                "우선순위": f"{config.PRIORITY_BADGE[c.priority]} {c.priority}",
                "유형": c.label,
                "처리 시급도": round(c.priority_score * 100),  # 내부 0~1 → 화면 0~100
                "이번 기간": c.count,
                "이전 기간": c.prev_count,
                "추세": c.trend,
                "심각도": c.severity,
                "카테고리 · 제품": f"{c.category} · {c.product}",
            } for c in clusters], columns=columns)
            if clusters:
                edited = st.data_editor(
                    table, hide_index=True, width="stretch", key=f"editor_{start}_{end}_{ss['editor_nonce']}",
                    height=(config.TABLE_VISIBLE_ROWS + 1) * config.TABLE_ROW_PX + 3,
                    column_config={
                        "발행": st.column_config.CheckboxColumn("발행", help="체크한 유형만 티켓 발행"),
                        "처리 시급도": st.column_config.NumberColumn(help="0~100, 높을수록 먼저 처리"),
                        "이번 기간": st.column_config.NumberColumn(format="%d건"),
                        "이전 기간": st.column_config.NumberColumn(format="%d건", help=f"{ps:%m/%d}~{pe:%m/%d}"),
                    },
                    disabled=[c for c in table.columns if c != "발행"],
                )
                to_issue = [c for c, flag in zip(clusters, edited["발행"]) if flag]
            else:
                st.caption("이 기간의 유형은 모두 티켓으로 발행되었거나 처리 완료되었습니다.")
                to_issue = []
            issue_clicked = c_issue.button("티켓 발행", type="primary", width="stretch", disabled=not to_issue)

            if issue_clicked and has_api_key():
                # 현재 개발자(me)를 담당자로 바로 발행 → outputs/tickets.json. 그사이 다른 개발자가 발행한 유형은 건너뛴다
                tickets, next_no = stage1.load_tickets()
                # 처리 전이 아닌 유형은 건너뛴다 ('취소' 티켓만 걸린 유형은 처리 전 → 다시 발행 가능)
                taken = {cid for cid, state in stage1.cluster_states(tickets, period_txt).items()
                         if state != config.CLUSTER_STATE_NEW}
                chosen = sorted((c for c in to_issue if c.cluster_id not in taken),
                                key=lambda c: (config.PRIORITY_ORDER[c.priority], -c.priority_score))
                with st.spinner(f"티켓 {len(chosen)}건 작성 중…"):
                    for c in chosen:
                        samples = stage1.sample_texts(df_all, c, config.TICKET_SAMPLE_VOCS)
                        ticket = stage1.ticketize(c, samples, period_txt, next_no, assignee=me)
                        tickets[ticket.ticket_id] = ticket
                        next_no += 1
                        stage1.save_tickets(tickets, next_no)
                        ss["current_ticket"] = ss["current_ticket"] or ticket.ticket_id
                ss["editor_nonce"] += 1  # 표를 체크 없이 새로 그린다 (발행된 유형은 표에서 빠진다)
                st.rerun()
            thresholds = " · ".join(f"{round(t * 100)} 이상 {label}" for t, label in config.PRIORITY_THRESHOLDS[:-1])
            st.caption(
                f"· 이전 기간 = 선택한 기간 바로 앞의 같은 길이 기간 ({ps:%m/%d}~{pe:%m/%d})  \n"
                f"· 처리 시급도 = 건수 {config.W_COUNT:.0%} + 증가율 {config.W_SURGE:.0%} + 심각도 {config.W_SEVERITY:.0%}를 합친 점수 "
                f"({thresholds} · 그 아래 {config.PRIORITY_THRESHOLDS[-1][1]})  \n"
                "· 심각도: 5 신체 위해(발열·화상) · 4 기능 불가 · 3 성능 저하(배터리 등) · 2 부정확·불편 · 1 단순 불만"
            )

            # 3. 발행된 티켓 카드
            render_ticket_cards()
            # 4. 대표 VOC 보기 (내 티켓의 유형만)
            render_representative_vocs()
            # 5. VOC 유사도 (의미 공간 · LLM 분류 vs 임베딩 분류) — 색·범례는 발행 여부와 상관없이 전체 유형
            render_voc_map(cur, all_clusters)

# ════════════════════════════ 탭 2: 의심 위치 ════════════════════════════

NO_MY_TICKET = "내가 맡은 티켓이 없습니다. VOC 대시보드에서 유형을 골라 티켓을 발행하세요"


def get_result(ticket) -> dict | None:
    """코드 찾기 결과: 세션 → 없으면 outputs/stage2/{ticket_id}.json (개발자를 바꿨다 돌아와도 다시 실행 안 함)."""
    result = ss["stage2"].get(ticket.ticket_id)
    if result and not stage2_result_ok(result):  # 예전 모델로 만든 결과(예: repo 없는 Impact)는 쓰지 않는다
        del ss["stage2"][ticket.ticket_id]
        result = None
    if result is None:
        result = stage2.load_result(ticket)
        if result:
            ss["stage2"][ticket.ticket_id] = result
    return result


def folder_option(f: dict) -> str:
    return f"{f['repo']} / {f['path']}"


def render_folder_pick(sel: str, wait: dict) -> None:
    """폴더 확인(interrupt 대기): '수정 라인을 찾을 폴더'를 고르고 [수정 라인 찾기]로 그래프를 재개한다.
    끝나서 검증된 수정 라인이 있으면 3️⃣ 탭으로 이동, 실패하거나 0건이면 2️⃣ 탭에 남아 안내.
    기본 선택 = 1위 레포지토리에서 감점되지 않은 의심도 상위 2개 폴더, 최대 MAX_SELECTED_FOLDERS개."""
    st.markdown("#### 수정 라인을 찾을 폴더")
    graph = cached_graph()
    if not stage2.is_waiting(graph, wait["thread_id"]):  # 서버 재시작 등으로 체크포인터에 그 실행이 없다
        st.warning("이전 코드 찾기 상태가 없어졌습니다. 다시 [코드 찾기]를 눌러 주세요")
        return
    payload = wait["payload"]
    folders = payload["folders"]
    by_label = {folder_option(f): f for f in folders}
    top = payload["repos"][0]["name"] if payload["repos"] else None
    default = [folder_option(f) for f in folders
               if f["repo"] == top and not f["penalized"]][:config.TOP_FOLDERS_FOR_LINES]
    picked = st.multiselect("수정 라인을 찾을 폴더", list(by_label), default=default,
                            max_selections=config.MAX_SELECTED_FOLDERS, key=f"pick_folders_{wait['thread_id']}",
                            label_visibility="collapsed")
    st.caption(f"고른 폴더 안의 파일에서만 수정 라인을 찾습니다 · 최대 {config.MAX_SELECTED_FOLDERS}개 · "
               "기본은 1위 레포지토리에서 감점되지 않은 의심도 상위 2개")
    if st.button("수정 라인 찾기", type="primary", disabled=not picked,
                 key=f"resume_{wait['thread_id']}") and has_api_key():
        chosen = [{"repo": by_label[o]["repo"], "path": by_label[o]["path"]} for o in picked]
        found = 0
        try:
            with st.spinner("수정 라인을 찾는 중…"):
                for event in stage2.resume_stage2(graph, wait["thread_id"], chosen):
                    if event[0] == "done":  # 끝까지 완료됐을 때만 저장
                        ss["stage2"][sel] = event[1]
                        stage2.save_result(sel, event[1])
                        ss["stage2_wait"].pop(sel, None)
                        found = sum(r.verified for r in event[1].get("line_recs", []))
        except Exception as e:  # noqa: BLE001 — 오류 화면 대신 2️⃣ 탭에 안내
            ss["stage2_wait"].pop(sel, None)
            ss["stage2_notice"][sel] = f"수정 라인 찾기에 실패했습니다. 다시 [코드 찾기]를 눌러 주세요 ({str(e)[:80]})"
        else:
            if found:
                ss["goto_lines_tab"] = True  # 다음 실행에서 3️⃣ 탭을 연다
            else:
                ss["stage2_notice"][sel] = ("고른 폴더에서 검증된 수정 라인을 찾지 못했습니다. "
                                            "[코드 찾기]를 다시 눌러 다른 폴더를 골라 보세요")
        st.rerun()  # 폴더 카드에 원인 후보·함께 확인 색을 칠해 다시 그린다


with tab2:
    if not my_tickets:  # 티켓 선택 목록에는 현재 개발자의 티켓만
        st.info(NO_MY_TICKET)
    else:
        ids = [t.ticket_id for t in my_tickets]
        cur_id = ss["current_ticket"] if ss["current_ticket"] in ids else ids[0]
        c_sel, c_run = st.columns([4, 1.4], vertical_alignment="bottom")
        labels = {f"{i} [{ss['tickets'][i].priority}] {ss['tickets'][i].problem}": i for i in ids}
        picked_label = c_sel.selectbox("티켓", list(labels), index=ids.index(cur_id))
        sel = labels[picked_label]
        ss["current_ticket"] = sel
        ticket = ss["tickets"][sel]

        if c_run.button("코드 찾기", type="primary", width="stretch") and has_api_key():  # 내 티켓에서만
            cached_vectorstore()
            graph = cached_graph()  # @st.cache_resource — 체크포인터가 rerun 사이에 유지된다
            # 다시 누르면 실행 번호를 올려 새 thread로 처음부터 (이전 대기 상태와 섞이지 않게)
            run_no = ss["stage2_run_no"].get(sel, 0) + 1
            ss["stage2_run_no"][sel] = run_no
            thread_id = f"{sel}:{run_no}"
            ss["stage2_wait"].pop(sel, None)
            with st.spinner("코드 위치를 찾는 중…"):  # ④ 레포지토리 라우팅 → ⑤ 폴더 의심도까지 돌고 폴더 확인에서 멈춘다
                for event in stage2.stream_stage2(graph, ticket, thread_id):
                    if event[0] == "interrupt":
                        ss["stage2_wait"][sel] = {"thread_id": thread_id, "payload": event[1],
                                                  "state": stage2.waiting_state(graph, thread_id)}
                    elif event[0] == "done":  # 고를 폴더가 없어 멈추지 않고 끝난 경우
                        ss["stage2"][sel] = event[1]
                        stage2.save_result(sel, event[1])

        notice = ss["stage2_notice"].pop(sel, None)  # 수정 라인 찾기가 실패했거나 0건이면 3️⃣로 가지 않고 여기서 안내
        if notice:
            st.warning(notice)
        wait = ss["stage2_wait"].get(sel)  # 폴더 확인 대기 중이면 그 시점 state(수정 라인·영향 범위 없음)를 보여 준다
        result = wait["state"] if wait else get_result(ticket)
        if not result:
            st.caption("코드 찾기를 누르면 ④ 레포지토리 라우팅 → ⑤ 폴더 의심도까지 실행하고 멈춥니다. "
                       "수정 라인을 찾을 폴더를 고르면 ⑥ 라인 탐색 → ⑦ 영향 범위를 찾습니다.")
        else:
            cands = result.get("repo_candidates", [])
            sources = result.get("repo_sources", {})
            st.markdown("#### 레포지토리 후보")
            if not cands:
                st.warning("점수 0.1 이상인 레포지토리 후보가 없습니다.")
            else:
                # 후보마다 한 블록: 1줄 이름(굵게) + 점수 막대 / 2줄 이유 전체 (잘림 없음)
                for c in cands:  # 후보마다 테두리 상자: 이름 · 막대 / 이유(본문 글씨) / 코드 출처
                    with st.container(border=True):
                        c_name, c_bar = st.columns([1, 3], vertical_alignment="center")
                        c_name.markdown(f"**{c.name}**")
                        c_bar.progress(min(max(c.score, 0.0), 1.0), text=f"{c.score:.2f}")
                        st.markdown(c.reason)
                        if c.name in sources:
                            st.caption(f"코드 출처: {sources[c.name]}")
                # 어느 레포지토리의 폴더를 볼지 (이름만, 기본 1위)
                names = [c.name for c in cands]
                picked = st.segmented_control("폴더를 볼 레포지토리", names, default=names[0], required=True,
                                              label_visibility="collapsed",
                                              key=f"repo_pick_{sel}")
                repo_name = picked if picked in names else names[0]

                st.markdown("#### 폴더 후보")
                render_folder_cards(stage2.build_folder_cards(result, repo_name), repo_name,
                                    stage2.repo_branch(repo_name), ticket.cluster_label)

                # 폴더 점수 구성 — 키워드 vs 함수 의미(LLM 요약) vs 코드 원문 임베딩 비교
                folders = [f for f in result.get("folder_scores", []) if f.repo == repo_name]
                if folders:
                    with st.expander("점수 구성 · 키워드 vs 함수 의미 비교"):
                        st.caption(f"점수 = ({config.W_KEYWORD}×키워드 + {config.W_SEMANTIC}×함수 의미 + "
                                   f"{config.W_RECENT}×최근 변경) × 역할보정. 함수 의미 = LLM이 함수 동작을 요약한 문장과 "
                                   "티켓 원인 문장의 임베딩 유사도. 코드 원문 = 함수 코드를 그대로 임베딩한 유사도(비교용, 점수에 안 들어감).")
                        show_table(pd.DataFrame([{
                            "폴더": f.path, "점수": f.score, "키워드": f.keyword, "함수 의미": f.semantic,
                            "코드 원문(비교)": f.semantic_code, "최근 변경": f.recent,
                            "가장 비슷한 함수": f.top_unit, "함수 동작 요약 (LLM)": f.top_summary,
                        } for f in folders]), key=f"folder_breakdown_{sel}_{repo_name}")

            if wait:  # 폴더 확인 — 사람이 고른 폴더에서만 수정 라인을 찾는다
                render_folder_pick(sel, wait)

# ════════════════════════════ 탭 3: 수정 라인 ════════════════════════════

def code_windows(n_lines: int, ranges: list[tuple[int, int]]) -> list[tuple[int, int, list[int]]]:
    """추천 범위 앞뒤 CODE_CONTEXT_LINES줄 창. 겹치면 합친다. (창 시작, 창 끝, 강조 절대 줄)"""
    pad = config.CODE_CONTEXT_LINES
    wins: list[list] = []
    for s, e in sorted(ranges):
        ws, we = max(1, s - pad), min(n_lines, e + pad)
        if wins and ws <= wins[-1][1] + 1:
            wins[-1][1] = max(wins[-1][1], we)
            wins[-1][2] += list(range(s, e + 1))
        else:
            wins.append([ws, we, list(range(s, e + 1))])
    return [(a, b, h) for a, b, h in wins]


def render_code(ref, ranges: list[tuple[int, int]]) -> None:
    """코드 뷰: 파일 전체를 읽지 않고 FileRef + read_lines로 추천 범위 앞뒤 CODE_CONTEXT_LINES줄 창만 읽는다."""
    for ws, we, hl in code_windows(ref.n_lines, ranges):
        chunk = stage2.read_lines(ref, ws, we, numbered=False) + "\n"
        fmt = HtmlFormatter(linenos="table", linenostart=ws, hl_lines=[n - ws + 1 for n in hl],
                            noclasses=True, hl_color="#FFF1A8", style="default")
        st.html(f'<div class="vocai-code" style="font-size:13px;overflow-x:auto;border:1px solid #E3E6EF;border-radius:8px;'
                f'padding:6px;background:#FFFFFF;margin-bottom:8px">{highlight(chunk, PythonLexer(), fmt)}</div>')


with tab3:
    sel = ss["current_ticket"]  # 사이드바 '내 티켓' · 2️⃣ 탭 선택과 같은 티켓 (항상 현재 개발자의 티켓)
    result = get_result(ss["tickets"][sel]) if sel in ss["tickets"] else None
    if not my_tickets:
        st.info(NO_MY_TICKET)
    elif sel in ss["stage2_wait"]:  # 폴더 확인 전 (코드 찾기가 폴더 후보에서 멈춰 있음)
        st.info("의심 위치 탭에서 폴더를 고르면 수정 라인이 나옵니다")
    elif not result:
        st.info("2️⃣ 탭에서 코드 찾기를 실행하면 검증된 수정 라인이 여기에 나옵니다.")
    else:
        ticket = ss["tickets"][sel]
        st.caption(f"{ticket.ticket_id} · {ticket.problem}")
        recs = [r for r in result.get("line_recs", []) if r.verified]  # 검증된 라인만
        if not recs:
            st.warning("검증을 통과한 라인이 없습니다. 다시 실행해 보세요.")
        # 위에서 아래로, 모두 가로 폭 전체: 수정 위치 요약 표 → 파일별 상세(expander) → 함께 확인할 곳 → 다운로드
        recs = sorted(recs, key=lambda r: -r.confidence)
        if recs:
            st.markdown("#### 수정 위치")
            show_table(pd.DataFrame([{"레포지토리": r.repo, "파일": r.path, "수정 라인": stage2.fmt_line_ranges(r.ranges),
                                      "기준 커밋": r.commit_sha} for r in recs]), key=f"tbl_lines_{sel}")
            st.markdown("#### 파일별 상세")
            for idx, r in enumerate(recs):
                with st.expander(f"{r.path} · {stage2.fmt_line_ranges(r.ranges)}", expanded=(idx == 0)):
                    render_code(stage2.file_ref(r.repo, r.path), r.ranges)
                    with st.container(key=f"card_evidence_{idx}"):  # 흰 배경 카드 안 근거 표
                        st.table(pd.DataFrame(stage2.evidence_rows(ticket, r), columns=["항목", "내용"]).set_index("항목"))
                    if len(r.last_authors) > 1:  # 근거 표의 마지막 수정자는 한 줄 요약 — 범위별은 접어 둔다
                        with st.expander(f"마지막 수정자 상세 ({len(r.last_authors)}개 범위)", expanded=False):
                            with st.container(key=f"lastauth_{idx}"):  # 같은 사람·날짜끼리, 최근 날짜 순
                                st.markdown("\n".join(f"- {md_escape(g)}" for g in stage2.last_author_groups(r)))
            st.markdown("#### 함께 확인할 곳")
            impacts = result.get("impacts", [])  # 티켓 전체 기준 (파일별로 나누지 않음)
            if impacts:
                show_table(pd.DataFrame([{"레포지토리": i.repo, "파일": i.path, "라인": i.line, "심볼": i.symbol,
                                          "관계": i.relation} for i in impacts]), key=f"tbl_impacts_{sel}")
            else:
                st.caption("같은 레포지토리의 다른 파일에서 참조하는 곳이 없습니다.")
        actions = st.container(horizontal=True, gap="small")  # [티켓 Markdown 다운로드] [처리 완료] 나란히
        actions.download_button("티켓 Markdown 다운로드", stage2.ticket_to_markdown(ticket, result),
                                file_name=f"{ticket.ticket_id}.md", mime="text/markdown")
        if actions.button("처리 완료", key=f"done_tab3_{ticket.ticket_id}"):
            confirm_complete(ticket.ticket_id)
