"""VOCAI — 고객의 목소리 → 고칠 곳 (Streamlit 화면). CSS 주입은 이 파일 한 곳에서만 한다."""

import html
import os
import time
from datetime import date

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
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

STEP_LABELS = ["① 수집", "② 클러스터", "③ 티켓", "④ 레포 라우팅", "⑤ 폴더 의심도", "⑥ 라인 탐색·검증·커밋 근거", "⑦ 영향 범위"]
STEP_NODES = {
    "④ 레포 라우팅": ["route_repos"],
    "⑤ 폴더 의심도": ["score_folders"],
    "⑥ 라인 탐색·검증·커밋 근거": ["search_lines", "verify_lines", "add_commit_evidence"],
    "⑦ 영향 범위": ["analyze_impact"],
}
VIEW_RAW = "📄 원본 VOC"
VIEW_CLUSTER = "🧩 클러스터 분석 결과"
VIEW_OPTIONS = [VIEW_RAW, VIEW_CLUSTER]

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
    """(임베딩, t-SNE 3D 좌표)."""
    vectors = cached_voc_vectors(voc_ids)
    return vectors, stage1.project_3d(vectors)


@st.cache_data(show_spinner=False)
def cached_embed_clusters(voc_ids: tuple[str, ...]):
    """임베딩 클러스터링 결과 (VOC별 이름, {k: 실루엣})."""
    vectors = cached_voc_vectors(voc_ids)
    ids, sil_by_k = stage1.cluster_embeddings(vectors)
    texts = cached_vocs().set_index("voc_id").loc[list(voc_ids), "text"].tolist()
    return stage1.embedding_cluster_names(vectors, ids, texts), sil_by_k


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
ss.setdefault("tickets", {})         # {ticket_id: Ticket}
ss.setdefault("ticket_secs", None)
ss.setdefault("next_ticket_no", config.FIRST_TICKET_NO)
ss.setdefault("stage2", {})          # {ticket_id: 최종 state}
ss.setdefault("current_ticket", None)
ss.setdefault("voc_view", VIEW_RAW)  # 1️⃣ 탭 보기 전환 (segmented_control key)


def badge(priority: str) -> str:
    """고정 라벨만 들어가는 HTML 배지 (LLM 텍스트 아님)."""
    color = config.PRIORITY_COLOR.get(priority, "#9AA0A6")
    return f'<span class="vocai-badge" style="background:{color}">{priority}</span>'


def has_api_key() -> bool:
    if os.getenv("OPENAI_API_KEY"):
        return True
    st.error("`.env`에 OPENAI_API_KEY가 없습니다.")
    return False


def daily_bar_chart(daily: pd.DataFrame, title: str) -> None:
    """일별 누적 막대. 범례는 차트 위 가로, 날짜는 MM/DD, 아래 여백을 줘 눈금이 잘리지 않게 한다."""
    order = daily.groupby("group")["count"].sum().sort_values(ascending=False).index.tolist()
    if config.OTHER_LABEL in order:  # '기타'는 항상 마지막 · 회색
        order = [g for g in order if g != config.OTHER_LABEL] + [config.OTHER_LABEL]
    palette = px.colors.qualitative.Bold
    colors = {g: palette[i % len(palette)] for i, g in enumerate(order)}
    colors[config.OTHER_LABEL] = config.OTHER_COLOR
    with st.container(key="card_chart"):
        st.markdown(f"**{title}**")
        fig = px.bar(daily, x="date", y="count", color="group", labels={"count": "건수", "group": ""},
                     category_orders={"group": order}, color_discrete_map=colors)
        fig.update_layout(barmode="stack", height=380, margin=dict(l=10, r=10, t=40, b=90),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title=None),
                          plot_bgcolor="#FFFFFF")
        # x축 제목 "날짜"는 눈금(10/03) 아래에 간격(standoff)을 두고, 범례는 차트 위라 겹치지 않는다
        fig.update_xaxes(title=dict(text="날짜", standoff=20), tickformat="%m/%d", dtick=86_400_000, tickangle=0)
        st.plotly_chart(fig, width="stretch")


def render_ticket_table() -> None:
    """발행된 티켓: 표 한 개(우선순위 순, 5줄 고정 + 스크롤) + 티켓 상세 접기. 티켓이 없으면 숨긴다."""
    tickets = stage1.sort_tickets(ss["tickets"].values())
    if not tickets:
        return
    st.subheader("발행된 티켓")
    st.dataframe(pd.DataFrame([stage1.ticket_summary_row(t) for t in tickets]), hide_index=True, width="stretch",
                 height=(config.TABLE_VISIBLE_ROWS + 1) * config.TABLE_ROW_PX + 3,
                 column_config={"건수": st.column_config.NumberColumn(format="%d건"),
                                "문제점": st.column_config.TextColumn(width="large")})
    with st.expander("티켓 상세"):
        for t in tickets:
            st.markdown(f"**{t.ticket_id}** · {t.problem}")
            st.markdown(f"발생 조건: {', '.join(t.conditions)}  \n기술 키워드: {', '.join(t.tech_keywords)}")
            for v in t.sample_vocs:
                st.text(f"대표 VOC · {v}")
            st.divider()


def hover_text(text: str, width: int = 28) -> str:
    """Plotly 호버용: HTML 이스케이프 후 width 글자마다 줄바꿈."""
    text = html.escape(text)
    return "<br>".join(text[i:i + width] for i in range(0, len(text), width))


def render_voc_map(cur: pd.DataFrame, clusters) -> None:
    """VOC 임베딩 3D 지도 + 선택한 VOC의 유사 VOC. 색은 LLM 또는 임베딩 클러스터, 위치는 임베딩."""
    st.subheader("🧭 VOC 유사도 지도")
    st.caption(f"위치는 VOC 문장 임베딩({config.VOC_EMBEDDING_MODEL})을 t-SNE로 3D 투영한 것입니다. "
               "가까울수록 고객 표현이 비슷합니다. 색은 LLM 분류 또는 임베딩만으로 묶은 분류(KMeans)입니다.")
    c_on, c_color = st.columns([1, 3], vertical_alignment="center")
    if not c_on.toggle("지도 보기", key="voc_map_on"):
        return
    color_by = c_color.radio("색 기준", ["LLM 분류", "임베딩 분류"], horizontal=True,
                             key="voc_map_color", label_visibility="collapsed")
    if not has_api_key():
        return

    cur = cur.sort_values("voc_id").reset_index(drop=True)
    voc_ids = tuple(cur["voc_id"])
    with st.spinner("VOC 임베딩 · t-SNE 투영 · 임베딩 클러스터링 중…"):
        vectors, coords = cached_voc_map(voc_ids)
        emb_labels, sil_by_k = cached_embed_clusters(voc_ids)
    cur = cur.assign(x=coords[:, 0], y=coords[:, 1], z=coords[:, 2], emb=emb_labels)
    labels = cur["cluster"].tolist()

    palette = px.colors.qualitative.Bold
    if color_by == "LLM 분류":
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
            hovertext=[f"<b>{v}</b><br>LLM: {html.escape(a)}<br>임베딩: {html.escape(b)}<br>{hover_text(t)}"
                       for v, a, b, t in zip(part["voc_id"], part["cluster"], part["emb"], part["text"])],
            hoverinfo="text",
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
        hovertext=[f"<b>{v}</b> · 유사도 {s:.3f}<br>{hover_text(t)}"
                   for v, t, (_, s) in zip(nb["voc_id"], nb["text"], neighbors)],
        hoverinfo="text",
    ))
    fig.add_trace(go.Scatter3d(
        x=[p["x"]], y=[p["y"]], z=[p["z"]], mode="markers", name="기준 VOC",
        marker=dict(size=config.MAP_HIGHLIGHT_SIZE, symbol="diamond", color="#202124", line=dict(color="#FFFFFF", width=1)),
        hovertext=[f"<b>{p['voc_id']}</b> (기준)<br>{hover_text(p['text'])}"], hoverinfo="text",
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
    st.dataframe(pd.DataFrame([{
        "순위": r, "VOC": cur.at[j, "voc_id"], "유사도": s, "LLM 클러스터": labels[j],
        "같은 LLM 클러스터": "✓" if labels[j] == labels[pick] else "⚠ 다름",
        "임베딩 클러스터": emb_labels[j], "원문": cur.at[j, "text"],
    } for r, (j, s) in enumerate(neighbors, 1)]), hide_index=True, width="stretch",
        column_config={"유사도": st.column_config.ProgressColumn(format="%.3f", min_value=0.0, max_value=1.0)})

    odd = stage1.label_disagreements(vectors, labels, config.DISAGREE_K)
    with st.expander(f"임베딩 이웃과 LLM 라벨이 다른 VOC {len(odd)}건 (경계·오분류 후보)"):
        if odd:
            st.dataframe(pd.DataFrame([{
                "VOC": cur.at[o["idx"], "voc_id"], "LLM 라벨": o["label"],
                "이웃 다수 라벨": f"{o['neighbor_label']} ({o['agree']}/{config.DISAGREE_K})",
                "원문": cur.at[o["idx"], "text"],
            } for o in odd]), hide_index=True, width="stretch")
        else:
            st.caption("모든 VOC가 이웃 다수와 같은 클러스터입니다.")

    render_cluster_comparison(vectors, labels, emb_labels, sil_by_k)


METRIC_HELP = {
    "실루엣 ↑": "자기 묶음과는 가깝고 다른 묶음과는 먼 정도 (-1~1). 임베딩 분류는 이 값을 최대화하도록 k를 골랐으므로 유리하다.",
    "Davies-Bouldin ↓": "묶음끼리 겹치는 정도 (0~, 낮을수록 잘 분리).",
    "kNN 일관성 ↑": f"가장 비슷한 VOC {config.DISAGREE_K}개 중 같은 라벨 비율 평균 (0~1).",
    "기타 비율 ↓": "어느 묶음에도 못 들어가 티켓으로 넘어가지 못하는 VOC 비율.",
}


def render_cluster_comparison(vectors, llm_labels: list[str], emb_labels: list[str], sil_by_k: dict) -> None:
    """LLM 분류 vs 임베딩 분류: 내부 지표 · 두 분류 일치도 · k별 실루엣."""
    st.markdown("#### 📏 LLM 분류 vs 임베딩 분류")
    k = config.DISAGREE_K
    m_llm = stage1.clustering_metrics(vectors, llm_labels, k)
    m_emb = stage1.clustering_metrics(vectors, emb_labels, k)
    both = stage1.agreement_metrics(llm_labels, emb_labels)

    def fmt(v, digits=3):
        return "–" if v is None else f"{v:.{digits}f}"

    c1, c2, c3 = st.columns(3)
    c1.metric("클러스터 수 (LLM · 임베딩)", f"{m_llm['n_clusters']} · {m_emb['n_clusters']}")
    c2.metric("두 분류 일치 ARI", fmt(both["ari"]), help="라벨 이름과 무관하게 같은 VOC끼리 묶였는지 (1 = 완전 일치, 0 = 무작위 수준)")
    c3.metric("두 분류 일치 NMI", fmt(both["nmi"]), help="한 분류를 알면 다른 분류를 얼마나 알 수 있는지 (0~1)")

    st.dataframe(pd.DataFrame([
        {"지표": "실루엣 ↑", "LLM": fmt(m_llm["silhouette"]), "임베딩": fmt(m_emb["silhouette"])},
        {"지표": "Davies-Bouldin ↓", "LLM": fmt(m_llm["davies_bouldin"]), "임베딩": fmt(m_emb["davies_bouldin"])},
        {"지표": "kNN 일관성 ↑", "LLM": fmt(m_llm["knn_consistency"]), "임베딩": fmt(m_emb["knn_consistency"])},
        {"지표": "기타 비율 ↓", "LLM": f"{m_llm['other_ratio']:.1%}", "임베딩": f"{m_emb['other_ratio']:.1%}"},
    ]).assign(설명=lambda d: d["지표"].map(METRIC_HELP)), hide_index=True, width="stretch")

    if sil_by_k:
        with st.expander(f"임베딩 분류의 k 선택 — k = {config.MIN_CLUSTERS}~{config.MAX_CLUSTERS} 중 실루엣 최대"):
            sil = pd.DataFrame({"k": list(sil_by_k), "실루엣": list(sil_by_k.values())})
            fig_k = px.line(sil, x="k", y="실루엣", markers=True)
            best = sil.loc[sil["실루엣"].idxmax()]
            fig_k.add_scatter(x=[best["k"]], y=[best["실루엣"]], mode="markers", name=f"선택 k={int(best['k'])}",
                              marker=dict(size=12, color="#D93025"))
            fig_k.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="#FFFFFF")
            st.plotly_chart(fig_k, width="stretch")


# ════════════════════════════ 헤더 ════════════════════════════

df_all = cached_vocs()
min_d, max_d = df_all["date"].min(), df_all["date"].max()
period = ss.get("period", (config.ANALYSIS_START, config.ANALYSIS_END))

st.markdown('<div class="vocai-header"><span class="vocai-title">VOCAI</span></div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["1️⃣ VOC 대시보드", "2️⃣ 레포 지도", "3️⃣ 수정 라인"])

# ════════════════════════════ 탭 1: VOC 대시보드 ════════════════════════════

with tab1:
    c_date, _, c_btn = st.columns([2, 3, 1.2], vertical_alignment="bottom")
    picked = c_date.date_input("기간", value=(config.ANALYSIS_START, config.ANALYSIS_END),
                               min_value=min_d, max_value=max_d, format="YYYY-MM-DD")
    if isinstance(picked, (tuple, list)) and len(picked) == 2:
        start, end = picked
    else:
        start, end = period
    if (start, end) != period:
        ss["period"] = (start, end)
        st.rerun()
    key = (start, end)
    # 누르면 보기도 '클러스터 분석 결과'로 바꾼다 (콜백은 위젯이 그려지기 전에 실행됨)
    run_clicked = c_btn.button("🔍 클러스터 분석", type="primary", width="stretch",
                               on_click=lambda: ss.update(voc_view=VIEW_CLUSTER))

    view = st.segmented_control("보기", VIEW_OPTIONS, key="voc_view", required=True,
                                label_visibility="collapsed")

    if run_clicked and has_api_key():
        with st.spinner("VOC를 클러스터로 묶는 중… (같은 기간은 캐시 사용)"):
            cached_llm()
            t0 = time.perf_counter()
            clusters, assignments = stage1.run_stage1(start, end, df_all)
            ss["stage1"][key] = {"clusters": clusters, "assignments": assignments,
                                 "secs": round(time.perf_counter() - t0, 1)}

    s1 = ss["stage1"].get(key)  # 분석 결과는 세션에 남아 보기를 바꿔도 LLM을 다시 부르지 않는다
    kpi = cached_kpis(start, end)
    ps, pe = data_utils.prev_period(start, end)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("총 VOC", f"{kpi['total']:,}건",
              delta=f"{kpi['change_pct']:+.1f}%" if kpi["change_pct"] is not None else None,
              delta_color="inverse",
              help=f"이전 기간({ps:%m/%d}~{pe:%m/%d}) 대비 · 이전 기간 {kpi['prev_total']:,}건")
    if s1:
        urgent = [c for c in s1["clusters"] if c.priority == "긴급"]
        k2.metric("VOC 유형", f"{len(s1['clusters'])}개", help="클러스터 수")
        k3.metric("긴급 유형", f"{len(urgent)}개", help="우선순위가 '긴급'인 VOC 유형 수")
        k4.metric("긴급 건수", f"{sum(c.count for c in urgent):,}건", help="긴급 유형에 속한 VOC 건수 합계")
    else:
        for col, label in [(k2, "VOC 유형"), (k3, "긴급 유형"), (k4, "긴급 건수")]:
            col.metric(label, "–", help="클러스터 분석 후 표시")

    cur = data_utils.filter_period(df_all, start, end)

    if view == VIEW_RAW:
        daily_bar_chart(data_utils.daily_counts(cur, "channel"), "일별 VOC 건수 (채널별 누적)")
        st.subheader("VOC 원문")
        st.dataframe(data_utils.voc_table(cur), hide_index=True, width="stretch", height=320)
    elif not s1:
        st.info("🔍 클러스터 분석을 누르면 이 기간의 VOC 유형과 우선순위가 여기에 나옵니다.")
    else:
        render_ticket_table()
        # 제목 줄: 왼쪽 섹션 제목 · 오른쪽 끝 발행 버튼 (버튼은 표의 체크 결과를 본 뒤 아래에서 채운다)
        c_title, c_issue = st.columns([4, 1], vertical_alignment="center")
        c_title.subheader("클러스터")
        st.caption("체크한 유형만 티켓 발행 · 담당 파트로 전달")
        clusters = sorted(s1["clusters"], key=lambda c: -c.priority_score)  # 처리 시급도 높은 순
        table = pd.DataFrame([{
            "발행": False,
            "우선순위": f"{config.PRIORITY_BADGE[c.priority]} {c.priority}",
            "클러스터": c.label,
            "영역 · 제품": f"{c.category} · {c.product}",
            "이번 기간": c.count,
            "이전 기간": c.prev_count,
            "추세": c.trend,
            "처리 시급도": round(c.priority_score * 100),  # 내부 0~1 → 화면 0~100
            "심각도": c.severity,
        } for c in clusters])
        edited = st.data_editor(
            table, hide_index=True, width="stretch", key=f"editor_{start}_{end}",
            height=(config.TABLE_VISIBLE_ROWS + 1) * config.TABLE_ROW_PX + 3,
            column_config={
                "발행": st.column_config.CheckboxColumn("발행", help="체크한 유형만 티켓 발행"),
                "이번 기간": st.column_config.NumberColumn(format="%d건"),
                "이전 기간": st.column_config.NumberColumn(format="%d건", help=f"{ps:%m/%d}~{pe:%m/%d}"),
                "처리 시급도": st.column_config.NumberColumn(help="0~100, 높을수록 먼저 처리"),
            },
            disabled=[c for c in table.columns if c != "발행"],
        )
        any_checked = bool(edited["발행"].any())
        issue_clicked = c_issue.button("🎫 티켓 발행", type="primary", width="stretch", disabled=not any_checked)
        thresholds = " · ".join(f"{round(t * 100)} 이상 {label}" for t, label in config.PRIORITY_THRESHOLDS[:-1])
        st.caption(
            f"· 이전 기간 = 선택한 기간 바로 앞의 같은 길이 기간 ({ps:%m/%d}~{pe:%m/%d}). 날짜는 선택한 기간에 따라 자동으로 바뀝니다.  \n"
            f"· 처리 시급도 = 건수 {config.W_COUNT:.0%} + 증가율 {config.W_SURGE:.0%} + 심각도 {config.W_SEVERITY:.0%}를 합친 점수 "
            f"({thresholds} · 그 아래 {config.PRIORITY_THRESHOLDS[-1][1]})  \n"
            "· 심각도: 5 신체 위해(발열·화상) · 4 기능 불가 · 3 성능 저하(배터리 등) · 2 부정확·불편 · 1 단순 불만"
        )

        cur = cur.assign(cluster=cur["voc_id"].map(s1["assignments"]).fillna(config.OTHER_LABEL))
        chart_df = cur.assign(cluster=data_utils.top_groups(cur["cluster"], config.CHART_TOP_TYPES, config.OTHER_LABEL))
        daily_bar_chart(data_utils.daily_counts(chart_df, "cluster"),
                        f"일별 VOC 건수 (유형별 누적 · 상위 {config.CHART_TOP_TYPES}개 + 기타)")

        df_idx = df_all.set_index("voc_id")
        for c in clusters:
            with st.expander(f"{config.PRIORITY_BADGE[c.priority]} {c.label} — 대표 VOC 보기"):
                for vid in c.voc_ids[:config.SAMPLE_VOCS_PER_CLUSTER]:
                    st.text(f"{vid} · {df_idx.loc[vid, 'text']}")

        if issue_clicked:
            chosen = [c for c, flag in zip(clusters, edited["발행"]) if flag]
            issued = {(t.period, t.cluster_id) for t in ss["tickets"].values()}
            period_txt = f"{start}~{end}"
            chosen = [c for c in chosen if (period_txt, c.cluster_id) not in issued]
            chosen.sort(key=lambda c: (config.PRIORITY_ORDER[c.priority], -c.priority_score))
            if not chosen:
                st.warning("발행할 클러스터를 체크하세요. (이미 발행된 클러스터는 건너뜁니다)")
            elif has_api_key():
                t0 = time.perf_counter()
                with st.spinner(f"티켓 {len(chosen)}건 작성 중…"):
                    for c in chosen:
                        samples = stage1.sample_texts(df_all, c, config.TICKET_SAMPLE_VOCS)
                        ticket = stage1.ticketize(c, samples, period_txt, ss["next_ticket_no"])
                        ss["tickets"][ticket.ticket_id] = ticket
                        ss["next_ticket_no"] += 1
                        ss["current_ticket"] = ss["current_ticket"] or ticket.ticket_id
                ss["ticket_secs"] = round(time.perf_counter() - t0, 1)
                st.rerun()  # 발행된 티켓 표(클러스터 위)에 바로 보이도록

        render_voc_map(cur, clusters)

# ════════════════════════════ 탭 2: 레포 지도 ════════════════════════════

with tab2:
    if not ss["tickets"]:
        st.info("1️⃣ 탭에서 클러스터를 분석하고 티켓을 발행하면 여기서 코드를 좁혀 갑니다.")
    else:
        ids = list(ss["tickets"])
        cur_id = ss["current_ticket"] if ss["current_ticket"] in ids else ids[0]
        c_sel, c_run = st.columns([4, 1.4], vertical_alignment="bottom")
        labels = {f"{i} [{ss['tickets'][i].priority}] {ss['tickets'][i].problem}": i for i in ids}
        picked_label = c_sel.selectbox("티켓", list(labels), index=ids.index(cur_id))
        sel = labels[picked_label]
        ss["current_ticket"] = sel
        ticket = ss["tickets"][sel]

        if c_run.button("▶ 코드 좁혀 가기", type="primary", width="stretch") and has_api_key():
            cached_vectorstore()
            graph = cached_graph()
            with st.spinner("코드 위치를 찾는 중…"):  # 단계별 소요 시간은 state timings → 사이드바에 표시
                for event in stage2.stream_stage2(graph, ticket):
                    if event[0] == "done":
                        ss["stage2"][sel] = event[1]

        result = ss["stage2"].get(sel)
        if not result:
            st.caption("▶ 코드 좁혀 가기를 누르면 ④ 레포 라우팅 → ⑤ 폴더 의심도 → ⑥ 라인 탐색 → ⑦ 영향 범위를 차례로 실행합니다.")
        else:
            with st.container(key="card_repos"):
                st.markdown("#### 레포 후보")
                cands = result.get("repo_candidates", [])
                sources = result.get("repo_sources", {})
                if not cands:
                    st.warning("점수 0.1 이상인 레포 후보가 없습니다.")
                else:
                    # 후보마다 한 블록: 1줄 이름 · 점수 막대 · 점수 / 2줄 이유 전체 (표 셀에 넣으면 잘린다)
                    for c in cands:
                        c_name, c_bar, c_score = st.columns([2, 5, 1], vertical_alignment="center")
                        c_name.markdown(f"**{c.name}**")
                        c_bar.progress(min(max(c.score, 0.0), 1.0))
                        c_score.markdown(f"{c.score:.2f}")
                        st.caption(c.reason)
                        if c.name in sources:
                            st.caption(f"코드 출처: {sources[c.name]}")
            if cands:
                with st.container(key="card_folders"):
                    st.markdown("#### 폴더 의심도")
                    repo_sel = st.radio("레포", [c.name for c in cands], horizontal=True, label_visibility="collapsed")
                    folders = [f for f in result.get("folder_scores", []) if f.repo == repo_sel]
                    if folders:
                        fdf = pd.DataFrame([{"폴더": f.path, "점수": f.score, "근거": f.note,
                                             "구분": "감점" if f.role else "일반"} for f in folders])
                        fig = px.bar(fdf.iloc[::-1], x="점수", y="폴더", orientation="h", text="점수",
                                     color="구분", hover_data=["근거"],
                                     color_discrete_map={"일반": "#1428A0", "감점": "#9AA0A6"})
                        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
                        fig.update_layout(height=60 + 42 * len(fdf), margin=dict(l=10, r=40, t=10, b=10),
                                          xaxis=dict(range=[0, 1.1]), showlegend=False, plot_bgcolor="#FFFFFF")
                        st.plotly_chart(fig, width="stretch")
                        for f in folders:
                            st.caption(("⚠ " if f.role else "") + f"{f.path} — {f.note}")
                        with st.expander("점수 구성 · 키워드 vs 함수 의미 비교"):
                            st.caption(f"점수 = ({config.W_KEYWORD}×키워드 + {config.W_SEMANTIC}×함수 의미 + "
                                       f"{config.W_RECENT}×최근 변경) × 역할보정. 함수 의미 = LLM이 함수 동작을 요약한 문장과 "
                                       "티켓 원인 문장의 임베딩 유사도. 코드 원문 = 함수 코드를 그대로 임베딩한 유사도(비교용, 점수에 안 들어감).")
                            st.dataframe(pd.DataFrame([{
                                "폴더": f.path, "점수": f.score, "키워드": f.keyword, "함수 의미": f.semantic,
                                "코드 원문(비교)": f.semantic_code, "최근 변경": f.recent,
                                "가장 비슷한 함수": f.top_unit, "함수 동작 요약 (LLM)": f.top_summary,
                            } for f in folders]), hide_index=True, width="stretch")
                    else:
                        st.caption("폴더 점수가 없습니다.")

# ════════════════════════════ 탭 3: 수정 라인 ════════════════════════════

def code_windows(lines: list[str], ranges: list[tuple[int, int]]) -> list[tuple[int, int, list[int]]]:
    """추천 범위 앞뒤 CODE_CONTEXT_LINES줄 창. 겹치면 합친다. (창 시작, 창 끝, 강조 절대 줄)"""
    pad = config.CODE_CONTEXT_LINES
    wins: list[list] = []
    for s, e in sorted(ranges):
        ws, we = max(1, s - pad), min(len(lines), e + pad)
        if wins and ws <= wins[-1][1] + 1:
            wins[-1][1] = max(wins[-1][1], we)
            wins[-1][2] += list(range(s, e + 1))
        else:
            wins.append([ws, we, list(range(s, e + 1))])
    return [(a, b, h) for a, b, h in wins]


def render_code(lines: list[str], ranges: list[tuple[int, int]]) -> None:
    for ws, we, hl in code_windows(lines, ranges):
        chunk = "\n".join(lines[ws - 1:we]) + "\n"
        fmt = HtmlFormatter(linenos="table", linenostart=ws, hl_lines=[n - ws + 1 for n in hl],
                            noclasses=True, hl_color="#FFF1A8", style="default")
        st.html(f'<div style="font-size:13px;overflow-x:auto;border:1px solid #E3E6EF;border-radius:8px;'
                f'padding:6px;background:#FFFFFF;margin-bottom:8px">{highlight(chunk, PythonLexer(), fmt)}</div>')


with tab3:
    sel = ss["current_ticket"]
    result = ss["stage2"].get(sel) if sel else None
    if not result:
        st.info("2️⃣ 탭에서 ▶ 코드 좁혀 가기를 실행하면 검증된 수정 라인이 여기에 나옵니다.")
    else:
        ticket = ss["tickets"][sel]
        st.caption(f"{ticket.ticket_id} · {ticket.problem}")
        recs = [r for r in result.get("line_recs", []) if r.verified]  # 검증된 라인만
        if not recs:
            st.warning("검증을 통과한 라인이 없습니다. 다시 실행해 보세요.")
        # 위에서 아래로 한 줄씩, 모두 가로 폭 전체: 요약 → 코드 → 근거 (추천 파일마다 한 블록)
        for r in recs:
            st.markdown("#### 수정 위치")
            with st.container(border=True):
                st.markdown(f"저장소: `{r.repo}`  \n"
                            f"파일: **`{r.path}`**  \n"
                            f"수정 라인: **{stage2.fmt_line_ranges(r.ranges)}**  \n"
                            f"기준 커밋: `{r.commit_sha}`")
            st.markdown("#### 코드")
            render_code(stage2.read_lines(stage2.repo_dir(r.repo), r.path), r.ranges)
            st.markdown("#### 근거")
            st.table(pd.DataFrame(stage2.evidence_rows(ticket, r), columns=["항목", "내용"]).set_index("항목"))
        if recs:
            st.markdown("#### 함께 확인할 곳")
            impacts = result.get("impacts", [])
            if impacts:
                st.dataframe(pd.DataFrame([{"저장소": i.repo, "파일": i.path, "라인": i.line, "심볼": i.symbol,
                                            "관계": i.relation} for i in impacts]), hide_index=True, width="stretch")
            else:
                st.caption("같은 저장소의 다른 파일에서 참조하는 곳이 없습니다.")
        st.download_button("티켓 Markdown 다운로드", stage2.ticket_to_markdown(ticket, result),
                           file_name=f"{ticket.ticket_id}.md", mime="text/markdown")

# ════════════════════════════ 사이드바 ════════════════════════════

with st.sidebar:
    st.markdown("### 🔎 VOCAI")
    cur_ticket = ss["tickets"].get(ss["current_ticket"]) if ss["current_ticket"] else None
    st.markdown("**현재 티켓**")
    if cur_ticket:
        st.markdown(f"{badge(cur_ticket.priority)} **{cur_ticket.ticket_id}**", unsafe_allow_html=True)
        st.caption(cur_ticket.problem)
        st.caption(f"👤 {cur_ticket.owner_part} · {cur_ticket.assignee}  \n📅 예상 완료일 {cur_ticket.due_date}")
    else:
        st.caption("없음")

    s1 = ss["stage1"].get(tuple(ss.get("period", (config.ANALYSIS_START, config.ANALYSIS_END))))
    timings = (ss["stage2"].get(ss["current_ticket"]) or {}).get("timings", {}) if cur_ticket else {}
    secs = {
        "① 수집": 0.0 if not df_all.empty else None,
        "② 클러스터": s1["secs"] if s1 else None,
        "③ 티켓": ss["ticket_secs"] if ss["tickets"] else None,
    }
    for label, nodes in STEP_NODES.items():
        secs[label] = round(sum(timings[n] for n in nodes if n in timings), 2) if any(n in timings for n in nodes) else None

    st.markdown("**단계 진행**")
    for label in STEP_LABELS:
        done = secs[label] is not None
        st.markdown(f"{'✅' if done else '⬜'} {label}")

    st.markdown("**단계별 소요 시간 (초)**")
    time_df = pd.DataFrame([{"단계": k, "초": v} for k, v in secs.items() if v is not None and k != "① 수집"])
    if time_df.empty:
        st.caption("아직 실행 기록이 없습니다.")
    else:
        st.dataframe(time_df, hide_index=True, width="stretch")
