"""VOCAI — 고객의 목소리 → 고칠 곳 (Streamlit 화면). CSS 주입은 이 파일 한 곳에서만 한다."""

import os
import time
from datetime import date

import pandas as pd
import plotly.express as px
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
.block-container {padding-top: 1.6rem;}
.vocai-header {display:flex; align-items:center; gap:14px; flex-wrap:wrap;
  background:#FFFFFF; border:1px solid #E3E6EF; border-radius:12px; padding:14px 20px; margin-bottom:10px;}
.vocai-title {font-size:1.5rem; font-weight:800; color:#1428A0;}
.vocai-sub {color:#5F6368; font-size:0.95rem;}
.vocai-chip {background:#EEF1FB; color:#1428A0; border-radius:999px; padding:2px 10px; font-size:0.8rem; font-weight:600;}
.vocai-period {margin-left:auto; color:#3C4043; font-size:0.9rem;}
.vocai-badge {display:inline-block; color:#FFFFFF; border-radius:6px; padding:1px 8px; font-size:0.8rem; font-weight:700;}
[class*="st-key-card"] {background:#FFFFFF; border:1px solid #E3E6EF; border-radius:12px; padding:14px 16px;}
[data-testid="stMetric"] {background:#FFFFFF; border:1px solid #E3E6EF; border-radius:12px; padding:12px 16px;}
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
NEXT_NODE = {
    "route_repos": "score_folders", "score_folders": "search_lines", "search_lines": "verify_lines",
    "verify_lines": "add_commit_evidence", "add_commit_evidence": "analyze_impact",
}

# ════════════════════════════ 캐시 ════════════════════════════

@st.cache_data
def cached_vocs() -> pd.DataFrame:
    return data_utils.load_vocs()


@st.cache_data
def cached_kpis(start: date, end: date) -> dict:
    df = cached_vocs()
    ps, pe = data_utils.prev_period(start, end)
    return data_utils.compute_kpis(data_utils.filter_period(df, start, end), data_utils.filter_period(df, ps, pe))


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


def badge(priority: str) -> str:
    """고정 라벨만 들어가는 HTML 배지 (LLM 텍스트 아님)."""
    color = config.PRIORITY_COLOR.get(priority, "#9AA0A6")
    return f'<span class="vocai-badge" style="background:{color}">{priority}</span>'


def fmt_ranges(ranges) -> str:
    return " · ".join(f"L{s}" if s == e else f"L{s}–{e}" for s, e in ranges)


def has_api_key() -> bool:
    if os.getenv("OPENAI_API_KEY"):
        return True
    st.error("`.env`에 OPENAI_API_KEY가 없습니다.")
    return False


# ════════════════════════════ 헤더 ════════════════════════════

df_all = cached_vocs()
min_d, max_d = df_all["date"].min(), df_all["date"].max()
period = ss.get("period", (config.ANALYSIS_START, config.ANALYSIS_END))

st.markdown(
    f"""<div class="vocai-header">
      <span class="vocai-title">🔎 VOCAI</span>
      <span class="vocai-sub">고객의 목소리 → 고칠 곳</span>
      <span class="vocai-chip">교육용 가상 데이터</span>
      <span class="vocai-period">분석 기간 {period[0]:%m/%d} ~ {period[1]:%m/%d}</span>
    </div>""",
    unsafe_allow_html=True,
)

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
    run_clicked = c_btn.button("🔍 클러스터 분석", type="primary", width="stretch")

    if run_clicked and has_api_key():
        with st.spinner("VOC를 클러스터로 묶는 중… (같은 기간은 캐시 사용)"):
            cached_llm()
            t0 = time.perf_counter()
            clusters, assignments = stage1.run_stage1(start, end, df_all)
            ss["stage1"][key] = {"clusters": clusters, "assignments": assignments,
                                 "secs": round(time.perf_counter() - t0, 1)}

    s1 = ss["stage1"].get(key)
    kpi = cached_kpis(start, end)
    ps, pe = data_utils.prev_period(start, end)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("총 VOC", f"{kpi['total']:,}건",
              delta=f"{kpi['total'] - kpi['prev_total']:+,}건", delta_color="inverse")
    k2.metric("직전 기간 대비", f"{kpi['change_pct']:+.1f}%" if kpi["change_pct"] is not None else "–",
              help=f"직전 기간 {ps:%m/%d}~{pe:%m/%d} · {kpi['prev_total']}건")
    k3.metric("리뷰 · 고객센터", f"{kpi['review_pct']:.1f}% 리뷰",
              help=f"리뷰 {kpi['review']}건 / 고객센터 {kpi['call_center']}건")
    k4.metric("평균 평점", f"{kpi['avg_rating']:.2f}")
    if s1:
        k5, k6, _, _ = st.columns(4)
        k5.metric("클러스터 수", f"{len(s1['clusters'])}개")
        k6.metric("긴급 클러스터", f"{sum(c.priority == '긴급' for c in s1['clusters'])}개")

    cur = data_utils.filter_period(df_all, start, end)
    with st.container(key="card_chart"):
        if s1:
            cur = cur.assign(cluster=cur["voc_id"].map(s1["assignments"]).fillna(config.OTHER_LABEL))
            daily = data_utils.daily_counts(cur, "cluster")
            title = "일별 VOC 건수 (클러스터별 누적)"
        else:
            daily = data_utils.daily_counts(cur, "channel")
            title = "일별 VOC 건수 (채널별 누적)"
        fig = px.bar(daily, x="date", y="count", color="group", title=title,
                     labels={"date": "날짜", "count": "건수", "group": ""})
        fig.update_layout(barmode="stack", height=330, margin=dict(l=10, r=10, t=50, b=10),
                          legend=dict(orientation="h", y=-0.2), plot_bgcolor="#FFFFFF")
        st.plotly_chart(fig, width="stretch")

    if not s1:
        st.subheader("VOC 원문")
        st.dataframe(data_utils.voc_table(cur), hide_index=True, width="stretch", height=320)
    else:
        st.subheader("클러스터")
        clusters = s1["clusters"]
        table = pd.DataFrame([{
            "발행": False,
            "우선순위": f"{config.PRIORITY_BADGE[c.priority]} {c.priority}",
            "클러스터": c.label,
            "영역 · 제품": f"{c.category} · {c.product}",
            "건수": c.count,
            "직전": c.prev_count,
            "추세": c.trend,
            "점수": c.priority_score,
            "심각도": c.severity,
        } for c in clusters])
        edited = st.data_editor(
            table, hide_index=True, width="stretch", key=f"editor_{start}_{end}",
            column_config={
                "발행": st.column_config.CheckboxColumn("발행", help="체크한 클러스터만 티켓 발행"),
                "점수": st.column_config.NumberColumn(format="%.2f"),
            },
            disabled=[c for c in table.columns if c != "발행"],
        )

        df_idx = df_all.set_index("voc_id")
        for c in clusters:
            with st.expander(f"{config.PRIORITY_BADGE[c.priority]} {c.label} — 대표 VOC 보기"):
                for vid in c.voc_ids[:config.SAMPLE_VOCS_PER_CLUSTER]:
                    st.text(f"{vid} · {df_idx.loc[vid, 'text']}")

        c_info, c_issue = st.columns([4, 1.2], vertical_alignment="center")
        c_info.caption("체크한 클러스터만 티켓으로 발행되어 담당 파트로 전달됩니다. (자동 전체 발행 없음)")
        if c_issue.button("🎫 티켓 발행", type="primary", width="stretch"):
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

    for t in ss["tickets"].values():
        with st.container(key=f"card_ticket_{t.ticket_id}"):
            st.markdown(f"{badge(t.priority)} &nbsp;**{t.ticket_id}**", unsafe_allow_html=True)
            st.markdown(f"**{t.problem}**")
            st.caption(f"{t.count}건 {t.trend} · {t.owner_part} · {t.product} · {t.period}")
            st.markdown(f"발생 조건: {', '.join(t.conditions)}  \n기술 키워드: {', '.join(t.tech_keywords)}")

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
            with st.status(f"{config.NODE_LABELS['route_repos']} 중…", expanded=True) as status:
                for event in stage2.stream_stage2(graph, ticket):
                    if event[0] == "node":
                        _, node, secs = event
                        st.write(f"✓ {config.NODE_LABELS[node]} ({secs:.1f}초)")
                        nxt = NEXT_NODE.get(node)
                        if nxt:
                            status.update(label=f"{config.NODE_LABELS[nxt]} 중…")
                    else:
                        ss["stage2"][sel] = event[1]
                status.update(label="진행 상황 ✓ (④ → ⑤ → ⑥ → ⑦)", state="complete", expanded=False)

        result = ss["stage2"].get(sel)
        if not result:
            st.caption("▶ 코드 좁혀 가기를 누르면 ④ 레포 라우팅 → ⑤ 폴더 의심도 → ⑥ 라인 탐색 → ⑦ 영향 범위를 차례로 실행합니다.")
        else:
            with st.container(key="card_repos"):
                st.markdown("#### 레포 후보")
                cands = result.get("repo_candidates", [])
                if not cands:
                    st.warning("점수 0.1 이상인 레포 후보가 없습니다.")
                else:
                    st.dataframe(
                        pd.DataFrame([{"레포": c.name, "점수": c.score, "이유": c.reason} for c in cands]),
                        hide_index=True, width="stretch",
                        column_config={"점수": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")},
                    )
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
        for idx, r in enumerate(recs):
            st.markdown(f"### `{r.path}` @ `{r.commit_sha}` &nbsp; 👉 {fmt_ranges(r.ranges)} 수정 추천")
            st.caption(f"{r.repo} · 신뢰도 {r.confidence:.2f}")
            c_code, c_side = st.columns([3, 2])
            with c_code:
                lines = stage2.read_lines(stage2.repo_dir(r.repo), r.path)
                render_code(lines, r.ranges)
                st.markdown(r.reason)
            with c_side:
                with st.container(key=f"card_evidence_{idx}"):
                    st.markdown("**📌 근거**")
                    icons = {"voc": "🗣", "commit": "🕒", "keyword": "🔤"}
                    for e in r.evidence:
                        st.markdown(f"{icons[e.kind]} {e.detail}" + (f" `{e.ref}`" if e.ref else ""))
        impacts = result.get("impacts", [])
        if recs:
            with st.container(key="card_impacts"):
                st.markdown("**🔗 함께 확인할 곳**")
                if impacts:
                    for i in impacts:
                        st.markdown(f"- `{i.path}:{i.line}` — {i.relation}")
                else:
                    st.caption("다른 파일에서 참조하는 곳이 없습니다.")
        st.download_button("⬇ 티켓 Markdown 다운로드", stage2.ticket_to_markdown(ticket, result),
                           file_name=f"{ticket.ticket_id}.md", mime="text/markdown")

# ════════════════════════════ 사이드바 ════════════════════════════

with st.sidebar:
    st.markdown("### 🔎 VOCAI")
    cur_ticket = ss["tickets"].get(ss["current_ticket"]) if ss["current_ticket"] else None
    st.markdown("**현재 티켓**")
    if cur_ticket:
        st.markdown(f"{badge(cur_ticket.priority)} **{cur_ticket.ticket_id}**", unsafe_allow_html=True)
        st.caption(cur_ticket.problem)
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
