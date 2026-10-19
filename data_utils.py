"""VOC 로드 · 기간 필터 · KPI/차트 집계. (streamlit import 금지)"""

from datetime import date, timedelta

import pandas as pd

import config

CHANNEL_LABELS = {"review": "리뷰", "call_center": "고객센터"}


def load_vocs() -> pd.DataFrame:
    """data/vocs.csv를 읽어 created_at·date 열을 날짜형으로 만든다."""
    df = pd.read_csv(config.VOCS_CSV, encoding="utf-8")
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["date"] = df["created_at"].dt.date
    return df


def prev_period(start: date, end: date) -> tuple[date, date]:
    """분석 기간과 같은 길이의 직전 기간."""
    days = (end - start).days + 1
    return start - timedelta(days=days), start - timedelta(days=1)


def filter_period(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """start~end(양 끝 포함) VOC만 남긴다."""
    mask = (df["date"] >= start) & (df["date"] <= end)
    return df.loc[mask].copy()


def compute_kpis(cur: pd.DataFrame, prev: pd.DataFrame) -> dict:
    """KPI 4개: 총 건수 · 직전 대비 증감률 · 채널 비율 · 평균 평점."""
    total = len(cur)
    prev_total = len(prev)
    change_pct = (total - prev_total) / prev_total * 100 if prev_total else None
    review = int((cur["channel"] == "review").sum())
    call = int((cur["channel"] == "call_center").sum())
    return {
        "total": total,
        "prev_total": prev_total,
        "change_pct": change_pct,
        "review": review,
        "call_center": call,
        "review_pct": review / total * 100 if total else 0.0,
        "avg_rating": float(cur["rating"].mean()) if total else 0.0,
    }


def daily_counts(df: pd.DataFrame, group_col: str = "channel") -> pd.DataFrame:
    """일별 · 그룹별 건수 (long 형식: date, group, count)."""
    if df.empty:
        return pd.DataFrame(columns=["date", "group", "count"])
    out = df.groupby(["date", group_col]).size().reset_index(name="count")
    out = out.rename(columns={group_col: "group"})
    if group_col == "channel":
        out["group"] = out["group"].map(CHANNEL_LABELS).fillna(out["group"])
    return out.sort_values("date")


def voc_table(df: pd.DataFrame) -> pd.DataFrame:
    """화면용 VOC 원문 표 (최신순)."""
    out = df.sort_values("created_at", ascending=False)[
        ["voc_id", "created_at", "channel", "rating", "text"]
    ].copy()
    out["channel"] = out["channel"].map(CHANNEL_LABELS).fillna(out["channel"])
    return out
