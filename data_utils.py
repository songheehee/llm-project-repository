"""VOC 로드 · 기간 필터 · KPI/차트 집계. (streamlit import 금지)"""

from datetime import date, timedelta

import pandas as pd
import yaml

import config

CHANNEL_LABELS = {"review": "리뷰", "call_center": "고객센터"}


def load_vocs() -> pd.DataFrame:
    """config.VOCS_CSV를 읽어 created_at·date 열을 날짜형으로 만든다."""
    df = pd.read_csv(config.VOCS_CSV, encoding="utf-8")
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["date"] = df["created_at"].dt.date
    return df


def load_answer_labels() -> dict[str, str]:
    """정답 라벨 {voc_id: 정답 주제} — 화면의 ARI 계산에만 쓴다(LLM에는 넣지 않음). 파일이 없으면 빈 dict."""
    if not config.ANSWER_CSV.exists():
        return {}
    df = pd.read_csv(config.ANSWER_CSV, encoding="utf-8")
    return dict(zip(df["voc_id"], df[config.ANSWER_LABEL_COL]))


def load_users() -> list[dict]:
    """data/users.yaml의 개발자 목록 [{id, name, part}] — 사이드바 '현재 개발자'(로그인 대신)."""
    data = yaml.safe_load(config.USERS_YAML.read_text(encoding="utf-8"))
    return list(data.get("users", []))


def prev_period(start: date, end: date) -> tuple[date, date]:
    """분석 기간과 같은 길이의 직전 기간."""
    days = (end - start).days + 1
    return start - timedelta(days=days), start - timedelta(days=1)


def filter_period(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """start~end(양 끝 포함) VOC만 남긴다."""
    mask = (df["date"] >= start) & (df["date"] <= end)
    return df.loc[mask].copy()


def compute_kpis(cur: pd.DataFrame, prev: pd.DataFrame) -> dict:
    """총 건수 · 직전 기간 건수 · 직전 대비 증감률. (유형별 KPI는 클러스터 분석 후 화면에서 계산)"""
    total = len(cur)
    prev_total = len(prev)
    change_pct = (total - prev_total) / prev_total * 100 if prev_total else None
    return {"total": total, "prev_total": prev_total, "change_pct": change_pct}


def daily_counts(df: pd.DataFrame, group_col: str = "channel", weekly: bool = False) -> pd.DataFrame:
    """일별(또는 주별) · 그룹별 건수 (long 형식: date, group, count). 주별이면 date = 그 주 월요일."""
    if df.empty:
        return pd.DataFrame(columns=["date", "group", "count"])
    key = df["date"]
    if weekly:
        key = pd.to_datetime(df["date"]).dt.to_period("W-SUN").dt.start_time.dt.date
    out = df.assign(date=key).groupby(["date", group_col]).size().reset_index(name="count")
    out = out.rename(columns={group_col: "group"})
    if group_col == "channel":
        out["group"] = out["group"].map(CHANNEL_LABELS).fillna(out["group"])
    return out.sort_values("date")


def top_groups(values: pd.Series, n: int, other: str, keep: tuple[str, ...] = ()) -> pd.Series:
    """건수 상위 n개 값만 남기고 나머지는 other로 묶는다 (차트 색 수 제한용).
    keep 값은 상위 계산에서 빼고 묶지도 않는다 (예: '미분류')."""
    top = [v for v in values.value_counts().index if v != other and v not in keep][:n]
    return values.where(values.isin(top) | values.isin(keep), other)


DISPLAY_COLUMNS = {"voc_id": "VOC ID", "created_at": "작성 일시", "channel": "채널", "text": "원문"}


def to_display_vocs(df: pd.DataFrame) -> pd.DataFrame:
    """VOC를 보여 주는 모든 표의 공통 규칙 (화면용 복사본만, 원본 DataFrame·CSV 열 이름은 그대로):
    열 이름 한글화(VOC ID · 작성 일시 · 채널 · 원문), 작성 일시 "YYYY-MM-DD HH:MM", 채널 리뷰/고객센터."""
    out = df.copy()
    if "created_at" in out:
        out["created_at"] = pd.to_datetime(out["created_at"]).dt.strftime("%Y-%m-%d %H:%M")
    if "channel" in out:
        out["channel"] = out["channel"].map(CHANNEL_LABELS).fillna(out["channel"])
    return out.rename(columns=DISPLAY_COLUMNS)


def voc_table(df: pd.DataFrame) -> pd.DataFrame:
    """화면용 VOC 원문 표 (최신순, 한글 열 이름)."""
    return to_display_vocs(df.sort_values("created_at", ascending=False)[["voc_id", "created_at", "channel", "text"]])
