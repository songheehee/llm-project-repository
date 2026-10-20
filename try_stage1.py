"""Step 2·3 터미널 확인: 클러스터 표 · 정답 일치율 · 1순위 클러스터 티켓.

실행: python try_stage1.py [--no-cache] [--no-ticket]
"""

import sys

import pandas as pd

import config
import data_utils
import stage1

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def purity(pred: list[str], true: list[str]) -> float:
    """예측 묶음마다 다수 정답 라벨 비율 — 한 묶음에 다른 정답이 섞일수록 낮다."""
    joined = pd.DataFrame({"pred": pred, "true": true})
    return joined.groupby("pred")["true"].agg(lambda s: s.value_counts().iloc[0]).sum() / len(joined)


def compare_methods(df: pd.DataFrame, assignments: dict[str, str], answers: pd.Series) -> None:
    """분석 기간 VOC를 LLM 분류 vs 임베딩 분류로 비교한다. 정답 지표는 여기서만 계산."""
    cur = data_utils.filter_period(df, config.ANALYSIS_START, config.ANALYSIS_END)
    cur = cur.sort_values("voc_id").reset_index(drop=True)
    vectors = stage1.embed_vocs(cur)
    ids, sil_by_k = stage1.cluster_embeddings(vectors)
    emb = stage1.embedding_cluster_names(vectors, ids, cur["text"].tolist())
    llm = cur["voc_id"].map(assignments).fillna(config.OTHER_LABEL).tolist()
    true = answers.reindex(cur["voc_id"]).tolist()

    print(f"\n[LLM vs 임베딩] 분석 기간 {len(cur)}건 · 임베딩 {config.VOC_EMBEDDING_MODEL}")
    print("  k별 실루엣: " + " ".join(f"{k}:{s:.3f}" for k, s in sil_by_k.items()))
    print(f"  {'지표':<22}{'LLM':>8}{'임베딩':>8}")
    m_llm = stage1.clustering_metrics(vectors, llm, config.DISAGREE_K)
    m_emb = stage1.clustering_metrics(vectors, emb, config.DISAGREE_K)
    a_llm = stage1.agreement_metrics(llm, true)
    a_emb = stage1.agreement_metrics(emb, true)
    rows = [
        ("클러스터 수", m_llm["n_clusters"], m_emb["n_clusters"], "{:>8d}"),
        ("실루엣 ↑ (내부)", m_llm["silhouette"], m_emb["silhouette"], "{:>8.3f}"),
        ("Davies-Bouldin ↓ (내부)", m_llm["davies_bouldin"], m_emb["davies_bouldin"], "{:>8.3f}"),
        ("kNN 일관성 ↑ (내부)", m_llm["knn_consistency"], m_emb["knn_consistency"], "{:>8.3f}"),
        ("ARI ↑ (정답)", a_llm["ari"], a_emb["ari"], "{:>8.3f}"),
        ("NMI ↑ (정답)", a_llm["nmi"], a_emb["nmi"], "{:>8.3f}"),
        ("순도 ↑ (정답)", purity(llm, true), purity(emb, true), "{:>8.3f}"),
    ]
    for name, a, b, fmt in rows:
        fa = fmt.format(a) if a is not None else f"{'–':>8}"
        fb = fmt.format(b) if b is not None else f"{'–':>8}"
        print(f"  {name:<22}{fa}{fb}")
    both = stage1.agreement_metrics(llm, emb)
    print(f"  LLM↔임베딩 일치: ARI {both['ari']:.3f} · NMI {both['nmi']:.3f}")
    print("  임베딩 클러스터 → 정답(다수결):")
    tab = pd.crosstab(pd.Series(emb, name="emb"), pd.Series(true, name="true"))
    for name, row in tab.iterrows():
        print(f"    {name:<28} {row.idxmax()} ({row.max()}/{row.sum()})")


# 현재 데이터(config.VOCS_CSV = vocs.csv) 기준 기대값 — 데이터를 바꾸면 여기와 CLAUDE.md 정답을 함께 갱신
EXPECTED_PRIORITY = {"운동 중 기기 발열": "긴급", "연결 끊김 시 발열": "높음", "배터리 소모": "높음"}  # 정답 주제 → 우선순위
EXPECTED_TOP = "운동 중 기기 발열"     # 처리 시급도 1위 유형의 정답 주제
# 데이터와 무관한 일반 기준
TYPE_COUNT_TOLERANCE = 3              # 유형 수 = 분석 기간 정답 주제 수 ± 이 값
MIN_ARI = 0.85                        # 분석 기간 LLM 분류 vs 정답 ARI
MAX_OTHER_RATIO = 0.10                # '기타' 비율 상한


def check_criteria(df: pd.DataFrame, clusters, assignments: dict[str, str], answers: pd.Series) -> bool:
    """1000건 기준 정답 확인: 꼭 맞아야 하는 것(우선순위)과 허용 범위(유형 수·ARI·기타)를 PASS/FAIL로 출력."""
    cur = data_utils.filter_period(df, config.ANALYSIS_START, config.ANALYSIS_END)
    pred = cur["voc_id"].map(assignments).fillna(config.OTHER_LABEL)
    true = answers.reindex(cur["voc_id"]).values
    tab = pd.crosstab(pred.values, true)
    major = {c.label: tab.loc[c.label].idxmax() for c in clusters if c.label in tab.index}  # 유형 → 다수 정답 주제
    by_topic = {}
    for c in clusters:
        by_topic.setdefault(major.get(c.label), []).append(c)
    n_true = len(set(true))
    ari = stage1.agreement_metrics(pred.tolist(), list(true))["ari"]
    other = (pred == config.OTHER_LABEL).mean()
    ranked = sorted(clusters, key=lambda c: -c.priority_score)

    checks = [(f"처리 시급도 1위 = {EXPECTED_TOP}", bool(ranked) and major.get(ranked[0].label) == EXPECTED_TOP)]
    for topic, prio in EXPECTED_PRIORITY.items():
        got = [c.priority for c in by_topic.get(topic, [])]
        checks.append((f"{topic} → {prio} (실제 {', '.join(got) or '없음'})", got == [prio]))
    urgent = [c for c in clusters if c.priority == "긴급"]
    n_urgent_expected = sum(p == "긴급" for p in EXPECTED_PRIORITY.values())
    checks.append((f"긴급 유형 {n_urgent_expected}개 (실제 {len(urgent)}개)", len(urgent) == n_urgent_expected))
    others_high = [c.label for c in clusters if c.priority in ("긴급", "높음") and major.get(c.label) not in EXPECTED_PRIORITY]
    checks.append((f"나머지 유형은 보통·낮음 (예외: {others_high or '없음'})", not others_high))
    checks.append((f"유형 수 {len(clusters)}개 = 정답 {n_true}개 ± {TYPE_COUNT_TOLERANCE}",
                   abs(len(clusters) - n_true) <= TYPE_COUNT_TOLERANCE))
    checks.append((f"ARI {ari:.3f} ≥ {MIN_ARI}", ari >= MIN_ARI))
    checks.append((f"기타 {other:.1%} ≤ {MAX_OTHER_RATIO:.0%}", other <= MAX_OTHER_RATIO))

    print(f"\n[정답 기준] 분석 기간 {len(cur)}건")
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    passed = all(ok for _, ok in checks)
    print(f"  → {'모두 통과' if passed else '실패 항목 있음'}")
    return passed


def main() -> None:
    use_cache = "--no-cache" not in sys.argv
    df = data_utils.load_vocs()
    start, end = config.ANALYSIS_START, config.ANALYSIS_END
    clusters, assignments = stage1.run_stage1(start, end, df, use_cache=use_cache)

    print(f"\n[클러스터] {start} ~ {end}")
    print(f"{'ID':<4}{'우선순위':<6}{'라벨':<20}{'영역':<6}{'심각도':>4}{'이번':>6}{'이전':>6}  {'추세':<12}{'시급도':>5}")
    for c in clusters:
        print(f"{c.cluster_id:<4}{c.priority:<6}{c.label:<20}{c.category:<6}{c.severity:>4}"
              f"{c.count:>6}{c.prev_count:>6}  {c.trend:<12}{round(c.priority_score * 100):>5}")
    others = sum(1 for v in assignments.values() if v == config.OTHER_LABEL)
    print(f"기타 분류: {others}건")

    # 정답 일치율 — 정답 라벨(config.ANSWER_CSV)은 여기서 확인용으로만 쓴다
    answers = pd.read_csv(config.ANSWER_CSV, encoding="utf-8").set_index("voc_id")[config.ANSWER_LABEL_COL]
    check_criteria(df, clusters, assignments, answers)
    pred = pd.Series(assignments)
    joined = pd.DataFrame({"pred": pred, "true": answers.reindex(pred.index)})
    mapping = joined.groupby("pred")["true"].agg(lambda s: s.value_counts().index[0])
    acc = (joined["pred"].map(mapping) == joined["true"]).mean()
    print("\n[정답 대응] 예측 라벨 → 정답 라벨(다수결)")
    for p, t in mapping.items():
        print(f"  {p} → {t}")
    print(f"일치율: {acc:.1%}")

    compare_methods(df, assignments, answers)

    if "--no-ticket" not in sys.argv and clusters:
        top = clusters[0]
        samples = stage1.sample_texts(df, top, config.TICKET_SAMPLE_VOCS)
        ticket = stage1.ticketize(top, samples, f"{start}~{end}", config.FIRST_TICKET_NO)
        print("\n[티켓]")
        for k, v in ticket.model_dump(exclude={"voc_ids"}).items():
            print(f"  {k}: {v}")
        print(f"  voc_ids: {len(ticket.voc_ids)}건")


if __name__ == "__main__":
    main()
