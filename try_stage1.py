"""Step 2·3 터미널 확인: 클러스터 표 · 정답 일치율 · 1순위 클러스터 티켓.

실행: python try_stage1.py [--no-cache]
"""

import sys

import pandas as pd

import config
import data_utils
import stage1

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    use_cache = "--no-cache" not in sys.argv
    df = data_utils.load_vocs()
    start, end = config.ANALYSIS_START, config.ANALYSIS_END
    clusters, assignments = stage1.run_stage1(start, end, df, use_cache=use_cache)

    print(f"\n[클러스터] {start} ~ {end}")
    print(f"{'ID':<4}{'우선순위':<6}{'라벨':<20}{'영역':<6}{'심각도':>4}{'건수':>6}{'직전':>6}  {'추세':<10}{'점수':>5}")
    for c in clusters:
        print(f"{c.cluster_id:<4}{c.priority:<6}{c.label:<20}{c.category:<6}{c.severity:>4}"
              f"{c.count:>6}{c.prev_count:>6}  {c.trend:<10}{c.priority_score:>5.2f}")
    others = sum(1 for v in assignments.values() if v == config.OTHER_LABEL)
    print(f"기타 분류: {others}건")

    # 정답 일치율 — answer_labels.csv는 여기서 확인용으로만 쓴다
    answers = pd.read_csv(config.ANSWER_CSV, encoding="utf-8").set_index("voc_id")["true_label"]
    pred = pd.Series(assignments)
    joined = pd.DataFrame({"pred": pred, "true": answers.reindex(pred.index)})
    mapping = joined.groupby("pred")["true"].agg(lambda s: s.value_counts().index[0])
    acc = (joined["pred"].map(mapping) == joined["true"]).mean()
    print("\n[정답 대응] 예측 라벨 → 정답 라벨(다수결)")
    for p, t in mapping.items():
        print(f"  {p} → {t}")
    print(f"일치율: {acc:.1%}")

    if clusters:
        top = clusters[0]
        samples = stage1.sample_texts(df, top, config.TICKET_SAMPLE_VOCS)
        ticket = stage1.ticketize(top, samples, f"{start}~{end}", config.FIRST_TICKET_NO)
        print("\n[티켓]")
        for k, v in ticket.model_dump(exclude={"voc_ids"}).items():
            print(f"  {k}: {v}")
        print(f"  voc_ids: {len(ticket.voc_ids)}건")


if __name__ == "__main__":
    main()
