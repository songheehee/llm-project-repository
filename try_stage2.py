"""Step 4·5 터미널 확인: TKT-0001 → 레포 · 폴더 · 라인 · 커밋 근거 · 영향.

실행: python try_stage2.py   (클러스터는 outputs/ 캐시 사용, 없으면 stage1 실행)
"""

import sys

import config
import data_utils
import stage1
import stage2

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    df = data_utils.load_vocs()
    start, end = config.ANALYSIS_START, config.ANALYSIS_END
    clusters, _ = stage1.run_stage1(start, end, df)
    top = clusters[0]
    ticket = stage1.ticketize(top, stage1.sample_texts(df, top, config.TICKET_SAMPLE_VOCS),
                              f"{start}~{end}", config.FIRST_TICKET_NO)
    print(f"[티켓] {ticket.ticket_id} [{ticket.priority}] {ticket.problem} · {ticket.owner_part}")
    print(f"  code_terms: {ticket.code_terms}\n")

    graph = stage2.build_stage2_graph()
    result = {}
    for event in stage2.stream_stage2(graph, ticket):
        if event[0] == "node":
            print(f"✓ {config.NODE_LABELS[event[1]]} ({event[2]:.1f}초)")
        else:
            result = event[1]

    print("\n[코드 출처]")
    for name, src in result.get("repo_sources", {}).items():
        print(f"  {name:<18} {src}")
    print("\n[④ 레포 후보]")
    for c in result["repo_candidates"]:
        print(f"  {c.name:<16} {c.score:.2f}  {c.reason}")
    print("\n[⑤ 폴더 의심도]")
    for f in result["folder_scores"]:
        print(f"  {f.repo}/{f.path:<10} {f.score:.2f}  (키워드 {f.keyword:.2f} · 최근 {f.recent})  {f.note}")
    print("\n[⑥ 수정 라인 — 검증됨]")
    for r in result["line_recs"]:
        rng = ", ".join(f"L{s}" if s == e else f"L{s}–{e}" for s, e in r.ranges)
        print(f"  {r.repo}/{r.path} {rng} @ {r.commit_sha}  (신뢰도 {r.confidence})")
        print(f"    이유: {r.reason}")
        for e in r.evidence:
            print(f"    근거[{e.kind}] {e.detail} {e.ref}")
    print("\n[⑦ 영향 범위]")
    for i in result["impacts"]:
        print(f"  {i.path}:{i.line}  {i.relation}")
    if result.get("errors"):
        print("\n[검증 실패 기록]", *result["errors"], sep="\n  ")
    print(f"\nretry={result.get('retry')} timings={result.get('timings')}")


if __name__ == "__main__":
    main()
