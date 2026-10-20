"""VOCAI 설정 상수 — 경로 · 기간 · 가중치 · 임계값 · 모델명은 모두 여기 한 곳에 둔다."""

from datetime import date
from pathlib import Path

# ── 경로 ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAG_DIR = DATA_DIR / "rag"
VOCS_CSV = DATA_DIR / "vocs_1000.csv"            # 원본 311건은 "vocs.csv"
ANSWER_CSV = DATA_DIR / "vocs_1000_labels.csv"   # 정답 확인 전용 (앱·LLM에 넣지 않음). 원본은 "answer_labels.csv"
ANSWER_LABEL_COL = "topic_name"                  # 원본은 "true_label"
DATASET = VOCS_CSV.stem                          # 캐시 파일 이름에 넣어 데이터셋끼리 섞이지 않게 한다
REPOS_YAML = DATA_DIR / "repos.yaml"
SAMPLE_REPOS_DIR = BASE_DIR / "sample_repos"
OUTPUTS_DIR = BASE_DIR / "outputs"

# ── 기간 ─────────────────────────────────────────────
ANALYSIS_START = date(2026, 10, 3)
ANALYSIS_END = date(2026, 10, 14)
PREV_START = date(2026, 9, 21)
PREV_END = date(2026, 10, 2)

# ── 모델 ─────────────────────────────────────────────
CHAT_MODEL = "gpt-5.4-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MAX_RETRIES = 2  # 구조화 출력 실패 시 재시도 횟수

# ── 클러스터링 ───────────────────────────────────────
ASSIGN_BATCH_SIZE = 50
MIN_CLUSTERS = 5
MAX_CLUSTERS = 20
OTHER_LABEL = "기타"
MIN_CLUSTER_SIZE = 3          # 분석 기간 건수가 이보다 적은 클러스터는 '기타'로 돌린다
SAMPLE_VOCS_PER_CLUSTER = 3   # 화면 대표 VOC
TICKET_SAMPLE_VOCS = 10       # 티켓화 프롬프트에 넣는 VOC 수
FIRST_TICKET_NO = 412         # VOC-0412부터

# ── VOC 유사도 지도 (임베딩 · t-SNE 3D 투영) ───────────────
VOC_EMBEDDING_MODEL = "text-embedding-3-large"  # VOC 지도·임베딩 클러스터링용 (RAG는 EMBEDDING_MODEL)
VOC_EMBEDDINGS_TEMPLATE = "voc_embeddings_{dataset}_{model}.json"  # outputs/ 안 {voc_id: 벡터} 캐시 (데이터셋·모델별)
EMBED_BATCH_SIZE = 100
PROJECTION_SEED = 42
TSNE_PERPLEXITY = 30
SIMILAR_TOP_K = 5             # 선택한 VOC와 가장 비슷한 VOC 수
MAP_POINT_SIZE = 2.5          # 지도 VOC 점 크기
MAP_HIGHLIGHT_SIZE = 6        # 기준·유사 VOC 강조 점 크기
DISAGREE_K = 7                # 라벨 불일치 판정에 쓰는 이웃 수 (다수결)
KMEANS_N_INIT = 10            # 임베딩 클러스터링: k마다 KMeans 초기화 횟수 (k는 MIN~MAX_CLUSTERS 중 실루엣 최대)
EMBED_NAME_CHARS = 18         # 임베딩 클러스터 이름에 쓰는 대표 문장 길이

# ── 급증률 · 추세 ─────────────────────────────────────
TREND_UP = 1.3
TREND_DOWN = 0.7

# ── 우선순위 ─────────────────────────────────────────
W_COUNT = 0.3
W_SURGE = 0.3
W_SEVERITY = 0.4
SURGE_LOG_DIVISOR = 3.0
PRIORITY_THRESHOLDS = [  # (최소 점수, 라벨) — 위에서부터 판정
    (0.80, "긴급"),
    (0.45, "높음"),
    (0.23, "보통"),
    (0.0, "낮음"),
]
PRIORITY_ORDER = {"긴급": 0, "높음": 1, "보통": 2, "낮음": 3}
PRIORITY_BADGE = {"긴급": "🔴", "높음": "🟠", "보통": "🟡", "낮음": "⚪"}
PRIORITY_COLOR = {"긴급": "#D93025", "높음": "#F28B00", "보통": "#E3B500", "낮음": "#9AA0A6"}

# ── 레포 라우팅 ──────────────────────────────────────
REPO_MIN_SCORE = 0.1
REPO_MAX_CANDIDATES = 3
RAG_CHUNK_SIZE = 300
RAG_CHUNK_OVERLAP = 50
RAG_TOP_K = 3

# ── 폴더 의심도 ──────────────────────────────────────
W_KEYWORD = 0.6
W_RECENT = 0.4
RECENT_DAYS_STRONG = 14   # 이 안에 커밋 → 1.0
RECENT_DAYS_WEAK = 30     # 이 안에 커밋 → 0.5
RECENT_SCORE_STRONG = 1.0
RECENT_SCORE_WEAK = 0.5
ROLE_FACTORS = {"display_only": 0.5}
ROLE_NOTES = {"display_only": "표시 전용 폴더 → 감점"}

# ── 라인 탐색 · 검증 · 영향 ──────────────────────────
TOP_FOLDERS_FOR_LINES = 2
TOP_FILES_FOR_LINES = 3
GIT_LOG_PATCH_CHARS = 1500   # 파일당 최근 변경 요약 길이
COMMIT_EVIDENCE_DAYS = 30
MAX_LINE_RETRY = 1           # verify 실패 시 search_lines 재시도 횟수
CODE_CONTEXT_LINES = 5       # 코드 뷰 앞뒤 줄 수

# ── 화면 표기 ────────────────────────────────────────
NODE_LABELS = {
    "route_repos": "④ 레포 라우팅",
    "score_folders": "⑤ 폴더 의심도",
    "search_lines": "⑥ 라인 탐색",
    "verify_lines": "⑥ 라인 검증",
    "add_commit_evidence": "⑥ 커밋 근거",
    "analyze_impact": "⑦ 영향 범위",
}
