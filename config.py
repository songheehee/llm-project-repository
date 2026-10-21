"""VOCAI 설정 상수 — 경로 · 기간 · 가중치 · 임계값 · 모델명은 모두 여기 한 곳에 둔다."""

import hashlib
from datetime import date
from pathlib import Path

# ── 경로 ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAG_DIR = DATA_DIR / "rag"
VOCS_CSV = DATA_DIR / "vocs.csv"            # VOC 입력 — 건수는 이 CSV를 읽을 때 정해진다 (지금은 테스트용 1000건)
ANSWER_CSV = DATA_DIR / "vocs_labels.csv"   # 정답 확인 전용 (앱·LLM에 넣지 않음, try_stage1에서만)
ANSWER_LABEL_COL = "topic_name"                  # ANSWER_CSV의 정답 라벨 열


def _dataset_key(path: Path) -> str:
    """캐시 파일 이름용: 파일 이름 + 내용 해시(줄바꿈 무시). 같은 이름에 다른 데이터가 들어와도 캐시가 섞이지 않는다."""
    if not path.exists():
        return path.stem
    digest = hashlib.sha1(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:8]
    return f"{path.stem}_{digest}"


DATASET = _dataset_key(VOCS_CSV)
REPOS_YAML = DATA_DIR / "repos.yaml"
USERS_YAML = DATA_DIR / "users.yaml"            # 개발자 목록 (사이드바 '현재 개발자' — 로그인 대신)
SAMPLE_REPOS_DIR = BASE_DIR / "sample_repos"
MIRRORS_DIR = BASE_DIR / "mirrors"            # repos.yaml에 url이 있는 저장소의 로컬 미러 (clone → 분석 전마다 pull)
GIT_SYNC_TIMEOUT_S = 60                      # clone · pull 제한 시간. 실패하면 sample_repos로 대체
OUTPUTS_DIR = BASE_DIR / "outputs"
TICKETS_JSON = OUTPUTS_DIR / "tickets.json"      # 발행된 티켓 + 다음 티켓 번호 (앱 시작·새로고침해도 유지)
STAGE2_RESULTS_DIR = OUTPUTS_DIR / "stage2"      # 코드 찾기 결과 — 티켓 id별 json (개발자를 바꿨다 돌아와도 다시 실행 안 함)

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
MAX_CLUSTERS = 14
OTHER_LABEL = "기타"
MIN_CLUSTER_SIZE = 3          # 분석 기간 건수가 이보다 적은 클러스터는 '기타'로 돌린다
MIN_CLUSTER_SHARE = 0.02      # 프롬프트 안내: 분석 기간 VOC의 이 비율 미만이 될 묶음은 따로 만들지 말고 합친다 (건수는 데이터에 따라 동적)
REP_VOC_MAX = 3               # '대표 VOC 보기' 티켓마다 보여 줄 원문 수 (유형 중심에 가까운 순)
TICKET_SAMPLE_VOCS = 10       # 티켓화 프롬프트에 넣는 VOC 수
PROBLEM_MAX_CHARS = 40        # 티켓 문제점(개발자 언어 한 줄) 최대 글자 수 — 표에서 잘리지 않게
REASON_MAX_CHARS = 60         # 수정 라인 추천 사유 최대 글자 수 (명사형 한 문장)
FIRST_TICKET_NO = 1           # 티켓 번호 시작 (발행 순서대로 TKT-0001, TKT-0002 …)
TICKET_ID_PREFIX = "TKT"      # 티켓은 VOC 하나가 아니라 VOC 유형 하나에 대한 것 — 개별 VOC ID(V0001…)와 구분
GRADES = {                      # users.yaml grade 코드 ↔ 뜻 (이번 범위에서는 화면에 표시하지 않음)
    "CL1": "사원급 (고졸 등)",
    "CL2": "대리·선임급 (대졸 신입 포함)",
    "CL3": "과장·차장급 (책임급)",
    "CL4": "부장급 (수석급)",
}
TICKET_STATUS_OPEN = "진행 중"  # 이번 범위의 티켓 상태는 이 값 하나 (완료 처리 기능 없음)
DUE_DAYS = {"긴급": 3, "높음": 7, "보통": 14, "낮음": 30}  # 예상 완료일 = 발행일 + 우선순위별 기한(일)

# ── 1️⃣ 대시보드 표시 ─────────────────────────────────
CHART_TOP_TYPES = 5           # 일별 차트: 건수 상위 N개 유형만 색으로 나누고 나머지 유형은 '기타', LLM 미분류는 '미분류'
OTHER_COLOR = "#BDC1C6"       # 차트 '기타' 색
UNASSIGNED_LABEL = "미분류"    # 차트 표시명: LLM이 어느 유형에도 넣지 못한 VOC (분류 결과의 '기타')
UNASSIGNED_COLOR = "#80868B"  # 차트 '미분류' 색 — '기타'보다 진한 회색
TABLE_VISIBLE_ROWS = 5        # 클러스터 표 높이 = N줄 + 헤더, 더 많으면 표 안에서 스크롤
TABLE_ROW_PX = 35             # st.data_editor 한 줄 높이(px)
RAW_TABLE_ROWS = 10           # 원본 VOC 원문 표 높이 = N줄 + 헤더
TICKET_CARDS_PER_ROW = 3      # 발행된 티켓 카드 한 줄 개수
HOVER_LINE_CHARS = 40         # 의미 공간 툴팁 원문 줄바꿈 글자 수
HOVER_MAX_CHARS = 120         # 의미 공간 툴팁 원문 최대 글자 수 (넘으면 …)
DATE_INPUT_WIDTH = 260        # 1️⃣ 기간 날짜 칸 고정 폭(px) — "2026/10/06 – 2026/10/20"이 끝까지 보이는 정도, 더 길게 하지 않는다
TICKET_CARD_HEIGHT = 220      # 티켓 카드 높이(px) 고정 — ID 줄 + 문제점 최대 2줄 + 담당 줄 + 날짜 줄이 스크롤 없이 들어가는 높이
# ── 기간 · 대용량 대비 (VOC가 수십만 건까지 커질 수 있다고 본다) ──
RECENT_PERIOD_DAYS = 14       # 기간 '최근 2주' = 오늘 − 14일 ~ 오늘 (데이터 마지막 날짜가 아니라 오늘 기준)
CLUSTER_MAX_VOCS = 5000       # 클러스터 분석 상한: 선택 기간 + 이전 기간 VOC 수 (LLM 호출 수가 건수에 비례)
TABLE_MAX_ROWS = 1000         # VOC 원문 표에 넘기는 최대 행 수 (최신순)
WEEKLY_CHART_MIN_DAYS = 90    # 선택 기간이 이 일수를 넘으면 주별로 묶어 그린다
SIMILARITY_MAX_POINTS = 2000  # VOC 유사도(의미 공간)에 그리는 최대 VOC 수 — 넘으면 무작위 표본(PROJECTION_SEED)
CELL_MAX_CHARS = 40           # 표(st.dataframe) 셀 글자 수 상한 — 넘으면 화면용 복사본만 앞 N자 + '…' (행 클릭 시 표 아래 전체)
APP_FONT = '"Source Sans 3", "Source Sans Pro", sans-serif'  # 앱 전체 글꼴 — CSS·st.html 공통 (코드 뷰만 고정폭)

# ── 2️⃣ 의심 위치 — 폴더 후보 ─────────────────────────
DEFAULT_DEPLOY_BRANCH = "main"  # repos.yaml에 branch가 없고 .git/HEAD도 못 읽을 때
FOLDER_CARD_COLUMNS = 3          # 폴더 카드 한 줄 개수
FOLDER_CARD_MAX_FILES = 5        # 폴더당 보여 줄 파일 최대 수
FILE_STATUS_ORDER = ["원인 후보", "함께 확인", "키워드 일치", "감점"]  # 폴더 카드에 그리는 상태 (위가 우선 적용·먼저 표시)
FILE_STATUS_NONE = "일반"       # 상태 없는 파일 — build_folder_cards에서 걸러 내고 그리지 않는다
FILE_STATUS_STYLE = {           # 폴더 후보 카드의 파일 알약 모양 (CSS)
    "원인 후보": "background:#E8622C; color:#FFFFFF; font-weight:700; border:1px solid #E8622C;",
    "함께 확인": "background:#F9D3C7; color:#202124; font-weight:700; border:1px solid #E8622C;",
    "키워드 일치": "background:#FFB547; color:#202124; font-weight:700; border:1px solid #FFB547;",
    "감점": "background:#FFFFFF; color:#D93025; border:1px dashed #D93025;",
}

# ── VOC 유사도 · 의미 공간 (임베딩 · t-SNE 3D 투영) ───────────────
VOC_EMBEDDING_MODEL = "text-embedding-3-large"  # VOC 유사도·임베딩 클러스터링용 (RAG는 EMBEDDING_MODEL)
VOC_EMBEDDINGS_TEMPLATE = "voc_embeddings_{dataset}_{model}.json"  # outputs/ 안 {voc_id: 벡터} 캐시 (데이터셋·모델별)
EMBED_BATCH_SIZE = 100
PROJECTION_SEED = 42
TSNE_PERPLEXITY = 30
SIMILAR_TOP_K = 5             # 선택한 VOC와 가장 비슷한 VOC 수
MAP_POINT_SIZE = 2.5          # 의미 공간 VOC 점 크기
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
W_KEYWORD = 0.3
W_SEMANTIC = 0.3          # 함수 의미(LLM 요약 임베딩) ↔ 티켓 유사도
W_RECENT = 0.4
# 함수 의미 점수: 함수마다 LLM이 '실제 동작'을 한 줄로 요약 → 임베딩 → 티켓 원인 문장과 코사인 유사도.
# 코드 원문 임베딩 점수도 같이 계산해 비교용으로만 보여 준다 (점수식에는 넣지 않음).
CODE_EMBEDDING_MODEL = "text-embedding-3-large"
CODE_UNITS_TEMPLATE = "code_units_{repo}_{model}.json"  # outputs/ 안 {파일::함수: {hash, summary, 벡터}} 캐시
SUMMARY_BATCH_SIZE = 20       # 함수 요약 LLM 1회에 넣는 함수 수
CODE_UNIT_MAX_CHARS = 2000    # 함수 하나를 요약·임베딩할 때 넣는 최대 글자 수
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
    "route_repos": "④ 레포지토리 라우팅",
    "score_folders": "⑤ 폴더 의심도",
    "search_lines": "⑥ 라인 탐색",
    "verify_lines": "⑥ 라인 검증",
    "add_commit_evidence": "⑥ 커밋 근거",
    "analyze_impact": "⑦ 영향 범위",
}
