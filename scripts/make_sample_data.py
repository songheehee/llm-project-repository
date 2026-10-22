"""Step 0: 레포 레지스트리(repos.yaml) · 개발자 목록(users.yaml) · RAG 문서(rag/*.md)를 data/ 아래에 생성한다.

실행: python scripts/make_sample_data.py
VOC 데이터는 만들지 않는다 — config.VOCS_CSV가 가리키는 CSV를 쓴다 (예: data/generate_vocs.py로 만든 vocs.csv).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


REPOS_YAML = """\
# VOCAI 레포 레지스트리 — 이 목록 밖의 저장소는 분석하지 않는다 (교육용 가상 데이터)
repos:
  - name: pulsefit-py
    url: https://github.com/SeungHyunLim00/test_repo_VOCAI.git  # 원격 미러 우선, 실패하면 path
    path: sample_repos/pulsefit-py
    product: 워치 앱
    description: >-
      PulseFit 워치 본체에서 도는 앱. 심박 센서(PPG)·혈중 산소 측정, 운동 세션 기록, 거리 계산,
      걸음 수·층수, 수면 측정, 알림 표시·진동 요청, 화면(UI) 표시, 배터리·전원 관리,
      앱 실행 관리(워치독·메모리), 기록 저장, 폰과의 블루투스 동기화 클라이언트를 담당한다.
    owner_parts: [센서 파트, 운동 파트, UI 파트, 전력 파트, 동기화 파트, 헬스 파트, 알림 파트, 플랫폼 파트]
    folder_roles:
      "ui/": display_only
      "watchfaces/": display_only
      "i18n/": display_only

  - name: pulsefit-sync
    path: sample_repos/pulsefit-sync
    product: 폰 연동
    description: >-
      워치와 스마트폰 사이 블루투스 연결·재연결, 데이터 전송 큐와 동기화 예약,
      폰 알림을 워치로 넘기는 알림 전달(relay)을 담당하는 모듈.
      연결 끊김 시 재연결 재시도 루프가 여기에 있다.
    owner_parts: [동기화 파트, 알림 파트]
    folder_roles: {}

  - name: pulsefit-mobile
    path: sample_repos/pulsefit-mobile
    product: 폰 앱
    description: >-
      스마트폰에 설치하는 PulseFit 앱. 폰 교체 시 예전 기록 가져오기, 워치·폰 걸음 수 합치기,
      체중 등 프로필 입력, 홈·설정 화면 표시를 담당한다. 워치 하드웨어를 직접 제어하지 않는다.
    owner_parts: [모바일 파트]
    folder_roles:
      "screens/": display_only

  - name: pulsefit-server
    path: sample_repos/pulsefit-server
    product: 클라우드 서버
    description: >-
      운동 기록을 받아 저장하고 통계·동기화 API를 제공하는 서버. 워치 하드웨어 동작과는 무관하다.
    owner_parts: [서버 파트]
    folder_roles: {}

  - name: pulsefit-firmware
    path: sample_repos/pulsefit-firmware
    product: 워치 펌웨어
    description: >-
      워치 하드웨어 칩 제어 펌웨어. 충전 접점 감지·충전 전류·만충 전압(배터리 셀 보호), 진동 모터 세기, GPS(GNSS) 수신 칩
      시작, 착용 감지(근접 센서로 주기적 재확인), 피부 온도 측정(측정·보고만 함)을 담당한다.
    owner_parts: [하드웨어 파트]
    folder_roles:
      "thermal/": report_only

  - name: pulsefit-web
    path: sample_repos/pulsefit-web
    product: 웹 스토어
    description: >-
      PulseFit 브랜드 사이트와 온라인 스토어(장바구니·쿠폰·배너). 워치·폰 앱 동작과는 무관하다.
    owner_parts: [웹 파트]
    folder_roles: {}
"""

PRODUCTS_MD = """\
# PulseFit 제품 정보 (사내 문서 · 교육용 가상 데이터)

## 제품 구성
- **워치 앱**: 손목에 차는 PulseFit 워치에서 동작하는 앱. 저장소 `pulsefit-py`.
- **폰 연동**: 워치와 스마트폰을 블루투스로 잇는 모듈. 저장소 `pulsefit-sync`.
- **클라우드 서버**: 운동 기록 저장과 통계·동기화 API. 저장소 `pulsefit-server`.
- **폰 앱**: 스마트폰에 설치하는 PulseFit 앱. 저장소 `pulsefit-mobile`.
- **워치 펌웨어**: 충전·진동 모터·GPS 칩 등 하드웨어 제어. 저장소 `pulsefit-firmware`.
- **웹 스토어**: 브랜드 사이트와 온라인 스토어. 저장소 `pulsefit-web`. 기기 VOC와는 무관하다.
- 스트랩·케이스 같은 기구(하드웨어 외형)는 코드 저장소가 없다.

## 워치 앱 기능 ↔ 폴더 (pulsefit-py)
- 심박 측정: `sensor/` — 광학 심박 센서(PPG)를 켜고 값을 읽는다. 운동 중에는 연속 측정을 한다. 혈중 산소(SpO2)도 여기.
- 운동 기록: `workout/` — 운동 세션 시작·종료, 거리 계산(GPS 보정 포함).
- 활동량: `activity/` — 가속도로 걸음 수를 세고, 기압계로 오른 층수를 센다. 하루 걸음 수 초기화.
- 수면: `sleep/` — 손목 움직임으로 잠든·깬 시각 감지, 수면 단계와 수면 점수.
- 알림: `notify/` — 폰에서 넘어온 알림을 배너·진동으로 내보낸다. 방해 금지 모드.
- 시스템: `system/` — 워치독(멈추면 재부팅), 메모리 부족 시 앱 종료.
- 화면 표시: `ui/` — 화면 구성, 경고 문구 표시. 하드웨어를 직접 제어하지 않는 표시 전용 계층이다.
- 워치 페이스·문구: `watchfaces/`, `i18n/` — 시계 화면 디자인과 번역 문구. 표시 전용.
- 전원 관리: `power/` — 배터리 잔량, 화면 켜짐 유지 시간, 절전 모드.
- 기록 저장: `storage/` — 측정 기록을 워치 내부에 저장.
- 동기화 클라이언트: `sync/` — 폰 연동 모듈과 통신하는 워치 쪽 클라이언트.

## 폰 연동 기능 ↔ 폴더 (pulsefit-sync)
- `bluetooth/` — BLE 연결, 연결 끊김 시 재연결 재시도 루프.
- `transfer/` — 워치 → 폰 전송 큐, 동기화 예약(언제 보낼지).
- `relay/` — 폰 알림을 워치로 전달. 묶음 전송 시간 창, 수신 확인 후 재전송.
- `protocol/` — 패킷 틀·CRC.

## 폰 앱 · 서버 · 펌웨어 기능 ↔ 폴더
- `pulsefit-mobile`: `sync/`(폰 교체 시 예전 기록 가져오기) · `health/`(워치·폰 걸음 수 합치기, 체중·프로필) · `screens/`(표시 전용) · `analytics/`(사용 통계).
- `pulsefit-server`: `api/`(업로드·통계·동기화, 중복 기록 판정) · `db/` · `auth/` · `jobs/`(리포트 메일).
- `pulsefit-firmware`: `charging/`(충전 접점 감지·충전 전류·온도 보호·만충 전압) · `haptics/`(진동 모터) · `gnss/`(GPS 칩 시작·보조 위성 정보) · `wear/`(착용 감지 — 근접 센서 적외선 LED로 주기적 재확인) · `thermal/`(온도 측정·보고만 함, 발열 원인 아님) · `boot/`.

## 사내 용어
- **PPG**: 광학 심박 센서. LED를 켜서 혈류를 읽는다. 켜져 있는 시간이 길수록 전력과 열이 늘어난다.
- **폴링 주기(측정 주기, 샘플링 주기)**: 센서를 깨워 값을 읽는 간격(초). 값이 작을수록 센서가 자주 깨어나 전력 소모와 발열이 커진다.
- **연속 측정**: 운동 세션 동안 PPG를 계속 켜 두고 폴링 주기마다 값을 읽는 모드.
- **재연결 재시도 루프**: 폰과 연결이 끊겼을 때 다시 붙으려고 반복 시도하는 루프. 대기 없이 돌면 무선 칩이 쉬지 못한다.
- **화면 켜짐 유지 시간**: 손목을 들었을 때 화면이 켜져 있는 시간. 길수록 배터리 소모가 커진다.
- **GPS 보정**: 위성 신호가 약할 때 거리 값을 보정하는 로직.
- **AGPS(보조 위성 정보)**: 폰에서 받아 GPS를 빨리 잡게 돕는 정보. 오래되면 콜드 스타트라 위치 잡는 데 오래 걸린다.
- **디바운스**: 접점·버튼 신호가 일정 시간 유지돼야 인정하는 처리. 너무 짧으면 상태가 깜빡인다.
- **묶음 전송 시간 창**: 알림·데이터를 모았다가 한 번에 보내는 대기 시간. 길면 늦게 도착한다.
- **워치독**: 메인 루프가 정해진 시간 안에 응답하지 않으면 기기를 다시 시작하는 장치.
"""

DEPARTMENTS_MD = """\
# 개발 파트 안내 (사내 문서 · 교육용 가상 데이터)

## 센서 파트
- 담당: 심박 센서(PPG) 드라이버, 측정 주기(폴링 주기), 연속 측정, 신호 필터
- 저장소·폴더: `pulsefit-py` / `sensor/`
- 이런 VOC가 오면: 운동 중 발열, 심박 수치 이상, 센서 과다 동작

## 전력 파트
- 담당: 배터리 잔량 계산, 화면 켜짐 유지 시간, 절전 모드
- 저장소·폴더: `pulsefit-py` / `power/`
- 이런 VOC가 오면: 배터리 급소모 (충전이 안 되는 문제는 하드웨어 파트)

## UI 파트
- 담당: 화면 구성, 메뉴·버튼 배치, 경고 문구 표시 (표시만 하고 하드웨어는 제어하지 않음)
- 저장소·폴더: `pulsefit-py` / `ui/`
- 이런 VOC가 오면: 버튼·글씨 크기, 메뉴 찾기 어려움

## 운동 파트
- 담당: 운동 세션 관리, 거리 계산, GPS 보정
- 저장소·폴더: `pulsefit-py` / `workout/`
- 이런 VOC가 오면: 거리 오차, 운동 기록 누락

## 동기화 파트
- 담당: 워치-폰 블루투스 연결, 연결 끊김 시 재연결 재시도, 데이터 전송
- 저장소·폴더: `pulsefit-py` / `sync/`, `pulsefit-sync` 전체
- 이런 VOC가 오면: 연결 끊김, 연결 끊김 상태의 발열·배터리 소모, 동기화 실패

## 서버 파트
- 담당: 운동 기록 저장 API, 통계 API, 데이터베이스
- 저장소·폴더: `pulsefit-server` 전체
- 이런 VOC가 오면: 앱에서 통계가 안 보임, 기록이 서버에 안 올라감, 같은 기록이 두 번 저장됨

## 헬스 파트
- 담당: 걸음 수·층수 계산, 수면 감지·수면 단계·수면 점수
- 저장소·폴더: `pulsefit-py` / `activity/`, `sleep/`
- 이런 VOC가 오면: 걸음 수 부정확, 수면 시간·단계 오류, 낮잠 미인식

## 알림 파트
- 담당: 폰 알림을 워치로 전달, 워치 알림 배너·진동 요청, 방해 금지 모드
- 저장소·폴더: `pulsefit-sync` / `relay/`, `pulsefit-py` / `notify/`
- 이런 VOC가 오면: 알림 지연·누락·중복, 방해 금지 중 진동

## 플랫폼 파트
- 담당: 앱 실행 관리, 워치독 재부팅, 메모리 부족 시 앱 종료
- 저장소·폴더: `pulsefit-py` / `system/`
- 이런 VOC가 오면: 앱 멈춤·강제 종료, 화면 멈춤, 혼자 재부팅

## 모바일 파트
- 담당: 폰 앱 화면, 폰 교체 시 기록 가져오기, 워치·폰 걸음 수 합치기, 프로필(체중) 입력
- 저장소·폴더: `pulsefit-mobile` 전체
- 이런 VOC가 오면: 폰 앱에 기록이 안 보임, 워치와 폰 숫자가 다름, 폰에서 입력한 값이 워치에 반영 안 됨

## 하드웨어 파트
- 담당: 충전 접점 감지·충전 전류·만충 전압(배터리 셀 보호), 진동 모터 세기, GPS 칩 시작(위치 잡는 시간), 착용 감지(근접 센서), 온도 측정
- 저장소·폴더: `pulsefit-firmware` 전체
- 이런 VOC가 오면: 충전 불량·느린 충전, 충전 중 발열, 배터리 부풂, 자는 동안 시계가 따뜻해짐(착용 감지), 진동이 약함, GPS 잡는 데 오래 걸림
- 참고: 온도 측정(thermal/)은 값을 읽어 알려 주기만 한다. 운동 중 발열은 센서 파트, 연결 끊김 발열은 동기화 파트

## 기구 파트
- 담당: 스트랩·케이스·버클 등 외형 설계 (코드 저장소 없음)
- 이런 VOC가 오면: 착용감, 스트랩 피부 트러블, 스트랩 고리·교체 버튼

## 웹 파트
- 담당: 브랜드 사이트, 온라인 스토어(장바구니·쿠폰·배너)
- 저장소·폴더: `pulsefit-web` 전체
- 이런 VOC가 오면: 구매·결제·쿠폰 문의 (기기 사용 VOC는 해당 없음)
"""


# 개발자 목록 — 파트별 이름 (교육용 가상 인물). 사람 목록은 users.yaml 한 곳 (사이드바 '현재 개발자' — 로그인 대신)
USER_PARTS = [
    ("센서 파트", ["김서준", "이수아", "박도윤"]), ("전력 파트", ["최하린", "정민재"]), ("UI 파트", ["강예은", "조현우"]),
    ("운동 파트", ["윤채원", "장우진"]), ("동기화 파트", ["임서윤", "한지훈", "오다인"]), ("서버 파트", ["신유준", "권나연"]),
    ("헬스 파트", ["황시윤", "송지안"]), ("알림 파트", ["전하준", "홍예린"]), ("플랫폼 파트", ["고은우", "문소율"]),
    ("모바일 파트", ["양지호", "배서현"]), ("하드웨어 파트", ["백승민", "노하윤"]), ("기구 파트", ["남태오", "서지유"]),
    ("웹 파트", ["유건우", "류가은"]),
    ("SCSA", ["강은성", "윤지현", "이온유", "이종협", "이지호", "이하나", "임승현", "정동은", "정회수", "조명현",
              "천예은", "홍송희", "황지희"]),
]
SCSA_PART, SCSA_GRADE = "SCSA", "CL2"   # SCSA(대졸 신입 과정)는 모두 CL2
# 파트 순서대로 번갈아 쓰는 직급 패턴 — 파트의 첫 사람은 항상 CL3 이상, CL1~CL4가 섞이도록
GRADE_PATTERNS = [["CL4", "CL2", "CL1"], ["CL3", "CL1", "CL2"]]


def users_yaml() -> str:
    """users.yaml 내용 (id는 순서대로 u01…, 같은 입력이면 항상 같은 결과)."""
    lines = ["# 개발자 목록 (교육용 가상 인물) — 사람 목록은 이 파일 한 곳. 사이드바 '현재 개발자'로 고른다 (로그인 대신)",
             "# grade: CL1 사원급 · CL2 대리·선임급(대졸 신입 포함) · CL3 과장·차장급(책임급) · CL4 부장급(수석급) — config.GRADES",
             "users:"]
    n = 0
    for i, (part, names) in enumerate(USER_PARTS):
        for j, name in enumerate(names):
            n += 1
            grade = SCSA_GRADE if part == SCSA_PART else GRADE_PATTERNS[i % 2][j % 3]
            lines.append(f"  - {{id: u{n:02d}, name: {name}, part: {part}, grade: {grade}}}")
    return "\n".join(lines) + "\n"



def main() -> None:
    config.RAG_DIR.mkdir(parents=True, exist_ok=True)
    config.REPOS_YAML.write_text(REPOS_YAML, encoding="utf-8")
    (config.RAG_DIR / "products.md").write_text(PRODUCTS_MD, encoding="utf-8")
    (config.RAG_DIR / "departments.md").write_text(DEPARTMENTS_MD, encoding="utf-8")
    config.USERS_YAML.write_text(users_yaml(), encoding="utf-8")
    print("repos.yaml · users.yaml · rag/products.md · rag/departments.md 생성 완료 (VOC CSV는 따로 준비 — config.VOCS_CSV)")


if __name__ == "__main__":
    main()
