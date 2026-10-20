"""Step 0: 교육용 가상 VOC · 레포 레지스트리 · RAG 문서를 data/ 아래에 생성한다.

실행: python scripts/make_sample_data.py
random.seed(42)로 매번 같은 결과가 나온다.
"""

import csv
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

# ── 클러스터별 문장 재료 (고객 말투, 코드 용어 금지, 클러스터끼리 표현이 겹치지 않게) ──

WORKOUT_HEAT = {
    "label": "운동 중 기기 발열",
    "n_analysis": 91,
    "n_prev": 10,
    "cores": [
        "업데이트하고 러닝하면 손목이 뜨거워요",
        "운동 시작하고 10분만 지나면 시계 뒷면이 너무 뜨거워요",
        "자전거 탈 때 워치가 달아올라서 손목이 따가워요",
        "헬스장에서 운동 모드 켜면 시계가 뜨끈뜨끈해져요",
        "달리기 기록 켜 두면 손목에 열감이 확 올라와요",
        "등산하면서 운동 측정했더니 시계가 손난로처럼 뜨거웠어요",
        "업데이트 이후로 운동만 하면 손목이 화끈거려요",
        "줄넘기 운동 중에 시계가 너무 뜨거워서 풀어 버렸어요",
        "러닝 30분 했는데 손목에 빨갛게 자국이 남았어요",
        "운동할 때마다 시계 뒤판이 뜨거워서 깜짝 놀라요",
        "수영 말고 실내 사이클 할 때도 시계가 엄청 뜨거워져요",
        "인터벌 트레이닝 하면 워치가 데일 것처럼 뜨거워요",
        "요가 세션만 켜도 손목이 후끈해져요",
        "마라톤 연습 중에 시계 때문에 손목이 너무 뜨거웠어요",
    ],
    "prefixes": ["", "요즘 ", "어제 ", "이번 주에 ", "솔직히 ", "아니 "],
    "endings": ["", " 화상 입을까 봐 무서워요.", " 운동을 못 하겠어요.", " 이거 정상인가요?", " 빨리 고쳐 주세요.", " 아이도 쓰는데 걱정돼요."],
}

BATTERY = {
    "label": "배터리 급소모",
    "n_analysis": 58,
    "n_prev": 36,
    "cores": [
        "하루도 못 가고 배터리가 바닥나요",
        "아침에 차고 나가면 점심도 안 돼서 배터리가 다 닳아요",
        "충전하고 반나절이면 배터리가 다 닳아요",
        "배터리가 예전보다 두 배는 빨리 닳아요",
        "배터리가 빨리 닳아서 하루를 못 버텨요",
        "배터리가 한두 시간 만에 훅 닳아 버려요",
        "가만히 차고만 있어도 배터리가 쭉쭉 닳아요",
        "배터리가 빨리 닳아서 충전기를 매일 두 번씩 꽂아야 해요",
        "배터리 오래간다고 해서 샀는데 하루를 못 버텨요",
        "배터리가 금방 닳아서 저녁이면 꺼져 있어요",
        "배터리가 금방금방 닳아서 하루 종일 못 써요",
        "주말 여행 갔다가 배터리가 다 닳아서 시계를 못 썼어요",
        "절전 모드를 켜도 배터리가 금방 닳아요",
    ],
    "prefixes": ["", "요새 ", "최근에 ", "며칠째 ", "진짜 "],
    "endings": ["", " 너무 불편해요.", " 충전이 귀찮아요.", " 배터리 불량인가요?", " 개선 부탁드려요."],
}

DISCONNECT_HEAT = {
    "label": "연결 끊김 시 발열",
    "n_analysis": 24,
    "n_prev": 0,
    "cores": [
        "폰이랑 연결이 끊기면 시계가 따뜻해져요",
        "휴대폰을 집에 두고 나오면 연결 끊김 표시가 뜨고 시계가 뜨거워져요",
        "블루투스 연결이 끊긴 상태로 두면 손목이 뜨거워져요",
        "폰과 연결 끊김 알림 뜬 뒤로 시계가 계속 뜨끈해요",
        "폰 배터리가 나가서 연결이 끊기니까 워치가 뜨거워지더라고요",
        "연결 끊김 상태에서 시계가 손목을 데울 만큼 달아올라요",
        "폰을 멀리 두면 연결이 끊기면서 시계가 점점 뜨거워져요",
        "비행기 모드 켜서 폰 연결이 끊기면 워치에서 열이 나요",
        "연결이 끊긴 채로 있으면 시계가 뜨거워서 손목이 화끈해요",
        "폰이랑 연결 끊김이 생기면 시계가 혼자 열을 내요",
        "지하철에서 폰 연결이 끊기는 순간부터 시계가 데워져요",
        "회사에 폰 두고 회의 갔더니 연결 끊김 뜨고 손목이 뜨끈했어요",
    ],
    "prefixes": ["", "오늘 ", "어제 ", "자꾸 "],
    "endings": ["", " 이상해요.", " 왜 그런 거죠?", " 무서워서 풀어 놨어요.", " 데일까 봐 걱정돼요."],
}

SCREEN_UX = {
    "label": "화면 사용성",
    "n_analysis": 29,
    "n_prev": 29,
    "cores": [
        "메뉴 버튼이 너무 작아요",
        "메뉴 버튼이 작아서 자꾸 옆 버튼이 눌려요",
        "설정 메뉴가 너무 깊이 숨어 있어요",
        "메뉴를 몇 번 넘겨야 원하는 기능이 나와요",
        "메뉴 아이콘이 다 비슷하게 생겨서 헷갈려요",
        "알림 지우는 버튼을 메뉴에서 찾기 어려워요",
        "위젯 순서 바꾸는 메뉴를 못 찾겠어요",
        "메뉴에서 뒤로 가기 버튼이 어디 있는지 모르겠어요",
        "메뉴 배치가 직관적이지 않아요",
        "메뉴 이름이 무슨 뜻인지 모르겠어요",
        "메뉴 버튼 글씨가 작아서 읽기 힘들어요",
        "메뉴 목록 스크롤이 너무 민감해서 자꾸 지나쳐요",
        "원하는 메뉴까지 가는 단계가 너무 많아요",
    ],
    "prefixes": ["", "음 ", "개인적으로 ", "부모님이 쓰시는데 "],
    "endings": ["", " 조금만 크게 해 주세요.", " 불편해요.", " 개선되면 좋겠어요."],
}

GPS_DISTANCE = {
    "label": "GPS 거리 오차",
    "n_analysis": 17,
    "n_prev": 17,
    "cores": [
        "달린 거리가 실제보다 짧게 나와요",
        "트랙 5바퀴 돌았는데 1.6km로 찍혀요",
        "걷기 거리가 지도 앱이랑 500m나 차이 나요",
        "위치가 지그재그로 잡혀서 거리가 엉망이에요",
        "공원 한 바퀴가 매번 다른 거리로 기록돼요",
        "10km 대회 나갔는데 9.2km로 나왔어요",
        "터널 지나면 거리가 엉뚱하게 튀어요",
        "러닝 코스 거리랑 비교하면 시계 거리가 항상 모자라요",
        "산책 코스 거리가 지도 앱보다 짧게 나와요",
        "같은 길을 걸어도 거리가 들쭉날쭉해요",
        "출발하고 한참 지나야 거리가 올라가기 시작해요",
        "실제로 달린 것보다 거리가 적게 쌓여요",
    ],
    "prefixes": ["", "음 ", "아쉽게도 ", "매번 "],
    "endings": ["", " 정확하게 해 주세요.", " 기록이 의미가 없어요.", " 아쉬워요."],
}

CLUSTERS = [WORKOUT_HEAT, BATTERY, DISCONNECT_HEAT, SCREEN_UX, GPS_DISTANCE]


def make_text(spec: dict, rng: random.Random, i: int) -> str:
    """문장 템플릿 + 앞말 + 어미를 섞어 한 건의 VOC 문장을 만든다."""
    core = spec["cores"][i % len(spec["cores"])]
    prefix = rng.choice(spec["prefixes"])
    ending = rng.choice(spec["endings"])
    text = f"{prefix}{core}."
    if ending:
        text += ending
    return text.replace("..", ".")


def make_vocs(rng: random.Random) -> list[dict]:
    rows = []
    periods = [(config.ANALYSIS_START, "n_analysis"), (config.PREV_START, "n_prev")]
    for spec in CLUSTERS:
        for start, key in periods:
            for i in range(spec[key]):
                day = start + timedelta(days=i % 12)
                ts = datetime(day.year, day.month, day.day,
                              rng.randint(0, 23), rng.randint(0, 59), rng.randint(0, 59))
                rows.append({
                    "created_at": ts.isoformat(),
                    "channel": "call_center" if i % 3 == 2 else "review",
                    "text": make_text(spec, rng, i),
                    "true_label": spec["label"],
                })
    rows.sort(key=lambda r: r["created_at"])
    for n, row in enumerate(rows, start=1):
        row["voc_id"] = f"V{n:04d}"
    return rows


REPOS_YAML = """\
# VOCAI 레포 레지스트리 — 이 목록 밖의 저장소는 분석하지 않는다 (교육용 가상 데이터)
repos:
  - name: pulsefit-py
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
      워치 하드웨어 칩 제어 펌웨어. 충전 접점 감지·충전 전류, 진동 모터 세기, GPS(GNSS) 수신 칩
      시작, 착용 감지, 피부 온도 측정(측정·보고만 함)을 담당한다.
    owner_parts: [하드웨어 파트]
    folder_roles: {}

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
- `pulsefit-firmware`: `charging/`(충전 접점 감지·충전 전류·온도 보호) · `haptics/`(진동 모터) · `gnss/`(GPS 칩 시작·보조 위성 정보) · `wear/`(착용 감지) · `thermal/`(온도 측정·보고만 함, 발열 원인 아님) · `boot/`.

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
- 담당자: 김서준
- 담당: 심박 센서(PPG) 드라이버, 측정 주기(폴링 주기), 연속 측정, 신호 필터
- 저장소·폴더: `pulsefit-py` / `sensor/`
- 이런 VOC가 오면: 운동 중 발열, 심박 수치 이상, 센서 과다 동작

## 전력 파트
- 담당자: 박지민
- 담당: 배터리 잔량 계산, 화면 켜짐 유지 시간, 절전 모드
- 저장소·폴더: `pulsefit-py` / `power/`
- 이런 VOC가 오면: 배터리 급소모 (충전이 안 되는 문제는 하드웨어 파트)

## UI 파트
- 담당자: 이하은
- 담당: 화면 구성, 메뉴·버튼 배치, 경고 문구 표시 (표시만 하고 하드웨어는 제어하지 않음)
- 저장소·폴더: `pulsefit-py` / `ui/`
- 이런 VOC가 오면: 버튼·글씨 크기, 메뉴 찾기 어려움

## 운동 파트
- 담당자: 정우진
- 담당: 운동 세션 관리, 거리 계산, GPS 보정
- 저장소·폴더: `pulsefit-py` / `workout/`
- 이런 VOC가 오면: 거리 오차, 운동 기록 누락

## 동기화 파트
- 담당자: 최유나
- 담당: 워치-폰 블루투스 연결, 연결 끊김 시 재연결 재시도, 데이터 전송
- 저장소·폴더: `pulsefit-py` / `sync/`, `pulsefit-sync` 전체
- 이런 VOC가 오면: 연결 끊김, 연결 끊김 상태의 발열·배터리 소모, 동기화 실패

## 서버 파트
- 담당자: 한도윤
- 담당: 운동 기록 저장 API, 통계 API, 데이터베이스
- 저장소·폴더: `pulsefit-server` 전체
- 이런 VOC가 오면: 앱에서 통계가 안 보임, 기록이 서버에 안 올라감, 같은 기록이 두 번 저장됨

## 헬스 파트
- 담당자: 윤서아
- 담당: 걸음 수·층수 계산, 수면 감지·수면 단계·수면 점수
- 저장소·폴더: `pulsefit-py` / `activity/`, `sleep/`
- 이런 VOC가 오면: 걸음 수 부정확, 수면 시간·단계 오류, 낮잠 미인식

## 알림 파트
- 담당자: 강민호
- 담당: 폰 알림을 워치로 전달, 워치 알림 배너·진동 요청, 방해 금지 모드
- 저장소·폴더: `pulsefit-sync` / `relay/`, `pulsefit-py` / `notify/`
- 이런 VOC가 오면: 알림 지연·누락·중복, 방해 금지 중 진동

## 플랫폼 파트
- 담당자: 조예린
- 담당: 앱 실행 관리, 워치독 재부팅, 메모리 부족 시 앱 종료
- 저장소·폴더: `pulsefit-py` / `system/`
- 이런 VOC가 오면: 앱 멈춤·강제 종료, 화면 멈춤, 혼자 재부팅

## 모바일 파트
- 담당자: 임지호
- 담당: 폰 앱 화면, 폰 교체 시 기록 가져오기, 워치·폰 걸음 수 합치기, 프로필(체중) 입력
- 저장소·폴더: `pulsefit-mobile` 전체
- 이런 VOC가 오면: 폰 앱에 기록이 안 보임, 워치와 폰 숫자가 다름, 폰에서 입력한 값이 워치에 반영 안 됨

## 하드웨어 파트
- 담당자: 오세훈
- 담당: 충전 접점 감지·충전 전류, 진동 모터 세기, GPS 칩 시작(위치 잡는 시간), 착용 감지, 온도 측정
- 저장소·폴더: `pulsefit-firmware` 전체
- 이런 VOC가 오면: 충전 불량·느린 충전, 진동이 약함, GPS 잡는 데 오래 걸림
- 참고: 온도 측정(thermal/)은 값을 읽어 알려 주기만 한다. 운동 중 발열은 센서 파트, 연결 끊김 발열은 동기화 파트

## 기구 파트
- 담당자: 신아름
- 담당: 스트랩·케이스·버클 등 외형 설계 (코드 저장소 없음)
- 이런 VOC가 오면: 착용감, 스트랩 피부 트러블, 스트랩 고리·교체 버튼

## 웹 파트
- 담당자: 문태오
- 담당: 브랜드 사이트, 온라인 스토어(장바구니·쿠폰·배너)
- 저장소·폴더: `pulsefit-web` 전체
- 이런 VOC가 오면: 구매·결제·쿠폰 문의 (기기 사용 VOC는 해당 없음)
"""


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rng = random.Random(42)
    random.seed(42)
    config.RAG_DIR.mkdir(parents=True, exist_ok=True)

    rows = make_vocs(rng)
    # 원본 311건 파일에 직접 쓴다 (config.VOCS_CSV는 다른 데이터셋을 가리킬 수 있음)
    write_csv(config.DATA_DIR / "vocs.csv", rows, ["voc_id", "created_at", "channel", "text"])
    write_csv(config.DATA_DIR / "answer_labels.csv", rows, ["voc_id", "true_label"])
    config.REPOS_YAML.write_text(REPOS_YAML, encoding="utf-8")
    (config.RAG_DIR / "products.md").write_text(PRODUCTS_MD, encoding="utf-8")
    (config.RAG_DIR / "departments.md").write_text(DEPARTMENTS_MD, encoding="utf-8")

    print(f"vocs.csv {len(rows)}건 · answer_labels.csv · repos.yaml · rag/*.md 생성 완료")


if __name__ == "__main__":
    main()
