"""Step 0: 분석 대상 가상 저장소 6개를 만든다.

  pulsefit-py       워치 앱        pulsefit-sync    폰 연동 (BLE·전송·알림 전달)
  pulsefit-mobile   폰 앱          pulsefit-server  클라우드 서버
  pulsefit-firmware 워치 펌웨어     pulsefit-web     브랜드 사이트·스토어 (VOC와 무관한 더미)

실행: python scripts/build_sample_repos.py
sample_repos/ 를 지우고 다시 만든다. 이 폴더 밖은 건드리지 않는다.
커밋 날짜·작성자는 환경 변수로 지정한다 (전역 git 설정에 기대지 않음).

VOC 주제(vocs_1000) ↔ 원인 코드 (교육용 정답 메모 — 앱·LLM에 넣지 않는다)
  운동 중 기기 발열      pulsefit-py       sensor/heart_rate.py      POLL_INTERVAL_S = 0.05 (10/02 커밋)
  연결 끊김 시 발열      pulsefit-sync     bluetooth/reconnect.py    RETRY_DELAY_S = 0 (09/28 커밋)
  배터리 소모            pulsefit-py       power/battery.py          SCREEN_ON_TIMEOUT_S = 8 (09/25 커밋)
  메뉴·UI 사용성         pulsefit-py       ui/screens.py             MENU_BUTTON_SIZE_PX = 28
  GPS 거리 오차          pulsefit-py       workout/distance.py       WEAK_SIGNAL_CORRECTION = 0.97
                         pulsefit-firmware gnss/receiver.py          AGPS_MAX_AGE_H (GPS 잡는 시간)
  심박·센서 측정 부정확  pulsefit-py       sensor/filters.py · sensor/spo2.py
  폰 앱 동기화 지연·누락 pulsefit-sync     transfer/scheduler.py     SYNC_ONLY_WHEN_APP_FOREGROUND (08/20 커밋)
                         pulsefit-server   api/sync.py               중복 판정 키 uploaded_at
  수면 측정 오류         pulsefit-py       sleep/tracker.py          SLEEP_ONSET_STILL_MIN · MIN_SLEEP_DURATION_MIN
  걸음 수 부정확         pulsefit-py       activity/pedometer.py     STEP_ACCEL_THRESHOLD_G = 0.8 (08/14 커밋) · utcnow
  알림 지연·누락         pulsefit-sync     relay/notification_forwarder.py  BATCH_WINDOW_S = 120 (09/12 커밋)
                         pulsefit-py       notify/dispatcher.py      방해 금지 중 진동 · 중복 미제거
  충전 불량              pulsefit-firmware charging/charger.py       CONTACT_DEBOUNCE_MS = 40 (08/27 커밋)
  앱 멈춤·강제 종료      pulsefit-py       system/watchdog.py · system/memory.py
  착용감·스트랩·피부     (코드 저장소 없음 — 기구 파트)
함정: pulsefit-py ui/warnings.py · i18n/ (발열 문구만), pulsefit-firmware thermal/ (온도 측정·보고만)
"""

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

AUTHORS = {
    "sensor": ("김센서", "sensor@pulsefit.example"),
    "power": ("박전력", "power@pulsefit.example"),
    "ui": ("이화면", "ui@pulsefit.example"),
    "sync": ("최동기", "sync@pulsefit.example"),
    "health": ("정수면", "health@pulsefit.example"),
    "notify": ("한알림", "notify@pulsefit.example"),
    "mobile": ("서모바", "mobile@pulsefit.example"),
    "server": ("오서버", "server@pulsefit.example"),
    "hw": ("장펌웨", "firmware@pulsefit.example"),
    "web": ("강웹", "web@pulsefit.example"),
    "init": ("PulseFit Bot", "bot@pulsefit.example"),
}

# ════════════════════════════ pulsefit-py ════════════════════════════

HEART_RATE_FINAL = '''"""심박 센서 측정 모듈.

워치의 광학 심박 센서(PPG)에서 값을 읽어 필터를 거쳐 콜백으로 전달한다.
"""

import threading
import time

from sensor.driver import PpgDriver
from sensor.filters import MovingAverageFilter

# 평상시 측정 주기 (초)
IDLE_INTERVAL_S = 5.0

# 운동 중 연속 측정
# 측정 주기 (초) — 값이 작을수록 센서가 자주 깨어난다
POLL_INTERVAL_S = 0.05


class HeartRateMonitor:
    """심박 센서를 켜고 끄며 측정값을 콜백으로 넘긴다."""

    def __init__(self, on_value):
        self.driver = PpgDriver()
        self.filter = MovingAverageFilter(window=5)
        self.on_value = on_value
        self._running = threading.Event()
        self._thread = None

    def measure_once(self):
        """평상시 1회 측정."""
        raw = self.driver.read()
        self.on_value(raw)

    def start_continuous(self):
        """운동 세션용 연속 측정을 시작한다."""
        self._running.set()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        """연속 측정 루프."""
        self.driver.power_on()
        try:
            while self._running.is_set():
                raw = self.driver.read()
                self.filter.push(raw)
                value = self.filter.value()
                if value is not None:
                    self.on_value(value)
                # 다음 측정까지 대기
                time.sleep(POLL_INTERVAL_S)
        finally:
            self.driver.power_off()

    def stop(self):
        """연속 측정을 멈춘다."""
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=1.0)
'''

HEART_RATE_INITIAL = HEART_RATE_FINAL.replace("POLL_INTERVAL_S = 0.05", "POLL_INTERVAL_S = 1.0")

DRIVER = '''"""PPG 센서 드라이버 (하드웨어 추상화 계층)."""

import random


class PpgDriver:
    """광학 심박 센서의 LED와 수광부를 제어한다."""

    def __init__(self, led_current_ma: float = 12.0):
        self.led_current_ma = led_current_ma
        self.powered = False
        self.read_count = 0

    def power_on(self):
        """센서 LED를 켠다. 켜져 있는 동안 전류가 계속 흐른다."""
        self.powered = True

    def power_off(self):
        """센서 LED를 끈다."""
        self.powered = False

    def read(self) -> float:
        """센서에서 원시 심박 값을 한 번 읽는다.

        꺼져 있으면 잠깐 켰다가 읽은 뒤 다시 끈다.
        """
        self.read_count += 1
        was_on = self.powered
        if not was_on:
            self.power_on()
        value = 60 + random.random() * 40
        if not was_on:
            self.power_off()
        return value
'''

FILTERS = '''"""센서 신호 필터."""

from collections import deque


class MovingAverageFilter:
    """최근 N개 값의 이동 평균을 낸다."""

    def __init__(self, window: int = 5):
        self.window = window
        self._values = deque(maxlen=window)

    def push(self, raw: float):
        """새 원시 값을 넣는다. 비정상 값은 버린다."""
        if raw <= 0 or raw > 250:
            return
        self._values.append(raw)

    def value(self):
        """창이 다 차면 평균을, 아니면 None을 돌려준다."""
        if len(self._values) < self.window:
            return None
        return sum(self._values) / len(self._values)

    def reset(self):
        self._values.clear()
'''

SESSION = '''"""운동 세션 관리."""

from datetime import datetime

from sensor.heart_rate import HeartRateMonitor
from workout.distance import DistanceTracker


class WorkoutSession:
    """운동 시작부터 종료까지 심박·거리를 기록한다."""

    def __init__(self, kind: str):
        self.kind = kind
        self.samples = []
        self.hr = HeartRateMonitor(on_value=self._on_heart_rate)
        self.distance = DistanceTracker()
        self.started_at = None

    def start(self):
        """운동을 시작한다."""
        self.started_at = datetime.now()
        self.distance.start()
        self.hr.start_continuous()

    def _on_heart_rate(self, value):
        self.samples.append(value)

    def finish(self):
        self.hr.stop()
        self.distance.stop()
        return {"kind": self.kind, "samples": len(self.samples)}
'''

DISTANCE = '''"""GPS 기반 이동 거리 계산."""

import math

# GPS 신호가 약할 때 보정 계수
WEAK_SIGNAL_CORRECTION = 0.97
EARTH_RADIUS_M = 6_371_000


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    """두 좌표 사이 거리(m)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


class DistanceTracker:
    """GPS 좌표를 받아 누적 거리를 계산한다."""

    def __init__(self):
        self.total_m = 0.0
        self._last = None
        self.active = False

    def start(self):
        self.active = True
        self._last = None

    def add_fix(self, lat, lon, weak_signal=False):
        if not self.active:
            return
        if self._last is not None:
            step = haversine_m(*self._last, lat, lon)
            if weak_signal:
                step *= WEAK_SIGNAL_CORRECTION
            self.total_m += step
        self._last = (lat, lon)

    def stop(self):
        self.active = False
        return self.total_m
'''

SYNC_BLUETOOTH = '''"""워치 쪽 블루투스 동기화 클라이언트."""

import json

# 한 번에 보낼 최대 기록 수
BATCH_LIMIT = 20


class SyncClient:
    """폰 연동 모듈(pulsefit-sync)로 기록을 보낸다."""

    def __init__(self, transport):
        self.transport = transport
        self.pending = []

    def enqueue(self, record: dict):
        self.pending.append(record)

    def flush(self) -> int:
        """대기 중인 기록을 묶어서 보낸다. 연결이 없으면 다음에 보낸다."""
        if not self.transport.connected:
            return 0
        sent = 0
        while self.pending:
            batch = self.pending[:BATCH_LIMIT]
            self.transport.send(json.dumps(batch, ensure_ascii=False))
            del self.pending[:BATCH_LIMIT]
            sent += len(batch)
        return sent
'''

WARNINGS_INITIAL = '''"""화면 경고 문구 모음 — 표시만 담당한다 (하드웨어 제어 없음)."""

LOW_BATTERY_TEXT = "배터리가 부족합니다."
DISCONNECTED_TEXT = "휴대폰과 연결이 끊겼습니다."
STORAGE_FULL_TEXT = "저장 공간이 부족합니다."


def show_low_battery(screen):
    """배터리 부족 배너를 띄운다."""
    screen.show_banner(LOW_BATTERY_TEXT, level="info")


def show_disconnected(screen):
    """연결 끊김 배너를 띄운다."""
    screen.show_banner(DISCONNECTED_TEXT, level="info")


def show_storage_full(screen):
    screen.show_banner(STORAGE_FULL_TEXT, level="info")
'''

WARNINGS_FINAL = '''"""화면 경고 문구 모음 — 표시만 담당한다 (하드웨어 제어 없음)."""

LOW_BATTERY_TEXT = "배터리가 부족합니다."
DISCONNECTED_TEXT = "휴대폰과 연결이 끊겼습니다."
STORAGE_FULL_TEXT = "저장 공간이 부족합니다."

# 발열 경고 문구 — heat warning
HEAT_WARNING_TITLE = "기기 발열 주의"
HEAT_WARNING_BODY = "기기 온도가 높습니다. 발열이 계속되면 잠시 시계를 풀어 주세요."
HEAT_WARNING_SEVERE = "심한 발열이 감지되었습니다. 운동을 멈추고 시계를 풀어 주세요."
HEAT_WARNING_TEMPERATURE_C = 41      # 이 temperature 이상이면 발열 경고 표시
HEAT_SEVERE_TEMPERATURE_C = 45       # 이 temperature 이상이면 심한 발열 경고 표시


def heat_warning_text(temperature_c: float):
    """temperature 값에 맞는 발열(heat) 경고 문구를 돌려준다. 문구만 고른다."""
    if temperature_c >= HEAT_SEVERE_TEMPERATURE_C:
        return f"{HEAT_WARNING_TITLE}: {HEAT_WARNING_SEVERE}"
    if temperature_c >= HEAT_WARNING_TEMPERATURE_C:
        return f"{HEAT_WARNING_TITLE}: {HEAT_WARNING_BODY}"
    return None


def show_heat_warning(screen, temperature_c: float):
    """발열 경고를 화면에 띄운다. temperature 값은 다른 모듈에서 받아 표시만 한다."""
    text = heat_warning_text(temperature_c)
    if text:
        screen.show_banner(text, level="heat")


def show_low_battery(screen):
    """배터리 부족 배너를 띄운다."""
    screen.show_banner(LOW_BATTERY_TEXT, level="info")


def show_disconnected(screen):
    """연결 끊김 배너를 띄운다."""
    screen.show_banner(DISCONNECTED_TEXT, level="info")


def show_storage_full(screen):
    screen.show_banner(STORAGE_FULL_TEXT, level="info")
'''

SCREENS = '''"""워치 화면 구성."""

# 메뉴 버튼 크기 (px)
MENU_BUTTON_SIZE_PX = 28
FONT_SIZE_PX = 14
MENU_ITEMS = ["운동", "심박", "기록", "설정"]


class Screen:
    """화면에 위젯과 배너를 그린다."""

    def __init__(self, width: int = 396, height: int = 484):
        self.width = width
        self.height = height
        self.banners = []
        self.widgets = []

    def show_banner(self, text: str, level: str = "info"):
        """화면 위쪽에 배너를 띄운다."""
        self.banners.append({"text": text, "level": level})

    def clear_banners(self):
        self.banners.clear()

    def layout_menu(self):
        """메뉴 버튼을 세로로 배치한다."""
        y = 40
        for item in MENU_ITEMS:
            self.widgets.append({"label": item, "y": y, "size": MENU_BUTTON_SIZE_PX})
            y += MENU_BUTTON_SIZE_PX + 8
        return self.widgets
'''

BATTERY_INITIAL = '''"""배터리 · 전원 관리."""

# 손목을 들었을 때 화면 켜짐 유지 시간 (초)
SCREEN_ON_TIMEOUT_S = 5
LOW_BATTERY_PERCENT = 15
POWER_SAVE_PERCENT = 5


class BatteryManager:
    """배터리 잔량을 읽고 절전 모드를 결정한다."""

    def __init__(self, gauge):
        self.gauge = gauge
        self.power_save = False

    def percent(self) -> int:
        return int(self.gauge.read_percent())

    def is_low(self) -> bool:
        return self.percent() <= LOW_BATTERY_PERCENT

    def update_power_mode(self):
        """잔량이 매우 낮으면 절전 모드로 들어간다."""
        self.power_save = self.percent() <= POWER_SAVE_PERCENT
        return self.power_save

    def screen_timeout(self) -> int:
        """절전 모드에서는 화면 켜짐 시간을 줄인다."""
        if self.power_save:
            return max(2, SCREEN_ON_TIMEOUT_S // 2)
        return SCREEN_ON_TIMEOUT_S
'''

BATTERY_FINAL = BATTERY_INITIAL.replace("SCREEN_ON_TIMEOUT_S = 5", "SCREEN_ON_TIMEOUT_S = 8")

STORAGE_RECORDS = '''"""워치 내부 기록 저장."""

import json
from pathlib import Path

MAX_RECORDS = 500


class RecordStore:
    """운동·심박 기록을 JSON 파일로 저장한다."""

    def __init__(self, path: str = "records.json"):
        self.path = Path(path)
        self.records = []

    def load(self):
        if self.path.exists():
            self.records = json.loads(self.path.read_text(encoding="utf-8"))
        return self.records

    def add(self, record: dict):
        """기록을 추가하고 오래된 기록은 지운다."""
        self.records.append(record)
        if len(self.records) > MAX_RECORDS:
            self.records = self.records[-MAX_RECORDS:]

    def save(self):
        self.path.write_text(json.dumps(self.records, ensure_ascii=False), encoding="utf-8")
'''

PY_README = "# pulsefit-py\n\nPulseFit 워치 앱 (교육용 가상 저장소).\n"

# ════════════════════════════ pulsefit-sync ════════════════════════════

RECONNECT_INITIAL = '''"""폰 연결이 끊겼을 때 다시 붙는 재연결 재시도 루프."""

import time

from bluetooth.link import BleLink

# 재연결 재시도 대기 (초)
RETRY_DELAY_S = 2.0


class Reconnector:
    """연결될 때까지 재연결을 시도한다."""

    def __init__(self, link: BleLink):
        self.link = link
        self.attempts = 0

    def run(self) -> int:
        """연결될 때까지 재시도하고 시도 횟수를 돌려준다."""
        while not self.link.is_connected():
            self.attempts += 1
            if self.link.try_connect():
                break
            time.sleep(RETRY_DELAY_S)
        return self.attempts
'''

RECONNECT_FINAL = '''"""폰 연결이 끊겼을 때 다시 붙는 재연결 재시도 루프."""

import time

from bluetooth.link import BleLink

# 재연결 재시도 대기 (초) — 빠른 재연결을 위해 대기 없이 바로 재시도
RETRY_DELAY_S = 0


class Reconnector:
    """연결될 때까지 재연결을 시도한다."""

    def __init__(self, link: BleLink):
        self.link = link
        self.attempts = 0
        self.started = time.monotonic()

    def run(self) -> int:
        """연결될 때까지 재시도하고 시도 횟수를 돌려준다."""
        while not self.link.is_connected():
            self.attempts += 1
            if self.link.try_connect():
                break
        return self.attempts
'''

BLE_LINK = '''"""블루투스 저전력(BLE) 링크."""


class BleLink:
    """폰과의 BLE 연결 상태를 관리한다."""

    def __init__(self, device_id: str):
        self.device_id = device_id
        self._connected = False
        self.rssi = None

    def is_connected(self) -> bool:
        return self._connected

    def try_connect(self) -> bool:
        """한 번 연결을 시도한다. 폰이 범위 밖이면 실패한다."""
        self._connected = self.rssi is not None and self.rssi > -90
        return self._connected

    def disconnect(self):
        self._connected = False

    def send(self, payload: bytes) -> int:
        if not self._connected:
            raise ConnectionError("연결되어 있지 않습니다")
        return len(payload)
'''

TRANSFER = '''"""워치 → 폰 데이터 전송 큐."""

from collections import deque

MAX_QUEUE = 1000


class TransferQueue:
    """연결이 끊겨 있는 동안 보낼 데이터를 쌓아 둔다."""

    def __init__(self, link):
        self.link = link
        self.queue = deque(maxlen=MAX_QUEUE)

    def put(self, payload: bytes):
        self.queue.append(payload)

    def drain(self) -> int:
        """연결되어 있으면 쌓인 데이터를 모두 보낸다."""
        sent = 0
        while self.queue and self.link.is_connected():
            self.link.send(self.queue.popleft())
            sent += 1
        return sent
'''

# ════════════════════════════ pulsefit-server ════════════════════════════

STATS_API = '''"""운동 통계 API."""

from db.records import RecordRepository


def weekly_summary(repo: RecordRepository, user_id: str) -> dict:
    """사용자의 최근 7일 운동 요약을 돌려준다."""
    rows = repo.recent(user_id, days=7)
    total_m = sum(r["distance_m"] for r in rows)
    avg_hr = sum(r["avg_hr"] for r in rows) / len(rows) if rows else None
    return {
        "user_id": user_id,
        "sessions": len(rows),
        "total_km": round(total_m / 1000, 2),
        "avg_heart_rate": avg_hr,
    }


def monthly_ranking(repo: RecordRepository, limit: int = 10) -> list[dict]:
    """이번 달 누적 거리 순위."""
    rows = repo.monthly_totals()
    rows.sort(key=lambda r: r["distance_m"], reverse=True)
    return rows[:limit]
'''

UPLOAD_API = '''"""워치 기록 업로드 API."""

from db.records import RecordRepository

REQUIRED_FIELDS = ("user_id", "started_at", "distance_m", "avg_hr")


def upload_record(repo: RecordRepository, payload: dict) -> dict:
    """업로드된 운동 기록을 검증하고 저장한다."""
    missing = [f for f in REQUIRED_FIELDS if f not in payload]
    if missing:
        return {"ok": False, "missing": missing}
    record_id = repo.insert(payload)
    return {"ok": True, "record_id": record_id}
'''

DB_RECORDS = '''"""운동 기록 저장소 (DB 접근 계층)."""

from datetime import datetime, timedelta


class RecordRepository:
    """운동 기록을 메모리 DB에 저장한다 (교육용)."""

    def __init__(self):
        self._rows = []

    def insert(self, row: dict) -> int:
        self._rows.append(dict(row, id=len(self._rows) + 1))
        return len(self._rows)

    def recent(self, user_id: str, days: int = 7) -> list[dict]:
        since = datetime.now() - timedelta(days=days)
        return [
            r for r in self._rows
            if r["user_id"] == user_id and datetime.fromisoformat(r["started_at"]) >= since
        ]

    def monthly_totals(self) -> list[dict]:
        totals = {}
        for r in self._rows:
            totals[r["user_id"]] = totals.get(r["user_id"], 0) + r["distance_m"]
        return [{"user_id": u, "distance_m": d} for u, d in totals.items()]
'''

# ═══════════════════════ pulsefit-py · 추가 모듈 ═══════════════════════

SPO2 = '''"""혈중 산소(SpO2) 측정 — PPG 센서의 적색·적외선 LED를 함께 쓴다."""

from sensor.driver import PpgDriver

# 측정 시간 (초) — 이 시간 동안 손목을 움직이지 않아야 한다
SPO2_MEASURE_SECONDS = 30
# 신호 품질이 이 값 미만이면 측정 실패로 처리한다 (0~1)
SPO2_MIN_SIGNAL_QUALITY = 0.92
# 정상 범위 밖 값은 버린다
SPO2_VALID_RANGE = (70, 100)


class Spo2Meter:
    """SpO2를 한 번 측정한다. 신호가 나쁘면 None을 돌려준다."""

    def __init__(self, driver: PpgDriver | None = None):
        self.driver = driver or PpgDriver()
        self.failures = 0

    def measure(self, samples: list[float], quality: float):
        """샘플 묶음과 신호 품질로 SpO2(%)를 계산한다."""
        if quality < SPO2_MIN_SIGNAL_QUALITY:
            self.failures += 1
            return None
        if not samples:
            self.failures += 1
            return None
        value = round(sum(samples) / len(samples))
        low, high = SPO2_VALID_RANGE
        if not low <= value <= high:
            self.failures += 1
            return None
        return value
'''

PEDOMETER_INITIAL = '''"""가속도 기반 걸음 수 계산 (만보기)."""

from datetime import datetime

# 걸음으로 인정하는 가속도 크기 (g) — 낮을수록 작은 흔들림도 걸음으로 센다
STEP_ACCEL_THRESHOLD_G = 1.1
# 걸음 사이 최소 간격 (ms) — 이보다 짧으면 같은 걸음으로 본다
MIN_STEP_GAP_MS = 250
# 팔 흔들림이 없을 때(유모차·카트) 보조 판정 사용 여부
USE_CADENCE_FALLBACK = False


class Pedometer:
    """가속도 피크를 세어 하루 걸음 수를 만든다."""

    def __init__(self):
        self.steps_today = 0
        self._last_peak_ms = None
        self._day = datetime.utcnow().date()

    def on_accel(self, magnitude_g: float, t_ms: int):
        """가속도 크기 하나를 받는다. 피크면 걸음으로 센다."""
        self._roll_day()
        if magnitude_g < STEP_ACCEL_THRESHOLD_G:
            return
        if self._last_peak_ms is not None and t_ms - self._last_peak_ms < MIN_STEP_GAP_MS:
            return
        self._last_peak_ms = t_ms
        self.steps_today += 1

    def _roll_day(self):
        """날짜가 바뀌면 걸음 수를 0으로 되돌린다."""
        today = datetime.utcnow().date()
        if today != self._day:
            self._day = today
            self.steps_today = 0

    def summary(self) -> dict:
        return {"date": self._day.isoformat(), "steps": self.steps_today}
'''

PEDOMETER_FINAL = PEDOMETER_INITIAL.replace(
    "STEP_ACCEL_THRESHOLD_G = 1.1", "STEP_ACCEL_THRESHOLD_G = 0.8")

FLOORS = '''"""기압계 기반 오른 층수 계산."""

# 한 층 높이 (m)
FLOOR_HEIGHT_M = 3.0
# 층 오름으로 인정하는 최소 고도 변화 (m)
MIN_ALTITUDE_DELTA_M = 4.5
# 기압계 값 흔들림을 줄이는 평활 계수
SMOOTHING = 0.2


class FloorCounter:
    """고도가 오른 만큼 층수를 센다. 내려가는 건 세지 않는다."""

    def __init__(self):
        self.floors = 0
        self._base_alt = None
        self._smoothed = None

    def on_altitude(self, altitude_m: float):
        if self._smoothed is None:
            self._smoothed = altitude_m
        self._smoothed += SMOOTHING * (altitude_m - self._smoothed)
        if self._base_alt is None:
            self._base_alt = self._smoothed
            return
        delta = self._smoothed - self._base_alt
        if delta >= MIN_ALTITUDE_DELTA_M:
            self.floors += int(delta // FLOOR_HEIGHT_M)
            self._base_alt = self._smoothed
        elif delta < 0:
            self._base_alt = self._smoothed

    def reset(self):
        self.floors = 0
        self._base_alt = None
'''

SLEEP_TRACKER = '''"""수면 시작·종료 감지."""

from datetime import datetime, timedelta

# 움직임이 이 시간(분) 이상 없으면 잠든 것으로 본다
SLEEP_ONSET_STILL_MIN = 10
# 움직임 크기 기준 (g) — 이보다 작으면 '가만히 있음'
STILL_MOTION_G = 0.05
# 이 시간(분)보다 짧은 수면은 기록하지 않는다
MIN_SLEEP_DURATION_MIN = 180
# 깸으로 인정하는 연속 움직임 시간 (분)
WAKE_MOTION_MIN = 15


class SleepTracker:
    """손목 움직임만으로 잠든 시각과 깬 시각을 정한다."""

    def __init__(self):
        self.still_since = None
        self.asleep_at = None
        self.sessions = []
        self._moving_since = None

    def on_motion(self, now: datetime, motion_g: float):
        if motion_g < STILL_MOTION_G:
            self._moving_since = None
            if self.still_since is None:
                self.still_since = now
            onset = timedelta(minutes=SLEEP_ONSET_STILL_MIN)
            if self.asleep_at is None and now - self.still_since >= onset:
                self.asleep_at = self.still_since
            return
        self.still_since = None
        if self.asleep_at is None:
            return
        if self._moving_since is None:
            self._moving_since = now
        if now - self._moving_since >= timedelta(minutes=WAKE_MOTION_MIN):
            self._close(self._moving_since)

    def _close(self, woke_at: datetime):
        minutes = (woke_at - self.asleep_at).total_seconds() / 60
        if minutes >= MIN_SLEEP_DURATION_MIN:
            self.sessions.append({"start": self.asleep_at, "end": woke_at, "minutes": round(minutes)})
        self.asleep_at = None
        self._moving_since = None
'''

SLEEP_STAGES = '''"""수면 단계(얕은 잠·깊은 잠·렘) 분류와 수면 점수."""

# 깊은 잠 판정: 심박이 평소 안정 심박의 이 비율 이하일 때
DEEP_SLEEP_HR_RATIO = 0.72
# 렘 판정: 심박 변동이 이 값(bpm) 이상일 때
REM_HR_VARIATION_BPM = 8
# 수면 점수 가중치
SCORE_WEIGHTS = {"duration": 0.5, "deep": 0.3, "wake": 0.2}
TARGET_SLEEP_MIN = 420


def classify(epochs: list[dict], resting_bpm: float) -> list[str]:
    """30초 단위 구간마다 단계를 붙인다. epoch = {"bpm": .., "var": ..}."""
    stages = []
    for e in epochs:
        if e["bpm"] <= resting_bpm * DEEP_SLEEP_HR_RATIO:
            stages.append("deep")
        elif e["var"] >= REM_HR_VARIATION_BPM:
            stages.append("rem")
        else:
            stages.append("light")
    return stages


def sleep_score(stages: list[str], wake_count: int) -> int:
    """0~100 수면 점수."""
    if not stages:
        return 0
    minutes = len(stages) / 2
    deep_ratio = stages.count("deep") / len(stages)
    duration = min(minutes / TARGET_SLEEP_MIN, 1.0)
    wake = max(0.0, 1 - wake_count / 10)
    total = (SCORE_WEIGHTS["duration"] * duration + SCORE_WEIGHTS["deep"] * min(deep_ratio / 0.2, 1.0)
             + SCORE_WEIGHTS["wake"] * wake)
    return round(total * 100)
'''

NOTIFY_DISPATCHER = '''"""폰에서 넘어온 알림을 화면 배너와 진동으로 내보낸다."""

from notify.dnd import DoNotDisturb

# 알림 미리보기 최대 글자 수 — 넘으면 잘린다
PREVIEW_MAX_CHARS = 40
# 보관할 알림 개수
HISTORY_LIMIT = 50


class NotificationDispatcher:
    """알림 하나를 받아 화면과 진동 모터로 보낸다."""

    def __init__(self, screen, haptics, dnd: DoNotDisturb):
        self.screen = screen
        self.haptics = haptics
        self.dnd = dnd
        self.history = []

    def deliver(self, note: dict):
        """note = {"id", "app", "text", "pattern"}."""
        text = note["text"][:PREVIEW_MAX_CHARS]
        if not self.dnd.active():
            self.screen.show_banner(text, level="notify")
        # 진동은 화면 표시와 따로 울린다
        self.haptics.buzz(pattern=note.get("pattern", "short"))
        self.history.append(note)
        del self.history[:-HISTORY_LIMIT]

    def dismiss(self, note_id: str):
        """워치에서 지운 알림을 목록에서 뺀다."""
        self.history = [n for n in self.history if n["id"] != note_id]

    def clear_all(self):
        self.history.clear()
'''

NOTIFY_DND = '''"""방해 금지 모드 일정."""

from datetime import datetime, time

DEFAULT_START = time(22, 0)
DEFAULT_END = time(7, 0)


class DoNotDisturb:
    """정해진 시간대 또는 수동으로 방해 금지를 켠다."""

    def __init__(self, start: time = DEFAULT_START, end: time = DEFAULT_END):
        self.start = start
        self.end = end
        self.manual = False
        self.scheduled = True

    def active(self, now: datetime | None = None) -> bool:
        if self.manual:
            return True
        if not self.scheduled:
            return False
        t = (now or datetime.now()).time()
        if self.start <= self.end:
            return self.start <= t < self.end
        return t >= self.start or t < self.end

    def toggle(self):
        self.manual = not self.manual
        return self.manual
'''

WATCHDOG = '''"""시스템 워치독 — 메인 루프가 멈추면 기기를 다시 시작한다."""

import time

# 메인 루프가 이 시간(초) 동안 신호를 안 보내면 재부팅한다
WATCHDOG_TIMEOUT_S = 3.0
# 재부팅 전에 기록을 저장할지 여부
SAVE_BEFORE_REBOOT = False


class Watchdog:
    """메인 루프가 주기적으로 kick()을 불러야 한다."""

    def __init__(self, reboot_fn, save_fn=None):
        self.reboot_fn = reboot_fn
        self.save_fn = save_fn
        self._last_kick = time.monotonic()
        self.reboots = 0

    def kick(self):
        self._last_kick = time.monotonic()

    def check(self) -> bool:
        """시간 초과면 재부팅하고 True를 돌려준다."""
        if time.monotonic() - self._last_kick < WATCHDOG_TIMEOUT_S:
            return False
        if SAVE_BEFORE_REBOOT and self.save_fn:
            self.save_fn()
        self.reboots += 1
        self.reboot_fn()
        return True
'''

MEMORY = '''"""메모리 부족 시 앱 정리."""

# 남은 메모리가 이 값(MB) 아래로 내려가면 앱을 종료한다
LOW_MEMORY_KILL_MB = 24
# 종료 대상에서 빼는 앱
PROTECTED_APPS = {"clock", "settings"}


class MemoryGuard:
    """메모리가 부족하면 가장 많이 쓰는 앱부터 끈다 (화면에 떠 있는 앱 포함)."""

    def __init__(self, app_manager):
        self.app_manager = app_manager
        self.killed = []

    def check(self, free_mb: float) -> list[str]:
        if free_mb >= LOW_MEMORY_KILL_MB:
            return []
        apps = sorted(self.app_manager.running(), key=lambda a: -a["mem_mb"])
        killed = []
        for app in apps:
            if app["name"] in PROTECTED_APPS:
                continue
            self.app_manager.kill(app["name"])
            killed.append(app["name"])
            free_mb += app["mem_mb"]
            if free_mb >= LOW_MEMORY_KILL_MB:
                break
        self.killed.extend(killed)
        return killed
'''

WATCHFACE_CLASSIC = '''"""아날로그 워치 페이스."""

import math

HAND_COLORS = {"hour": "#FFFFFF", "minute": "#FFFFFF", "second": "#FF5A36"}
TICK_COUNT = 12


def hand_angle(unit: str, h: int, m: int, s: int) -> float:
    """시침·분침·초침 각도(도)."""
    if unit == "hour":
        return (h % 12 + m / 60) * 30
    if unit == "minute":
        return (m + s / 60) * 6
    return s * 6


def tick_positions(cx: int, cy: int, radius: int) -> list[tuple[int, int]]:
    """눈금 좌표."""
    out = []
    for i in range(TICK_COUNT):
        a = math.radians(i * 360 / TICK_COUNT)
        out.append((round(cx + radius * math.sin(a)), round(cy - radius * math.cos(a))))
    return out


def render(screen, h: int, m: int, s: int):
    for unit in ("hour", "minute", "second"):
        screen.widgets.append({"hand": unit, "angle": hand_angle(unit, h, m, s),
                               "color": HAND_COLORS[unit]})
'''

WATCHFACE_DIGITAL = '''"""디지털 워치 페이스."""

COMPLICATIONS = ["날짜", "걸음", "날씨"]
TIME_FONT_PX = 64


def format_time(h: int, m: int, use_24h: bool = True) -> str:
    if use_24h:
        return f"{h:02d}:{m:02d}"
    suffix = "오전" if h < 12 else "오후"
    return f"{suffix} {(h % 12) or 12}:{m:02d}"


def render(screen, h: int, m: int, data: dict):
    screen.widgets.append({"text": format_time(h, m), "size": TIME_FONT_PX})
    for i, name in enumerate(COMPLICATIONS):
        screen.widgets.append({"label": name, "value": data.get(name, "-"), "slot": i})
'''

STRINGS_KO_INITIAL = '''"""화면 문구 (한국어) — 표시용 리소스만 둔다."""

STRINGS = {
    "workout.start": "운동 시작",
    "workout.finish": "운동 종료",
    "battery.low": "배터리가 부족합니다",
    "sync.done": "동기화 완료",
    "sleep.summary": "지난밤 수면",
    "steps.goal": "오늘 걸음 목표 달성!",
}


def t(key: str) -> str:
    return STRINGS.get(key, key)
'''

STRINGS_KO_FINAL = STRINGS_KO_INITIAL.replace(
    '    "steps.goal": "오늘 걸음 목표 달성!",\n',
    '    "steps.goal": "오늘 걸음 목표 달성!",\n'
    '    "heat.title": "기기 발열 주의",\n'
    '    "heat.body": "기기가 뜨거우면 잠시 풀어 두세요",\n'
    '    "heat.workout": "운동 중 발열이 감지되었습니다",\n')

TIMEUTIL = '''"""시간 관련 도움 함수."""

from datetime import datetime, timedelta, timezone

KST = timezone(timedelta(hours=9))


def now_kst() -> datetime:
    return datetime.now(KST)


def minutes_between(a: datetime, b: datetime) -> float:
    return abs((b - a).total_seconds()) / 60


def format_duration(seconds: int) -> str:
    h, rest = divmod(int(seconds), 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
'''

LOG = '''"""워치 내부 로그."""

LEVELS = ("debug", "info", "warn", "error")
MAX_LINES = 200


class RingLog:
    """최근 로그만 남기는 링 버퍼."""

    def __init__(self):
        self.lines = []

    def write(self, level: str, msg: str):
        if level not in LEVELS:
            level = "info"
        self.lines.append(f"[{level}] {msg}")
        del self.lines[:-MAX_LINES]

    def dump(self) -> str:
        return "\\n".join(self.lines)
'''

# ═══════════════════════ pulsefit-sync · 추가 모듈 ═══════════════════════

SCHEDULER_INITIAL = '''"""워치 → 폰 동기화 예약."""

# 백그라운드 동기화 주기 (분)
BACKGROUND_SYNC_INTERVAL_MIN = 15
# 폰 앱이 화면에 떠 있을 때만 동기화할지 여부
SYNC_ONLY_WHEN_APP_FOREGROUND = False


class SyncScheduler:
    """언제 전송 큐를 비울지 정한다."""

    def __init__(self, queue):
        self.queue = queue
        self._last_sync_min = None

    def should_sync(self, now_min: int, app_foreground: bool) -> bool:
        if SYNC_ONLY_WHEN_APP_FOREGROUND and not app_foreground:
            return False
        if self._last_sync_min is None:
            return True
        return now_min - self._last_sync_min >= BACKGROUND_SYNC_INTERVAL_MIN

    def tick(self, now_min: int, app_foreground: bool) -> int:
        """예약 시각이면 큐를 비우고 보낸 개수를 돌려준다."""
        if not self.should_sync(now_min, app_foreground):
            return 0
        self._last_sync_min = now_min
        return self.queue.drain()
'''

SCHEDULER_FINAL = SCHEDULER_INITIAL.replace(
    "BACKGROUND_SYNC_INTERVAL_MIN = 15", "BACKGROUND_SYNC_INTERVAL_MIN = 60").replace(
    "# 폰 앱이 화면에 떠 있을 때만 동기화할지 여부\nSYNC_ONLY_WHEN_APP_FOREGROUND = False",
    "# 폰 앱이 화면에 떠 있을 때만 동기화할지 여부 — 폰 배터리 절약\nSYNC_ONLY_WHEN_APP_FOREGROUND = True")

FORWARDER_INITIAL = '''"""폰 알림을 워치로 전달한다."""

import time

# 알림을 모아서 보내는 시간 창 (초)
BATCH_WINDOW_S = 0
# 워치 수신 확인을 기다리는 시간 (초) — 넘으면 다시 보낸다
ACK_TIMEOUT_S = 5


class NotificationForwarder:
    """폰에서 받은 알림을 묶어 BLE로 보낸다."""

    def __init__(self, link):
        self.link = link
        self.buffer = []
        self._window_started = None

    def on_phone_notification(self, note: dict):
        self.buffer.append(note)
        if self._window_started is None:
            self._window_started = time.monotonic()
        if time.monotonic() - self._window_started >= BATCH_WINDOW_S:
            self.flush()

    def flush(self) -> int:
        sent = 0
        for note in self.buffer:
            self.link.send(repr(note).encode("utf-8"))
            sent += 1
        self.buffer.clear()
        self._window_started = None
        return sent

    def on_ack_timeout(self, note: dict):
        """수신 확인이 없으면 새 알림처럼 다시 넣는다."""
        self.buffer.append(dict(note))
'''

FORWARDER_FINAL = FORWARDER_INITIAL.replace(
    "# 알림을 모아서 보내는 시간 창 (초)\nBATCH_WINDOW_S = 0",
    "# 알림을 모아서 보내는 시간 창 (초) — 무선 전송 횟수를 줄여 전력 절감\nBATCH_WINDOW_S = 120")

PACKET = '''"""BLE 패킷 틀 (헤더 · 길이 · 본문 · CRC)."""

from protocol.crc import crc16

HEADER = b"PF"
VERSION = 2
MAX_PAYLOAD = 180


def encode(payload: bytes) -> bytes:
    if len(payload) > MAX_PAYLOAD:
        raise ValueError("payload too large")
    body = bytes([VERSION, len(payload)]) + payload
    return HEADER + body + crc16(body).to_bytes(2, "big")


def decode(frame: bytes) -> bytes:
    if not frame.startswith(HEADER):
        raise ValueError("bad header")
    body, crc = frame[2:-2], int.from_bytes(frame[-2:], "big")
    if crc16(body) != crc:
        raise ValueError("crc mismatch")
    return body[2:2 + body[1]]
'''

CRC = '''"""CRC-16/CCITT."""


def crc16(data: bytes, init: int = 0xFFFF) -> int:
    crc = init
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) if crc & 0x8000 else (crc << 1)
            crc &= 0xFFFF
    return crc
'''

# ═══════════════════════ pulsefit-mobile (폰 앱) ═══════════════════════

HISTORY_IMPORT = '''"""폰을 바꿨을 때 서버에서 예전 기록을 내려받는다."""

from datetime import date, timedelta

# 가져올 기간 (일) — 이보다 오래된 기록은 새 폰으로 넘어오지 않는다
IMPORT_DAYS = 30
PAGE_SIZE = 100


class HistoryImporter:
    """새 폰에 로그인하면 한 번 실행된다."""

    def __init__(self, api, local_db):
        self.api = api
        self.local_db = local_db

    def run(self, user_id: str, today: date) -> int:
        since = today - timedelta(days=IMPORT_DAYS)
        page, total = 0, 0
        while True:
            rows = self.api.list_records(user_id, since=since.isoformat(), page=page, size=PAGE_SIZE)
            if not rows:
                break
            for row in rows:
                self.local_db.upsert(row)
            total += len(rows)
            page += 1
        return total
'''

STEPS_MERGE = '''"""워치 걸음 수와 폰 걸음 수 합치기."""

# 같은 시간대에 둘 다 있으면 어느 쪽을 쓸지
PREFER_SOURCE = "phone"


def merge_hourly(watch: dict[int, int], phone: dict[int, int]) -> dict[int, int]:
    """시간(0~23)별 걸음 수를 합친다. 겹치면 PREFER_SOURCE 값을 쓴다."""
    merged = {}
    for hour in range(24):
        w, p = watch.get(hour), phone.get(hour)
        if w is None and p is None:
            continue
        if w is None or p is None:
            merged[hour] = w if p is None else p
        else:
            merged[hour] = p if PREFER_SOURCE == "phone" else w
    return merged


def daily_total(hourly: dict[int, int]) -> int:
    return sum(hourly.values())
'''

WEIGHT = '''"""체중 입력 · 워치로 보내기."""

# 워치로 보내는 프로필 항목
WATCH_PUSH_FIELDS = ("height_cm", "birth_year", "gender")


class ProfileService:
    """폰 앱에서 바꾼 프로필을 저장하고 워치로 보낸다."""

    def __init__(self, store, watch_link):
        self.store = store
        self.watch_link = watch_link

    def update(self, user_id: str, **fields):
        profile = self.store.get(user_id) | fields
        self.store.put(user_id, profile)
        push = {k: profile[k] for k in WATCH_PUSH_FIELDS if k in profile}
        if push:
            self.watch_link.send_profile(push)
        return profile

    def set_weight(self, user_id: str, kg: float):
        return self.update(user_id, weight_kg=round(kg, 1))
'''

DASHBOARD = '''"""폰 앱 홈 화면 카드 구성 — 표시만 한다."""

CARDS = ["오늘 걸음", "심박", "수면", "운동 기록", "배터리"]


def build_cards(summary: dict) -> list[dict]:
    out = []
    for name in CARDS:
        out.append({"title": name, "value": summary.get(name, "-")})
    return out


def empty_state_text(has_watch: bool) -> str:
    if not has_watch:
        return "워치를 연결하면 기록이 여기에 표시됩니다."
    return "아직 기록이 없어요. 오늘 첫 운동을 시작해 보세요."
'''

SETTINGS_SCREEN = '''"""폰 앱 설정 화면 항목 — 표시만 한다."""

SECTIONS = {
    "알림": ["앱별 알림 전달", "방해 금지 연동", "진동 세기"],
    "건강": ["수면 측정", "심박 상시 측정", "걸음 목표"],
    "기기": ["워치 페이스", "펌웨어 업데이트", "배터리 절약"],
}


def section_titles() -> list[str]:
    return list(SECTIONS)


def items(section: str) -> list[str]:
    return SECTIONS.get(section, [])
'''

EVENTS = '''"""앱 사용 통계 이벤트 (제품 분석용)."""

from datetime import datetime

ALLOWED_EVENTS = {"app_open", "card_tap", "workout_view", "settings_change"}


class EventLogger:
    def __init__(self, sink):
        self.sink = sink

    def log(self, name: str, **props):
        if name not in ALLOWED_EVENTS:
            return False
        self.sink.append({"name": name, "at": datetime.now().isoformat(), **props})
        return True
'''

# ═══════════════════════ pulsefit-server · 추가 모듈 ═══════════════════════

SYNC_API = '''"""폰 앱 동기화 API — 워치 기록 묶음을 받는다."""

from db.records import RecordRepository


def dedupe_key(row: dict) -> tuple:
    """같은 기록인지 판정하는 키."""
    return (row["user_id"], row["uploaded_at"])


def sync_batch(repo: RecordRepository, rows: list[dict]) -> dict:
    """이미 받은 기록은 건너뛰고 새 기록만 저장한다."""
    seen = {dedupe_key(r) for r in repo._rows if "uploaded_at" in r}
    inserted, skipped = 0, 0
    for row in rows:
        key = dedupe_key(row)
        if key in seen:
            skipped += 1
            continue
        repo.insert(row)
        seen.add(key)
        inserted += 1
    return {"inserted": inserted, "skipped": skipped}
'''

TOKENS = '''"""로그인 토큰 발급 (교육용 단순 구현)."""

import hashlib
import secrets
from datetime import datetime, timedelta

TOKEN_TTL_H = 24


def issue(user_id: str) -> dict:
    raw = secrets.token_hex(16)
    return {
        "user_id": user_id,
        "token_hash": hashlib.sha256(raw.encode()).hexdigest(),
        "expires_at": (datetime.now() + timedelta(hours=TOKEN_TTL_H)).isoformat(),
        "token": raw,
    }


def is_expired(token: dict, now: datetime | None = None) -> bool:
    return (now or datetime.now()) >= datetime.fromisoformat(token["expires_at"])
'''

WEEKLY_REPORT_INITIAL = '''"""주간 리포트 메일 발송 배치."""

SUBJECT = "[PulseFit] 이번 주 운동 리포트"
SEND_WEEKDAY = 0  # 월요일


def build_body(summary: dict) -> str:
    return (
        f"이번 주 {summary['sessions']}번 운동했어요.\\n"
        f"총 {summary['total_km']}km를 달렸습니다."
    )


def should_send(weekday: int, opted_in: bool) -> bool:
    return opted_in and weekday == SEND_WEEKDAY
'''

WEEKLY_REPORT_FINAL = WEEKLY_REPORT_INITIAL.replace(
    'SUBJECT = "[PulseFit] 이번 주 운동 리포트"', 'SUBJECT = "[PulseFit] 한 주 운동을 돌아봐요"')

# ═══════════════════════ pulsefit-firmware (워치 펌웨어) ═══════════════════════

CHARGER_INITIAL = '''"""충전 관리 — 충전 접점 감지 · 충전 전류 · 온도 보호."""

# 충전 접점이 이 시간(ms) 동안 붙어 있어야 '충전 중'으로 본다
CONTACT_DEBOUNCE_MS = 300
# 충전 전류 (mA)
CHARGE_CURRENT_MA = 200
# 배터리 온도가 이 값(℃) 이상이면 충전을 잠시 멈춘다
CHARGE_PAUSE_TEMP_C = 38
CHARGE_RESUME_TEMP_C = 35


class Charger:
    """충전 거치대 접점과 충전 칩(PMIC)을 다룬다."""

    def __init__(self, pmic):
        self.pmic = pmic
        self.charging = False
        self.paused = False
        self._contact_ms = 0

    def on_contact(self, touching: bool, dt_ms: int):
        """접점 상태를 받는다. 떨어지면 바로 충전 중 표시를 끈다."""
        self._contact_ms = self._contact_ms + dt_ms if touching else 0
        self.charging = self._contact_ms >= CONTACT_DEBOUNCE_MS
        if self.charging and not self.paused:
            self.pmic.set_current(CHARGE_CURRENT_MA)
        elif not self.charging:
            self.pmic.set_current(0)

    def on_battery_temp(self, temp_c: float):
        if temp_c >= CHARGE_PAUSE_TEMP_C:
            self.paused = True
            self.pmic.set_current(0)
        elif self.paused and temp_c <= CHARGE_RESUME_TEMP_C:
            self.paused = False
            if self.charging:
                self.pmic.set_current(CHARGE_CURRENT_MA)
'''

CHARGER_FINAL = CHARGER_INITIAL.replace(
    "# 충전 접점이 이 시간(ms) 동안 붙어 있어야 '충전 중'으로 본다\nCONTACT_DEBOUNCE_MS = 300",
    "# 충전 접점이 이 시간(ms) 동안 붙어 있어야 '충전 중'으로 본다 — 반응 속도 개선\nCONTACT_DEBOUNCE_MS = 40")

MOTOR = '''"""진동 모터(햅틱) 제어."""

# 패턴별 진동 길이 (ms)
VIBRATION_MS = {"short": 120, "long": 400, "alarm": 600}
# 모터 세기 (0~1)
DEFAULT_STRENGTH = 0.35
# 알람 진동 반복 횟수
ALARM_REPEAT = 1


class HapticMotor:
    def __init__(self, pwm):
        self.pwm = pwm
        self.strength = DEFAULT_STRENGTH

    def buzz(self, pattern: str = "short"):
        ms = VIBRATION_MS.get(pattern, VIBRATION_MS["short"])
        repeat = ALARM_REPEAT if pattern == "alarm" else 1
        for _ in range(repeat):
            self.pwm.pulse(duty=self.strength, ms=ms)

    def set_strength(self, level: float):
        self.strength = min(max(level, 0.0), 1.0)
'''

GNSS = '''"""GNSS(GPS) 수신 칩 제어."""

# 보조 위성 정보(AGPS)가 이 시간(시간)보다 오래되면 콜드 스타트한다
AGPS_MAX_AGE_H = 72
# 위치를 못 잡으면 포기하는 시간 (초)
FIX_TIMEOUT_S = 180
# 운동 중 위치 갱신 주기 (초)
FIX_RATE_S = 1


class GnssReceiver:
    def __init__(self, chip):
        self.chip = chip
        self.agps_age_h = None

    def start(self) -> str:
        """위치 잡기를 시작한다. 보조 정보가 오래됐으면 콜드 스타트."""
        if self.agps_age_h is None or self.agps_age_h > AGPS_MAX_AGE_H:
            self.chip.cold_start(timeout_s=FIX_TIMEOUT_S)
            return "cold"
        self.chip.hot_start(timeout_s=FIX_TIMEOUT_S)
        return "hot"

    def update_agps(self, age_h: float):
        """폰에서 받은 보조 위성 정보의 나이를 기록한다."""
        self.agps_age_h = age_h

    def stop(self):
        self.chip.power_off()
'''

THERMAL_INITIAL = '''"""피부·배터리 온도 측정 — 값을 읽어서 알려 주기만 한다 (발열을 만들지 않음).

온도 센서는 전력 소모가 매우 작아(0.1mW 수준) 측정 주기를 바꿔도 발열과는 무관하다.
"""

# 온도 측정 주기 (초)
TEMP_POLL_INTERVAL_S = 60
# 이 온도(℃) 이상이면 화면 쪽에 경고를 요청한다
REPORT_THRESHOLD_C = 41


class ThermalMonitor:
    """온도 값을 읽어 구독자에게 넘긴다."""

    def __init__(self, temp_sensor):
        self.temp_sensor = temp_sensor
        self.subscribers = []
        self.last_c = None

    def subscribe(self, fn):
        self.subscribers.append(fn)

    def sample(self) -> float:
        self.last_c = self.temp_sensor.read_celsius()
        if self.last_c >= REPORT_THRESHOLD_C:
            for fn in self.subscribers:
                fn(self.last_c)
        return self.last_c
'''

THERMAL_FINAL = THERMAL_INITIAL.replace("TEMP_POLL_INTERVAL_S = 60", "TEMP_POLL_INTERVAL_S = 30")

BOOT_VERSION = '''"""부트로더 · 펌웨어 버전 확인."""

FIRMWARE_VERSION = (3, 4, 1)
MIN_BOOTLOADER = (1, 2, 0)


def parse(version: str) -> tuple[int, ...]:
    return tuple(int(x) for x in version.split("."))


def can_update(current: str, target: str) -> bool:
    return parse(target) > parse(current)


def bootloader_ok(bootloader: str) -> bool:
    return parse(bootloader) >= MIN_BOOTLOADER
'''

WRIST_DETECT = '''"""착용 감지 — 손목에 차고 있는지 판단한다 (소프트웨어로 착용감은 바꾸지 못한다)."""

# 근접 값이 이 값 이상이면 착용 중
WORN_PROXIMITY = 0.6
# 착용 해제로 판단하기까지 기다리는 시간 (초)
OFF_WRIST_DELAY_S = 10


class WristDetector:
    def __init__(self):
        self.worn = False
        self._off_since = None

    def update(self, proximity: float, now_s: float) -> bool:
        if proximity >= WORN_PROXIMITY:
            self.worn = True
            self._off_since = None
        else:
            if self._off_since is None:
                self._off_since = now_s
            if now_s - self._off_since >= OFF_WRIST_DELAY_S:
                self.worn = False
        return self.worn
'''

# ═══════════════════════ pulsefit-web (브랜드 사이트 · 스토어, 더미) ═══════════════════════

CART = '''"""온라인 스토어 장바구니."""

FREE_SHIPPING_KRW = 50_000
SHIPPING_FEE_KRW = 3_000


class Cart:
    def __init__(self):
        self.items = {}

    def add(self, sku: str, price: int, qty: int = 1):
        item = self.items.setdefault(sku, {"price": price, "qty": 0})
        item["qty"] += qty

    def remove(self, sku: str):
        self.items.pop(sku, None)

    def subtotal(self) -> int:
        return sum(i["price"] * i["qty"] for i in self.items.values())

    def total(self, discount: int = 0) -> int:
        sub = max(self.subtotal() - discount, 0)
        return sub if sub >= FREE_SHIPPING_KRW else sub + SHIPPING_FEE_KRW
'''

COUPONS = '''"""쿠폰 적용."""

COUPONS = {
    "WELCOME10": {"rate": 0.10, "min_krw": 30_000},
    "STRAP5000": {"amount": 5_000, "min_krw": 20_000},
}


def discount(code: str, subtotal: int) -> int:
    c = COUPONS.get(code.upper())
    if not c or subtotal < c["min_krw"]:
        return 0
    if "rate" in c:
        return int(subtotal * c["rate"])
    return c["amount"]
'''

BANNERS_INITIAL = '''"""메인 페이지 배너 (CMS)."""

BANNERS = [
    {"id": "launch", "title": "PulseFit 3 출시", "link": "/products/pulsefit-3"},
    {"id": "strap", "title": "새 스트랩 컬렉션", "link": "/accessories/straps"},
]


def active_banners(limit: int = 3) -> list[dict]:
    return BANNERS[:limit]
'''

BANNERS_FINAL = BANNERS_INITIAL.replace(
    '    {"id": "strap", "title": "새 스트랩 컬렉션", "link": "/accessories/straps"},\n',
    '    {"id": "strap", "title": "새 스트랩 컬렉션", "link": "/accessories/straps"},\n'
    '    {"id": "chuseok", "title": "추석 맞이 최대 20% 할인", "link": "/events/chuseok"},\n')

# ════════════════════════════ 커밋 계획 ════════════════════════════

INIT_DATE = "2026-08-01T10:00:00+09:00"

REPOS = {
    "pulsefit-py": {
        "README.md": PY_README,
        "sensor/__init__.py": "",
        "sensor/heart_rate.py": HEART_RATE_INITIAL,
        "sensor/driver.py": DRIVER,
        "sensor/filters.py": FILTERS,
        "workout/__init__.py": "",
        "workout/session.py": SESSION,
        "workout/distance.py": DISTANCE,
        "sync/__init__.py": "",
        "sync/bluetooth.py": SYNC_BLUETOOTH,
        "ui/__init__.py": "",
        "ui/warnings.py": WARNINGS_INITIAL,
        "ui/screens.py": SCREENS,
        "power/__init__.py": "",
        "power/battery.py": BATTERY_INITIAL,
        "storage/__init__.py": "",
        "storage/records.py": STORAGE_RECORDS,
        "sensor/spo2.py": SPO2,
        "activity/__init__.py": "",
        "activity/pedometer.py": PEDOMETER_INITIAL,
        "activity/floors.py": FLOORS,
        "sleep/__init__.py": "",
        "sleep/tracker.py": SLEEP_TRACKER,
        "sleep/stages.py": SLEEP_STAGES,
        "notify/__init__.py": "",
        "notify/dispatcher.py": NOTIFY_DISPATCHER,
        "notify/dnd.py": NOTIFY_DND,
        "system/__init__.py": "",
        "system/watchdog.py": WATCHDOG,
        "system/memory.py": MEMORY,
        "watchfaces/__init__.py": "",
        "watchfaces/classic.py": WATCHFACE_CLASSIC,
        "watchfaces/digital.py": WATCHFACE_DIGITAL,
        "i18n/__init__.py": "",
        "i18n/strings_ko.py": STRINGS_KO_INITIAL,
        "utils/__init__.py": "",
        "utils/timeutil.py": TIMEUTIL,
        "utils/log.py": LOG,
    },
    "pulsefit-sync": {
        "README.md": "# pulsefit-sync\n\n워치-폰 블루투스 연동 모듈 (교육용 가상 저장소).\n",
        "bluetooth/__init__.py": "",
        "bluetooth/reconnect.py": RECONNECT_INITIAL,
        "bluetooth/link.py": BLE_LINK,
        "transfer/__init__.py": "",
        "transfer/queue.py": TRANSFER,
        "transfer/scheduler.py": SCHEDULER_INITIAL,
        "relay/__init__.py": "",
        "relay/notification_forwarder.py": FORWARDER_INITIAL,
        "protocol/__init__.py": "",
        "protocol/packet.py": PACKET,
        "protocol/crc.py": CRC,
    },
    "pulsefit-mobile": {
        "README.md": "# pulsefit-mobile\n\nPulseFit 폰 앱 (교육용 가상 저장소).\n",
        "sync/__init__.py": "",
        "sync/history_import.py": HISTORY_IMPORT,
        "health/__init__.py": "",
        "health/steps_merge.py": STEPS_MERGE,
        "health/profile.py": WEIGHT,
        "screens/__init__.py": "",
        "screens/dashboard.py": DASHBOARD,
        "screens/settings_screen.py": SETTINGS_SCREEN,
        "analytics/__init__.py": "",
        "analytics/events.py": EVENTS,
    },
    "pulsefit-server": {
        "README.md": "# pulsefit-server\n\n운동 기록 서버 (교육용 가상 저장소).\n",
        "api/__init__.py": "",
        "api/stats.py": STATS_API,
        "api/upload.py": UPLOAD_API,
        "api/sync.py": SYNC_API,
        "db/__init__.py": "",
        "db/records.py": DB_RECORDS,
        "auth/__init__.py": "",
        "auth/tokens.py": TOKENS,
        "jobs/__init__.py": "",
        "jobs/weekly_report.py": WEEKLY_REPORT_INITIAL,
    },
    "pulsefit-firmware": {
        "README.md": "# pulsefit-firmware\n\nPulseFit 워치 펌웨어 · 하드웨어 제어 (교육용 가상 저장소).\n",
        "charging/__init__.py": "",
        "charging/charger.py": CHARGER_INITIAL,
        "haptics/__init__.py": "",
        "haptics/motor.py": MOTOR,
        "gnss/__init__.py": "",
        "gnss/receiver.py": GNSS,
        "thermal/__init__.py": "",
        "thermal/monitor.py": THERMAL_INITIAL,
        "wear/__init__.py": "",
        "wear/wrist_detect.py": WRIST_DETECT,
        "boot/__init__.py": "",
        "boot/version.py": BOOT_VERSION,
    },
    "pulsefit-web": {
        "README.md": "# pulsefit-web\n\nPulseFit 브랜드 사이트 · 온라인 스토어 (교육용 가상 저장소).\n",
        "shop/__init__.py": "",
        "shop/cart.py": CART,
        "shop/coupons.py": COUPONS,
        "cms/__init__.py": "",
        "cms/banners.py": BANNERS_INITIAL,
    },
}

# (날짜, 저장소, 작성자 키, {경로: 내용}, 메시지)
CHANGES = [
    ("2026-08-14T15:10:00+09:00", "pulsefit-py", "health",
     {"activity/pedometer.py": PEDOMETER_FINAL}, "걸음 감지 민감도 상향: 가속도 기준 1.1g → 0.8g"),
    ("2026-08-20T10:30:00+09:00", "pulsefit-sync", "sync",
     {"transfer/scheduler.py": SCHEDULER_FINAL}, "폰 배터리 절약: 앱이 켜져 있을 때만 동기화"),
    ("2026-08-27T17:45:00+09:00", "pulsefit-firmware", "hw",
     {"charging/charger.py": CHARGER_FINAL}, "충전 감지 반응 속도 개선: 접점 디바운스 300ms → 40ms"),
    ("2026-09-12T13:00:00+09:00", "pulsefit-sync", "notify",
     {"relay/notification_forwarder.py": FORWARDER_FINAL}, "알림 묶음 전송으로 무선 전력 절감"),
    ("2026-09-18T09:20:00+09:00", "pulsefit-web", "web",
     {"cms/banners.py": BANNERS_FINAL}, "추석 할인 배너 추가"),
    ("2026-09-22T16:00:00+09:00", "pulsefit-firmware", "hw",
     {"thermal/monitor.py": THERMAL_FINAL}, "피부 온도 측정 주기 60초 → 30초"),
    ("2026-09-30T11:30:00+09:00", "pulsefit-py", "ui",
     {"i18n/strings_ko.py": STRINGS_KO_FINAL}, "발열 안내 문구 번역 추가"),
    ("2026-10-01T10:00:00+09:00", "pulsefit-server", "server",
     {"jobs/weekly_report.py": WEEKLY_REPORT_FINAL}, "주간 리포트 메일 제목 변경"),
    ("2026-09-25T14:20:00+09:00", "pulsefit-py", "power",
     {"power/battery.py": BATTERY_FINAL}, "화면 켜짐 유지 시간 5초 → 8초로 변경"),
    ("2026-09-28T11:05:00+09:00", "pulsefit-py", "ui",
     {"ui/warnings.py": WARNINGS_FINAL}, "발열 경고 문구 추가"),
    ("2026-09-28T16:40:00+09:00", "pulsefit-sync", "sync",
     {"bluetooth/reconnect.py": RECONNECT_FINAL}, "재연결 속도 개선: 재시도 대기 제거"),
    ("2026-10-02T18:30:00+09:00", "pulsefit-py", "sensor",
     {"sensor/heart_rate.py": HEART_RATE_FINAL}, "운동 중 심박 정확도 개선: 측정 주기 단축"),
]


def git(repo: Path, *args: str, date: str | None = None, author: str = "init") -> None:
    env = os.environ.copy()
    if date:
        name, email = AUTHORS[author]
        env.update({
            "GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date,
            "GIT_AUTHOR_NAME": name, "GIT_AUTHOR_EMAIL": email,
            "GIT_COMMITTER_NAME": name, "GIT_COMMITTER_EMAIL": email,
        })
    subprocess.run(["git", "-C", str(repo), *args], env=env, check=True,
                   capture_output=True, text=True, encoding="utf-8")


def write_files(repo: Path, files: dict[str, str]) -> None:
    for rel, content in files.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")


def _force_remove(func, path, _exc):
    """Windows에서 .git 읽기 전용 파일 삭제용."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def main() -> None:
    root = config.SAMPLE_REPOS_DIR
    if root.exists():
        shutil.rmtree(root, onerror=_force_remove)
    root.mkdir(parents=True)

    for name, files in REPOS.items():
        repo = root / name
        repo.mkdir()
        git(repo, "init", "-q", "-b", "main")
        git(repo, "config", "core.autocrlf", "false")
        write_files(repo, files)
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", "초기 커밋", date=INIT_DATE)

    for date, name, author, files, message in sorted(CHANGES, key=lambda c: c[0]):
        repo = root / name
        write_files(repo, files)
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", message, date=date, author=author)

    # 정답 줄 번호 확인
    hr_lines = (root / "pulsefit-py/sensor/heart_rate.py").read_text(encoding="utf-8").splitlines()
    session_lines = (root / "pulsefit-py/workout/session.py").read_text(encoding="utf-8").splitlines()
    assert hr_lines[16] == "POLL_INTERVAL_S = 0.05", hr_lines[16]
    assert hr_lines[44].strip() == "while self._running.is_set():", hr_lines[44]
    assert hr_lines[51].strip() == "time.sleep(POLL_INTERVAL_S)", hr_lines[51]
    assert session_lines[22].strip() == "self.hr.start_continuous()", session_lines[22]

    print("sample_repos/ 생성 완료: " + ", ".join(REPOS))
    print("  heart_rate.py L17 =", hr_lines[16])
    print("  session.py    L23 =", session_lines[22].strip())


if __name__ == "__main__":
    main()
