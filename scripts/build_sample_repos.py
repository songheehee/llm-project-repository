"""Step 0: 분석 대상 가상 저장소 3개(pulsefit-py / pulsefit-sync / pulsefit-server)를 만든다.

실행: python scripts/build_sample_repos.py
sample_repos/ 를 지우고 다시 만든다. 이 폴더 밖은 건드리지 않는다.
커밋 날짜·작성자는 환경 변수로 지정한다 (전역 git 설정에 기대지 않음).
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
    },
    "pulsefit-sync": {
        "README.md": "# pulsefit-sync\n\n워치-폰 블루투스 연동 모듈 (교육용 가상 저장소).\n",
        "bluetooth/__init__.py": "",
        "bluetooth/reconnect.py": RECONNECT_INITIAL,
        "bluetooth/link.py": BLE_LINK,
        "transfer/__init__.py": "",
        "transfer/queue.py": TRANSFER,
    },
    "pulsefit-server": {
        "README.md": "# pulsefit-server\n\n운동 기록 서버 (교육용 가상 저장소).\n",
        "api/__init__.py": "",
        "api/stats.py": STATS_API,
        "api/upload.py": UPLOAD_API,
        "db/__init__.py": "",
        "db/records.py": DB_RECORDS,
    },
}

# (날짜, 저장소, 작성자 키, {경로: 내용}, 메시지)
CHANGES = [
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

    for date, name, author, files, message in CHANGES:
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
