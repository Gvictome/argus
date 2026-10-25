"""
Thermal monitor tests.

The enclosure claim ("stays below the throttle point while detecting")
rests on these numbers, so they lock in: the throttle word decodes the
way the firmware documents it, the soak summary fails on a throttle or
a soft-limit crossing, the CSV log is written per day with the enclosure
label, and the API keeps its shape on a machine with no sensors at all.
"""

import csv

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.config import settings
from src.system.thermal import ThermalMonitor, decode_throttled


def _monitor(tmp_path, temps, throttle=0, **kw):
    seq = iter(temps)
    return ThermalMonitor(
        node="Node A", enclosure="HighPi Pro 5S closed", interval_s=30,
        log_dir=tmp_path, fps_source=lambda: 12.34,
        readers={"temp": lambda: next(seq), "mhz": lambda: 2400,
                 "fan": lambda: 3100, "throttle": lambda: throttle},
        **kw,
    )


class TestDecode:
    def test_nothing_set(self):
        assert decode_throttled(0) == {"hex": "0x0", "now": [], "since_boot": []}

    def test_now_and_since_boot_bits(self):
        # 0x50005: under-voltage and throttled now, both also since boot.
        d = decode_throttled(0x50005)
        assert d["now"] == ["under_voltage", "throttled"]
        assert d["since_boot"] == ["under_voltage", "throttled"]

    def test_no_vcgencmd(self):
        assert decode_throttled(None) == {"hex": None, "now": [], "since_boot": []}


class TestSummary:
    def test_cool_run_passes(self, tmp_path):
        m = _monitor(tmp_path, [55.0, 61.5, 58.0])
        for _ in range(3):
            m.sample()
        s = m.summary()
        assert (s["temp_min_c"], s["temp_max_c"], s["temp_avg_c"]) == (55.0, 61.5, 58.2)
        assert s["throttled_samples"] == 0
        assert s["pass"] is True

    def test_soft_limit_fails(self, tmp_path):
        m = _monitor(tmp_path, [70.0, 80.2])
        m.sample(); m.sample()
        assert m.summary()["pass"] is False

    def test_throttle_fails_even_when_cool(self, tmp_path):
        m = _monitor(tmp_path, [60.0], throttle=0x4)
        m.sample()
        s = m.summary()
        assert s["throttled_samples"] == 1
        assert s["pass"] is False

    def test_no_sensor_never_passes(self, tmp_path):
        m = _monitor(tmp_path, [None])
        m.sample()
        assert m.summary()["pass"] is False

    def test_history_is_bounded(self, tmp_path):
        m = _monitor(tmp_path, [50.0] * 10, history_hours=60 / 3600)  # 2 samples at 30 s
        for _ in range(10):
            m.sample()
        assert len(m.samples) == 2


class TestCsvLog:
    def test_one_file_per_day_with_enclosure(self, tmp_path):
        m = _monitor(tmp_path, [57.0, 58.0])
        m.sample(); m.sample()
        files = list(tmp_path.glob("thermal-*.csv"))
        assert len(files) == 1
        rows = list(csv.DictReader(files[0].open()))
        assert len(rows) == 2
        assert rows[0]["enclosure"] == "HighPi Pro 5S closed"
        assert rows[0]["soc_temp_c"] == "57.0"
        assert rows[0]["detect_fps"] == "12.3"

    def test_dead_fps_source_does_not_stop_logging(self, tmp_path):
        m = _monitor(tmp_path, [57.0])
        m.fps_source = lambda: 1 / 0
        s = m.sample()
        assert s.detect_fps is None and s.soc_temp_c == 57.0


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "AUTH_REQUIRED", False)
    monkeypatch.setattr(settings, "DB_PATH", tmp_path / "t.db")
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setattr(settings, "THERMAL_SAMPLE_S", 3600.0)
    with TestClient(create_app()) as c:
        yield c


class TestApi:
    def test_status_shape(self, client):
        r = client.get("/api/system/thermal")
        assert r.status_code == 200
        body = r.json()
        for key in ("node", "enclosure", "interval_s", "limits_c", "current", "throttled", "summary"):
            assert key in body
        assert body["limits_c"] == {"soft": 80.0, "hard": 85.0}

    def test_history_and_csv(self, client):
        r = client.get("/api/system/thermal/history?minutes=30")
        assert r.status_code == 200
        assert set(r.json()) == {"interval_s", "summary", "samples"}
        c = client.get("/api/system/thermal.csv?minutes=30")
        assert c.status_code == 200
        assert c.headers["content-type"].startswith("text/csv")
        assert c.text.splitlines()[0].startswith("ts,node,enclosure,soc_temp_c")

    def test_disabled_is_503(self, monkeypatch, tmp_path):
        monkeypatch.setattr(settings, "AUTH_REQUIRED", False)
        monkeypatch.setattr(settings, "DB_PATH", tmp_path / "t2.db")
        monkeypatch.setattr(settings, "THERMAL_ENABLED", False)
        with TestClient(create_app()) as c:
            assert c.get("/api/system/thermal").status_code == 503
