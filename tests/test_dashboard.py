"""
Dashboard serving tests.

The dashboard is the surface a judge actually looks at, and it is served
by the same app as the API. These lock in that it is reachable, that
serving it did not break the JSON health contract, and that its assets
are self-contained.
"""

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.config import BASE_DIR, settings

DASHBOARD = BASE_DIR / "static" / "index.html"

HTML_ACCEPT = {"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9"}


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "AUTH_REQUIRED", False)
    monkeypatch.setattr(settings, "DB_PATH", tmp_path / "dash.db")
    with TestClient(create_app()) as c:
        yield c


class TestServing:
    def test_dashboard_path_serves_html(self, client):
        r = client.get("/dashboard")

        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]
        assert "<title>" in r.text

    def test_root_serves_html_to_a_browser(self, client):
        r = client.get("/", headers=HTML_ACCEPT)

        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]

    def test_root_still_serves_json_to_api_clients(self, client):
        """
        The health contract predates the dashboard and other things poll
        it. Content negotiation must not have quietly replaced it.
        """
        r = client.get("/")

        assert r.status_code == 200
        assert r.json() == {"status": "online", "service": "THE EYE"}

    def test_root_serves_json_for_explicit_json_accept(self, client):
        r = client.get("/", headers={"Accept": "application/json"})

        assert r.json()["status"] == "online"


@pytest.mark.skipif(not DASHBOARD.exists(), reason="dashboard not installed")
class TestPageContents:
    """
    Static checks on the page itself. Cheap, and they catch the failures
    that only show up at a booth.
    """

    @pytest.fixture(scope="class")
    def page(self):
        return DASHBOARD.read_text(encoding="utf-8")

    def test_no_external_resources(self, page):
        """
        The demo unit runs offline. A CDN script or webfont would work on
        a laptop and fail silently at the booth, which is the worst
        possible time to discover it.
        """
        external = re.findall(r'(?:src|href)\s*=\s*"(https?://[^"]+)"', page)

        assert external == [], f"external resources would break offline: {external}"

    def test_talks_to_the_real_endpoints(self, page):
        """/api/faces is gone: face recognition was deprecated 2026-09-14."""
        for path in ("/api/status", "/api/detection/status", "/api/events",
                     "/api/camera/stream", "/api/auth/login"):
            assert path in page, f"dashboard never calls {path}"

    def test_shows_live_throughput_and_events(self, page):
        """The page exists to answer 'is it running and did it see me'."""
        assert "detect_fps" in page, "no live frame rate on the page"
        assert "open_tracks" in page, "nothing shows a detection in progress"
        assert "setInterval" in page, "page never refreshes"

    def test_stream_carries_a_token(self, page):
        """
        An <img> cannot set an Authorization header. If the dashboard does
        not put the token in the query string, video is blank whenever
        auth is on -- and auth is always on through the tunnel.
        """
        assert 'q.set("token"' in page or "token=" in page

    def test_face_names_are_not_interpolated_as_html(self, page):
        """
        Enrolled names are attacker-influenced text. They must reach the
        DOM via textContent, never innerHTML.
        """
        assert "textContent" in page
        assert not re.search(r"innerHTML\s*=\s*[^;]*\bf\.name\b", page)
        assert not re.search(r"innerHTML\s*\+=", page)

    def test_token_is_not_persisted_to_local_storage(self, page):
        """
        sessionStorage dies with the tab; localStorage would leave a live
        token on a shared demo laptop.
        """
        assert "sessionStorage" in page
        # Match calls, not the word: the comment explaining this choice
        # names localStorage, and prose is not a security finding.
        assert not re.search(r"\blocalStorage\s*\.", page)

    def test_has_a_viewport_for_phones(self, page):
        assert 'name="viewport"' in page
