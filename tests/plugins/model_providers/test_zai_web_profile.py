"""The z-api (chat.z.ai browser bridge) provider profile.

z-api answers bare ``/models`` in Gemini's shape, which the default parser reads as an empty
catalog; the profile asks for the codex catalog instead. Its base URL is loopback, and URL->
provider inference is a netloc substring match, so it must not claim other local servers.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


@pytest.fixture
def profile():
    import model_tools  # noqa: F401  (runs plugin discovery)
    import providers

    return providers.get_provider_profile("z-api")


@pytest.fixture
def bridge():
    """A loopback stand-in answering like z-api: codex catalog only with ``client_version``."""
    seen: list[tuple[str, str | None]] = []
    catalog = {"models": [{"slug": "glm-web"}, {"slug": "gemini-web"}, {"slug": "GLM-5.3"}, {"slug": "GLM-5.2"}]}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            seen.append((self.path, self.headers.get("Authorization")))
            body = catalog if "client_version=" in self.path else {"models": [{"name": "models/glm-web"}]}
            payload = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/v1", seen, catalog
    server.shutdown()


def test_fetch_models_lists_the_picker_behind_glm_web(profile, bridge):
    base_url, seen, catalog = bridge

    assert profile.fetch_models(api_key="bridge-token", base_url=base_url) == ["glm-web", "GLM-5.3", "GLM-5.2"]
    assert seen[-1][1] == "Bearer bridge-token"

    catalog["models"] = []  # z-api serves no catalog until a turn has read the picker
    assert profile.fetch_models(api_key="bridge-token", base_url=base_url) == ["glm-web"]

    unreachable = base_url.replace(base_url.split(":")[-1].split("/")[0], "9")
    assert profile.fetch_models(api_key="bridge-token", base_url=unreachable, timeout=1.0) is None


def test_url_inference_claims_only_the_bridge_port(profile):
    from agent.model_metadata import _infer_provider_from_url

    assert _infer_provider_from_url(profile.base_url) == profile.name
    for other in ("http://127.0.0.1:11434/v1", "http://127.0.0.1:8080/v1", "http://localhost:1234/v1"):
        assert _infer_provider_from_url(other) != profile.name
