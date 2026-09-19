"""Z.ai web bridge (z-api) provider profile.

z-api drives a signed-in chat.z.ai tab through a Chromium extension and serves it on
loopback as an OpenAI Responses endpoint (``/v1/responses``), authenticated with its
``BRIDGE_TOKEN``. Tools are taught in-prompt and parsed back into ``function_call``
items, so the stock Responses transport works unchanged. What differs from a real API:

- ``prompt_cache_key`` pins a Hermes session to one browser conversation; later turns
  send only the new items.
- No token streaming: the reply is scraped once it settles, and a silent model is
  paced by a ``response.in_progress`` keepalive every 20s (``stream_keepalive_seconds``).
- One Z.ai identity runs one conversation at a time. Concurrent calls (title
  generation, background review, parallel subagents) queue behind each other in the
  bridge instead of failing.
- ``glm-web`` answers with whatever model the Z.ai picker has selected; any other id
  selects that picker entry for the turn.
"""

import json
import logging
import urllib.request

from providers import register_provider
from providers.base import ProviderProfile

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "glm-web"
# Slugs z-api treats as "leave the picker alone" (``gemini-web`` is its apollo-api spelling).
_PICKER_UNCHANGED = frozenset({DEFAULT_MODEL, "gemini-web"})


class ZaiWebProfile(ProviderProfile):
    """z-api browser bridge: Responses wire, Codex-shaped catalog, no reasoning dial."""

    def supported_reasoning_efforts(self, model: str | None) -> tuple[str, ...]:
        """The web picker exposes no effort control, so reasoning fields are omitted."""
        return ()

    def fetch_models(
        self, *, api_key: str | None = None, base_url: str | None = None, timeout: float = 8.0
    ) -> list[str] | None:
        """Z.ai's picker as z-api caches it, with ``glm-web`` first.

        Bare ``/models`` answers in Gemini's shape, which the default parser reads as an
        empty catalog. With ``client_version`` z-api serves its codex catalog instead,
        from cache and without opening a tab; that catalog stays empty until a turn has
        read the picker.
        """
        from hermes_cli.urllib_security import open_credentialed_url

        base = (base_url or self.base_url).rstrip("/")
        req = urllib.request.Request(f"{base}/models?client_version=hermes", headers={"Accept": "application/json"})
        if api_key:
            req.add_header("Authorization", f"Bearer {api_key}")
        try:
            with open_credentialed_url(req, timeout=timeout) as resp:
                entries = json.loads(resp.read().decode()).get("models", [])
        except Exception as exc:
            logger.debug("fetch_models(%s): %s", self.name, exc)
            return None
        slugs = [e["slug"] for e in entries if isinstance(e, dict) and isinstance(e.get("slug"), str)]
        return [DEFAULT_MODEL, *(slug for slug in slugs if slug not in _PICKER_UNCHANGED)]


zai_web = ZaiWebProfile(
    name="zai-web", aliases=("z-api", "zai-api", "zai-bridge"), api_mode="codex_responses",
    display_name="Z.ai Web (z-api bridge)",
    description="chat.z.ai through the local z-api browser bridge (unofficial)",
    signup_url="https://chat.z.ai/",
    # The key is z-api's BRIDGE_TOKEN (``zai-api print-token``).
    env_vars=("ZAI_BRIDGE_TOKEN", "ZAI_BRIDGE_BASE_URL"),
    base_url="http://127.0.0.1:3230/v1",
    # URL->provider inference matches this as a substring of the netloc; a bare
    # 127.0.0.1 would claim every loopback endpoint (Ollama, llama.cpp, LM Studio).
    hostname="127.0.0.1:3230",
    stream_keepalive_seconds=20.0,
    fallback_models=(DEFAULT_MODEL,),
)

register_provider(zai_web)
