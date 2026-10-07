"""Inworld TTS + instant voice cloning over the official REST API.

Endpoints (docs.inworld.ai, checked 2026-09):
  POST https://api.inworld.ai/voices/v1/voices:clone   -> {"voice": {"voiceId": ...}}
  POST https://api.inworld.ai/tts/v1/voice             -> {"audioContent": <base64>}
Auth: "Authorization: Basic <INWORLD_API_KEY>" (the key is already Base64 credentials).
"""
from __future__ import annotations

import base64
import io
import os
import wave
from pathlib import Path

import requests

from services.audio_provider import AudioProvider
from services.config import InworldConfig
from utils.retry import PermanentError, RateLimitError, TransientError

API_BASE = "https://api.inworld.ai"
MAX_TEXT_CHARS = 2000


def _raise_for_status(resp: requests.Response, action: str) -> None:
    if resp.ok:
        return
    try:
        detail = resp.json().get("message") or resp.text
    except ValueError:
        detail = resp.text
    detail = (detail or "").strip()[:500]
    msg = f"Inworld {action} failed: HTTP {resp.status_code}. {detail}"
    if resp.status_code in (401, 403):
        raise PermanentError(msg + " Check INWORLD_API_KEY in .env.")
    if resp.status_code == 429:
        raise RateLimitError(msg)
    if resp.status_code >= 500:
        raise TransientError(msg)
    raise PermanentError(msg)


def pcm_to_wav(pcm: bytes, sample_rate: int, channels: int = 1) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


class InworldAudioProvider(AudioProvider):
    name = "inworld"

    def __init__(self, cfg: InworldConfig, sample_rate: int, language: str = "en-US",
                 api_key: str | None = None, session: requests.Session | None = None):
        self.cfg = cfg
        self.sample_rate = sample_rate
        self.language = language
        self.api_key = api_key or os.environ.get("INWORLD_API_KEY", "").strip()
        if not self.api_key:
            raise PermanentError("INWORLD_API_KEY is not set. Copy .env.example to .env and add your Inworld key.")
        self.session = session or requests.Session()

    def _post(self, path: str, payload: dict, action: str) -> dict:
        try:
            resp = self.session.post(
                f"{API_BASE}{path}", json=payload, timeout=self.cfg.timeout_seconds,
                headers={"Authorization": f"Basic {self.api_key}", "Content-Type": "application/json"},
            )
        except (requests.Timeout, requests.ConnectionError) as exc:
            raise TransientError(f"Inworld {action}: network error ({exc.__class__.__name__}: {exc})") from exc
        _raise_for_status(resp, action)
        return resp.json()

    def clone_voice(self, sample_path: Path, display_name: str, language: str) -> str:
        payload = {
            "displayName": display_name[:100],
            "languageCode": language,
            "voiceSamples": [{"audioData": base64.b64encode(sample_path.read_bytes()).decode("ascii")}],
            "audioProcessingConfig": {"removeBackgroundNoise": True},
        }
        data = self._post("/voices/v1/voices:clone", payload, "voice cloning")
        voice_id = (data.get("voice") or {}).get("voiceId")
        if not voice_id:
            raise PermanentError(f"Inworld voice cloning returned no voiceId: {str(data)[:300]}")
        return voice_id

    def generate(self, text: str, voice_id: str, output_path: Path) -> Path:
        if len(text) > MAX_TEXT_CHARS:
            raise PermanentError(f"Scene text is {len(text)} chars; Inworld allows {MAX_TEXT_CHARS} per request.")
        payload = {
            "text": text,
            "voiceId": voice_id,
            "modelId": self.cfg.model_id,
            "language": self.language,
            "temperature": self.cfg.temperature,
            "audioConfig": {
                "audioEncoding": "LINEAR16",
                "sampleRateHertz": self.sample_rate,
                "speakingRate": self.cfg.speaking_rate,
            },
        }
        data = self._post("/tts/v1/voice", payload, "speech synthesis")
        content = data.get("audioContent")
        if not content:
            raise TransientError("Inworld speech synthesis returned no audioContent")
        audio = base64.b64decode(content)
        if not audio.startswith(b"RIFF"):  # raw PCM without a header
            audio = pcm_to_wav(audio, self.sample_rate)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = output_path.with_suffix(".part")
        tmp.write_bytes(audio)
        os.replace(tmp, output_path)
        return output_path
