"""Inworld provider against a fake HTTP session: no network, no paid calls."""
import base64
import wave

import pytest

from services.config import InworldConfig
from services.inworld import InworldAudioProvider
from utils.retry import PermanentError, TransientError


class FakeResponse:
    def __init__(self, status, body):
        self.status_code, self._body = status, body
        self.ok = 200 <= status < 300
        self.text = str(body)

    def json(self):
        return self._body


class FakeSession:
    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []

    def post(self, url, json, timeout, headers):
        self.calls.append({"url": url, "json": json, "headers": headers})
        return self.responses.pop(0)


def provider(session):
    return InworldAudioProvider(InworldConfig(), 48000, api_key="KEY", session=session)


def test_synthesis_request_shape_and_raw_pcm_is_wrapped(tmp_path):
    pcm = b"\x00\x00" * 48000  # 1 s of silence, no WAV header
    s = FakeSession(FakeResponse(200, {"audioContent": base64.b64encode(pcm).decode()}))
    out = provider(s).generate("Hello world.", "ws__voice", tmp_path / "a.wav")
    call = s.calls[0]
    assert call["url"] == "https://api.inworld.ai/tts/v1/voice"
    assert call["headers"]["Authorization"] == "Basic KEY"
    assert call["json"]["voiceId"] == "ws__voice" and call["json"]["modelId"] == "inworld-tts-2"
    assert call["json"]["audioConfig"]["audioEncoding"] == "LINEAR16"
    with wave.open(str(out)) as w:
        assert w.getframerate() == 48000 and w.getnframes() == 48000


def test_clone_returns_voice_id(tmp_path):
    sample = tmp_path / "s.wav"
    sample.write_bytes(b"RIFFfake")
    s = FakeSession(FakeResponse(200, {"voice": {"voiceId": "ws__me_123"}}))
    assert provider(s).clone_voice(sample, "Me", "en-US") == "ws__me_123"
    body = s.calls[0]["json"]
    assert s.calls[0]["url"].endswith("/voices/v1/voices:clone")
    assert body["languageCode"] == "en-US" and body["voiceSamples"][0]["audioData"]


@pytest.mark.parametrize("status, exc", [(401, PermanentError), (400, PermanentError),
                                         (429, TransientError), (503, TransientError)])
def test_error_classification(tmp_path, status, exc):
    s = FakeSession(FakeResponse(status, {"message": "nope"}))
    with pytest.raises(exc):
        provider(s).generate("Hi.", "v", tmp_path / "a.wav")


def test_missing_key_is_permanent(monkeypatch):
    monkeypatch.delenv("INWORLD_API_KEY", raising=False)
    with pytest.raises(PermanentError):
        InworldAudioProvider(InworldConfig(), 48000)


def test_text_limit(tmp_path):
    with pytest.raises(PermanentError):
        provider(FakeSession()).generate("x" * 2001, "v", tmp_path / "a.wav")
