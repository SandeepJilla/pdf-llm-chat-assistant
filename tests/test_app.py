import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pdf_chat_app as app_module  # noqa: E402


class FakeUpload:
    def __init__(self, name, data):
        self.filename = name
        self._buf = io.BytesIO(data)

    def read(self, n=-1):
        return self._buf.read(n)


def test_health_degraded_without_key(monkeypatch):
    monkeypatch.setattr(app_module, "OPENROUTER_API_KEY", "XXXXX_API_KEY_XXXXX")
    client = app_module.app.test_client()
    body = client.get("/health").get_json()
    assert body["status"] == "degraded"
    assert body["api_ok"] is False


def test_health_ok_with_key(monkeypatch):
    monkeypatch.setattr(app_module, "OPENROUTER_API_KEY", "sk-or-v1-test")
    client = app_module.app.test_client()
    assert client.get("/health").get_json()["status"] == "ok"


def test_parse_text_file():
    name, text = app_module.parse_uploaded_file(FakeUpload("notes.txt", b"hello world"))
    assert name == "notes.txt"
    assert text == "hello world"


def test_parse_unsupported_extension():
    _, text = app_module.parse_uploaded_file(FakeUpload("malware.exe", b"data"))
    assert text.startswith("[Unsupported file type")


def test_parse_truncates_large_file(monkeypatch):
    monkeypatch.setattr(app_module, "MAX_FILE_BYTES", 10)
    _, text = app_module.parse_uploaded_file(FakeUpload("big.txt", b"x" * 50))
    assert text.endswith("[FILE TRUNCATED]")


def test_parse_invalid_pdf_does_not_raise():
    _, text = app_module.parse_uploaded_file(FakeUpload("bad.pdf", b"not a pdf"))
    assert text.startswith("[")


def test_set_model_requires_value():
    client = app_module.app.test_client()
    resp = client.post("/set-model", json={"model": ""})
    assert resp.status_code == 400


def test_models_endpoint_returns_current():
    client = app_module.app.test_client()
    body = client.get("/models").get_json()
    assert "models" in body and "current" in body
