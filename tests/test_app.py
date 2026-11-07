"""End-to-end tests of the Streamlit app using Streamlit's headless AppTest."""

import io

import pytest
import streamlit
from PIL import Image
from streamlit.testing.v1 import AppTest

from tests.conftest import ROOT, SAMPLES

APP = str(ROOT / "app" / "app.py")
TIMEOUT = 300  # first run loads the model


def new_app():
    return AppTest.from_file(APP, default_timeout=TIMEOUT).run()


def click(at, text):
    next(b for b in at.button if text in b.label).click().run()
    return at


def result_text(at):
    boxes = [m.value for m in at.markdown if '<div class="prediction-box' in m.value]
    assert len(boxes) == 1, "expected exactly one prediction result"
    return boxes[0]


def assert_clean(at):
    assert not at.exception, [e.value for e in at.exception]


class FakeUpload(io.BytesIO):
    def __init__(self, data: bytes, name: str):
        super().__init__(data)
        self.name = name


def fake_uploader(monkeypatch, data, name):
    monkeypatch.setattr(streamlit, "file_uploader", lambda *a, **k: FakeUpload(data, name))


def encode(image: Image.Image, fmt: str) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()


def test_startup():
    at = new_app()
    assert_clean(at)
    assert not at.error
    labels = [b.label for b in at.button]
    assert any("Normal Sample" in l for l in labels)
    assert not any("Analyze" in l for l in labels)  # nothing selected yet


@pytest.mark.parametrize("button,label", [("Normal Sample", "NORMAL"),
                                          ("Pneumonia Sample", "PNEUMONIA")])
def test_sample_then_analyze(button, label):
    """Regression: the sample used to be lost on the rerun caused by Analyze."""
    at = click(new_app(), button)
    click(at, "Analyze")
    assert_clean(at)
    assert at.session_state["selected_sample"] == label.lower()
    text = result_text(at)
    assert ("PNEUMONIA DETECTED" in text) == (label == "PNEUMONIA")
    # Grad-CAM pair rendered and caption names the explained class
    assert any(f"predicted class ({label})" in c.value for c in at.caption)
    # sidebar logo + selected input image + Grad-CAM original + heatmap
    assert len(at.get("imgs")) == 4


def test_switch_samples():
    at = click(new_app(), "Normal Sample")
    click(at, "Analyze")
    assert "NORMAL" in result_text(at)
    click(at, "Pneumonia Sample")
    assert at.session_state["selected_sample"] == "pneumonia"
    click(at, "Analyze")
    assert_clean(at)
    assert "PNEUMONIA DETECTED" in result_text(at)


@pytest.mark.parametrize("fmt,mode,name", [
    ("JPEG", "RGB", "xray.jpg"),
    ("PNG", "RGB", "xray.png"),
    ("PNG", "L", "gray.png"),
    ("JPEG", "L", "gray.jpeg"),
    ("PNG", "RGBA", "rgba.png"),
])
def test_upload_then_analyze(monkeypatch, fmt, mode, name):
    data = encode(Image.open(SAMPLES["PNEUMONIA"]).convert(mode), fmt)
    fake_uploader(monkeypatch, data, name)
    at = new_app()
    assert_clean(at)
    click(at, "Analyze")
    assert_clean(at)
    assert at.session_state["image_source"] == name
    assert "PNEUMONIA" in result_text(at)


@pytest.mark.parametrize("data,name", [
    (b"this is not an image", "fake.png"),
    (encode(Image.open(SAMPLES["NORMAL"]), "JPEG")[:2000], "truncated.jpg"),
    (b"", "empty.jpeg"),
])
def test_invalid_upload_shows_error(monkeypatch, data, name):
    fake_uploader(monkeypatch, data, name)
    at = new_app()
    assert_clean(at)
    assert any("Could not read this file" in e.value for e in at.error)
    assert not any("Analyze" in b.label for b in at.button)
