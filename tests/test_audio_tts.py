"""Tests for Text-to-Speech (TTS) narration synthesis and audio generation."""

import pytest
from pathlib import Path
from src.audio_generator import generate_tts_audio, generate_scene_narration
from src.script_generator import SceneScript


def test_generate_tts_audio_clean_text(tmp_path):
    """Verify TTS generates audio file with valid size for spoken narration."""
    out_mp3 = tmp_path / "narration_test.mp3"
    result = generate_tts_audio(
        text="Welcome to Green Valley Residency, Chandigarh.",
        output_path=out_mp3,
        tld="co.in",
        lang="en",
        slow=False,
        engine_mode="auto"
    )
    assert result is not None
    assert result.exists()
    assert result.stat().st_size > 500


def test_generate_tts_audio_empty_text(tmp_path):
    """Verify empty text returns None without crashing."""
    out_mp3 = tmp_path / "empty.mp3"
    result = generate_tts_audio(
        text="",
        output_path=out_mp3
    )
    assert result is None
    assert not out_mp3.exists()


def test_generate_scene_narration(tmp_path):
    """Verify generate_scene_narration properly integrates with SceneScript."""
    scene = SceneScript(
        scene_index=1,
        scene_type="intro",
        category_tag="PROPERTY OVERVIEW",
        headline="Ocean Crest Luxury Villa",
        subheadline="Goa Coast • ₹6.5 Crore",
        caption="Welcome to luxury living",
        voiceover="Welcome to Ocean Crest Luxury Villa located along the serene Goa Coast.",
        duration=4.0
    )
    out_file = tmp_path / "scene_1_narration.mp3"
    result = generate_scene_narration(scene, out_file)
    assert result is not None
    assert result.exists()
    assert result.stat().st_size > 500
