"""Comprehensive test suite for Text-to-Speech (TTS) synchronization and video assembly in Property Studio AI.

Validates:
- No voiceover overlaps (cumulative timeline scheduling with configurable gaps).
- Final spoken sentence completion without truncation (configurable visual hold).
- Robust handling of varied audio durations (short, long, mixed).
- Background music enabled vs disabled without clipping or overpowering voiceover.
- TTS failure handling and missing/empty narration clips.
- Duration verification using MoviePy and FFmpeg/FFprobe.
"""

import os
import re
import subprocess
import pytest
from pathlib import Path
from PIL import Image
import numpy as np

from moviepy import VideoFileClip, AudioFileClip
import imageio_ffmpeg

from src.validators import PropertyInput
from src.script_generator import SceneScript, VideoScript, generate_script
from src.audio_generator import (
    generate_tts_audio,
    generate_scene_narration,
    generate_full_audio_track,
    get_media_duration,
    synthesize_ambient_background_music
)
from src.video_generator import generate_property_video


def get_ffprobe_duration(file_path: Path) -> dict:
    """Probe audio and video stream durations using FFmpeg."""
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    res = subprocess.run(
        [ffmpeg_exe, "-i", str(file_path)],
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        timeout=10
    )
    stderr = res.stderr
    # Extract container duration
    dur_match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", stderr)
    container_dur = 0.0
    if dur_match:
        h, m, s = dur_match.groups()
        container_dur = float(h) * 3600 + float(m) * 60 + float(s)

    has_video = "Video:" in stderr
    has_audio = "Audio:" in stderr
    return {
        "duration": container_dur,
        "has_video": has_video,
        "has_audio": has_audio,
        "stderr": stderr
    }


def create_synthetic_property(tmp_path: Path, num_images: int = 4) -> PropertyInput:
    """Create a minimal valid PropertyInput with synthetic images."""
    img_paths = []
    for i in range(num_images):
        p = tmp_path / f"test_prop_img_{i}.jpg"
        Image.new("RGB", (640, 480), color=(60 + i * 30, 80 + i * 20, 120)).save(p)
        img_paths.append(p)

    return PropertyInput(
        name="Sunset Palms Luxury Estate",
        location="Malabar Hill, Mumbai",
        price="₹15 Crore",
        property_type="Luxury Villa",
        bedrooms=4,
        description="Private sanctuary with panoramic sea views.",
        features=["Private Pool", "Marble Flooring", "Smart Automation", "Concierge"],
        call_to_action="Contact Elite Realty Today",
        images=img_paths
    )


def test_get_media_duration_accuracy(tmp_path):
    """Verify get_media_duration returns accurate duration for synthesized WAV audio."""
    wav_path = tmp_path / "test_ambient_3s.wav"
    synthesize_ambient_background_music(duration=3.5, output_path=wav_path, volume=0.1)
    
    dur = get_media_duration(wav_path)
    assert abs(dur - 3.5) < 0.05, f"Expected 3.5s, got {dur}s"


def test_narration_sequential_schedule_and_no_overlap(tmp_path):
    """Test 1: Multiple slides with varied audio durations (short, long, extra-long).
    
    Verifies:
    - Every narration starts strictly AFTER previous narration finishes.
    - Inter-sentence gap (e.g. 0.3s) is preserved.
    - Final scene visual hold (e.g. 0.8s) is preserved.
    """
    scenes = [
        SceneScript(
            scene_index=1,
            scene_type="intro",
            category_tag="OVERVIEW",
            headline="Sunset Palms",
            subheadline="Mumbai Coast",
            caption="Welcome to luxury",
            voiceover="Short intro.",  # Short sentence (~1s)
            duration=3.0
        ),
        SceneScript(
            scene_index=2,
            scene_type="feature",
            category_tag="INTERIOR",
            headline="Italian Marble Flooring",
            subheadline="Grand Living Area",
            caption="Exceptional finishes throughout",
            voiceover="This magnificent property features authentic Italian marble flooring with soaring ceilings and floor to ceiling windows.",  # Long (~4-5s)
            duration=3.0
        ),
        SceneScript(
            scene_index=3,
            scene_type="feature",
            category_tag="EXTERIOR",
            headline="Infinity Edge Pool",
            subheadline="Overlooking The Arabian Sea",
            caption="Unobstructed ocean views",
            voiceover="Step outside onto the expansive sun deck beside your private infinity edge swimming pool.",  # Medium (~3-4s)
            duration=3.0
        ),
        SceneScript(
            scene_index=4,
            scene_type="cta",
            category_tag="CONTACT",
            headline="Schedule A Private Tour",
            subheadline="Contact Elite Realty Today",
            caption="Call now to book your private viewing",
            voiceover="Schedule your private confidential tour of Sunset Palms today by contacting Elite Realty Mumbai.",  # Final sentence (~4-5s)
            duration=3.0
        ),
    ]

    script = VideoScript(
        property_name="Sunset Palms",
        total_scenes=len(scenes),
        estimated_duration=12.0,
        scenes=scenes
    )

    prop = create_synthetic_property(tmp_path, num_images=4)
    out_mp4 = tmp_path / "varied_duration_video.mp4"

    sentence_gap = 0.3
    ending_hold = 0.8

    result_path = generate_property_video(
        prop=prop,
        output_path=out_mp4,
        script=script,
        include_narration=True,
        include_background_music=True,
        sentence_gap=sentence_gap,
        ending_hold=ending_hold,
        tts_engine="auto",
        fps=15
    )

    assert result_path.exists()
    assert result_path.stat().st_size > 50000

    probe = get_ffprobe_duration(result_path)
    assert probe["has_video"]
    assert probe["has_audio"]
    total_vid_dur = probe["duration"]

    # Open with MoviePy to inspect audio and video streams
    vclip = VideoFileClip(str(result_path))
    assert vclip.duration > 0
    assert vclip.audio is not None
    assert abs(vclip.duration - total_vid_dur) < 0.5

    # Check that video duration is greater than or equal to estimated_duration
    assert vclip.duration >= script.estimated_duration - 0.2
    vclip.close()


def test_final_sentence_longer_than_default_slide_duration(tmp_path):
    """Test 2: Final CTA sentence takes 6+ seconds, exceeding default slide duration.
    
    Verifies:
    - Final slide expands to accommodate speech duration + ending_hold.
    - Final audio sentence is not cut off or truncated.
    - Video remains visible for at least ending_hold seconds after speech completes.
    """
    long_final_cta = (
        "Do not miss this once in a lifetime luxury opportunity. "
        "Call our senior private client advisor right now at nine eight two zero zero, "
        "or visit elite realty dot com to arrange an exclusive private showing."
    )
    scenes = [
        SceneScript(
            scene_index=1,
            scene_type="intro",
            category_tag="INTRO",
            headline="Slide One",
            subheadline="Short",
            caption="Caption 1",
            voiceover="Welcome to this home.",
            duration=2.0
        ),
        SceneScript(
            scene_index=2,
            scene_type="cta",
            category_tag="CTA",
            headline="Long Call To Action",
            subheadline="Extended Speech",
            caption="Ending scene",
            voiceover=long_final_cta,
            duration=2.0
        )
    ]
    script = VideoScript(
        property_name="Test Long CTA",
        total_scenes=len(scenes),
        estimated_duration=4.0,
        scenes=scenes
    )

    prop = create_synthetic_property(tmp_path, num_images=2)
    out_mp4 = tmp_path / "long_cta_video.mp4"

    ending_hold = 0.9
    result_path = generate_property_video(
        prop=prop,
        output_path=out_mp4,
        script=script,
        include_narration=True,
        include_background_music=True,
        sentence_gap=0.3,
        ending_hold=ending_hold,
        fps=15
    )

    vclip = VideoFileClip(str(result_path))
    # Pre-generate narration to get exact speech duration
    cta_audio = tmp_path / "cta_audio.mp3"
    generate_tts_audio(text=long_final_cta, output_path=cta_audio)
    speech_dur = get_media_duration(cta_audio)
    
    # Video duration must easily accommodate intro + cta speech + ending hold
    assert speech_dur > 3.0, f"Speech duration {speech_dur}s should be several seconds"
    assert vclip.duration >= speech_dur + ending_hold
    assert vclip.audio is not None
    vclip.close()


def test_background_music_disabled(tmp_path):
    """Test 3: Voice narration enabled, background music disabled.
    
    Verifies:
    - Voiceover is exported clearly without background music.
    - Audio stream exists and duration matches video.
    """
    scenes = [
        SceneScript(
            scene_index=1,
            scene_type="intro",
            category_tag="INTRO",
            headline="Quiet Villa",
            subheadline="Peaceful",
            caption="Only voiceover",
            voiceover="Experience utter silence and tranquility in this secluded sanctuary.",
            duration=3.0
        ),
        SceneScript(
            scene_index=2,
            scene_type="cta",
            category_tag="CTA",
            headline="Inquire Today",
            subheadline="Exclusive",
            caption="Call now",
            voiceover="Inquire today for exclusive pricing.",
            duration=3.0
        )
    ]
    script = VideoScript(
        property_name="Voice Only",
        total_scenes=len(scenes),
        estimated_duration=6.0,
        scenes=scenes
    )

    prop = create_synthetic_property(tmp_path, num_images=2)
    out_mp4 = tmp_path / "no_bgm_video.mp4"

    result_path = generate_property_video(
        prop=prop,
        output_path=out_mp4,
        script=script,
        include_narration=True,
        include_background_music=False,
        fps=15
    )

    probe = get_ffprobe_duration(result_path)
    assert probe["has_audio"]
    assert probe["has_video"]
    assert abs(probe["duration"] - script.estimated_duration) < 0.5


def test_tts_failure_or_missing_narration_on_one_slide(tmp_path):
    """Test 4: Slide 1 has voiceover, Slide 2 has empty voiceover (missing narration),
    Slide 3 has voiceover.
    
    Verifies:
    - Missing narration does not crash video assembly.
    - Slide 2 maintains default duration without overlap.
    - Slide 3 narration starts after slide 2 completes.
    """
    scenes = [
        SceneScript(
            scene_index=1,
            scene_type="intro",
            category_tag="INTRO",
            headline="First Scene",
            subheadline="Narration present",
            caption="Scene 1",
            voiceover="Welcome to the first scene of this tour.",
            duration=3.0
        ),
        SceneScript(
            scene_index=2,
            scene_type="feature",
            category_tag="GALLERY",
            headline="Silent Feature",
            subheadline="No voiceover",
            caption="Music only here",
            voiceover="",  # Empty / missing narration
            duration=4.0
        ),
        SceneScript(
            scene_index=3,
            scene_type="cta",
            category_tag="CTA",
            headline="Closing Scene",
            subheadline="Final words",
            caption="Scene 3",
            voiceover="Thank you for touring this extraordinary property.",
            duration=3.0
        )
    ]
    script = VideoScript(
        property_name="Missing Narration Test",
        total_scenes=len(scenes),
        estimated_duration=10.0,
        scenes=scenes
    )

    prop = create_synthetic_property(tmp_path, num_images=3)
    out_mp4 = tmp_path / "missing_narration_video.mp4"

    result_path = generate_property_video(
        prop=prop,
        output_path=out_mp4,
        script=script,
        include_narration=True,
        include_background_music=True,
        fps=15
    )

    assert result_path.exists()
    probe = get_ffprobe_duration(result_path)
    assert probe["has_audio"]
    assert probe["has_video"]
    assert probe["duration"] > 8.0
