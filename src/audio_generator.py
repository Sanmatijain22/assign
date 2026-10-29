"""Audio and narration generator for Property Studio AI.

Provides:
- Text-to-Speech (TTS) narration using gTTS with fallback.
- 100% Royalty-Free procedural ambient background music generator (synthesized locally via NumPy/WAV).
- Audio mixing: combines voiceover narration and background music with automatic ducking.
- Graceful fallbacks ensuring video generation succeeds even if audio fails.
"""

from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import math
import struct
import wave
import numpy as np

from .script_generator import SceneScript, VideoScript


def get_media_duration(file_path: Union[str, Path, None]) -> float:
    """Measure exact duration of an audio or video file in seconds.
    
    Uses wave header parsing for fast WAV duration, MoviePy AudioFileClip for general formats,
    and imageio_ffmpeg probing as fallback. Returns 0.0 on missing or unreadable files.
    """
    if not file_path:
        return 0.0
    p = Path(file_path)
    if not p.exists() or p.stat().st_size == 0:
        return 0.0

    # 1. Fast WAV header reading
    if p.suffix.lower() == ".wav":
        try:
            with wave.open(str(p), "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                if rate > 0:
                    return float(frames) / float(rate)
        except Exception:
            pass

    # 2. MoviePy AudioFileClip
    try:
        from moviepy import AudioFileClip
        clip = AudioFileClip(str(p))
        dur = float(clip.duration) if clip.duration is not None else 0.0
        clip.close()
        if dur > 0.0:
            return dur
    except Exception:
        pass

    # 3. Fallback: probe via imageio_ffmpeg
    try:
        import subprocess
        import re
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        res = subprocess.run(
            [ffmpeg_exe, "-i", str(p)],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            timeout=5
        )
        match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", res.stderr)
        if match:
            h, m, s = match.groups()
            return float(h) * 3600 + float(m) * 60 + float(s)
    except Exception:
        pass

    return 0.0


def synthesize_ambient_background_music(
    duration: float,
    output_path: Path,
    sample_rate: int = 44100,
    volume: float = 0.18
) -> Path:
    """Generate a warm, peaceful, elegant ambient chord progression as a stereo WAV file.
    
    100% procedural, offline, and royalty-free. No external music files or copyrights needed.
    Uses gentle musical chords (C maj7 -> Am9 -> F maj7 -> G sus4) with soft envelope fading.
    """
    num_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    
    # 4-chord sequence looping across the duration
    chords = [
        [261.63, 329.63, 392.00, 493.88],  # C maj7 (C4, E4, G4, B4)
        [220.00, 261.63, 329.63, 392.00],  # Am7 (A3, C4, E4, G4)
        [174.61, 220.00, 261.63, 329.63],  # F maj7 (F3, A3, C4, E4)
        [196.00, 246.94, 293.66, 392.00],  # G (G3, B3, D4, G4)
    ]
    
    chord_len = 5.0  # seconds per chord
    audio_left = np.zeros(num_samples, dtype=np.float32)
    audio_right = np.zeros(num_samples, dtype=np.float32)

    for i in range(int(math.ceil(duration / chord_len))):
        start_t = i * chord_len
        end_t = min(duration, (i + 1) * chord_len)
        idx_start = int(start_t * sample_rate)
        idx_end = int(end_t * sample_rate)
        seg_len = idx_end - idx_start
        if seg_len <= 0:
            break
            
        chord = chords[i % len(chords)]
        seg_t = np.linspace(0, (idx_end - idx_start) / sample_rate, seg_len, endpoint=False)
        
        # Soft attack and release envelope
        envelope = np.ones(seg_len, dtype=np.float32)
        attack_len = int(0.6 * sample_rate)
        release_len = int(1.2 * sample_rate)
        
        if attack_len < seg_len:
            envelope[:attack_len] = np.sin(np.linspace(0, np.pi / 2, attack_len))
        if release_len < seg_len:
            envelope[-release_len:] = np.cos(np.linspace(0, np.pi / 2, release_len))

        # Sum fundamental sine waves + warm harmonics
        chord_signal_l = np.zeros(seg_len, dtype=np.float32)
        chord_signal_r = np.zeros(seg_len, dtype=np.float32)
        
        for freq_idx, freq in enumerate(chord):
            # Slightly detuned stereo voices for richness
            pan = (freq_idx / (len(chord) - 1)) * 0.4 + 0.3  # pan slightly
            voice_l = np.sin(2 * np.pi * freq * seg_t) + 0.35 * np.sin(2 * np.pi * (freq * 2) * seg_t)
            voice_r = np.sin(2 * np.pi * (freq * 1.002) * seg_t) + 0.35 * np.sin(2 * np.pi * (freq * 2 * 1.002) * seg_t)
            
            chord_signal_l += voice_l * (1.0 - pan)
            chord_signal_r += voice_r * pan
            
        chord_signal_l *= envelope * 0.25
        chord_signal_r *= envelope * 0.25

        audio_left[idx_start:idx_end] += chord_signal_l
        audio_right[idx_start:idx_end] += chord_signal_r

    # Overall fade-in and fade-out
    fade_in_samples = int(1.5 * sample_rate)
    fade_out_samples = int(2.5 * sample_rate)
    if num_samples > fade_in_samples:
        audio_left[:fade_in_samples] *= np.linspace(0, 1, fade_in_samples)
        audio_right[:fade_in_samples] *= np.linspace(0, 1, fade_in_samples)
    if num_samples > fade_out_samples:
        audio_left[-fade_out_samples:] *= np.linspace(1, 0, fade_out_samples)
        audio_right[-fade_out_samples:] *= np.linspace(1, 0, fade_out_samples)

    # Master volume
    audio_left = np.clip(audio_left * volume, -0.95, 0.95)
    audio_right = np.clip(audio_right * volume, -0.95, 0.95)

    # Convert to 16-bit PCM WAV
    int_left = (audio_left * 32767).astype(np.int16)
    int_right = (audio_right * 32767).astype(np.int16)
    
    interleaved = np.empty((num_samples * 2,), dtype=np.int16)
    interleaved[0::2] = int_left
    interleaved[1::2] = int_right

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), 'wb') as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(interleaved.tobytes())

    return output_path


def generate_tts_audio(
    text: str,
    output_path: Path,
    tld: str = 'co.in',
    lang: str = 'en',
    slow: bool = False,
    engine_mode: str = 'auto',
    voice_gender: str = 'female'
) -> Optional[Path]:
    """Generate Text-to-Speech (TTS) narration audio.
    
    Supports:
    - Online high-fidelity neural speech via gTTS (with regional accents: co.in, com, co.uk, com.au).
    - 100% offline Windows native speech synthesis via pyttsx3 (SAPI5).
    - Automatic graceful fallback if internet is unavailable or gTTS times out.
    """
    clean_text = text.strip() if text else ""
    if not clean_text:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)

    import re
    # Add a natural breathing pause between sentences so they don't rush together
    spoken_text = re.sub(r'([.?!])\s+', r'\1 ... ', clean_text)

    # 1. Try online gTTS if requested or in auto mode
    if engine_mode in ('auto', 'gtts'):
        try:
            from gtts import gTTS
            tts = gTTS(text=spoken_text, lang=lang, tld=tld, slow=slow)
            tts.save(str(output_path))
            if output_path.exists() and output_path.stat().st_size > 100:
                return output_path
        except Exception:
            # If strictly online was requested, don't fall back to offline unless in auto mode
            if engine_mode == 'gtts':
                return None

    # 2. Offline pyttsx3 synthesis (either in auto fallback or explicit offline mode)
    try:
        import pyttsx3
        engine = pyttsx3.init()
        # Set speaking rate (words per minute)
        engine.setProperty('rate', 140 if slow else 165)
        
        # Configure gender/voice
        voices = engine.getProperty('voices')
        if voices:
            if voice_gender.lower() == 'female':
                female_voices = [v for v in voices if 'zira' in v.name.lower() or 'female' in v.name.lower()]
                if female_voices:
                    engine.setProperty('voice', female_voices[0].id)
            else:
                male_voices = [v for v in voices if 'david' in v.name.lower() or 'male' in v.name.lower()]
                if male_voices:
                    engine.setProperty('voice', male_voices[0].id)
        
        # Output as WAV if pyttsx3
        wav_target = output_path.with_suffix('.wav')
        engine.save_to_file(clean_text, str(wav_target))
        engine.runAndWait()
        
        if wav_target.exists() and wav_target.stat().st_size > 100:
            return wav_target
    except Exception:
        pass

    return None


def generate_scene_narration(
    scene: SceneScript,
    output_path: Path,
    tld: str = 'co.in',
    lang: str = 'en',
    slow: bool = False,
    engine_mode: str = 'auto',
    voice_gender: str = 'female'
) -> Optional[Path]:
    """Generate TTS narration MP3/WAV for a single scene using gTTS or pyttsx3."""
    return generate_tts_audio(
        text=scene.voiceover,
        output_path=output_path,
        tld=tld,
        lang=lang,
        slow=slow,
        engine_mode=engine_mode,
        voice_gender=voice_gender
    )


def generate_full_audio_track(
    script: VideoScript,
    temp_dir: Path,
    include_narration: bool = True,
    include_background_music: bool = True,
    narration_schedule: Optional[List[Dict[str, Any]]] = None,
    scene_start_times: Optional[List[float]] = None,
    total_duration: Optional[float] = None,
    tld: str = 'co.in',
    lang: str = 'en',
    slow: bool = False,
    engine_mode: str = 'auto',
    voice_gender: str = 'female',
    bg_music_volume: float = 0.12,
    scene_audio_files: Optional[List[Optional[Path]]] = None,
    sentence_gap: float = 0.3,
    ending_hold: float = 0.8,
) -> Optional[Path]:
    """Assemble complete video audio track with narration and royalty-free ambient music.
    
    Synchronizes narration clips to accurate sequential start times based on actual measured
    audio durations, ensuring zero voiceover overlap and keeping background music softly ducked.
    """
    if not include_narration and not include_background_music:
        return None

    final_audio_path = temp_dir / "final_soundtrack.mp3"

    try:
        from moviepy import AudioFileClip, CompositeAudioClip
        audio_clips = []

        # 1. Narration clips
        if include_narration:
            # If a pre-calculated narration schedule was passed, use it directly
            if narration_schedule:
                for item in narration_schedule:
                    gen_path = item.get("audio_path")
                    dur = item.get("duration", 0.0)
                    start_t = float(item.get("start_time", 0.0))
                    if gen_path and Path(gen_path).exists() and dur > 0.0:
                        try:
                            clip = AudioFileClip(str(gen_path))
                            # Never truncate or subclip: narration plays in full
                            if hasattr(clip, "with_start"):
                                positioned = clip.with_start(start_t)
                            else:
                                positioned = clip.set_start(start_t)
                            audio_clips.append(positioned)
                        except Exception:
                            pass
            else:
                # Build schedule dynamically from scenes or pre-generated scene_audio_files
                curr_audio_time = 0.0
                total_scenes = len(script.scenes)
                for idx, scene in enumerate(script.scenes):
                    if scene_audio_files and idx < len(scene_audio_files) and scene_audio_files[idx]:
                        gen_path = scene_audio_files[idx]
                    else:
                        scene_audio_path = temp_dir / f"narration_scene_{idx+1}.mp3"
                        gen_path = generate_scene_narration(
                            scene=scene,
                            output_path=scene_audio_path,
                            tld=tld,
                            lang=lang,
                            slow=slow,
                            engine_mode=engine_mode,
                            voice_gender=voice_gender
                        )

                    dur = get_media_duration(gen_path)
                    if gen_path and gen_path.exists() and dur > 0.0:
                        try:
                            clip = AudioFileClip(str(gen_path))
                            if hasattr(clip, "with_start"):
                                positioned = clip.with_start(curr_audio_time)
                            else:
                                positioned = clip.set_start(curr_audio_time)
                            audio_clips.append(positioned)
                        except Exception:
                            pass
                        # Sequential advancement with non-overlapping gap
                        is_last = (idx == total_scenes - 1)
                        hold = ending_hold if is_last else sentence_gap
                        curr_audio_time += max(scene.duration, dur + hold)
                    else:
                        curr_audio_time += scene.duration

        # Determine effective total audio duration
        if total_duration is None or total_duration <= 0:
            if audio_clips:
                max_end = max((getattr(c, "end", None) or (getattr(c, "start", 0) + c.duration)) for c in audio_clips)
                total_duration = max(script.estimated_duration, max_end + ending_hold)
            else:
                total_duration = max(script.estimated_duration, 5.0)

        # 2. Ambient background music clip (mixed softly to never overpower speech)
        if include_background_music and total_duration > 0:
            bg_wav_path = temp_dir / "ambient_bg.wav"
            synthesize_ambient_background_music(
                duration=total_duration,
                output_path=bg_wav_path,
                volume=bg_music_volume
            )
            if bg_wav_path.exists():
                try:
                    bg_clip = AudioFileClip(str(bg_wav_path))
                    if hasattr(bg_clip, "with_duration"):
                        bg_clip = bg_clip.with_duration(total_duration)
                    else:
                        bg_clip = bg_clip.set_duration(total_duration)
                    audio_clips.append(bg_clip)
                except Exception:
                    pass

        if not audio_clips:
            return None

        # 3. Composite audio track with total_duration
        composite = CompositeAudioClip(audio_clips)
        if hasattr(composite, "with_duration"):
            composite = composite.with_duration(total_duration)
        else:
            composite = composite.set_duration(total_duration)

        composite.write_audiofile(str(final_audio_path), fps=44100, logger=None)

        # Cleanly close clips
        for c in audio_clips:
            try:
                c.close()
            except Exception:
                pass
        try:
            composite.close()
        except Exception:
            pass

        if final_audio_path.exists() and final_audio_path.stat().st_size > 0:
            return final_audio_path

    except Exception:
        return None

    return None


