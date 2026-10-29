"""Video generation engine for Property Studio AI.

Orchestrates:
- Scene frame rendering with readable text overlays and category badges.
- Ken Burns zoom effect (slow animated pan+zoom per scene).
- Smooth crossfade transitions between property scenes.
- Audio synchronization (voiceover narration and ambient background music).
- Cinematic fade-to-black ending.
- H.264 MP4 export at 1280x720 resolution.
- Temporary file management and cleanup.
"""

from pathlib import Path
from typing import Optional, Callable
import tempfile
import shutil
import numpy as np
from PIL import Image

from moviepy import ImageClip, CompositeVideoClip, AudioFileClip, concatenate_videoclips
import moviepy.video.fx as vfx

from .validators import PropertyInput
from .script_generator import VideoScript, generate_script
from .image_processor import render_scene_frame
from .audio_generator import generate_full_audio_track, generate_scene_narration, get_media_duration


class VideoGenerationError(Exception):
    """Raised when video compilation or encoding fails."""
    pass


def _make_ken_burns_clip(frame_array: np.ndarray, duration: float, fps: int, zoom_direction: str = "in") -> ImageClip:
    """Apply Ken Burns slow zoom-pan effect to a static image frame.

    Produces a smooth animated clip from the static frame by scaling from
    1.0 → 1.08 (zoom in) or 1.08 → 1.0 (zoom out), anchored to center.

    Parameters:
    - frame_array: RGB numpy array (H, W, 3)
    - duration: desired clip duration in seconds
    - fps: frames per second
    - zoom_direction: 'in' to slowly zoom in, 'out' to zoom out

    Returns:
    - An ImageClip with the Ken Burns effect baked in as a function clip.
    """
    h, w = frame_array.shape[:2]
    zoom_start = 1.0 if zoom_direction == "in" else 1.08
    zoom_end   = 1.08 if zoom_direction == "in" else 1.0

    def make_frame(t: float):
        progress = t / duration if duration > 0 else 0.0
        # Ease in-out via smoothstep
        progress = progress * progress * (3 - 2 * progress)
        scale = zoom_start + (zoom_end - zoom_start) * progress

        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize frame
        pil_img = Image.fromarray(frame_array)
        resample = getattr(Image.Resampling, 'LANCZOS', Image.LANCZOS)
        pil_resized = pil_img.resize((new_w, new_h), resample)
        arr_resized = np.array(pil_resized)

        # Center-crop back to original (w, h)
        x0 = (new_w - w) // 2
        y0 = (new_h - h) // 2
        cropped = arr_resized[y0:y0 + h, x0:x0 + w]

        return cropped

    clip = ImageClip(make_frame=make_frame, duration=duration, ismask=False)
    return clip


def generate_property_video(
    prop: PropertyInput,
    output_path: Path,
    script: Optional[VideoScript] = None,
    include_narration: bool = True,
    include_background_music: bool = True,
    fps: int = 24,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    tts_tld: str = "co.in",
    tts_lang: str = "en",
    tts_slow: bool = False,
    tts_engine: str = "auto",
    tts_gender: str = "female",
    bg_music_volume: float = 0.12,
    sentence_gap: float = 0.3,
    ending_hold: float = 0.8,
) -> Path:
    """Generate a complete cinematic promotional property video MP4.
    
    Parameters:
    - prop: Validated PropertyInput object containing details and image paths
    - output_path: Destination path for the generated MP4 file
    - script: Optional pre-generated VideoScript; if None, generates default template script
    - include_narration: Whether to generate TTS narration
    - include_background_music: Whether to synthesize royalty-free ambient music
    - fps: Output frames per second (default 24)
    - progress_callback: Optional function(progress_ratio: float, status_text: str)
    - sentence_gap: Configurable gap in seconds between narration lines (default 0.3s)
    - ending_hold: Configurable visual hold in seconds after the final spoken sentence (default 0.8s)
    
    Returns:
    - Path to verified output MP4 file
    """
    def update_progress(ratio: float, text: str):
        if progress_callback:
            progress_callback(ratio, text)

    update_progress(0.05, "Validating input and preparing script...")

    if script is None:
        script = generate_script(prop)

    if not script.scenes:
        raise VideoGenerationError("Script contains no scenes to render.")

    # Crossfade duration between scenes
    CROSSFADE_DURATION = 0.55  # seconds

    # Use an isolated temporary directory for intermediate files
    temp_dir = Path(tempfile.mkdtemp(prefix="prop_video_"))
    
    try:
        total_scenes = len(script.scenes)
        scene_audio_files = []
        actual_durations = []

        # Step 1: Pre-generate TTS narration clips and measure their actual audio durations
        if include_narration:
            update_progress(0.10, "Synthesizing voice narration and measuring audio clips...")
            for idx, scene in enumerate(script.scenes):
                clean_voice = scene.voiceover.strip() if scene.voiceover else ""
                if clean_voice:
                    scene_audio_path = temp_dir / f"narration_scene_{idx+1}.mp3"
                    progress = 0.10 + (0.15 * (idx / total_scenes))
                    update_progress(progress, f"Synthesizing voice for scene {idx + 1}/{total_scenes}...")
                    try:
                        generated_path = generate_scene_narration(
                            scene=scene,
                            output_path=scene_audio_path,
                            tld=tts_tld,
                            lang=tts_lang,
                            slow=tts_slow,
                            engine_mode=tts_engine,
                            voice_gender=tts_gender
                        )
                        speech_dur = get_media_duration(generated_path) if (generated_path and generated_path.exists()) else 0.0
                        scene_audio_files.append(generated_path)
                        actual_durations.append(speech_dur)
                    except Exception as e:
                        print(f"Failed to generate narration for scene {idx + 1}: {e}")
                        scene_audio_files.append(None)
                        actual_durations.append(0.0)
                else:
                    scene_audio_files.append(None)
                    actual_durations.append(0.0)
        else:
            scene_audio_files = [None] * total_scenes
            actual_durations = [0.0] * total_scenes

        # Step 2: Build strict sequential narration schedule & calculate slide durations
        min_slide_duration = 3.5
        slide_durations = []
        narration_schedule = []
        curr_time = 0.0

        for idx, scene in enumerate(script.scenes):
            is_last = (idx == total_scenes - 1)
            speech_dur = actual_durations[idx]
            audio_path = scene_audio_files[idx]

            if speech_dur > 0.0 and audio_path and audio_path.exists():
                hold = ending_hold if is_last else sentence_gap
                slide_dur = max(min_slide_duration, speech_dur + hold)
                narr_start = curr_time
                narr_end = curr_time + speech_dur
            else:
                slide_dur = max(min_slide_duration, 4.0)
                narr_start = curr_time
                narr_end = curr_time

            scene.duration = slide_dur
            slide_durations.append(slide_dur)
            narration_schedule.append({
                "index": idx,
                "audio_path": audio_path,
                "duration": speech_dur,
                "start_time": narr_start,
                "end_time": narr_end,
                "slide_start": curr_time,
                "slide_duration": slide_dur
            })
            curr_time += slide_dur

        total_video_duration = curr_time
        script.estimated_duration = total_video_duration

        # Step 3: Render video frames with Ken Burns effect and concatenate with crossfades
        update_progress(0.25, "Rendering cinematic scenes with Ken Burns zoom effect...")
        clips = []

        for idx, scene in enumerate(script.scenes):
            update_progress(
                0.25 + (0.40 * (idx / total_scenes)),
                f"Animating scene {idx + 1}/{total_scenes}: {scene.headline}..."
            )

            img_path = scene.image_path if hasattr(scene, "image_path") and scene.image_path else (prop.image_paths[idx % len(prop.image_paths)])
            if not Path(img_path).exists():
                raise VideoGenerationError(f"Image not found for scene {scene.scene_index}: {img_path}")

            # Render composite frame (with lower-third overlay, badge, etc.)
            frame_img = render_scene_frame(img_path, scene)
            frame_array = np.array(frame_img)

            slide_dur = slide_durations[idx]

            # Alternate Ken Burns direction per scene for visual variety
            zoom_dir = "in" if idx % 2 == 0 else "out"

            try:
                # Apply Ken Burns animated zoom
                ken_burns_clip = _make_ken_burns_clip(
                    frame_array=frame_array,
                    duration=slide_dur,
                    fps=fps,
                    zoom_direction=zoom_dir
                )
                clip = ken_burns_clip
            except Exception:
                # Fallback to static clip if Ken Burns fails
                clip = ImageClip(frame_array)
                if hasattr(clip, "with_duration"):
                    clip = clip.with_duration(slide_dur)
                else:
                    clip = clip.set_duration(slide_dur)

            # Apply crossfade-in for all scenes except the first
            if idx > 0:
                fade_t = min(CROSSFADE_DURATION, slide_dur * 0.15)
                try:
                    if hasattr(clip, "with_effects"):
                        clip = clip.with_effects([vfx.FadeIn(fade_t)])
                    elif hasattr(clip, "fadein"):
                        clip = clip.fadein(fade_t)
                except Exception:
                    pass

            # Apply crossfade-out for all scenes except the last
            if idx < total_scenes - 1:
                fade_t = min(CROSSFADE_DURATION, slide_dur * 0.15)
                try:
                    if hasattr(clip, "with_effects"):
                        clip = clip.with_effects([vfx.FadeOut(fade_t)])
                    elif hasattr(clip, "fadeout"):
                        clip = clip.fadeout(fade_t)
                except Exception:
                    pass

            # Fade to black on the very last scene
            if idx == total_scenes - 1:
                fade_end = min(1.2, slide_dur * 0.25)
                try:
                    if hasattr(clip, "with_effects"):
                        clip = clip.with_effects([vfx.FadeOut(fade_end)])
                    elif hasattr(clip, "fadeout"):
                        clip = clip.fadeout(fade_end)
                except Exception:
                    pass

            clips.append(clip)

        update_progress(0.68, "Concatenating cinematic scenes...")
        final_video = concatenate_videoclips(clips, method="compose")

        # Step 4: Assemble synchronized audio track (narration + ambient music)
        update_progress(0.74, "Synchronizing audio track with video timeline...")
        audio_path = None
        audio_clip = None
        try:
            audio_path = generate_full_audio_track(
                script=script,
                temp_dir=temp_dir,
                include_narration=include_narration,
                include_background_music=include_background_music,
                narration_schedule=narration_schedule,
                total_duration=total_video_duration,
                tld=tts_tld,
                lang=tts_lang,
                slow=tts_slow,
                engine_mode=tts_engine,
                voice_gender=tts_gender,
                bg_music_volume=bg_music_volume,
                sentence_gap=sentence_gap,
                ending_hold=ending_hold
            )
        except Exception:
            audio_path = None

        if audio_path and audio_path.exists():
            update_progress(0.78, "Muxing audio track with video timeline...")
            try:
                audio_clip = AudioFileClip(str(audio_path))
                if hasattr(final_video, "with_audio"):
                    final_video = final_video.with_audio(audio_clip)
                else:
                    final_video = final_video.set_audio(audio_clip)
            except Exception:
                pass

        # Step 5: Export MP4 video file
        update_progress(0.82, "Encoding cinematic H.264 MP4...")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        final_video.write_videofile(
            str(output_path),
            fps=fps,
            codec="libx264",
            audio_codec="aac" if (audio_path and audio_path.exists()) else None,
            preset="fast",
            threads=4,
            logger=None
        )

        # Cleanly close all MoviePy clips
        try:
            final_video.close()
            if audio_clip is not None:
                audio_clip.close()
            for c in clips:
                c.close()
        except Exception:
            pass

        # Verify output
        if not output_path.exists():
            raise VideoGenerationError(f"Video file was not created at {output_path}")

        file_size = output_path.stat().st_size
        if file_size == 0:
            output_path.unlink(missing_ok=True)
            raise VideoGenerationError("Generated MP4 file is empty (0 bytes).")

        update_progress(1.0, "Cinematic video generated successfully!")
        return output_path

    except Exception as e:
        if isinstance(e, VideoGenerationError):
            raise
        raise VideoGenerationError(f"Video encoding failed: {str(e)}") from e

    finally:
        # Safely remove intermediate temp files
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass
