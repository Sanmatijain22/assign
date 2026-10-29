"""Property Studio AI — Premium Real Estate Creative Studio.

A professional AI property promotional video generator built with Streamlit.
Converts property details and photos into 20-40 second branded marketing videos.
"""

from pathlib import Path
import json
import tempfile
import time
import sys
import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.validators import (
    validate_property_input,
    PropertyImage,
    ValidationError
)
from src.image_classifier import detect_image_category, DISPLAY_NAMES
from src.script_generator import generate_script
from src.video_generator import generate_property_video, VideoGenerationError
from src.audio_generator import generate_tts_audio

SAMPLE_JSON_PATH = PROJECT_ROOT / "sample_data" / "sample_property.json"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSS_PATH = PROJECT_ROOT / "assets" / "style.css"

# ─── Page Configuration ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Property Studio AI | Premium Video Generator",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_css():
    """Load external CSS stylesheet."""
    if CSS_PATH.exists():
        with open(CSS_PATH, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css()


def load_sample_property(preset: str = "green_valley"):
    """Load sample property data into session state."""
    if preset == "ocean_crest":
        target_json = PROJECT_ROOT / "sample_data" / "sample_property_villa.json"
        img_folder = PROJECT_ROOT / "assets" / "sample_images_villa"
        preset_title = "Ocean Crest Luxury Villa"
    else:
        target_json = PROJECT_ROOT / "sample_data" / "sample_property.json"
        img_folder = PROJECT_ROOT / "assets" / "sample_images"
        preset_title = "Green Valley Residency"

    if target_json.exists():
        with open(target_json, "r", encoding="utf-8") as f:
            data = json.load(f)
            st.session_state["inp_name"] = data.get("name", "")
            st.session_state["inp_location"] = data.get("location", "")
            st.session_state["inp_price"] = data.get("price", "")
            st.session_state["property_type"] = data.get("property_type", "2-BHK apartment")
            st.session_state["sel_proptype"] = data.get("property_type", "2-BHK apartment")
            st.session_state["inp_bedrooms"] = data.get("bedrooms", 2)
            st.session_state["inp_desc"] = data.get("description", "")
            st.session_state["inp_features"] = "\n".join(data.get("features", []))
            st.session_state["inp_cta"] = data.get("call_to_action", "")
            st.session_state["use_sample_images"] = True
            st.session_state["sample_images_dir"] = str(img_folder)
            st.session_state["sample_preset_name"] = preset_title


# ─── Initialize Session State ─────────────────────────────────────────────
if "initialized" not in st.session_state:
    st.session_state["initialized"] = True
    load_sample_property("green_valley")
    st.session_state["generated_video"] = None
    st.session_state["generated_script"] = None
    st.session_state["generating"] = False


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR — Navigation & Settings
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # Brand Mark
    st.markdown("""
    <div style="padding: 0.5rem 0 0.2rem 0;">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div style="font-size: 1.1rem; font-weight: 700; color: #FAFAFA; letter-spacing: -0.03em;">
                ✦ Property Studio
            </div>
            <span style="background: #27272A; color: #A1A1AA; font-size: 0.65rem; font-weight: 600;
                         padding: 2px 7px; border-radius: 9999px; border: 1px solid #3F3F46;">v2.0</span>
        </div>
        <div style="font-size: 0.75rem; color: #71717A; margin-top: 2px;">
            AI Video Production Studio
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Quick Start Presets
    st.markdown("""
    <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em;
                color: rgba(255,255,255,0.45); font-weight: 600; margin-bottom: 0.6rem;">
        Quick Start
    </div>
    """, unsafe_allow_html=True)

    col_pre1, col_pre2 = st.columns(2)
    with col_pre1:
        if st.button("🏢 Green Valley", use_container_width=True, key="btn_preset_gv"):
            load_sample_property("green_valley")
            st.rerun()
    with col_pre2:
        if st.button("🏖️ Ocean Crest", use_container_width=True, key="btn_preset_oc"):
            load_sample_property("ocean_crest")
            st.rerun()

    st.markdown("---")

    # Video & Voice Settings
    st.markdown("""
    <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em;
                color: rgba(255,255,255,0.45); font-weight: 600; margin-bottom: 0.6rem;">
        Production Settings
    </div>
    """, unsafe_allow_html=True)

    include_narration = st.checkbox("AI Voice Narration", value=True, key="chk_narration")

    tts_tld = "co.in"
    tts_slow = False
    tts_engine_val = "auto"
    tts_gender = "female"

    if include_narration:
        with st.expander("Voice Configuration", expanded=False):
            accent_choice = st.selectbox(
                "Accent",
                ["English (India)", "English (US)", "English (UK)", "English (Australia)"],
                index=0,
                key="sel_accent"
            )
            tld_map = {
                "English (India)": "co.in",
                "English (US)": "com",
                "English (UK)": "co.uk",
                "English (Australia)": "com.au"
            }
            tts_tld = tld_map.get(accent_choice, "co.in")

            engine_choice = st.selectbox(
                "Engine",
                ["Auto (gTTS + Offline)", "Online Natural (gTTS)", "Offline (Windows SAPI5)"],
                index=0,
                key="sel_engine"
            )
            if "Auto" in engine_choice:
                tts_engine_val = "auto"
            elif "Online" in engine_choice:
                tts_engine_val = "gtts"
            else:
                tts_engine_val = "offline"

            pace_choice = st.selectbox("Pacing", ["Normal", "Relaxed"], index=0, key="sel_pace")
            tts_slow = (pace_choice == "Relaxed")

            col_sync1, col_sync2 = st.columns(2)
            with col_sync1:
                sentence_gap_val = st.slider(
                    "Sentence Gap",
                    min_value=0.1, max_value=0.6, value=0.3, step=0.05,
                    help="Pause buffer between spoken lines (default 0.3s)",
                    key="sld_sentence_gap"
                )
            with col_sync2:
                ending_hold_val = st.slider(
                    "Ending Hold",
                    min_value=0.5, max_value=1.5, value=0.8, step=0.1,
                    help="Visual hold on screen after last word spoken (default 0.8s)",
                    key="sld_ending_hold"
                )
    else:
        sentence_gap_val = 0.3
        ending_hold_val = 0.8

    include_ambient = st.checkbox("Background Music", value=True, key="chk_ambient")
    bg_music_vol = 0.15
    if include_ambient:
        bg_music_vol = st.slider("Volume", min_value=0.05, max_value=0.35, value=0.15, step=0.05, key="sld_vol")

    use_llm = st.checkbox("LLM Script Enhancement", value=False, key="chk_llm")
    api_key = None
    if use_llm:
        api_key = st.text_input(
            "API Key",
            type="password",
            help="Gemini / OpenAI API Key for enhanced scripts",
            key="inp_api_key"
        )

    st.markdown("---")

    # About
    st.markdown("""
    <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em;
                color: rgba(255,255,255,0.45); font-weight: 600; margin-bottom: 0.4rem;">
        About
    </div>
    """, unsafe_allow_html=True)
    st.caption(
        "Property Studio AI transforms property listings and photographs "
        "into polished promotional videos with narration, captions, and ambient audio."
    )


# ═══════════════════════════════════════════════════════════════════════════
# MAIN CONTENT AREA
# ═══════════════════════════════════════════════════════════════════════════

# ─── Hero Header ──────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="shadcn-badge">
        <span class="shadcn-badge-dot"></span> AI Video Studio
    </div>
    <h1 class="hero-title">Property Studio</h1>
    <p class="hero-subtitle">High-impact promotional video generator for modern real estate listings.</p>
</div>
""", unsafe_allow_html=True)

# ─── Tabs ─────────────────────────────────────────────────────────────────
tab_create, tab_preview, tab_docs = st.tabs([
    "Create Video",
    "Preview & Download",
    "Architecture"
])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: CREATE VIDEO
# ═══════════════════════════════════════════════════════════════════════════
with tab_create:

    with st.container(border=True):
        st.markdown("""
        <div class="shadcn-card-header">
            <h3 class="shadcn-card-title">Property Specifications</h3>
            <p class="shadcn-card-description">Configure the listing details, key features, and price points.</p>
        </div>
        """, unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            name_val = st.text_input(
                "Property Name *",
                placeholder="e.g. Green Valley Residency",
                key="inp_name"
            )
            location_val = st.text_input(
                "Location *",
                placeholder="e.g. Chandigarh",
                key="inp_location"
            )
            price_val = st.text_input(
                "Price *",
                placeholder="e.g. ₹85 lakh",
                key="inp_price"
            )

        with col2:
            prop_types = [
                "2-BHK apartment", "3-BHK apartment", "Luxury Villa",
                "Penthouse", "Studio", "Commercial Office", "Plot / Land"
            ]
            current_type = st.session_state.get("property_type", "2-BHK apartment")
            type_idx = prop_types.index(current_type) if current_type in prop_types else 0
            property_type_val = st.selectbox(
                "Property Type *", prop_types, index=type_idx, key="sel_proptype"
            )
            bedrooms_val = st.number_input(
                "Bedrooms",
                min_value=0, max_value=20,
                key="inp_bedrooms"
            )
            cta_val = st.text_input(
                "Call to Action *",
                key="inp_cta"
            )

        col_desc, col_feat = st.columns(2)
        with col_desc:
            desc_val = st.text_area(
                "Description (optional)",
                placeholder="Brief overview of the property...",
                height=120,
                key="inp_desc"
            )
        with col_feat:
            features_val = st.text_area(
                "Features & Amenities * (one per line)",
                placeholder="Two bedrooms\nModular kitchen\nBalcony\nParking",
                height=120,
                key="inp_features"
            )

    # ── Section: Property Images ──────────────────────────────────────────
    st.markdown('<div style="height: 0.75rem;"></div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("""
        <div class="shadcn-card-header">
            <h3 class="shadcn-card-title">Property Photographs</h3>
            <p class="shadcn-card-description">Upload 4 to 8 high-resolution property photographs. Images are framed automatically to 16:9 widescreen.</p>
        </div>
        """, unsafe_allow_html=True)

        uploaded_files = st.file_uploader(
            "Choose property images",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            help="Drag and drop or browse. Supported formats: JPG, PNG, WEBP.",
            label_visibility="collapsed",
            key="uploader_images"
        )

        # Determine image sources
        selected_image_paths = []
        temp_dir_obj = None

        if uploaded_files:
            temp_dir_obj = tempfile.mkdtemp(prefix="uploaded_imgs_")
            for idx, f in enumerate(uploaded_files):
                p = Path(temp_dir_obj) / f"upload_{idx}_{f.name}"
                with open(p, "wb") as out_f:
                    out_f.write(f.getbuffer())
                selected_image_paths.append(p)
        elif st.session_state.get("use_sample_images"):
            sample_img_dir = Path(st.session_state.get(
                "sample_images_dir", str(PROJECT_ROOT / "assets" / "sample_images")
            ))
            preset_title = st.session_state.get("sample_preset_name", "Green Valley Residency")
            if sample_img_dir.exists():
                selected_image_paths = sorted(list(sample_img_dir.glob("*.jpg")))
                st.markdown(f"""
                <div style="display: flex; align-items: center; gap: 8px; padding: 0.5rem 0.85rem;
                            background: #F4F4F5; border: 1px solid #E4E4E7;
                            border-radius: 6px; margin-bottom: 0.8rem;">
                    <span style="font-size: 0.82rem; color: #18181B; font-weight: 500;">
                        ✓ {len(selected_image_paths)} sample photographs loaded for <strong>{preset_title}</strong>
                    </span>
                </div>
                """, unsafe_allow_html=True)

    # Image Previews & Category Mapping
    category_choices = [
        "Auto-Detect",
        "Property Exterior",
        "Living Room",
        "Modular Kitchen",
        "Bedroom",
        "Balcony View",
        "Designer Bathroom",
        "Parking & Garage",
        "Key Amenity"
    ]
    image_category_map = {}

    if selected_image_paths:
        st.markdown(f"""
        <div style="font-size: 0.82rem; font-weight: 600; color: #192B3A; margin: 0.8rem 0 0.5rem 0;
                    text-transform: uppercase; letter-spacing: 0.04em;">
            Image Gallery — {len(selected_image_paths)} photographs
        </div>
        """, unsafe_allow_html=True)

        img_cols = st.columns(min(len(selected_image_paths), 5))
        for idx, img_p in enumerate(selected_image_paths):
            with img_cols[idx % 5]:
                try:
                    img = Image.open(img_p)
                    st.image(img, use_container_width=True)
                    _, detected_disp, conf = detect_image_category(img_p)
                    def_idx = 0
                    if detected_disp in category_choices:
                        def_idx = category_choices.index(detected_disp)

                    chosen_cat = st.selectbox(
                        f"Image {idx + 1}",
                        category_choices,
                        index=def_idx,
                        key=f"img_cat_select_{idx}_{img_p.name}",
                        help=f"Detected: {detected_disp} ({conf:.0%})"
                    )
                    image_category_map[img_p] = chosen_cat
                except Exception:
                    st.error(f"Error loading {img_p.name}")

    # ── Section: Voice Preview ────────────────────────────────────────────
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    with st.expander("🎙️ Voice Preview — Test speech synthesis before rendering", expanded=False):
        sample_vo_text = st.text_input(
            "Text to preview:",
            value=f"Welcome to {name_val or 'your property'}, located in {location_val or 'a prime location'}.",
            key="tts_preview_input"
        )
        if st.button("▶ Preview Voice", key="btn_test_tts"):
            test_tts_path = OUTPUT_DIR / "tts_quick_preview.mp3"
            with st.spinner("Synthesizing speech..."):
                gen_res = generate_tts_audio(
                    text=sample_vo_text,
                    output_path=test_tts_path,
                    tld=tts_tld,
                    slow=tts_slow,
                    engine_mode=tts_engine_val,
                    voice_gender=tts_gender
                )
                if gen_res and gen_res.exists():
                    st.success("Speech synthesized successfully")
                    st.audio(str(gen_res), format="audio/mp3" if gen_res.suffix == ".mp3" else "audio/wav")
                else:
                    st.error("Failed to synthesize speech audio.")

    # ── Primary Action ────────────────────────────────────────────────────
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    generate_btn = st.button(
        "✦  Generate Property Video",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.get("generating", False),
        key="btn_generate"
    )

    if generate_btn:
        st.session_state["generating"] = True
        feature_list = [f.strip() for f in features_val.split("\n") if f.strip()]

        # Build PropertyImage objects with user-selected or detected categories
        property_image_objects = []
        for img_p in selected_image_paths:
            user_choice = image_category_map.get(img_p, "Auto-Detect")
            cat_to_use = None if user_choice == "Auto-Detect" else user_choice
            property_image_objects.append(PropertyImage(path=img_p, category=cat_to_use))

        # Validation
        try:
            prop_input = validate_property_input(
                name=name_val,
                location=location_val,
                price=price_val,
                property_type=property_type_val,
                features=feature_list,
                call_to_action=cta_val,
                image_paths=property_image_objects,
                bedrooms=bedrooms_val if bedrooms_val > 0 else None,
                description=desc_val
            )
        except ValidationError as ve:
            st.session_state["generating"] = False
            st.error(f"⚠️ {ve}")
            st.stop()

        # Script Generation
        with st.status("Generating script and captions...", expanded=True) as status:
            st.write("Analyzing property details...")
            script = generate_script(prop_input, use_llm=use_llm, api_key=api_key)
            st.session_state["generated_script"] = script
            st.write(f"✓ Script generated — {len(script.scenes)} scenes")
            status.update(label="Script ready", state="complete")

        # Video Generation with Progress Bar
        progress_bar = st.progress(0, text="Initializing video renderer...")
        status_box = st.empty()

        def ui_progress(ratio: float, msg: str):
            progress_bar.progress(int(ratio * 100), text=f"[{int(ratio * 100)}%] {msg}")
            status_box.info(msg)

        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c for c in prop_input.name if c.isalnum() or c in (' ', '_', '-')).rstrip()
        safe_name = safe_name.replace(" ", "_").lower()
        output_filename = f"{safe_name}_{timestamp_str}.mp4"
        out_mp4_path = OUTPUT_DIR / output_filename

        try:
            res_path = generate_property_video(
                prop=prop_input,
                output_path=out_mp4_path,
                script=script,
                include_narration=include_narration,
                include_background_music=include_ambient,
                fps=24,
                progress_callback=ui_progress,
                tts_tld=tts_tld,
                tts_slow=tts_slow,
                tts_engine=tts_engine_val,
                tts_gender=tts_gender,
                bg_music_volume=bg_music_vol,
                sentence_gap=sentence_gap_val,
                ending_hold=ending_hold_val
            )
            st.session_state["generated_video"] = res_path
            st.session_state["generating"] = False
            progress_bar.progress(100, text="Rendering complete")

            st.markdown("""
            <div class="success-banner">
                <span class="success-banner-icon">✓</span>
                <span class="success-banner-text">
                    Video generated successfully — switch to the Preview tab to watch and download.
                </span>
            </div>
            """, unsafe_allow_html=True)

        except VideoGenerationError as vge:
            st.session_state["generating"] = False
            st.error(f"Video Generation Error: {vge}")
        except Exception as ex:
            st.session_state["generating"] = False
            st.error(f"Unexpected error: {ex}")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: PREVIEW & DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════════
with tab_preview:

    current_video = st.session_state.get("generated_video")
    if not current_video or not Path(current_video).exists():
        sample_promo = OUTPUT_DIR / "green_valley_residency_promo.mp4"
        if sample_promo.exists():
            current_video = sample_promo

    if current_video and Path(current_video).exists():
        v_path = Path(current_video)

        st.markdown("""
        <div class="section-header">
            <span class="icon">▶</span> Video Preview
        </div>
        """, unsafe_allow_html=True)

        col_vid, col_meta = st.columns([3, 2])

        with col_vid:
            st.video(str(v_path))
            with open(v_path, "rb") as vid_file:
                st.download_button(
                    label="⬇  Download MP4",
                    data=vid_file,
                    file_name=v_path.name,
                    mime="video/mp4",
                    type="primary",
                    use_container_width=True,
                    key="btn_download_video"
                )

        with col_meta:
            file_mb = v_path.stat().st_size / (1024 * 1024)

            st.markdown("""
            <div class="section-header" style="margin-top: 0;">
                <span class="icon">📄</span> File Details
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="studio-card" style="padding: 1.2rem;">
                <table style="width: 100%; border-collapse: collapse; font-size: 0.85rem;">
                    <tr style="border-bottom: 1px solid #E7E1D9;">
                        <td style="padding: 0.5rem 0; color: #77736D; font-weight: 500;">Filename</td>
                        <td style="padding: 0.5rem 0; color: #192B3A; text-align: right;"><code>{v_path.name}</code></td>
                    </tr>
                    <tr style="border-bottom: 1px solid #E7E1D9;">
                        <td style="padding: 0.5rem 0; color: #77736D; font-weight: 500;">File Size</td>
                        <td style="padding: 0.5rem 0; color: #192B3A; text-align: right;">{file_mb:.2f} MB</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #E7E1D9;">
                        <td style="padding: 0.5rem 0; color: #77736D; font-weight: 500;">Resolution</td>
                        <td style="padding: 0.5rem 0; color: #192B3A; text-align: right;">1280 × 720</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #E7E1D9;">
                        <td style="padding: 0.5rem 0; color: #77736D; font-weight: 500;">Format</td>
                        <td style="padding: 0.5rem 0; color: #192B3A; text-align: right;">H.264 / AAC</td>
                    </tr>
                    <tr>
                        <td style="padding: 0.5rem 0; color: #77736D; font-weight: 500;">Status</td>
                        <td style="padding: 0.5rem 0; color: #3A5A3C; text-align: right; font-weight: 500;">
                            ● Ready to share
                        </td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

        # Script & Scene Breakdown
        script_obj = st.session_state.get("generated_script")
        if script_obj:
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            st.markdown("""
            <div class="section-header">
                Script & Scene Breakdown
            </div>
            """, unsafe_allow_html=True)

            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.markdown(f"""
                <div class="shadcn-metric-card">
                    <div class="shadcn-metric-label">Total Scenes</div>
                    <div class="shadcn-metric-value">{len(script_obj.scenes)}</div>
                    <div class="shadcn-metric-desc">Structured scene sequence</div>
                </div>
                """, unsafe_allow_html=True)
            with col_m2:
                st.markdown(f"""
                <div class="shadcn-metric-card">
                    <div class="shadcn-metric-label">Runtime</div>
                    <div class="shadcn-metric-value">{script_obj.estimated_duration:.1f}s</div>
                    <div class="shadcn-metric-desc">Synchronized audio & video</div>
                </div>
                """, unsafe_allow_html=True)
            with col_m3:
                st.markdown(f"""
                <div class="shadcn-metric-card">
                    <div class="shadcn-metric-label">Script Engine</div>
                    <div class="shadcn-metric-value">{script_obj.source.title()}</div>
                    <div class="shadcn-metric-desc">Factual template generator</div>
                </div>
                """, unsafe_allow_html=True)

            for s in script_obj.scenes:
                with st.expander(
                    f"Scene {s.scene_index} — {s.category_tag} — {s.headline} ({s.duration:.1f}s)",
                    expanded=False
                ):
                    sc_col1, sc_col2 = st.columns([1, 2])
                    with sc_col1:
                        st.markdown(f"""
                        <div class="shadcn-badge">{s.category_tag}</div>
                        <div style="margin-top: 0.6rem;">
                            <div style="font-size: 0.85rem; font-weight: 600; color: #09090B;">{s.headline}</div>
                            <div style="font-size: 0.78rem; color: #71717A; margin-top: 0.2rem;">{s.subheadline}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with sc_col2:
                        st.markdown(f"""
                        <div style="font-size: 0.8rem; margin-bottom: 0.5rem;">
                            <span style="color: #77736D; font-weight: 500;">Caption:</span>
                            <span style="color: #192B3A; font-style: italic;">&ldquo;{s.caption}&rdquo;</span>
                        </div>
                        <div style="font-size: 0.8rem;">
                            <span style="color: #77736D; font-weight: 500;">Voiceover:</span>
                            <span style="color: #192B3A; font-style: italic;">&ldquo;{s.voiceover}&rdquo;</span>
                        </div>
                        """, unsafe_allow_html=True)

                        # Inline Audio Player
                        scene_tts_path = OUTPUT_DIR / f"scene_{s.scene_index}_tts.mp3"
                        if not scene_tts_path.exists():
                            generate_tts_audio(
                                text=s.voiceover,
                                output_path=scene_tts_path,
                                tld=tts_tld,
                                slow=tts_slow,
                                engine_mode=tts_engine_val,
                                voice_gender=tts_gender
                            )
                        if scene_tts_path.exists() and scene_tts_path.stat().st_size > 0:
                            st.audio(
                                str(scene_tts_path),
                                format="audio/mp3" if scene_tts_path.suffix == ".mp3" else "audio/wav"
                            )

    else:
        # Empty state
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">🎬</div>
            <div class="empty-state-title">No Video Yet</div>
            <div class="empty-state-text">
                Create your first property video by filling in the details on the
                <strong>Create Video</strong> tab and clicking <strong>Generate</strong>.
                Your finished video will appear here for preview and download.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3: ARCHITECTURE & INFO
# ═══════════════════════════════════════════════════════════════════════════
with tab_docs:

    with st.container(border=True):
        st.markdown("""
        <div class="shadcn-card-header">
            <h3 class="shadcn-card-title">Technical Pipeline Architecture</h3>
            <p class="shadcn-card-description">Automated real estate video production pipeline components.</p>
        </div>
        <div style="line-height: 1.7; font-size: 0.875rem; color: #09090B;">
            <div style="margin-bottom: 1rem;">
                <strong>1. Input &amp; Validation</strong> 
                <code style="background: #F4F4F5; color: #18181B; border: 1px solid #E4E4E7; padding: 2px 6px; border-radius: 4px; font-size: 0.78rem;">src/validators.py</code>
                <br/><span style="color: #71717A;">Enforces strict property field rules and validates uploaded photography (4–8 images, 100×100px minimum).</span>
            </div>
            <div style="margin-bottom: 1rem;">
                <strong>2. Script &amp; Caption Binding</strong> 
                <code style="background: #F4F4F5; color: #18181B; border: 1px solid #E4E4E7; padding: 2px 6px; border-radius: 4px; font-size: 0.78rem;">src/script_generator.py</code>
                <br/><span style="color: #71717A;">Rule-based template logic that maps features strictly to corresponding scenes without hallucinating.</span>
            </div>
            <div style="margin-bottom: 1rem;">
                <strong>3. Image Composition &amp; Lower-Thirds</strong> 
                <code style="background: #F4F4F5; color: #18181B; border: 1px solid #E4E4E7; padding: 2px 6px; border-radius: 4px; font-size: 0.78rem;">src/image_processor.py</code>
                <br/><span style="color: #71717A;">Pillow-based typography composition. High-resolution cover crop to 1280×720 (16:9), category pill badges, and contrast gradient overlays.</span>
            </div>
            <div style="margin-bottom: 1rem;">
                <strong>4. Audio Synthesis &amp; Dynamic Sync</strong> 
                <code style="background: #F4F4F5; color: #18181B; border: 1px solid #E4E4E7; padding: 2px 6px; border-radius: 4px; font-size: 0.78rem;">src/audio_generator.py</code>
                <br/><span style="color: #71717A;">TTS speech generation with inter-sentence pacing. Procedural ambient background music synthesized offline. Timeline dynamically calibrated so slides match speech duration.</span>
            </div>
            <div>
                <strong>5. Video Rendering &amp; Encoding</strong> 
                <code style="background: #F4F4F5; color: #18181B; border: 1px solid #E4E4E7; padding: 2px 6px; border-radius: 4px; font-size: 0.78rem;">src/video_generator.py</code>
                <br/><span style="color: #71717A;">MoviePy 2.x and FFmpeg render H.264 MP4 with seamless crossfades and stereo AAC audio.</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
