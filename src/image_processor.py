"""Image processing and text overlay composition module for Property Studio AI.

Handles:
- Aspect-ratio preserving smart crop and resize to 1280x720 (16:9 widescreen).
- Cinematic vignette overlay for dramatic dark-edge film look.
- Premium glassmorphism lower-third card with glowing gold accent line.
- Gold gradient category badge pill.
- High-contrast, crisp typography with proper shadow depth.
- Professional top watermark bar.
"""

from pathlib import Path
from typing import Tuple, Optional, List
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
from .script_generator import SceneScript

TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
ASPECT_RATIO = TARGET_WIDTH / TARGET_HEIGHT  # 16:9 (1.777...)


def get_default_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """Load an available system font or fall back gracefully."""
    font_candidates = []
    if bold:
        font_candidates = [
            "C:/Windows/Fonts/segoeuib.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/calibrib.ttf",
            "C:/Windows/Fonts/tahomabd.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/System/Library/Fonts/HelveticaNeue-Bold.otf",
            "arialbd.ttf"
        ]
    else:
        font_candidates = [
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/calibri.ttf",
            "C:/Windows/Fonts/tahoma.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/HelveticaNeue.otf",
            "arial.ttf"
        ]

    for cand in font_candidates:
        try:
            return ImageFont.truetype(cand, size)
        except Exception:
            continue

    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


def wrap_text(text: str, font: ImageFont.ImageFont, max_width: int) -> List[str]:
    """Break text into multiple lines that fit within max_width pixels."""
    if not text:
        return []
    words = text.split()
    lines = []
    current_line = []

    for word in words:
        test_line = " ".join(current_line + [word])
        try:
            bbox = font.getbbox(test_line)
            w = bbox[2] - bbox[0]
        except Exception:
            w = len(test_line) * (font.size if hasattr(font, 'size') else 10) * 0.6
            
        if w <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]

    if current_line:
        lines.append(" ".join(current_line))
    return lines


def resize_and_crop_to_16_9(
    image_path: Path,
    target_w: int = TARGET_WIDTH,
    target_h: int = TARGET_HEIGHT,
    scale_extra: float = 1.0
) -> Image.Image:
    """Resize and center-crop an image to target dimensions without distortion.
    
    scale_extra > 1.0 lets the caller pre-enlarge for Ken Burns headroom.
    Preserves aspect ratio by scaling to cover target bounds, then center-cropping.
    """
    with Image.open(image_path) as img:
        img = img.convert("RGBA")
        src_w, src_h = img.size
        src_ratio = src_w / src_h
        target_ratio = target_w / target_h

        scaled_w = int(target_w * scale_extra)
        scaled_h = int(target_h * scale_extra)

        if src_ratio > target_ratio:
            new_h = scaled_h
            new_w = int(src_w * (scaled_h / src_h))
        else:
            new_w = scaled_w
            new_h = int(src_h * (scaled_w / src_w))

        resample_filter = getattr(Image.Resampling, 'LANCZOS', Image.LANCZOS)
        resized = img.resize((new_w, new_h), resample=resample_filter)

        left = max(0, (new_w - target_w) // 2)
        top = max(0, (new_h - target_h) // 2)
        right = left + target_w
        bottom = top + target_h

        cropped = resized.crop((left, top, right, bottom))
        return cropped


def apply_vignette(image: Image.Image) -> Image.Image:
    """Apply a cinematic radial vignette to darken the frame edges.
    
    Creates a soft oval gradient mask — centre stays bright, edges darken.
    """
    w, h = image.size
    if w < 2 or h < 2:
        return image.convert("RGBA")

    vignette = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(vignette)

    # Draw concentric ellipses from outside in, darkening edges
    steps = 80
    for i in range(steps):
        progress = i / steps           # 0 = outermost, 1 = centre
        alpha = int(210 * (1 - progress) ** 2.2)
        margin_x = min(int(w * 0.01 * i), (w - 1) // 2)
        margin_y = min(int(h * 0.01 * i), (h - 1) // 2)
        draw.ellipse(
            [(margin_x, margin_y), (w - margin_x, h - margin_y)],
            fill=alpha
        )

    # Blur for smoothness
    vignette = vignette.filter(ImageFilter.GaussianBlur(radius=40))

    # Build a black overlay at the vignette opacity
    black_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    black_draw = ImageDraw.Draw(black_layer)
    black_draw.rectangle([(0, 0), (w, h)], fill=(0, 0, 0, 255))

    # Invert mask: vignette high value = dark edge
    mask_arr = np.array(vignette, dtype=np.float32) / 255.0
    inv_mask = (1.0 - mask_arr) * 180  # max 180 opacity at corners
    inv_mask_img = Image.fromarray(inv_mask.astype(np.uint8), mode="L")

    black_layer.putalpha(inv_mask_img)

    result = image.copy().convert("RGBA")
    result = Image.alpha_composite(result, black_layer)
    return result


def draw_glow_line(draw: ImageDraw.Draw, x0: int, y: int, x1: int,
                   color: Tuple[int, int, int], width: int = 3, glow_radius: int = 6) -> None:
    """Draw a line with a soft outer glow effect (multiple passes, decreasing opacity)."""
    r, g, b = color
    for i in range(glow_radius, 0, -1):
        alpha = int(60 * (i / glow_radius) ** 1.5)
        draw.line([(x0, y - i), (x1, y - i)], fill=(r, g, b, alpha), width=1)
        draw.line([(x0, y + i), (x1, y + i)], fill=(r, g, b, alpha), width=1)
    # Core line
    draw.line([(x0, y), (x1, y)], fill=(r, g, b, 240), width=width)


def render_scene_frame(
    image_path: Path,
    scene: SceneScript,
    target_w: int = TARGET_WIDTH,
    target_h: int = TARGET_HEIGHT
) -> Image.Image:
    """Render a complete 1280x720 cinematic video frame.

    Includes:
    - Base cropped photograph
    - Cinematic radial vignette (dark edges, bright centre)
    - Top letterbox bar: gradient + 'PROPERTY STUDIO AI' watermark
    - Glassmorphism lower-third card (deep navy, frosted)
    - Glowing gold accent line above the card
    - Gold gradient category badge pill
    - Bold headline with drop shadow
    - Warm subheadline and wrapped caption
    """
    # --- Base image ---
    base_img = resize_and_crop_to_16_9(image_path, target_w, target_h)

    # --- Cinematic vignette ---
    base_img = apply_vignette(base_img)
    base_img = base_img.convert("RGBA")

    # Overlay canvas
    overlay = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # ── 1. Top Letterbox / Watermark Bar ──────────────────────────────────
    top_bar_height = 70
    for y_px in range(top_bar_height):
        progress = 1 - (y_px / top_bar_height)
        alpha = int(180 * progress ** 1.6)
        draw.line([(0, y_px), (target_w, y_px)], fill=(5, 4, 14, alpha))

    # Brand watermark text
    font_brand = get_default_font(15, bold=True)
    brand_x, brand_y = 42, 18
    # Shadow
    draw.text((brand_x + 1, brand_y + 1), "PROPERTY STUDIO AI", fill=(0, 0, 0, 160), font=font_brand)
    draw.text((brand_x, brand_y), "PROPERTY STUDIO AI", fill=(255, 255, 255, 220), font=font_brand)

    font_brand_sub = get_default_font(13, bold=False)
    draw.text((brand_x + 208, brand_y + 1), "✦  CINEMATIC REAL ESTATE", fill=(0, 0, 0, 120), font=font_brand_sub)
    draw.text((brand_x + 207, brand_y), "✦  CINEMATIC REAL ESTATE", fill=(218, 165, 32, 200), font=font_brand_sub)

    # ── 2. Lower-Third Glassmorphism Card ─────────────────────────────────
    margin_x = 42
    margin_bottom = 32
    card_height = 215
    card_y0 = target_h - card_height - margin_bottom
    card_y1 = target_h - margin_bottom
    card_width = target_w - 2 * margin_x

    # Card mask (rounded corners)
    card_mask = Image.new("L", (card_width, card_height), 0)
    card_mask_draw = ImageDraw.Draw(card_mask)
    card_mask_draw.rounded_rectangle([(0, 0), (card_width, card_height)], radius=18, fill=255)

    # Card background: deep navy glass with subtle blue-violet tint
    card_bg = Image.new("RGBA", (card_width, card_height), (8, 10, 28, 235))

    # Subtle inner highlight at the top of the card (glass shimmer)
    card_draw = ImageDraw.Draw(card_bg)
    for yi in range(12):
        shimmer_alpha = int(35 * (1 - yi / 12))
        card_draw.line([(0, yi), (card_width, yi)], fill=(255, 255, 255, shimmer_alpha))

    overlay.paste(card_bg, (margin_x, card_y0), card_mask)

    # ── 3. Glowing Gold Accent Line ────────────────────────────────────────
    glow_draw = ImageDraw.Draw(overlay)
    draw_glow_line(
        glow_draw,
        x0=margin_x + 12,
        y=card_y0,
        x1=margin_x + card_width - 12,
        color=(218, 165, 32),
        width=3,
        glow_radius=8
    )

    # ── 4. Card Content ────────────────────────────────────────────────────
    card_content_draw = ImageDraw.Draw(overlay)
    text_x = margin_x + 30
    curr_y = card_y0 + 16
    content_max_w = card_width - 60

    # Category Badge (Gold gradient pill)
    font_badge = get_default_font(11, bold=True)
    badge_text = scene.category_tag.upper().strip()

    try:
        bbox = font_badge.getbbox(badge_text)
        badge_w = bbox[2] - bbox[0] + 22
        badge_h = bbox[3] - bbox[1] + 10
    except Exception:
        badge_w = len(badge_text) * 8 + 22
        badge_h = 24

    badge_rect = [(text_x, curr_y), (text_x + badge_w, curr_y + badge_h)]

    # Badge background: gold gradient pill (simulated with two rect layers)
    badge_bg = Image.new("RGBA", (badge_w, badge_h), (0, 0, 0, 0))
    badge_bg_draw = ImageDraw.Draw(badge_bg)
    badge_bg_draw.rounded_rectangle([(0, 0), (badge_w, badge_h)], radius=6, fill=(218, 165, 32, 255))
    # Subtle lighter top-left shimmer
    for gi in range(badge_h // 2):
        shimmer = int(40 * (1 - gi / (badge_h / 2)))
        badge_bg_draw.line([(2, gi), (badge_w - 2, gi)], fill=(255, 220, 100, shimmer))

    overlay.paste(badge_bg, (text_x, curr_y), badge_bg)
    card_content_draw.text((text_x + 11, curr_y + 4), badge_text, fill=(10, 5, 20, 255), font=font_badge)

    curr_y += badge_h + 12

    # Headline (large, bold, pure white with drop shadow)
    font_headline = get_default_font(32, bold=True)
    # Drop shadow
    card_content_draw.text((text_x + 2, curr_y + 2), scene.headline, fill=(0, 0, 0, 200), font=font_headline)
    card_content_draw.text((text_x, curr_y), scene.headline, fill=(255, 255, 255, 255), font=font_headline)
    curr_y += 42

    # Subheadline (medium, warm light-blue)
    font_subhead = get_default_font(17, bold=True)
    card_content_draw.text((text_x, curr_y), scene.subheadline, fill=(180, 210, 240, 240), font=font_subhead)
    curr_y += 26

    # Thin separator line
    card_content_draw.line(
        [(text_x, curr_y + 3), (text_x + min(content_max_w, 400), curr_y + 3)],
        fill=(255, 255, 255, 30),
        width=1
    )
    curr_y += 10

    # Caption (wrapped, soft slate)
    font_cap = get_default_font(14, bold=False)
    caption_lines = wrap_text(scene.caption, font_cap, content_max_w)
    for line in caption_lines[:2]:
        card_content_draw.text((text_x, curr_y), line, fill=(190, 205, 225, 215), font=font_cap)
        curr_y += 20

    # ── 5. Composite final frame ───────────────────────────────────────────
    final_frame = Image.alpha_composite(base_img, overlay).convert("RGB")
    return final_frame


def save_scene_frame(image_path: Path, scene: SceneScript, output_path: Path) -> Path:
    """Render and save a processed scene frame to disk."""
    frame = render_scene_frame(image_path, scene)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.save(output_path, "JPEG", quality=95)
    return output_path
