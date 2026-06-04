from pathlib import Path
from typing import Any, Dict, List
import base64
import io
import re

from PIL import Image, ImageDraw, ImageFont, JpegImagePlugin, PdfImagePlugin  # noqa: F401


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _clean_markdown(text: str) -> str:
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text or "")
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = text.replace("**", "").replace("__", "")
    return text.strip()


def _clean_inline_markdown(text: str) -> str:
    text = re.sub(r"`([^`]+)`", r"\1", text or "")
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"^\s*[-*]\s+", "- ", text)
    return text.strip()


def _image_from_markdown_target(target: str) -> Image.Image | None:
    target = (target or "").strip().strip('"').strip("'")
    try:
        if target.startswith("data:image/"):
            _, encoded = target.split(",", 1)
            return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")

        image_path = Path(target)
        if image_path.exists() and image_path.is_file():
            return Image.open(image_path).convert("RGB")
    except Exception:
        return None
    return None


def _wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> List[str]:
    words = text.split()
    if not words:
        return [""]
    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def export_notes_pdf(
    notes_markdown: str,
    slides: List[Dict[str, Any]],
    output_path: Path,
    title: str = "EchoNotes AI Report",
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width, height = 1240, 1754  # A4-ish at 150 DPI
    margin = 80
    pages: List[Image.Image] = []
    page = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(page)
    title_font = _font(38, bold=True)
    heading_font = _font(25, bold=True)
    body_font = _font(21)
    small_font = _font(16)
    y = margin

    def new_page():
        nonlocal page, draw, y
        pages.append(page)
        page = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(page)
        y = margin

    def ensure(space: int):
        if y + space > height - margin:
            new_page()

    def draw_markdown_image(target: str) -> bool:
        nonlocal y
        img = _image_from_markdown_target(target)
        if img is None:
            return False

        max_w = width - 2 * margin
        max_h = 520
        img.thumbnail((max_w, max_h))
        ensure(img.height + 28)
        x = margin + max(0, (max_w - img.width) // 2)
        page.paste(img, (x, y))
        y += img.height + 28
        return True

    draw.text((margin, y), title, fill="#0f172a", font=title_font)
    y += 55
    draw.text((margin, y), "Generated from transcript, visual keyframes, and local AI notes.", fill="#475569", font=small_font)
    y += 45

    image_rendered = False
    for raw_line in (notes_markdown or "").splitlines():
        line = raw_line.strip()
        if not line:
            y += 14
            continue

        image_match = re.match(r"!\[[^\]]*\]\((.+)\)", line)
        if image_match:
            image_rendered = draw_markdown_image(image_match.group(1)) or image_rendered
            continue

        if line.startswith("#"):
            line = line.lstrip("#").strip()
            font = heading_font
            fill = "#0f172a"
            gap = 36
        else:
            line = _clean_inline_markdown(line)
            if not line:
                continue
            font = body_font
            fill = "#111827"
            gap = 29
        for wrapped in _wrap(draw, line, font, width - 2 * margin):
            ensure(gap + 8)
            draw.text((margin, y), wrapped, fill=fill, font=font)
            y += gap

    if not image_rendered:
        image_count = 0
        for slide in slides or []:
            image_path = Path(slide.get("image_path") or "")
            if image_count >= 6:
                break
            if image_path.exists():
                ensure(390)
                draw.text((margin, y), f"Visual context - {slide.get('timestamp_formatted', '')}", fill="#1e40af", font=heading_font)
                y += 35
                if draw_markdown_image(str(image_path)):
                    image_count += 1

    pages.append(page)
    pages[0].save(output_path, save_all=True, append_images=pages[1:])
    return output_path
