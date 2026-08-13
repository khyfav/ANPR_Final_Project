import argparse
import csv
import os
import random
import string
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

CONDITIONS = ["clean", "light_blur", "low_contrast", "rotated"]

STATE_CONFIG = {
    "simple": {
        "bg": (245, 245, 245),
        "border": (70, 70, 70),
        "text": (20, 20, 20),
        "accent": (220, 220, 220),
        "state_text": "SYNTHETIC",
        "bottom_text": "BASELINE",
        "patterns": ["AAA111"],
    },
    "florida": {
        "bg": (244, 244, 239),
        "border": (63, 132, 76),
        "text": (28, 49, 31),
        "accent": (245, 153, 62),
        "state_text": "FLORIDA",
        "bottom_text": "SUNSHINE STATE",
        "patterns": ["AAA111", "AAA1B2"],
    },
    "california": {
        "bg": (252, 252, 250),
        "border": (165, 165, 165),
        "text": (16, 36, 120),
        "accent": (195, 28, 49),
        "state_text": "CALIFORNIA",
        "bottom_text": "dmv.ca.gov",
        "patterns": ["1AAA111", "AAA1111"],
    },
    "texas": {
        "bg": (248, 248, 246),
        "border": (75, 75, 75),
        "text": (25, 25, 25),
        "accent": (36, 84, 162),
        "state_text": "TEXAS",
        "bottom_text": "The Texas Classic",
        "patterns": ["AAA1111", "A1A1111"],
    },
    "new_york": {
        "bg": (245, 245, 241),
        "border": (44, 73, 150),
        "text": (25, 31, 50),
        "accent": (224, 166, 41),
        "state_text": "NEW YORK",
        "bottom_text": "EXCELSIOR",
        "patterns": ["AAA1111", "AAA1B2"],
    },
}


def load_font(size, bold=False):
    candidates = []
    if bold:
        candidates = [
            r"C:\Windows\Fonts\arialbd.ttf",
            r"C:\Windows\Fonts\calibrib.ttf",
            r"C:\Windows\Fonts\bahnschrift.ttf",
            r"C:\Windows\Fonts\segoeuib.ttf",
        ]
    else:
        candidates = [
            r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\calibri.ttf",
            r"C:\Windows\Fonts\bahnschrift.ttf",
            r"C:\Windows\Fonts\segoeui.ttf",
        ]

    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                pass

    return ImageFont.load_default()


def random_plate_text(state_name, length=None):
    cfg = STATE_CONFIG[state_name]
    pattern = random.choice(cfg["patterns"])

    result = []
    for ch in pattern:
        if ch in ("A", "B"):
            result.append(random.choice(string.ascii_uppercase))
        elif ch in ("1", "2"):
            result.append(random.choice(string.digits))
        else:
            result.append(ch)

    text = "".join(result)

    if length is not None:
        alphabet = string.ascii_uppercase + string.digits
        if len(text) < length:
            text += "".join(random.choice(alphabet)
                            for _ in range(length - len(text)))
        elif len(text) > length:
            text = text[:length]

    return text


def apply_condition(img, condition):
    if condition == "light_blur":
        return img.filter(ImageFilter.GaussianBlur(radius=1.1))

    if condition == "low_contrast":
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(0.68)

    if condition == "rotated":
        angle = random.uniform(-3.5, 3.5)
        return img.rotate(angle, expand=False, fillcolor=(244, 244, 238))

    return img


def add_background_noise(img, amount=90):
    for _ in range(amount):
        x = random.randint(0, img.width - 1)
        y = random.randint(0, img.height - 1)
        c = random.randint(228, 247)
        img.putpixel((x, y), (c, c, c))
    return img


def draw_screw_holes(draw, width):
    for hx in (34, width - 34):
        draw.ellipse([(hx - 4, 30 - 4), (hx + 4, 30 + 4)],
                     fill=(120, 120, 120))


def draw_florida_outline(draw, center_x, center_y):
    pts = [
        (center_x - 30, center_y - 10), (center_x - 18, center_y - 15),
        (center_x - 10, center_y - 20), (center_x + 8, center_y - 18),
        (center_x + 18, center_y - 10), (center_x + 15, center_y - 2),
        (center_x + 10, center_y + 6), (center_x + 2, center_y + 10),
        (center_x - 10, center_y + 10), (center_x - 17, center_y + 18),
        (center_x - 20, center_y + 26), (center_x - 25, center_y + 18),
        (center_x - 25, center_y + 8), (center_x - 32, center_y + 2),
    ]
    draw.polygon(pts, outline=(144, 188, 142), fill=(237, 245, 232))
    draw.ellipse([(center_x - 6, center_y + 18), (center_x + 6,
                 center_y + 30)], fill=(242, 155, 61), outline=(210, 120, 40))


def draw_texas_star(draw, x, y, fill=(36, 84, 162)):
    pts = [
        (x, y - 8), (x + 2, y - 2), (x + 8, y - 2), (x + 3, y + 2), (x + 5, y + 8),
        (x, y + 4), (x - 5, y + 8), (x - 3, y + 2), (x - 8, y - 2), (x - 2, y - 2)
    ]
    draw.polygon(pts, fill=fill)


def draw_ny_landmarks(draw, width, base_y):
    blue = (52, 85, 170)
    gold = (223, 168, 54)
    draw.rectangle([(10, base_y - 8), (width - 10, base_y + 10)],
                   fill=(246, 229, 185))
    draw.line([(10, base_y - 8), (width - 10, base_y - 8)], fill=gold, width=2)
    draw.line([(10, base_y + 10), (width - 10, base_y + 10)],
              fill=blue, width=2)

    skyline = [
        (24, base_y + 8), (24, base_y), (34, base_y), (34, base_y + 8),
        (45, base_y + 8), (45, base_y - 3), (58, base_y - 3), (58, base_y + 8),
        (72, base_y + 8), (72, base_y - 6), (85, base_y - 6), (85, base_y + 8),
        (98, base_y + 8), (98, base_y - 1), (110, base_y - 1), (110, base_y + 8),
    ]
    draw.line(skyline, fill=blue, width=2)

    # simple statue / skyline cue on left
    draw.line([(145, base_y + 8), (150, base_y - 6),
              (155, base_y + 8)], fill=blue, width=2)
    draw.line([(150, base_y - 6), (150, base_y - 12)], fill=blue, width=2)


def draw_plate_base(draw, width, height, fill, border):
    draw.rounded_rectangle(
        [(4, 4), (width - 4, height - 4)],
        radius=10,
        outline=border,
        width=3,
        fill=fill,
    )


def draw_state_template(draw, state_name, width, height, cfg):
    bg = cfg["bg"]
    border = cfg["border"]
    accent = cfg["accent"]

    if state_name == "florida":
        draw_state_plate_common(draw, width, height, bg, border)
        state_font = load_font(17, bold=True)
        small_font = load_font(8, bold=False)
        bottom_font = load_font(10, bold=False)

        # Keep the header above the OCR row.
        header = "FLORIDA"
        box = draw.textbbox((0, 0), header, font=state_font)
        draw.text(
            ((width - (box[2] - box[0])) // 2, 7),
            header,
            font=state_font,
            fill=border,
        )

        # Florida-inspired artwork stays entirely below the character row.
        outline_x = 84
        outline_y = 95
        outline_pts = [
            (outline_x - 13, outline_y - 6),
            (outline_x - 5, outline_y - 9),
            (outline_x + 3, outline_y - 8),
            (outline_x + 9, outline_y - 4),
            (outline_x + 7, outline_y),
            (outline_x + 3, outline_y + 4),
            (outline_x - 3, outline_y + 4),
            (outline_x - 7, outline_y + 9),
            (outline_x - 9, outline_y + 14),
            (outline_x - 12, outline_y + 8),
            (outline_x - 12, outline_y + 2),
            (outline_x - 15, outline_y),
        ]
        draw.polygon(
            outline_pts,
            outline=(185, 210, 181),
            fill=(244, 248, 241),
        )

        orange_x = 116
        orange_y = 103
        draw.ellipse(
            [(orange_x - 5, orange_y - 5), (orange_x + 5, orange_y + 5)],
            fill=(244, 164, 77),
            outline=(220, 140, 58),
        )
        draw.polygon(
            [
                (orange_x + 2, orange_y - 4),
                (orange_x + 7, orange_y - 8),
                (orange_x + 5, orange_y - 1),
            ],
            fill=(96, 153, 95),
        )

        draw.text(
            (150, 94),
            "myflorida.com",
            font=small_font,
            fill=(128, 168, 128),
        )

        bottom = cfg["bottom_text"]
        bb = draw.textbbox((0, 0), bottom, font=bottom_font)
        draw.text(
            ((width - (bb[2] - bb[0])) // 2, 103),
            bottom,
            font=bottom_font,
            fill=(101, 151, 104),
        )

    elif state_name == "california":
        draw_state_plate_common(draw, width, height, bg, border)
        script_font = load_font(25, bold=True)
        plate_small = load_font(9, bold=False)

        header = "CALIFORNIA"
        hb = draw.textbbox((0, 0), header, font=script_font)
        draw.text(((width - (hb[2] - hb[0])) // 2, 6),
                  header, font=script_font, fill=accent)
        draw.text((22, 100), "dmv.ca.gov",
                  font=plate_small, fill=(110, 110, 110))
        draw.text((width - 74, 100), "APR",
                  font=plate_small, fill=(42, 80, 160))

    elif state_name == "texas":
        draw_state_plate_common(draw, width, height, bg, border)
        state_font = load_font(16, bold=True)
        bottom_font = load_font(10, bold=False)

        header = "TEXAS"
        hb = draw.textbbox((0, 0), header, font=state_font)
        draw.text(((width - (hb[2] - hb[0])) // 2, 9),
                  header, font=state_font, fill=(72, 72, 72))
        bb = draw.textbbox((0, 0), "The Texas Classic", font=bottom_font)
        draw.text(((width - (bb[2] - bb[0])) // 2, 100),
                  "The Texas Classic", font=bottom_font, fill=(80, 80, 80))
        draw_texas_star(draw, 34, 95, fill=accent)

    elif state_name == "new_york":
        draw_state_plate_common(draw, width, height, bg, border)
        # Excelsior-inspired top band and landmark strip
        draw.rectangle([(8, 8), (width - 8, 22)], fill=(44, 73, 150))
        state_font = load_font(18, bold=True)
        hb = draw.textbbox((0, 0), "NEW YORK", font=state_font)
        draw.text(((width - (hb[2] - hb[0])) // 2, 4),
                  "NEW YORK", font=state_font, fill=(248, 248, 248))
        draw_ny_landmarks(draw, width, 101)

    else:
        draw_state_plate_common(draw, width, height, bg, border)
        state_font = load_font(16, bold=True)
        bottom_font = load_font(10, bold=False)
        hb = draw.textbbox((0, 0), cfg["state_text"], font=state_font)
        draw.text(((width - (hb[2] - hb[0])) // 2, 9),
                  cfg["state_text"], font=state_font, fill=border)
        bb = draw.textbbox((0, 0), cfg["bottom_text"], font=bottom_font)
        draw.text(((width - (bb[2] - bb[0])) // 2, 100),
                  cfg["bottom_text"], font=bottom_font, fill=border)

    draw_screw_holes(draw, width)


def draw_state_plate_common(draw, width, height, bg, border):
    draw_plate_base(draw, width, height, bg, border)
    for y in range(10, height - 10):
        lift = 5 - int((y / height) * 7)
        color = tuple(max(0, min(255, c + lift)) for c in bg)
        draw.line([(8, y), (width - 8, y)], fill=color)


def draw_main_text(draw, text, state_name, cfg, width):
    text_color = cfg["text"]
    if state_name == "california":
        plate_font = load_font(52, bold=True)
        y = 33
    elif state_name == "new_york":
        plate_font = load_font(54, bold=True)
        y = 31
    elif state_name == "florida":
        plate_font = load_font(55, bold=True)
        y = 31
    else:
        plate_font = load_font(56, bold=True)
        y = 31

    bbox = draw.textbbox((0, 0), text, font=plate_font)
    tw = bbox[2] - bbox[0]
    tx = (width - tw) // 2

    draw.text((tx + 1, y + 1), text, font=plate_font, fill=(95, 95, 95))
    draw.text((tx, y), text, font=plate_font, fill=text_color)


def draw_plate(text, state_name="simple", condition="clean"):
    cfg = STATE_CONFIG[state_name]
    width, height = 360, 120
    img = Image.new("RGB", (width, height), cfg["bg"])
    draw = ImageDraw.Draw(img)

    draw_state_template(draw, state_name, width, height, cfg)
    draw_main_text(draw, text, state_name, cfg, width)

    img = add_background_noise(img, amount=90)
    img = apply_condition(img, condition)
    return img


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def generate_dataset(out_dir, states, count_per_state, length=None, seed=42):
    random.seed(seed)
    ensure_dir(out_dir)

    rows = []
    total = 0

    for state_name in states:
        if state_name not in STATE_CONFIG:
            raise ValueError(
                f"Unsupported state '{state_name}'. Supported: {list(STATE_CONFIG.keys())}")

        for i in range(count_per_state):
            condition = CONDITIONS[i % len(CONDITIONS)]
            plate_text = random_plate_text(state_name, length=length)
            img = draw_plate(plate_text, state_name=state_name,
                             condition=condition)

            filename = f"{state_name}_{i:04d}_{condition}.png"
            save_path = os.path.join(out_dir, filename)
            img.save(save_path)

            rows.append({
                "filename": filename,
                "plate_text": plate_text,
                "condition": condition,
                "state": state_name,
            })
            total += 1

    labels_path = os.path.join(out_dir, "labels.csv")
    with open(labels_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filename", "plate_text", "condition", "state"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {total} plates in {out_dir}")
    print(f"Saved labels to {labels_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic plate images.")
    parser.add_argument("--count", type=int, default=25,
                        help="Number of plates per state")
    parser.add_argument("--length", type=int, default=None,
                        help="Optional forced plate length override")
    parser.add_argument("--out", type=str,
                        default="data/multistate", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--states",
        nargs="+",
        default=["florida", "california", "texas", "new_york"],
        help="List of states/styles to generate",
    )
    args = parser.parse_args()

    generate_dataset(out_dir=args.out, states=args.states,
                     count_per_state=args.count, length=args.length, seed=args.seed)


if __name__ == "__main__":
    main()
