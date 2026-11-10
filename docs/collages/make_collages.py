"""Build the README header collages from the Riggs 1890 page scans.

Run from the repository root:
    python docs/collages/make_collages.py data/processed_images

Each panel is a crop of one scan page: (page number, left, top, width) in scan
pixels. The crop height follows from the panel shape.
"""
import sys
from pathlib import Path

from PIL import Image

OUT = Path(__file__).parent
W, H, GAP, BG = 1600, 380, 8, (252, 252, 251)


def page(scans: Path, number: int) -> Image.Image:
    return Image.open(scans / f"grammardictionar00riggrich_{number:04d}.jpg").convert("RGB")


def banner(name: str, panels: list[Image.Image]) -> None:
    n = len(panels)
    pw = (W - GAP * (n - 1)) // n
    out = Image.new("RGB", (pw * n + GAP * (n - 1), H), BG)
    for i, im in enumerate(panels):
        scale = max(pw / im.width, H / im.height)
        im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
        left, top = (im.width - pw) // 2, (im.height - H) // 2
        out.paste(im.crop((left, top, left + pw, top + H)), (i * (pw + GAP), 0))
    out.save(OUT / f"{name}.jpg", quality=84, optimize=True)


def crops(scans: Path, specs: list[tuple[int, int, int, int]]) -> list[Image.Image]:
    n = len(specs)
    pw = (W - GAP * (n - 1)) // n
    out = []
    for number, left, top, width in specs:
        height = round(width * H / pw)
        out.append(page(scans, number).crop((left, top, left + width, top + height)))
    return out


def main(scans: Path, root: Path) -> None:
    banner("nov2026_two_parts", crops(scans, [(95, 500, 430, 1400), (371, 500, 500, 1400), (100, 230, 330, 1000)]))
    banner("dataset_two_witnesses", crops(scans, [(100, 230, 560, 1000), (376, 230, 560, 1000)]))
    banner("rl_environment_entries", crops(scans, [(100, 230, 1500, 620), (172, 230, 900, 620), (280, 230, 600, 620), (355, 230, 700, 620)]))
    banner("tinker_grammar_tables", crops(scans, [(57, 330, 330, 1750), (40, 430, 2330, 1550)]))
    banner("pilot_dictionary_closeups", crops(scans, [(280, 1220, 300, 860), (355, 1220, 1500, 860), (172, 1220, 2000, 860)]))
    banner("grpo_etymology_text", crops(scans, [(40, 330, 330, 880), (40, 1100, 1000, 880), (40, 330, 1650, 880)]))
    banner("plan_english_dakota", crops(scans, [(430, 230, 400, 860), (411, 230, 1200, 860), (376, 1220, 1800, 860)]))
    banner("source_pages", [page(scans, n) for n in (9, 40, 57, 95, 100, 371)])
    banner("how_to_use_closeups", crops(scans, [(411, 1220, 400, 1000), (430, 1220, 1700, 1000)]))
    public = root / "Public"
    banner("model_progression_title_page", [Image.open(public / "grammar.jpg").convert("RGB")])
    banner("pipeline_dictionary_strip", [Image.open(public / "dictionary2.jpg").convert("RGB")])
    banner("reward_prepositions", [Image.open(public / "Prepositions.jpg").convert("RGB")])


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(__file__).resolve().parents[2])
