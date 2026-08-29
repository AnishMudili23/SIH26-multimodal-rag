"""Generate a themed image + audio demo corpus for the multimodal RAG.

For testing (and demoing) the image / audio / cross-modal paths without hand-
making assets. Everything is generated offline:
  - images  -> Pillow (text is real, so OCR + ImageBind vision both get signal)
  - audio   -> Windows System.Speech via PowerShell (no pip dependency)

Each asset is tied to one of the MS MARCO passages already in
`corpus_manifest.csv`; the script fills that row's `image_file` / `audio_file`
/ `image_timestamp` columns so `rag/ingest/corpus.py` links them (see load_manifest).

    python scripts/make_demo_assets.py           # generate + update the manifest
    python -m rag.ingest.corpus --src <raw> --reset # then re-index everything

Drop your own real screenshots / voice memos into  data/raw/images  or
data/raw/audio  as well — ingest picks them up; they just won't have a
passage link unless you add them to the manifest by filename.
"""

from __future__ import annotations

import csv
import re
import subprocess
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root on path

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from rag import config  # noqa: E402

IMG_DIR = config.RAW_DATA_DIR / "images"
AUD_DIR = config.RAW_DATA_DIR / "audio"

INK = (23, 23, 26)
MUTED = (110, 116, 124)
ACCENT = (37, 99, 235)
PANEL = (240, 242, 245)


# --------------------------------------------------------------------------
# Fonts
# --------------------------------------------------------------------------

def _font(size: int, bold: bool = False):
    for name in (("arialbd.ttf", "arial.ttf") if bold else ("arial.ttf",)):
        for base in (r"C:\Windows\Fonts", "/usr/share/fonts"):
            p = Path(base) / name
            if p.exists():
                return ImageFont.truetype(str(p), size)
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:  # noqa: BLE001
        return ImageFont.load_default()


# --------------------------------------------------------------------------
# Image primitives
# --------------------------------------------------------------------------

def _canvas(w=1100, h=740, bg=(255, 255, 255)):
    img = Image.new("RGB", (w, h), bg)
    return img, ImageDraw.Draw(img)


def _wrap(draw, text, font, x, y, max_w, fill=INK, leading=8):
    avg = draw.textlength("m", font=font) or 10
    per_line = max(10, int(max_w / avg))
    for line in text.split("\n"):
        for seg in textwrap.wrap(line, per_line) or [""]:
            draw.text((x, y), seg, font=font, fill=fill)
            y += font.size + leading
    return y


def make_report_cover(path, title, subtitle, date, org):
    img, d = _canvas()
    d.rectangle([0, 0, img.width, 12], fill=ACCENT)
    d.text((70, 90), org.upper(), font=_font(22, bold=True), fill=MUTED)
    y = _wrap(d, title, _font(52, bold=True), 70, 190, img.width - 140)
    _wrap(d, subtitle, _font(28), 70, y + 20, img.width - 140, fill=MUTED)
    d.text((70, img.height - 110), date, font=_font(24), fill=INK)
    d.text((70, img.height - 70), "OFFICIAL — INTERNAL DISTRIBUTION",
           font=_font(18, bold=True), fill=MUTED)
    img.save(path)


def make_slide(path, title, bullets):
    img, d = _canvas()
    d.text((70, 70), title, font=_font(44, bold=True), fill=INK)
    d.line([70, 140, img.width - 70, 140], fill=ACCENT, width=4)
    y = 190
    for b in bullets:
        d.ellipse([74, y + 12, 86, y + 24], fill=ACCENT)
        y = _wrap(d, b, _font(28), 110, y, img.width - 200) + 24
    img.save(path)


def make_email(path, sender, to, subject, body):
    img, d = _canvas(h=680)
    d.rectangle([0, 0, img.width, 130], fill=PANEL)
    hf, lf = _font(22, bold=True), _font(22)
    d.text((40, 22), "From:", font=hf, fill=MUTED); d.text((150, 22), sender, font=lf, fill=INK)
    d.text((40, 54), "To:", font=hf, fill=MUTED); d.text((150, 54), to, font=lf, fill=INK)
    d.text((40, 86), "Subject:", font=hf, fill=MUTED)
    d.text((150, 86), subject, font=_font(22, bold=True), fill=INK)
    _wrap(d, body, _font(24), 40, 170, img.width - 80, leading=10)
    img.save(path)


def make_chart(path, title, pairs, unit=""):
    img, d = _canvas()
    d.text((70, 60), title, font=_font(38, bold=True), fill=INK)
    x0, y0, chart_h, bar_w, gap = 130, 620, 420, 110, 60
    d.line([x0 - 20, y0, x0 + len(pairs) * (bar_w + gap), y0], fill=INK, width=3)
    top = max(v for _, v in pairs) or 1
    for i, (label, val) in enumerate(pairs):
        bx = x0 + i * (bar_w + gap)
        bh = int(chart_h * val / top)
        d.rectangle([bx, y0 - bh, bx + bar_w, y0], fill=ACCENT)
        d.text((bx, y0 - bh - 34), f"{val}{unit}", font=_font(24, bold=True), fill=INK)
        _wrap(d, label, _font(20), bx - 10, y0 + 14, bar_w + 40, fill=MUTED)
    img.save(path)


def make_dashboard(path, title, rows, clock):
    img, d = _canvas()
    d.rectangle([0, 0, img.width, 70], fill=INK)
    d.text((30, 20), title, font=_font(26, bold=True), fill=(255, 255, 255))
    d.text((img.width - 140, 20), clock, font=_font(28, bold=True), fill=(255, 255, 255))
    y = 120
    for k, v in rows:
        d.rectangle([40, y, img.width - 40, y + 80], outline=MUTED, width=2)
        d.text((60, y + 24), k, font=_font(26), fill=INK)
        d.text((img.width - 260, y + 20), v, font=_font(30, bold=True), fill=ACCENT)
        y += 104
    d.text((40, img.height - 50), f"Captured {clock}  ·  auto-refresh 5 min",
           font=_font(20), fill=MUTED)
    img.save(path)


def make_diagram(path, title, boxes, caption):
    img, d = _canvas(h=520)
    d.text((70, 50), title, font=_font(36, bold=True), fill=INK)
    bw, bh, y = 230, 120, 200
    x = 70
    for i, label in enumerate(boxes):
        d.rectangle([x, y, x + bw, y + bh], outline=INK, width=3, fill=PANEL)
        _wrap(d, label, _font(22, bold=True), x + 16, y + 24, bw - 24)
        if i < len(boxes) - 1:
            d.line([x + bw, y + bh // 2, x + bw + 60, y + bh // 2], fill=ACCENT, width=4)
            d.polygon([(x + bw + 60, y + bh // 2 - 10), (x + bw + 60, y + bh // 2 + 10),
                       (x + bw + 78, y + bh // 2)], fill=ACCENT)
        x += bw + 78
    _wrap(d, caption, _font(22), 70, y + bh + 60, img.width - 140, fill=MUTED)
    img.save(path)


# --------------------------------------------------------------------------
# Audio (offline TTS via Windows System.Speech)
# --------------------------------------------------------------------------

def tts(text: str, out_path: Path) -> bool:
    ps = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$s.Rate = -1; "
        f"$s.SetOutputToWaveFile('{out_path}'); "
        f"$s.Speak([string]@'\n{text}\n'@); $s.Dispose()"
    )
    try:
        subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                       check=True, capture_output=True, timeout=120)
        return out_path.exists() and out_path.stat().st_size > 1000
    except Exception as e:  # noqa: BLE001
        print(f"  TTS failed ({e}) — install/enable a Windows voice, or add clips manually")
        return False


# --------------------------------------------------------------------------
# Asset specs — each ties to a passage by a keyword found in its text
# --------------------------------------------------------------------------

IMAGE_SPECS = [
    dict(file="report_cover_hydro.png", kw=["hydropower", "waterway", "emission-free"],
         fn=lambda p: make_report_cover(
             p, "Hydropower Environmental Impact Review",
             "Waterway regulation, fish-stocking and habitat mitigation — Q2 2024",
             "12 June 2024", "Directorate of Energy Infrastructure Assessment")),
    dict(file="dashboard_1432.png", kw=["hydropower", "waterway"], ts="14:32",
         fn=lambda p: make_dashboard(
             p, "Hydro Generation — Live Status",
             [("Active plants", "18"), ("Output vs. plan", "97%"),
              ("Fish-stocking budget used", "62%"), ("Flow-regulation alerts", "2")],
             "14:32")),
    dict(file="slide_nuclear_cost.png", kw=["kwh", "refueling", "reload", "nuclear power plant"],
         fn=lambda p: make_slide(
             p, "Nuclear Baseload — Cost per kWh",
             ["Levelised cost driven by capital, not fuel",
              "Fuel: uranium mined, processed, fissioned for heat",
              "Heat raises steam; steam turns the turbine; turbine drives the generator",
              "Long build times are the main cost-recovery risk"])),
    dict(file="email_solar_review.png", kw=["solar", "pay for", "cost of the"],
         fn=lambda p: make_email(
             p, "r.menon@deia.gov", "assessment-team@deia.gov",
             "Re: Solar cost-recovery review — input needed Friday",
             "Team,\n\nThe solar cost-recovery assessment needs your section notes "
             "by Friday. Key question: does generation revenue cover the levelised "
             "cost over the panel lifetime, and how sensitive is that to the "
             "capacity-factor assumption?\n\nAttach figures per site.\n\n— R. Menon")),
    dict(file="chart_capacity_by_source.png", kw=["renewable resources", "biomass", "geothermal"],
         fn=lambda p: make_chart(
             p, "Generation Capacity by Source (GW)",
             [("Hydro", 34), ("Nuclear", 21), ("Wind", 18), ("Solar", 12), ("Gas", 27)],
             unit=" GW")),
    dict(file="diagram_steam_turbine.png", kw=["steam turbine", "condenser", "mechanical power"],
         fn=lambda p: make_diagram(
             p, "Steam Turbine — Energy Path",
             ["Boiler / reactor heat", "Steam turbine", "Condenser", "Generator"],
             "The condenser turns spent steam back to water and holds turbine "
             "back-pressure low, which is what keeps the cycle efficient.")),
    dict(file="slide_emissions.png", kw=["acid rain", "sulfur oxides", "nitrogen oxides", "greenhouse gas"],
         fn=lambda p: make_slide(
             p, "Fossil-Fuel Emissions — Briefing",
             ["Combustion releases CO2, sulfur oxides, nitrogen oxides",
              "Sulfur and nitrogen oxides drive acid rain",
              "Particulates and smoke degrade air quality",
              "Clean air is the stated public-health baseline"])),
    dict(file="slide_wind.png", kw=["wind energy", "wind power", "wind turbine", "wind is"],
         fn=lambda p: make_slide(
             p, "Wind Energy — Use Overview",
             ["Turbines convert the wind's kinetic energy to electricity",
              "Also used for water pumping and mechanical drive",
              "Output scales with the cube of wind speed",
              "Best paired with dispatchable backup"])),
    dict(file="slide_desalination.png", kw=["desalination", "osmosis", "membrane"],
         fn=lambda p: make_slide(
             p, "Reverse-Osmosis Desalination",
             ["Pressure forces seawater through a semi-permeable membrane",
              "Salts and impurities stay behind; fresh water passes",
              "Energy cost is dominated by the high-pressure pumps",
              "Common where surface freshwater is scarce"])),
    dict(file="slide_lng.png", kw=["liquefied natural gas", "compressed natural gas", "natural gas vehicles"],
         fn=lambda p: make_slide(
             p, "Liquefied Natural Gas — Uses",
             ["Natural gas cooled to liquid for dense storage and shipping",
              "Regasified for power generation and heating",
              "Used as a transport fuel for heavy vehicles and ships",
              "Bridges supply where pipelines do not reach"])),
    dict(file="report_cover_three_gorges.png", kw=["three gorges", "yangtze", "hubei"],
         fn=lambda p: make_report_cover(
             p, "Three Gorges Dam — Site Note",
             "World's largest hydroelectric dam, Yangtze River, Hubei Province, China",
             "March 2024", "Directorate of Energy Infrastructure Assessment")),
    dict(file="chart_pollution_types.png", kw=["types of pollution", "water pollution", "environmental pollution"],
         fn=lambda p: make_chart(
             p, "Recorded Incidents by Pollution Type (this quarter)",
             [("Air", 41), ("Water", 33), ("Land", 19)], unit="")),
]

AUDIO_SPECS = [
    dict(file="call_hydro_review.wav", kw=["hydropower", "waterway", "emission-free"],
         text="Briefing note on the hydropower environmental review. The assessment "
              "finds that constructing hydropower plants changes waterways and their "
              "natural conditions. The main impact during operation comes from "
              "regulating the flow of the water. Recommended mitigations are fish "
              "stocking and habitat restoration in the affected areas."),
    dict(file="call_solar_cost.wav", kw=["solar", "pay for", "cost of the"],
         text="Update on the solar cost recovery review. The question is whether "
              "generation revenue covers the levelised cost of the panels over their "
              "lifetime. The result is sensitive to the capacity factor we assume. "
              "Section notes are due Friday."),
    dict(file="call_nuclear_options.wav", kw=["kwh", "refueling", "reload", "nuclear power plant"],
         text="Summary of the nuclear baseload options. Nuclear power works by "
              "fission of uranium, which releases heat. The heat raises steam, the "
              "steam turns a turbine, and the turbine drives the generator. The cost "
              "per kilowatt hour is dominated by capital and build time, not fuel."),
    dict(file="call_emissions.wav", kw=["acid rain", "sulfur oxides", "nitrogen oxides", "greenhouse gas"],
         text="Briefing on fossil fuel emissions. Burning fossil fuels releases "
              "carbon dioxide, sulfur oxides and nitrogen oxides. The sulfur and "
              "nitrogen oxides contribute to acid rain. Smoke and particulates "
              "reduce air quality."),
    dict(file="call_wind.wav", kw=["wind energy", "wind power", "wind turbine", "wind is"],
         text="Note on wind energy. Wind turbines capture the kinetic energy in "
              "moving air and convert it to electricity. Wind is also used for "
              "pumping water and for mechanical work. Output rises sharply with wind "
              "speed."),
    dict(file="call_desalination.wav", kw=["desalination", "osmosis", "membrane"],
         text="Note on reverse osmosis desalination. High pressure forces seawater "
              "through a semi permeable membrane. Fresh water passes through and the "
              "salts are left behind. Most of the energy goes into the pumps."),
]


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def _load_rows() -> list[dict]:
    with config.CORPUS_MANIFEST_CSV.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _match_row(rows, keywords: list[str], used: set[str]) -> dict | None:
    """Highest-scoring passage for a keyword list (whole-word / phrase match).
    Prefers a passage not already claimed by another asset of the same kind."""
    best, best_score = None, 0
    for r in rows:
        hay = " " + re.sub(r"\s+", " ", r.get("passage_text", "").lower()) + " "
        score = sum(
            1 for k in keywords
            if re.search(r"(?<![a-z])" + re.escape(k.lower()) + r"(?![a-z])", hay)
        )
        if score == 0:
            continue
        if r["doc_id"] not in used:
            score += 0.5   # tie-break toward an unclaimed passage
        if score > best_score:
            best, best_score = r, score
    return best


def main() -> int:
    if not config.CORPUS_MANIFEST_CSV.exists():
        print(f"No manifest at {config.CORPUS_MANIFEST_CSV} — run rag/ingest/manifest.py first.")
        return 1
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    AUD_DIR.mkdir(parents=True, exist_ok=True)
    rows = _load_rows()
    fieldnames = list(rows[0].keys())
    for col in ("image_file", "audio_file", "image_timestamp"):
        if col not in fieldnames:
            fieldnames.append(col)
    # idempotent: this run regenerates every asset, so start from a clean slate
    for r in rows:
        r["image_file"] = r["audio_file"] = r["image_timestamp"] = ""

    made_img = made_aud = 0
    used_img: set[str] = set()
    used_aud: set[str] = set()

    print(f"Images -> {IMG_DIR}")
    for spec in IMAGE_SPECS:
        row = _match_row(rows, spec["kw"], used_img)
        if row is None:
            print(f"  skip {spec['file']} (no passage matches {spec['kw']})")
            continue
        out = IMG_DIR / spec["file"]
        spec["fn"](out)
        # image_file and image_timestamp are kept as position-aligned ";"-lists
        files = [x for x in row["image_file"].split(";") if x] + [spec["file"]]
        stamps = (row["image_timestamp"].split(";")
                  if row["image_timestamp"] else [])
        stamps += [""] * (len(files) - 1 - len(stamps)) + [spec.get("ts", "")]
        row["image_file"] = ";".join(files)
        row["image_timestamp"] = ";".join(stamps) if any(stamps) else ""
        used_img.add(row["doc_id"])
        made_img += 1
        print(f"  {spec['file']:32} -> {row['doc_id']}"
              + (f"  @ {spec['ts']}" if spec.get('ts') else ""))

    print(f"\nAudio -> {AUD_DIR}")
    for spec in AUDIO_SPECS:
        row = _match_row(rows, spec["kw"], used_aud)
        if row is None:
            print(f"  skip {spec['file']} (no passage matches {spec['kw']})")
            continue
        out = AUD_DIR / spec["file"]
        if tts(spec["text"], out):
            row["audio_file"] = ";".join(filter(None, [row["audio_file"], spec["file"]]))
            used_aud.add(row["doc_id"])
            made_aud += 1
            print(f"  {spec['file']:32} -> {row['doc_id']}")

    with config.CORPUS_MANIFEST_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"\nDone: {made_img} images, {made_aud} audio. Manifest updated.")
    print("Next:  python -m rag.ingest.corpus --src", config.RAW_DATA_DIR, "--reset")
    return 0


if __name__ == "__main__":
    sys.exit(main())
