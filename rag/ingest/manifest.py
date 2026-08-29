"""Build the demo corpus from the **AMI Meeting Corpus** (ROADMAP Phase 2).

AMI gives genuinely simultaneous, same-content signals for a meeting: the
audio, a time-aligned orthographic transcript, the slides that were on the
projector (JPEG + slide-change timestamps), and -- for the scenario meetings --
the participants' own minutes and final report. That is exactly what the PS's
cross-format-linking claim needs, with nothing paired after the fact.

License: **CC BY 4.0** (confirmed on the official corpus page). Attribution
only, commercial use allowed. This script prints the attribution block to add
to the repo once AMI files are committed.

For each selected meeting id this downloads into
``RAW_DATA_DIR/ami/<meeting_id>/``:

  * ``<mid>.Mix-Headset.wav``            the meeting audio (mixed headsets)
  * ``<mid>.Mix-Headset.wav.segments.json``   the real transcript, time-aligned
        -- a sidecar `extraction.extract_audio` reads instead of running Whisper
  * ``slides/<mid>.<start>__<end>.jpg``  slides; the filename is the on-screen
        window in seconds. OCR text is produced by our own pytesseract pass.
  * ``docs/*.txt``                       the meeting's minutes / summary emails
        (`.doc` -> text via ``antiword``); the shared final report lands under
        ``ami/_shared/``.

and writes a manifest CSV with columns

    doc_id, meeting_id, modality, source_file, text_preview, start_time, end_time

where ``doc_id`` groups every one of a meeting's chunks (transcript <-> slide <->
minutes) so cross-modal linking is queryable. This feeds the unchanged Phase 0
ingestion pipeline (`rag/ingest/corpus.py`).

    python -m rag.ingest.manifest                       # ES2002a, ES2002b
    python -m rag.ingest.manifest --meetings ES2002a ES2002b ES2003a
    python -m rag.ingest.manifest --skip-download       # rebuild manifest only
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from rag import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("ami")

_UA = {"User-Agent": "Mozilla/5.0 (offline-rag corpus builder)"}
MANIFEST_FIELDS = ["doc_id", "meeting_id", "modality", "source_file",
                   "text_preview", "start_time", "end_time"]

# Minutes-of-Nth-meeting  ->  meeting-id letter (scenario meetings run a..d)
_MINUTES_ORDINAL = {"kickoff": "a", "2nd": "b", "3rd": "c", "4th": "d"}
_SUM_RE = re.compile(r"([A-Z]{2}\d{4}[a-d])(ID|ME|PM|UI)sum", re.I)
_ROLE = {"ID": "industrial-designer", "ME": "marketing-expert",
         "PM": "project-manager", "UI": "user-interface"}


# --------------------------------------------------------------------------
# HTTP helpers (build-time only -- inference stays fully offline)
# --------------------------------------------------------------------------

def _get(url: str, timeout: int = 120) -> bytes:
    for attempt in range(3):
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, headers=_UA), timeout=timeout
            ) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            if attempt == 2:
                raise
            logger.warning("  retry %d for %s (%s)", attempt + 1, url, e)
    return b""


def _download(url: str, dest: Path, timeout: int = 300) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        logger.info("  cached  %s", dest.name)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("  GET     %s", url)
    dest.write_bytes(_get(url, timeout=timeout))
    logger.info("  saved   %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
    return dest


def _list_dir(url: str) -> list[str]:
    """File names from an Apache autoindex page."""
    html = _get(url, timeout=60).decode("utf-8", "replace")
    return [h for h in re.findall(r'href="([^"?/][^"]*)"', html)
            if not h.startswith("/")]


# --------------------------------------------------------------------------
# Manual transcript  (words + segments NXT XML -> time-aligned segments)
# --------------------------------------------------------------------------

def _annotations_zip() -> zipfile.ZipFile:
    cache = config.RAW_DATA_DIR / "ami" / "_cache" / "ami_public_manual_1.6.2.zip"
    _download(config.AMI_ANNOTATIONS_URL, cache, timeout=300)
    return zipfile.ZipFile(cache)


def _join_words(tokens: list[tuple[str, bool]]) -> str:
    """tokens: (text, is_punctuation). Punctuation joins with no leading space;
    "'s", "n't" style clitics too."""
    out = ""
    for text, is_punc in tokens:
        text = text.strip()
        if not text:
            continue
        if not out:
            out = text
        elif is_punc or text[0] in ",.;:!?)]}%" or text.startswith("'"):
            out += text
        elif out[-1] in "([{":
            out += text
        else:
            out += " " + text
    return out


def parse_transcript(zf: zipfile.ZipFile, meeting_id: str) -> list[dict]:
    """Ordered list of {speaker, start, end, text} for a meeting, from the
    manual `words` + `segments` annotation layers."""
    names = zf.namelist()
    segments: list[dict] = []
    for spk in "ABCDE":
        wname = f"words/{meeting_id}.{spk}.words.xml"
        sname = f"segments/{meeting_id}.{spk}.segments.xml"
        if wname not in names or sname not in names:
            continue

        words: dict[str, tuple[str, float | None, float | None, bool]] = {}
        wroot = ET.fromstring(zf.read(wname))
        for w in wroot:
            wid = w.attrib.get("{http://nite.sourceforge.net/}id")
            if not wid:
                continue
            txt = (w.text or "").strip()
            if not txt and w.tag.endswith("w"):
                continue
            st = w.attrib.get("starttime")
            en = w.attrib.get("endtime")
            is_punc = w.attrib.get("punc") == "true"
            words[wid] = (txt, float(st) if st else None,
                          float(en) if en else None, is_punc)
        order = list(words)  # words.xml is emitted in order

        sroot = ET.fromstring(zf.read(sname))
        for seg in sroot:
            st = seg.attrib.get("transcriber_start")
            en = seg.attrib.get("transcriber_end")
            child = seg.find("{http://nite.sourceforge.net/}child")
            if child is None:
                continue
            href = child.attrib.get("href", "")
            ids = re.findall(r"id\(([^)]+)\)", href)
            if not ids:
                continue
            if len(ids) == 1:
                wid_range = [ids[0]]
            else:
                try:
                    a, b = order.index(ids[0]), order.index(ids[-1])
                    wid_range = order[a:b + 1]
                except ValueError:
                    wid_range = ids
            toks = [(words[i][0], words[i][3]) for i in wid_range if i in words]
            text = _join_words(toks)
            if not text or not re.search(r"[A-Za-z0-9]", text):
                continue
            s = float(st) if st else (
                next((words[i][1] for i in wid_range if i in words and words[i][1] is not None), 0.0))
            e = float(en) if en else (
                next((words[i][2] for i in reversed(wid_range) if i in words and words[i][2] is not None), s))
            segments.append({"speaker": spk, "start": round(s, 3),
                             "end": round(e, 3), "text": text})

    segments.sort(key=lambda d: (d["start"], d["speaker"]))
    return segments


# --------------------------------------------------------------------------
# Per-meeting fetchers
# --------------------------------------------------------------------------

def fetch_audio(meeting_id: str, dest_dir: Path) -> Path:
    stream = config.AMI_AUDIO_STREAM
    fn = f"{meeting_id}.{stream}.wav"
    url = f"{config.AMI_MIRROR}/{meeting_id}/audio/{fn}"
    return _download(url, dest_dir / fn, timeout=600)


def fetch_slides(meeting_id: str, dest_dir: Path, limit: int | None = None
                 ) -> list[tuple[Path, float, float]]:
    """Download slide JPEGs from the mirror's slidesBackUp/ (the on-Edinburgh
    copy; slides/ points at a dead idiap.ch host). Filename is
    ``<start>__<end>.jpg`` -- the slide's on-screen window in seconds."""
    base = f"{config.AMI_MIRROR}/{meeting_id}/slidesBackUp/"
    try:
        names = [n for n in _list_dir(base) if n.lower().endswith(".jpg")]
    except Exception as e:  # noqa: BLE001
        logger.warning("  no slides for %s (%s)", meeting_id, e)
        return []
    out: list[tuple[Path, float, float]] = []
    dest_dir.mkdir(parents=True, exist_ok=True)
    for n in sorted(names)[: limit or None]:
        m = re.match(r"(\d+(?:\.\d+)?)__(\d+(?:\.\d+)?)\.jpg$", n)
        if not m:
            continue
        start, end = float(m.group(1)), float(m.group(2))
        local = dest_dir / f"{meeting_id}.{n}"
        try:
            _download(base + n, local, timeout=120)
            out.append((local, start, end))
        except Exception as e:  # noqa: BLE001
            logger.warning("  slide %s failed (%s)", n, e)
    logger.info("  slides: %d for %s", len(out), meeting_id)
    return out


def _antiword() -> str | None:
    for c in ("antiword", "antiword.exe"):
        p = shutil.which(c)
        if p:
            return p
    for p in (r"C:\Program Files\Git\mingw64\bin\antiword.exe",
              r"C:\Program Files\Git\usr\bin\antiword.exe",
              "/mingw64/bin/antiword"):
        if Path(p).exists():
            return p
    return None


def _doc_to_text(doc_path: Path, out_path: Path) -> Path | None:
    aw = _antiword()
    if not aw:
        logger.warning("  antiword not found -- skipping %s (install Git for "
                       "Windows or `apt install antiword`)", doc_path.name)
        return None
    try:
        r = subprocess.run([aw, "-w", "0", str(doc_path)],
                           capture_output=True, text=True, timeout=120)
        if r.returncode != 0 or not r.stdout.strip():
            logger.warning("  antiword failed on %s: %s", doc_path.name, r.stderr[:160])
            return None
        out_path.write_text(r.stdout, encoding="utf-8")
        return out_path
    except Exception as e:  # noqa: BLE001
        logger.warning("  antiword error on %s: %s", doc_path.name, e)
        return None


def fetch_docs(meeting_ids: list[str], series: str, raw_dir: Path
               ) -> list[tuple[Path, str, str]]:
    """Pull the scenario meetings' real shared-doc files.
    Returns (text_path, doc_id, meeting_id) -- meeting_id "" for series-level
    documents (the final report spans the whole a-d project)."""
    base = f"{config.AMI_MIRROR}/{series}a/shared-doc/"  # all a-d share one dir
    try:
        names = _list_dir(base)
    except Exception as e:  # noqa: BLE001
        logger.warning("  no shared-doc for %s (%s)", series, e)
        return []

    want_letters = {m[-1] for m in meeting_ids}
    cache = raw_dir / "_cache" / "shared-doc"
    cache.mkdir(parents=True, exist_ok=True)
    shared_dir = raw_dir / "_shared"
    shared_dir.mkdir(parents=True, exist_ok=True)
    out: list[tuple[Path, str, str]] = []

    for n in names:
        low = n.lower()
        target_mid: str | None = None
        doc_id: str | None = None
        label: str | None = None

        m = _SUM_RE.search(n)
        if low.endswith("sum.txt") and m:
            letter = m.group(1)[-1].lower()
            if letter not in want_letters:
                continue
            target_mid = f"{series}{letter}"
            doc_id = target_mid
            label = f"{target_mid}.summary-{_ROLE.get(m.group(2).upper(), 'role')}.txt"
            local = _download(base + n, cache / n, timeout=60)
            dest = (raw_dir / target_mid / "docs"); dest.mkdir(parents=True, exist_ok=True)
            txt_path = dest / label
            shutil.copy2(local, txt_path)
            out.append((txt_path, doc_id, target_mid))
            continue

        if "minutes.of." in low and low.endswith(".doc"):
            mo = re.search(r"minutes\.of\.([a-z0-9]+)-meeting", low)
            ordn = mo.group(1) if mo else ""
            letter = _MINUTES_ORDINAL.get(ordn)
            if not letter or letter not in want_letters:
                continue
            target_mid = f"{series}{letter}"
            doc_id = target_mid
            label = f"{target_mid}.minutes-{ordn}.txt"
            local = _download(base + n, cache / n, timeout=60)
            dest = (raw_dir / target_mid / "docs"); dest.mkdir(parents=True, exist_ok=True)
            txt = _doc_to_text(local, dest / label)
            if txt:
                out.append((txt, doc_id, target_mid))
            continue

        if "final.report" in low and low.endswith(".doc"):
            local = _download(base + n, cache / n, timeout=60)
            txt = _doc_to_text(local, shared_dir / f"{series}.final-report.txt")
            if txt:
                out.append((txt, series, ""))     # series-level doc_id
            continue

    logger.info("  docs: %d for %s", len(out), series)
    return out


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------

def _rel(p: Path, raw_dir: Path) -> str:
    try:
        return p.relative_to(raw_dir).as_posix()
    except ValueError:
        return p.name


def build(meetings: list[str], raw_dir: Path, manifest_path: Path,
          skip_download: bool = False, slide_limit: int | None = None) -> None:
    ami_root = raw_dir / "ami"
    ami_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    series = sorted({re.sub(r"[a-z]$", "", m) for m in meetings})
    zf = None if skip_download else _annotations_zip()

    for mid in meetings:
        logger.info("meeting %s", mid)
        mdir = ami_root / mid
        mdir.mkdir(parents=True, exist_ok=True)
        audio_fn = f"{mid}.{config.AMI_AUDIO_STREAM}.wav"
        audio_path = mdir / audio_fn

        if not skip_download:
            fetch_audio(mid, mdir)
            segs = parse_transcript(zf, mid)  # type: ignore[arg-type]
            (mdir / f"{audio_fn}.segments.json").write_text(
                json.dumps(segs, ensure_ascii=False, indent=0), encoding="utf-8")
            logger.info("  transcript: %d segments", len(segs))
            slides = fetch_slides(mid, mdir / "slides", limit=slide_limit)
        else:
            slides = []
            for jp in sorted((mdir / "slides").glob("*.jpg")):
                m = re.search(r"(\d+(?:\.\d+)?)__(\d+(?:\.\d+)?)\.jpg$", jp.name)
                if m:
                    slides.append((jp, float(m.group(1)), float(m.group(2))))

        if audio_path.exists():
            rows.append({"doc_id": mid, "meeting_id": mid, "modality": "audio",
                         "source_file": _rel(audio_path, raw_dir),
                         "text_preview": "", "start_time": "", "end_time": ""})
        for jp, s, e in slides:
            rows.append({"doc_id": mid, "meeting_id": mid, "modality": "image",
                         "source_file": _rel(jp, raw_dir), "text_preview": "",
                         "start_time": f"{s:.2f}", "end_time": f"{e:.2f}"})

    # shared documents (minutes / summaries / final report) -- real, not synthetic
    if not skip_download:
        for ser in series:
            for txt_path, doc_id, mid in fetch_docs(
                [m for m in meetings if m.startswith(ser)], ser, ami_root
            ):
                rows.append({"doc_id": doc_id, "meeting_id": mid,
                             "modality": "document",
                             "source_file": _rel(txt_path, raw_dir),
                             "text_preview": "", "start_time": "", "end_time": ""})
    else:
        for p in sorted(ami_root.rglob("docs/*.txt")) + sorted((ami_root / "_shared").glob("*.txt")):
            mid = p.parent.parent.name if p.parent.name == "docs" else ""
            doc_id = mid or p.stem.split(".")[0]
            rows.append({"doc_id": doc_id, "meeting_id": mid, "modality": "document",
                         "source_file": _rel(p, raw_dir), "text_preview": "",
                         "start_time": "", "end_time": ""})

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)

    by_mod: dict[str, int] = {}
    for r in rows:
        by_mod[r["modality"]] = by_mod.get(r["modality"], 0) + 1
    logger.info("wrote %d manifest rows -> %s  (%s)", len(rows), manifest_path,
                ", ".join(f"{k}:{v}" for k, v in sorted(by_mod.items())))
    _print_attribution(meetings)


def _print_attribution(meetings: list[str]) -> None:
    print("\n" + "=" * 70)
    print("AMI Meeting Corpus  --  license: CC BY 4.0")
    print("Add this attribution to README.md / the demo credits:")
    print("-" * 70)
    print("  Demo corpus: the AMI Meeting Corpus (http://groups.inf.ed.ac.uk/ami/corpus/),")
    print("  meetings " + ", ".join(meetings) + ". Licensed CC BY 4.0.")
    print("  J. Carletta et al., 'The AMI Meeting Corpus', 2005.")
    print("=" * 70 + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meetings", nargs="+", default=config.AMI_MEETINGS,
                    help="AMI meeting ids (default: %(default)s)")
    ap.add_argument("--out", type=Path, default=config.CORPUS_MANIFEST_CSV)
    ap.add_argument("--raw-dir", type=Path, default=config.RAW_DATA_DIR)
    ap.add_argument("--skip-download", action="store_true",
                    help="rebuild the manifest from already-downloaded files")
    ap.add_argument("--slide-limit", type=int, default=None,
                    help="cap slides per meeting (quick iteration)")
    args = ap.parse_args(argv)

    build(args.meetings, args.raw_dir, args.out,
          skip_download=args.skip_download, slide_limit=args.slide_limit)
    print("Next:  python -m rag.ingest.corpus --src", args.raw_dir / "ami", "--reset")
    return 0


if __name__ == "__main__":
    sys.exit(main())
