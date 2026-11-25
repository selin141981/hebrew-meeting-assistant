"""
השוואת מודלי תמלול על עברית אמיתית ועל עברית-הייטק (עברית מעורבת באנגלית).

שלושה מדדים:
    WER          - אחוז המילים שהמודל טעה בהן (נמוך = טוב)
    Term Recall  - מתוך המונחים הטכניים שנאמרו, כמה תומללו נכון (גבוה = טוב)
    RTF          - זמן עיבוד חלקי אורך השמע (מתחת ל-1 = מהיר מזמן אמת)

כל מודל נבדק פעמיים: בלי מילון מונחים, ועם מילון (initial_prompt).
כך רואים אם המילון באמת משפר, ובכמה.

מבנה התיקייה, תת-תיקייה לכל סוג הקלטה:
    data/benchmark/
        general/        <- עברית רגילה: הודעות קוליות, פודקאסט, שיחה
            clip01.wav
            clip01.txt  <- התמלול הנכון, מוקלד ידנית
        hitech/         <- פגישות ושיחות עם ז'רגון טכני
            standup01.m4a
            standup01.txt

שימוש:
    python benchmark.py
    python benchmark.py --models ivrit-ai/whisper-large-v3-turbo-ct2 --no-glossary-compare
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import jiwer

from textutils import glossary_prompt, load_glossary, normalize_he, term_hits
from transcribe import load_model, transcribe

AUDIO_EXT = {".wav", ".mp3", ".m4a", ".ogg", ".opus", ".flac", ".webm"}
DEFAULT_MODELS = [
    "small",                                 # Whisper רגיל, קטן
    "large-v3-turbo",                        # Whisper רגיל, גדול
    "ivrit-ai/whisper-large-v3-turbo-ct2",   # מאומן על עברית
]


def load_clips(data_dir: Path):
    """מחזיר רשימה של (קטגוריה, קובץ שמע, תמלול נכון)."""
    clips = []
    for audio in sorted(data_dir.rglob("*")):
        if audio.suffix.lower() not in AUDIO_EXT:
            continue
        ref = audio.with_suffix(".txt")
        if not ref.exists():
            print(f"אזהרה: אין תמלול נכון עבור {audio.name}, מדלג")
            continue
        category = audio.parent.name if audio.parent != data_dir else "general"
        clips.append((category, audio, ref.read_text(encoding="utf-8")))
    return clips


def pct(x):
    return "—" if x is None else f"{x:.1%}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/benchmark")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--glossary", default="glossary.txt")
    parser.add_argument("--no-glossary-compare", action="store_true",
                        help="להריץ רק בלי מילון (חוסך חצי מהזמן)")
    args = parser.parse_args()

    clips = load_clips(Path(args.data))
    if not clips:
        raise SystemExit(f"לא נמצאו זוגות שמע+txt ב-{args.data}")
    categories = sorted({c for c, _, _ in clips})
    print(f"{len(clips)} קטעים בקטגוריות: {', '.join(categories)}\n")

    terms = load_glossary(args.glossary)
    prompt = glossary_prompt(args.glossary)
    variants = [("no", None)]
    if prompt and not args.no_glossary_compare:
        variants.append(("yes", prompt))

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    rows, summary = [], []

    for model_name in args.models:
        model = load_model(model_name)
        for gl_label, gl_prompt in variants:
            print(f"== {model_name} | glossary={gl_label} ==")
            refs, hyps = defaultdict(list), defaultdict(list)
            said_total = hit_total = 0
            missed_all = defaultdict(int)
            audio_sec = proc_sec = 0.0

            for category, audio, ref in clips:
                r = transcribe(model, str(audio), initial_prompt=gl_prompt)
                ref_n, hyp_n = normalize_he(ref), normalize_he(r["text"])
                said, hit, missed = term_hits(ref, r["text"], terms)
                said_total += said
                hit_total += hit
                for m in missed:
                    missed_all[m] += 1
                refs[category].append(ref_n)
                hyps[category].append(hyp_n)
                audio_sec += r["duration_sec"]
                proc_sec += r["processing_sec"]
                clip_wer = jiwer.wer(ref_n, hyp_n)
                rows.append({
                    "model": model_name, "glossary": gl_label, "category": category,
                    "clip": audio.name, "wer": round(clip_wer, 4),
                    "terms_said": said, "terms_hit": hit, "missed_terms": ", ".join(missed),
                    "rtf": r["rtf"], "hypothesis": r["text"], "reference": ref.strip(),
                })
                print(f"  [{category}] {audio.name}: WER={clip_wer:.1%}  terms={hit}/{said}")

            # WER כולל מחושב על כל המילים יחד, לא ממוצע של אחוזים
            all_refs = [x for c in categories for x in refs[c]]
            all_hyps = [x for c in categories for x in hyps[c]]
            entry = {
                "model": model_name,
                "glossary": gl_label,
                "wer_all": jiwer.wer(all_refs, all_hyps),
                "wer_by_cat": {c: jiwer.wer(refs[c], hyps[c]) if refs[c] else None for c in categories},
                "term_recall": hit_total / said_total if said_total else None,
                "rtf": proc_sec / audio_sec if audio_sec else 0,
                "top_missed": sorted(missed_all.items(), key=lambda x: -x[1])[:5],
            }
            summary.append(entry)
            print(f"  סה\"כ: WER={pct(entry['wer_all'])}  Term Recall={pct(entry['term_recall'])}"
                  f"  RTF={entry['rtf']:.2f}\n")
        del model  # שחרור זיכרון לפני המודל הבא

    # קובץ מפורט, לכל קטע: מה נאמר, מה תומלל, אילו מונחים פוספסו
    with open(out_dir / "benchmark.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    # טבלה מוכנה ל-README
    cat_cols = " | ".join(f"WER {c} ↓" for c in categories)
    md = [f"| Model | Glossary | WER all ↓ | {cat_cols} | Term recall ↑ | RTF ↓ |",
          "|---|---|---|" + "---|" * len(categories) + "---|---|"]
    for e in sorted(summary, key=lambda e: e["wer_all"]):
        cats = " | ".join(pct(e["wer_by_cat"][c]) for c in categories)
        md.append(f"| `{e['model']}` | {e['glossary']} | {pct(e['wer_all'])} | {cats} | "
                  f"{pct(e['term_recall'])} | {e['rtf']:.2f} |")
    md.append("\n**Most-missed tech terms (best configuration):** " +
              ", ".join(f"{t} ({n}×)" for t, n in min(summary, key=lambda e: e["wer_all"])["top_missed"]))
    (out_dir / "benchmark.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
