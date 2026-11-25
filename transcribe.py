"""
שלב 1: תמלול קובץ שמע בעברית עם חותמות זמן.

שימוש:
    python transcribe.py recording.mp3
    python transcribe.py recording.mp3 --model small        # Whisper הרגיל, להשוואה
    python transcribe.py recording.mp3 --out transcript.json
    python transcribe.py recording.mp3 --glossary glossary.txt   # עם מילון מונחי הייטק
"""
import argparse
import json
import time
from pathlib import Path

from faster_whisper import WhisperModel

from textutils import glossary_prompt

# המודל של ivrit.ai: Whisper large-v3-turbo שאומן על עברית, בפורמט של faster-whisper
DEFAULT_MODEL = "ivrit-ai/whisper-large-v3-turbo-ct2"


def load_model(model_name: str = DEFAULT_MODEL) -> WhisperModel:
    # int8 רץ סביר גם על מחשב בלי כרטיס מסך. עם GPU אפשר לשנות ל-float16
    return WhisperModel(model_name, device="auto", compute_type="int8")


def transcribe(model: WhisperModel, audio_path: str, initial_prompt: str | None = None) -> dict:
    start = time.perf_counter()
    segments, info = model.transcribe(
        audio_path,
        language="he",
        beam_size=5,
        vad_filter=True,  # מדלג על שקט, חוסך זמן ומונע "הזיות" בקטעים ריקים
        initial_prompt=initial_prompt,  # מילון מונחים, "מכין" את המודל לז'רגון
    )
    seg_list = [
        {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
        for s in segments  # segments הוא גנרטור: התמלול קורה בפועל רק כאן
    ]
    elapsed = time.perf_counter() - start
    return {
        "audio": str(audio_path),
        "duration_sec": round(info.duration, 2),
        "processing_sec": round(elapsed, 2),
        # Real-Time Factor: מתחת ל-1 = מהיר מזמן אמת. חשוב לגרסה החיה בהמשך
        "rtf": round(elapsed / info.duration, 3) if info.duration else None,
        "segments": seg_list,
        "text": " ".join(s["text"] for s in seg_list),
    }


def fmt(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"


def main():
    parser = argparse.ArgumentParser(description="תמלול שמע בעברית")
    parser.add_argument("audio", help="נתיב לקובץ שמע (mp3, wav, m4a, ogg...)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", help="שמירת התוצאה כ-JSON")
    parser.add_argument("--glossary", help="קובץ מונחים, למשל glossary.txt")
    args = parser.parse_args()

    prompt = glossary_prompt(args.glossary) if args.glossary else None
    model = load_model(args.model)
    result = transcribe(model, args.audio, initial_prompt=prompt)

    for s in result["segments"]:
        print(f"[{fmt(s['start'])} - {fmt(s['end'])}] {s['text']}")
    print(f"\nאורך: {result['duration_sec']}s | עיבוד: {result['processing_sec']}s | RTF: {result['rtf']}")

    if args.out:
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"נשמר ל-{args.out}")


if __name__ == "__main__":
    main()
