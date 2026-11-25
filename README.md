# Hebrew Meeting Assistant

Live Hebrew transcription, speaker identification and AI-generated action items for meetings.

> 🎬 _Demo GIF goes here_ · 🔗 _Live demo link goes here_

## Why
Israeli tech meetings don't sound like Hebrew *or* English. They sound like this:
*"בוא נעשה sync על ה-deployment ונפתח ticket בג'ירה"*.
Speech models are trained mostly on monolingual audio, so this Hebrew-English mix and its tech
jargon are exactly where they break. This project measures that gap, tries to close it, and builds
a meeting assistant on top of the result, tested with real users.

## What makes this hard (and measured)
- **Code-switching:** English terms inside Hebrew sentences, written either in Latin or Hebrew letters.
- **Hebrew prefixes:** "הדיפלוימנט", "בג'ירה". The term-matching metric accounts for them.
- **Term recall metric:** of the tech terms actually said, how many were transcribed correctly
  (any accepted spelling counts). Plain WER unfairly punishes "דיפלוימנט" vs "deployment".

## Benchmark: which model actually works on Hebrew?
Tested on _N_ minutes of real Hebrew audio (meetings, voice notes, podcasts), hand-transcribed.

<!-- paste results/benchmark.md here -->
| Model | WER ↓ | RTF ↓ |
|---|---|---|
| _coming soon_ | | |

**WER** = word error rate (lower is better). **RTF** = processing time / audio length (below 1 = faster than real time).

## Roadmap
- [x] Hebrew transcription with timestamps
- [ ] Benchmark: stock Whisper vs. Hebrew fine-tuned models, general vs. tech speech
- [ ] Improve tech-term recall: glossary prompt → LLM post-correction → (maybe) small fine-tune
- [ ] Speaker diarization (who said what)
- [ ] LLM summary + action items (owner, task, deadline)
- [ ] Web UI
- [ ] Near-real-time mode
- [ ] Docker + public deployment
- [ ] Real-user pilot: what people actually needed, and what I changed because of it

## Run locally
```bash
pip install -r requirements.txt
python transcribe.py my_recording.mp3 --out transcript.json
python transcribe.py my_recording.mp3 --glossary glossary.txt
python benchmark.py
```

## Credits
Hebrew ASR model by [ivrit.ai](https://huggingface.co/ivrit-ai).
