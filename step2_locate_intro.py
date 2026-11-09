"""Step 2 - Remove the self-introduction (Sec. IV-B).

The self-introduction differs between speakers and can identify them, so only the three
pangram sentences are modelled. Whisper word timestamps locate where the first pangram starts
and where the last pangram ends; everything outside that span is discarded.
"""
import json
import re
from difflib import SequenceMatcher

import pandas as pd
from faster_whisper import WhisperModel

import config as C

TRANSCRIPTS = C.DATA / "transcripts_PRIVATE.json"   # contains spoken names - keep out of results/

# Script read by every participant after "Hello, my name is ...". The pangram span starts at the
# first words of the first pangram and runs to the last transcribed word (VAD trims the tail later).
FIRST_PANGRAM_START = "pack my box with five dozen"


def norm(w):
    w = w.lower().replace("5", "five")        # Whisper writes "five dozen" as "5 ,000" etc.
    return re.sub(r"[^a-z]", "", w)


def transcribe_all(ann):
    if TRANSCRIPTS.exists():
        return json.loads(TRANSCRIPTS.read_text())
    model = WhisperModel(C.WHISPER_MODEL, device="cpu", compute_type="int8",
                         download_root=str(C.MODEL_CACHE / "whisper"))
    out = {}
    for row in ann.itertuples():
        segments, _ = model.transcribe(str(C.WAV_DIR / f"{row.utt_id}.wav"), language="en",
                                       word_timestamps=True, beam_size=5, vad_filter=False)
        out[row.utt_id] = [{"w": w.word.strip(), "start": w.start, "end": w.end}
                           for s in segments for w in s.words]
        print(row.utt_id, " ".join(w["w"] for w in out[row.utt_id]), flush=True)
    TRANSCRIPTS.write_text(json.dumps(out, indent=1))
    return out


def best_match(words, target, lo=0):
    """Index in words where the target word sequence fits best (fuzzy), and its similarity."""
    t = " ".join(target)
    best = (None, 0.0)
    for i in range(lo, len(words)):
        cand = " ".join(words[i:i + len(target)])
        r = SequenceMatcher(None, cand, t).ratio()
        if r > best[1]:
            best = (i, r)
    return best


def anchor_index(toks):
    """Structural anchor: the intro is 'my name is X', the first pangram is '<pack> my box ...',
    so the pangram starts one word before the first 'my' that is not followed by 'name'."""
    for i, t in enumerate(toks[1:], start=1):
        if t == "my" and (i + 1 >= len(toks) or toks[i + 1] != "name"):
            # mis-transcriptions such as "back in my box" insert a filler word before "my"
            return i - 2 if i >= 2 and toks[i - 1] in {"in", "on", "a"} else i - 1
    return None


def locate(words):
    """Return (start_s, end_s, match similarity, intro text, tail text) of the pangram span."""
    words = [w for w in words if norm(w["w"])]
    toks = [norm(w["w"]) for w in words]
    i0, r0 = best_match(toks, FIRST_PANGRAM_START.split())
    anchor = anchor_index(toks)
    if anchor is not None and anchor <= 10:
        i0 = anchor
        r0 = SequenceMatcher(None, " ".join(toks[i0:i0 + 3]), "pack my box").ratio()
    start = max(words[i0]["start"] - C.INTRO_PAD_S, 0.0)
    end = words[-1]["end"] + C.INTRO_PAD_S
    intro = " ".join(w["w"] for w in words[:i0])
    tail = " ".join(w["w"] for w in words[-4:])
    return start, end, r0, intro, tail


def main():
    ann = pd.read_csv(C.ANNOTATIONS_CSV)
    transcripts = transcribe_all(ann)
    rows = []
    for row in ann.itertuples():
        s, e, r0, intro, tail = locate(transcripts[row.utt_id])
        rows.append({"utt_id": row.utt_id, "start_s": round(s, 3), "end_s": round(e, 3),
                     "match_start": round(r0, 3), "intro_words": len(intro.split()),
                     "tail_text": tail,
                     "needs_review": r0 < 0.6 or not 4 <= len(intro.split()) <= 7})
    df = pd.DataFrame(rows)
    df.to_csv(C.RESULTS / "intro_bounds.csv", index=False)
    print(df.describe().to_string())
    print("needs review:", df[df.needs_review].utt_id.tolist())


if __name__ == "__main__":
    main()
