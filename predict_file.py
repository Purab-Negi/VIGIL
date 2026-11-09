"""Predict rested vs fatigued for one recording with the VIGIL model (WavLM, no normalisation).

The recording must follow the corpus script: "Hello, my name is ..." followed by the three
pangrams. Any format ffmpeg can read works (mp3, m4a, webm, wav ...).

    .venv/Scripts/python.exe predict_file.py path/to/recording.m4a
    .venv/Scripts/python.exe predict_file.py data/wav16k/S04_fat.wav --exclude S04

--exclude trains without that speaker, so a dataset recording is tested as an unseen speaker.
Without it, the model is trained on all 66 speakers (cached in models/ after the first run).
"""
import argparse
import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg
import joblib
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad
from sklearn.model_selection import GroupKFold

import config as C
import step5_evaluate as E
from step2_locate_intro import locate
from step3_vad_segment import segment_bounds, speech_regions
from step4_extract_features import wavlm_extractor

FEATURE = "wavlm"


def load_audio(path):
    """Decode anything to mono 16 kHz, exactly as step 1 does."""
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "x.wav"
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(path),
                        "-ac", "1", "-ar", str(C.SAMPLE_RATE), "-c:a", "pcm_s16le", str(wav)], check=True)
        x, _ = sf.read(wav, dtype="float32")
        whisper = WhisperModel(C.WHISPER_MODEL, device="cpu", compute_type="int8",
                               download_root=str(C.MODEL_CACHE / "whisper"))
        segments, _ = whisper.transcribe(str(wav), language="en", word_timestamps=True, beam_size=5)
        words = [{"w": w.word.strip(), "start": w.start, "end": w.end} for s in segments for w in s.words]
    return x, words


def to_segments(x, words):
    """Steps 2-3: cut the self-introduction, remove silence, cut 3 s overlapping segments."""
    start, end, *_ = locate(words)
    sr = C.SAMPLE_RATE
    pangrams = x[int(start * sr): int(end * sr)]
    regions = speech_regions(pangrams, load_silero_vad(), min_silence_duration_ms=C.VAD_MIN_SILENCE_MS,
                             speech_pad_ms=C.VAD_SPEECH_PAD_MS)
    speech = np.concatenate([pangrams[s:e] for s, e in regions]) if regions else pangrams
    bounds = segment_bounds(len(speech), int(C.SEGMENT_S * sr), int(C.HOP_S * sr), int(C.MIN_TAIL_S * sr))
    return [speech[s:e] for s, e in bounds], start, len(speech) / sr


def train(exclude=None):
    """Same classifier as the evaluation: scaler -> PCA 95 % -> RBF-SVM, grid-searched by speaker folds."""
    cache = C.ROOT / "models" / f"vigil_{FEATURE}_all.joblib"
    if exclude is None and cache.exists():
        return joblib.load(cache)
    segs = pd.read_csv(C.RESULTS / "segments.csv")
    X = E.load_features(FEATURE, segs)
    keep = (segs.speaker_id != exclude).to_numpy()
    y, g = segs.label.to_numpy()[keep], segs.speaker_id.to_numpy()[keep]
    gs = E.GridSearchCV(E.base_pipeline(), C.SVM_GRID, cv=GroupKFold(C.INNER_FOLDS),
                        scoring=C.GRID_SCORING, n_jobs=-1, refit=False).fit(X[keep], y, groups=g)
    model = E.base_pipeline().set_params(**gs.best_params_, svc__probability=True,
                                         svc__random_state=C.SEED).fit(X[keep], y)
    if exclude is None:
        cache.parent.mkdir(exist_ok=True)
        joblib.dump(model, cache)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--exclude", metavar="SPEAKER", help="leave this speaker out of training, e.g. S04")
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    x, words = load_audio(args.audio)
    segments, intro_end, speech_s = to_segments(x, words)
    extract = wavlm_extractor()
    X = np.vstack([extract(s) for s in segments]).astype(np.float64)

    model = train(args.exclude)
    scores = model.decision_function(X)
    probs = model.predict_proba(X)[:, 1]

    print(f"\nFile: {args.audio}")
    print(f"Intro cut at {intro_end:.2f} s; {speech_s:.2f} s of pangram speech -> {len(segments)} segments")
    print(f"Trained on: {'all 66 speakers' if args.exclude is None else 'all speakers except ' + args.exclude}")
    for i, (s, p) in enumerate(zip(scores, probs)):
        print(f"  segment {i}: score {s:+.3f}  P(fatigued) {p:.2f}")
    mean = scores.mean()
    verdict = "FATIGUED" if mean > 0 else "RESTED"
    conf = probs.mean() if mean > 0 else 1 - probs.mean()
    print(f"\nPrediction: {verdict}  (mean score {mean:+.3f}, confidence {conf:.0%})")


if __name__ == "__main__":
    main()
