"""Step 3 - Signal Processing Layer (Sec. IV-C).

For each recording: keep only the pangram portion (intro removed in step 2), remove leading,
trailing and inter-sentence silence with a voice activity detector, estimate SNR, and cut the
remaining speech into overlapping fixed-length segments.
"""
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from silero_vad import get_speech_timestamps, load_silero_vad

import config as C


def speech_regions(x, model, **kw):
    ts = get_speech_timestamps(torch.from_numpy(x), model, sampling_rate=C.SAMPLE_RATE,
                               threshold=C.VAD_THRESHOLD, **kw)
    return [(t["start"], t["end"]) for t in ts]


def estimate_snr_db(x, frame=400):
    """Percentile SNR: loud (speech) frame energy vs quiet (background) frame energy, 25 ms frames."""
    n = len(x) // frame
    energy = np.mean(x[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-10
    return float(10 * np.log10(np.percentile(energy, 90) / np.percentile(energy, 10)))


def segment_bounds(n, seg, hop, min_tail):
    """Start/end sample indices of overlapping windows covering n samples."""
    if n <= seg:
        return [(0, n)]
    starts = list(range(0, n - seg + 1, hop))
    if n - (starts[-1] + seg) >= min_tail:
        starts.append(n - seg)          # end-aligned window so the tail is not discarded
    return [(s, s + seg) for s in starts]


def main():
    ann = pd.read_csv(C.ANNOTATIONS_CSV)
    bounds = pd.read_csv(C.RESULTS / "intro_bounds.csv").set_index("utt_id")
    C.SPEECH_DIR.mkdir(parents=True, exist_ok=True)
    model = load_silero_vad()
    sr = C.SAMPLE_RATE
    seg, hop, tail = int(C.SEGMENT_S * sr), int(C.HOP_S * sr), int(C.MIN_TAIL_S * sr)

    utt_rows, seg_rows = [], []
    for row in ann.itertuples():
        x, _ = sf.read(C.WAV_DIR / f"{row.utt_id}.wav", dtype="float32")
        snr = estimate_snr_db(x)

        b = bounds.loc[row.utt_id]
        pangrams = x[int(b.start_s * sr): int(b.end_s * sr)]
        regions = speech_regions(pangrams, model, min_silence_duration_ms=C.VAD_MIN_SILENCE_MS,
                                 speech_pad_ms=C.VAD_SPEECH_PAD_MS)
        speech = np.concatenate([pangrams[s:e] for s, e in regions]) if regions else pangrams
        sf.write(C.SPEECH_DIR / f"{row.utt_id}.wav", speech, sr, subtype="PCM_16")

        segs = segment_bounds(len(speech), seg, hop, tail)
        for i, (s, e) in enumerate(segs):
            seg_rows.append({"seg_id": f"{row.utt_id}_{i:02d}", "utt_id": row.utt_id,
                             "speaker_id": row.speaker_id, "label": row.label,
                             "start": s, "end": e})
        utt_rows.append({"utt_id": row.utt_id, "speaker_id": row.speaker_id, "label": row.label,
                         "pangram_span_s": round(len(pangrams) / sr, 3),
                         "speech_s": round(len(speech) / sr, 3),
                         "n_segments": len(segs), "snr_db": round(snr, 2),
                         "snr_flag": "low_snr" if snr < C.LOW_SNR_DB else "ok"})

    utt = pd.DataFrame(utt_rows)
    utt.to_csv(C.RESULTS / "utterances.csv", index=False)
    pd.DataFrame(seg_rows).to_csv(C.RESULTS / "segments.csv", index=False)
    print(utt.groupby("label")[["speech_s", "n_segments", "snr_db"]].describe().T.to_string())
    print(f"{len(seg_rows)} segments from {len(utt)} utterances; "
          f"{(utt.snr_flag != 'ok').sum()} flagged low SNR")


if __name__ == "__main__":
    main()
