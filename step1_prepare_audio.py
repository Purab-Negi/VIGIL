"""Step 1 - Data Acquisition Layer (Sec. IV-B/C).

Extracts the recordings, renames them to anonymous utterance codes, transcodes every file to
mono 16 kHz 16-bit PCM WAV, and runs the signal-quality checks (peak level, clipping).
The files are WebM/Opus despite the .mp3 extension, so ffmpeg does the decoding.
"""
import subprocess
import zipfile

import imageio_ffmpeg
import numpy as np
import pandas as pd
import soundfile as sf

import config as C


def main():
    ann = pd.read_csv(C.ANNOTATIONS_CSV)
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    C.WAV_DIR.mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    with zipfile.ZipFile(C.AUDIO_ZIP) as z:
        members = {m.split("/")[-1]: m for m in z.namelist() if not m.endswith("/")}
        for row in ann.itertuples():
            raw = C.RAW_DIR / f"{row.utt_id}.bin"   # anonymous name; original name stays in the private map
            if not raw.exists():
                raw.write_bytes(z.read(members[row.original_filename]))

    rows = []
    for row in ann.itertuples():
        raw = C.RAW_DIR / f"{row.utt_id}.bin"
        wav = C.WAV_DIR / f"{row.utt_id}.wav"
        if not wav.exists():
            subprocess.run(
                [ffmpeg, "-y", "-loglevel", "error", "-i", str(raw),
                 "-ac", "1", "-ar", str(C.SAMPLE_RATE), "-c:a", "pcm_s16le", str(wav)],
                check=True,
            )
        x, sr = sf.read(wav, dtype="float32")
        assert sr == C.SAMPLE_RATE and x.ndim == 1
        peak = float(np.max(np.abs(x))) + 1e-12
        clip_ratio = float(np.mean(np.abs(x) >= C.CLIP_THRESHOLD))
        flags = []
        if clip_ratio > C.CLIP_FLAG_RATIO:
            flags.append("clipping")
        if 20 * np.log10(peak) < C.LOW_PEAK_DBFS:
            flags.append("low_level")
        rows.append({
            "utt_id": row.utt_id, "speaker_id": row.speaker_id, "label": row.label,
            "wav_duration_s": round(len(x) / sr, 3),
            "peak_dbfs": round(20 * np.log10(peak), 2),
            "clip_ratio": clip_ratio,
            "signal_qc_flags": ";".join(flags) or "ok",
            "annotation_qc_flag": row.qc_flag,
        })

    qc = pd.DataFrame(rows)
    C.RESULTS.mkdir(exist_ok=True)
    qc.to_csv(C.RESULTS / "qc_signal.csv", index=False)
    print(qc["signal_qc_flags"].value_counts().to_string())
    print(f"Transcoded {len(qc)} recordings to {C.WAV_DIR}")


if __name__ == "__main__":
    main()
