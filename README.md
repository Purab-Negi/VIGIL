# VIGIL: voice-based fatigue detection (implementation of the paper)

This pipeline implements Sections IV and V of *VIGIL: Voice-Based Fatigue Detection Using
Self-Supervised Speech Embeddings* on the annotated corpus. That corpus is 66 speakers, each
recorded once rested and once fatigued, for 132 recordings.

| Step | Script | Paper section |
|---|---|---|
| 1 | `step1_prepare_audio.py`: unzip, rename to anonymous IDs, transcode to mono 16 kHz PCM WAV, peak and clipping QC | IV-B, IV-C |
| 2 | `step2_locate_intro.py`: use Whisper word timestamps to cut the self-introduction and keep only the 3 pangrams | IV-B |
| 3 | `step3_vad_segment.py`: silero-VAD silence removal, SNR estimate, 3 s segments with 50 % overlap | IV-C |
| 4 | `step4_extract_features.py`: frozen ECAPA-TDNN, WavLM-Base+ (layer 6, mean-pooled) and eGeMAPSv02 | IV-D |
| 5 | `step5_evaluate.py`: per-speaker rested-baseline normalisation (plus a no-normalisation control), scaler → PCA 95 % → RBF-SVM with inner grid search; LOSO vs segment-level random split; utterance aggregation; metrics with participant bootstrap CIs | IV-E to IV-H, V |

All hyperparameters the paper leaves open are in `config.py`.

## Run

```bash
.venv/Scripts/python.exe step1_prepare_audio.py
.venv/Scripts/python.exe step2_locate_intro.py
.venv/Scripts/python.exe step3_vad_segment.py
.venv/Scripts/python.exe step4_extract_features.py
.venv/Scripts/python.exe step5_evaluate.py
```

All outputs go to `results/`.

## Verify the accuracy by hand

```bash
.venv/Scripts/python.exe verify_accuracy.py              # recount 106/132 from the saved predictions
.venv/Scripts/python.exe verify_accuracy.py --refit S04  # re-run one LOSO fold from scratch
```

`results/manual_check_wavlm_loso.csv` has one row per recording (true label, predicted label,
correct), so the accuracy can be counted in Excel.

## Predict a single recording

```bash
.venv/Scripts/python.exe predict_file.py data/wav16k/S04_fat.wav --exclude S04   # dataset speaker, held out
.venv/Scripts/python.exe predict_file.py "C:/path/to/new_recording.m4a"          # a new person
```

## Decisions and caveats

- **`session_metadata_SYNTHETIC_v2.csv` is not used.** It is synthetic, so none of its values
  (hours slept, KSS, caffeine, illness, microphone) should be reported as study data.
- **Normalisation follows the paper as written.** Each speaker's mean rested embedding is
  subtracted, including for the held-out LOSO speaker. That uses the test speaker's condition
  labels, so the `rested_baseline` numbers are optimistic; the `none` control has no such leak.
- **Inner cross-validation is grouped by speaker** under LOSO, so model selection never sees a
  speaker's segments in both inner-train and inner-validation.
- **Segments versus the complete utterance.** Section IV-C says "overlapping segments" while IV-D
  says "complete utterance". Here each 3 s segment gets one embedding over its full length (no
  frame-level smoothing, unlike the base study), and segment scores are averaged into the
  utterance decision (IV-H step 6).
- **The random-split protocol** is 5-fold × 5-repeat stratified CV over segments, with
  predictions averaged over repeats.
- **The speech-duration baseline is supplementary** and not part of the paper. It shows how much
  can be learned from speaking duration alone.
