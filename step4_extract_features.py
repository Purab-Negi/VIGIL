"""Step 4 - Representation Layer: frozen encoders (Sec. IV-D).

Extracts one fixed-length vector per segment with three representations:
  ecapa   - SpeechBrain ECAPA-TDNN speaker embedding (192-d)
  wavlm   - WavLM-Base+ hidden states of an intermediate layer, mean-pooled over time (768-d)
  egemaps - openSMILE eGeMAPSv02 functionals (88-d), the interpretable classical baseline
All neural models run in inference mode only; nothing is trained here.
"""
import sys

import numpy as np
import pandas as pd
import soundfile as sf
import torch

import config as C


def load_segments():
    segs = pd.read_csv(C.RESULTS / "segments.csv")
    cache = {}
    for r in segs.itertuples():
        if r.utt_id not in cache:
            cache[r.utt_id], _ = sf.read(C.SPEECH_DIR / f"{r.utt_id}.wav", dtype="float32")
        yield r.seg_id, cache[r.utt_id][r.start:r.end]


def ecapa_extractor():
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy
    model = EncoderClassifier.from_hparams(
        source=C.ECAPA_SOURCE, savedir=str(C.MODEL_CACHE / "ecapa"),
        run_opts={"device": "cpu"}, local_strategy=LocalStrategy.COPY)
    model.eval()
    return lambda x: model.encode_batch(torch.from_numpy(x)[None]).squeeze().numpy()


def wavlm_extractor():
    from transformers import AutoFeatureExtractor, WavLMModel
    fe = AutoFeatureExtractor.from_pretrained(C.WAVLM_SOURCE, cache_dir=C.MODEL_CACHE)
    model = WavLMModel.from_pretrained(C.WAVLM_SOURCE, cache_dir=C.MODEL_CACHE).eval()

    def run(x):
        inputs = fe(x, sampling_rate=C.SAMPLE_RATE, return_tensors="pt")
        out = model(**inputs, output_hidden_states=True)
        return out.hidden_states[C.WAVLM_LAYER][0].mean(dim=0).numpy()
    return run


def egemaps_extractor():
    import opensmile
    smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                            feature_level=opensmile.FeatureLevel.Functionals)
    return lambda x: smile.process_signal(x, C.SAMPLE_RATE).to_numpy().ravel()


EXTRACTORS = {"ecapa": ecapa_extractor, "wavlm": wavlm_extractor, "egemaps": egemaps_extractor}


def main(names):
    C.FEAT_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    for name in names:
        extract = EXTRACTORS[name]()
        ids, X = [], []
        for seg_id, x in load_segments():
            ids.append(seg_id)
            X.append(extract(x))
        X = np.vstack(X).astype(np.float32)
        assert np.isfinite(X).all(), f"non-finite values in {name} features"
        np.savez(C.FEAT_DIR / f"{name}.npz", X=X, seg_id=np.array(ids, dtype=str))
        print(f"{name}: {X.shape}")


if __name__ == "__main__":
    main(sys.argv[1:] or C.FEATURE_SETS)
