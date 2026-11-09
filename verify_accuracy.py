"""Manual proof of the reported accuracy.

Part 1 recounts the accuracy from the saved LOSO predictions, one row per recording, and writes a
CSV you can open in Excel and count by hand.
Part 2 (optional, --refit S04) re-runs one LOSO fold from scratch: trains on the other 65
speakers, predicts the held-out speaker, and shows the result matches the saved prediction.

    .venv/Scripts/python.exe verify_accuracy.py
    .venv/Scripts/python.exe verify_accuracy.py --refit S04
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import config as C
import step5_evaluate as E

FEATURE, NORM, PROTOCOL = "wavlm", "none", "loso"


def recount():
    p = pd.read_csv(C.RESULTS / "predictions_segments.csv")
    p = p[(p.feature == FEATURE) & (p.normalisation == NORM) & (p.protocol == PROTOCOL)]

    # Each recording's decision = average of its segment scores; score > 0 means "fatigued".
    rec = p.groupby(["speaker_id", "utt_id", "label"], as_index=False).agg(
        n_segments=("score", "size"), mean_score=("score", "mean"),
        prob_fatigued=("prob_fatigued", "mean"))
    rec["true"] = rec.label.map({0: "rested", 1: "fatigued"})
    rec["predicted"] = np.where(rec.mean_score > 0, "fatigued", "rested")
    rec["correct"] = rec.true == rec.predicted
    out = C.RESULTS / "manual_check_wavlm_loso.csv"
    rec.drop(columns="label").round(4).to_csv(out, index=False)

    n, k = len(rec), int(rec.correct.sum())
    tr = rec[rec.true == "rested"]; tf = rec[rec.true == "fatigued"]
    print(f"Configuration: {FEATURE}, normalisation={NORM}, protocol={PROTOCOL}\n")
    print(rec[["speaker_id", "utt_id", "true", "predicted", "correct"]].head(10).to_string(index=False))
    print("...\n")
    print(f"Recordings: {n}  (rested {len(tr)}, fatigued {len(tf)})")
    print(f"Correct:    {k}")
    print(f"Accuracy  = {k}/{n} = {k / n:.4f}")
    print(f"Rested recall   = {int(tr.correct.sum())}/{len(tr)} = {tr.correct.mean():.4f}")
    print(f"Fatigued recall = {int(tf.correct.sum())}/{len(tf)} = {tf.correct.mean():.4f}")
    print(f"UAR = mean of the two recalls = {(tr.correct.mean() + tf.correct.mean()) / 2:.4f}")
    both = rec.groupby("speaker_id").correct.all().sum()
    print(f"Speakers with both recordings right: {both}/{rec.speaker_id.nunique()}")
    print(f"\nPer-recording sheet for hand checking: {out}")
    return rec


def refit(speaker, rec):
    """Independently re-run one LOSO fold and compare with the saved prediction."""
    segs = pd.read_csv(C.RESULTS / "segments.csv")
    X = E.load_features(FEATURE, segs)
    y, g = segs.label.to_numpy(), segs.speaker_id.to_numpy()
    te = g == speaker
    assert te.any(), f"unknown speaker {speaker}"
    print(f"\nRe-running LOSO fold for {speaker}: training on {len(set(g[~te]))} other speakers "
          f"({(~te).sum()} segments), testing on {te.sum()} segments of {speaker} ...")
    score, prob, info = E.fit_predict(X[~te], y[~te], X[te], GroupKFold(C.INNER_FOLDS), g[~te])
    fresh = segs[te].assign(score=score).groupby("utt_id").score.mean()
    print(f"chosen hyperparameters: {info}")
    for utt, s in fresh.items():
        saved = rec.set_index("utt_id").loc[utt]
        pred = "fatigued" if s > 0 else "rested"
        print(f"  {utt}: true={saved.true:8s} fresh prediction={pred:8s} (score {s:+.3f}) "
              f"| saved prediction={saved.predicted}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refit", metavar="SPEAKER", help="re-run one LOSO fold, e.g. S04")
    args = ap.parse_args()
    rec = recount()
    if args.refit:
        refit(args.refit, rec)
