"""Step 5 - Classification Engine + Evaluation and Storage Layer (Sec. IV-E..H, V).

For every representation x normalisation condition:
  * per-speaker baseline normalisation (Sec. IV-E) or the un-normalised control
  * StandardScaler -> PCA (fixed variance) -> RBF-SVM, hyperparameters by inner-CV grid search
    on the training data only (Sec. IV-F)
  * Protocol A: speaker-independent Leave-One-Subject-Out (folds from loso_folds.csv)
  * Protocol B: speaker-dependent random split at segment level (intentionally flawed, Sec. IV-G)
  * segment predictions aggregated to an utterance-level decision (Sec. IV-H step 6)
  * accuracy, UAR, macro-F1, ROC-AUC with participant-level bootstrap 95 % CIs,
    confusion matrices and per-participant accuracy (Sec. V-A)
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.model_selection import (GridSearchCV, GroupKFold, RepeatedStratifiedKFold,
                                     StratifiedKFold)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import config as C

METRICS = ["accuracy", "uar", "macro_f1", "roc_auc"]


# ---------------------------------------------------------------------------- data
def load_features(name, segs):
    d = np.load(C.FEAT_DIR / f"{name}.npz")
    order = pd.Series(np.arange(len(d["seg_id"])), index=d["seg_id"])
    return d["X"][order.loc[segs.seg_id].to_numpy()].astype(np.float64)


def normalise(X, segs, mode):
    """Sec. IV-E: subtract each participant's mean rested-session embedding (as written in the paper)."""
    if mode == "none":
        return X
    Xn = X.copy()
    for spk, idx in segs.groupby("speaker_id").indices.items():
        rested = idx[segs.label.to_numpy()[idx] == 0]
        Xn[idx] -= X[rested].mean(axis=0)
    return Xn


# ------------------------------------------------------------------------- model
def base_pipeline(use_pca=True):
    steps = [("scale", StandardScaler())]
    if use_pca:
        steps.append(("pca", PCA(n_components=C.PCA_VARIANCE, svd_solver="full")))
    steps.append(("svc", SVC(kernel="rbf")))
    return Pipeline(steps)


def fit_predict(Xtr, ytr, Xte, inner_cv, groups=None, use_pca=True):
    """Grid-search on the training data only, then refit with Platt scaling for confidence scores."""
    gs = GridSearchCV(base_pipeline(use_pca), C.SVM_GRID, cv=inner_cv, scoring=C.GRID_SCORING,
                      n_jobs=-1, refit=False)
    gs.fit(Xtr, ytr, groups=groups)
    model = clone(base_pipeline(use_pca)).set_params(
        **gs.best_params_, svc__probability=True, svc__random_state=C.SEED)
    model.fit(Xtr, ytr)
    info = {**gs.best_params_, "inner_uar": gs.best_score_}
    if use_pca:
        info["pca_components"] = int(model.named_steps["pca"].n_components_)
    return model.decision_function(Xte), model.predict_proba(Xte)[:, 1], info


def run_loso(X, segs, folds, use_pca=True):
    score = np.zeros(len(segs)); prob = np.zeros(len(segs)); log = []
    y, g = segs.label.to_numpy(), segs.speaker_id.to_numpy()
    for f in folds.itertuples():
        te = g == f.test_speaker
        tr = ~te
        s, p, info = fit_predict(X[tr], y[tr], X[te], GroupKFold(C.INNER_FOLDS), g[tr], use_pca)
        score[te], prob[te] = s, p
        log.append({"fold": f.fold, "test_speaker": f.test_speaker, **info})
    return score, prob, log


def run_random_split(X, segs, use_pca=True):
    """Speaker-dependent protocol: segments of one participant can land in train and test."""
    y = segs.label.to_numpy()
    score = np.zeros(len(segs)); prob = np.zeros(len(segs)); log = []
    outer = RepeatedStratifiedKFold(n_splits=C.RANDOM_SPLIT_FOLDS,
                                    n_repeats=C.RANDOM_SPLIT_REPEATS, random_state=C.SEED)
    for k, (tr, te) in enumerate(outer.split(X, y)):
        inner = StratifiedKFold(C.INNER_FOLDS, shuffle=True, random_state=C.SEED + k)
        s, p, info = fit_predict(X[tr], y[tr], X[te], inner, None, use_pca)
        score[te] += s / C.RANDOM_SPLIT_REPEATS   # every segment is tested once per repeat
        prob[te] += p / C.RANDOM_SPLIT_REPEATS
        log.append({"outer_fold": k, **info})
    return score, prob, log


# ----------------------------------------------------------------------- metrics
def metrics(y, score):
    pred = (score > 0).astype(int)
    tp = np.sum((pred == 1) & (y == 1)); tn = np.sum((pred == 0) & (y == 0))
    fp = np.sum((pred == 1) & (y == 0)); fn = np.sum((pred == 0) & (y == 1))
    rec1, rec0 = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    prec1, prec0 = tp / max(tp + fp, 1), tn / max(tn + fn, 1)
    f1 = lambda p, r: 0.0 if p + r == 0 else 2 * p * r / (p + r)
    ranks = rankdata(score)
    n1, n0 = np.sum(y == 1), np.sum(y == 0)
    auc = (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return {"accuracy": (tp + tn) / len(y), "uar": (rec0 + rec1) / 2,
            "macro_f1": (f1(prec0, rec0) + f1(prec1, rec1)) / 2, "roc_auc": auc}


def bootstrap_ci(df, rng):
    """Resample participants (not segments), since segments of one speaker are not independent."""
    idx_by_spk = [ix for ix in df.groupby("speaker_id").indices.values()]
    y, s = df.label.to_numpy(), df.score.to_numpy()
    boot = []
    for _ in range(C.BOOTSTRAP_ITERS):
        pick = rng.integers(0, len(idx_by_spk), len(idx_by_spk))
        ix = np.concatenate([idx_by_spk[i] for i in pick])
        boot.append(metrics(y[ix], s[ix]))
    boot = pd.DataFrame(boot)
    return {m: (boot[m].quantile(0.025), boot[m].quantile(0.975)) for m in METRICS}


def to_utterances(seg_pred):
    """Sec. IV-H step 6: mean segment score -> utterance decision; confidence = mean P(predicted class)."""
    u = seg_pred.groupby(["utt_id", "speaker_id", "label"], as_index=False).agg(
        score=("score", "mean"), prob_fatigued=("prob_fatigued", "mean"), n_segments=("score", "size"))
    u["pred"] = (u.score > 0).astype(int)
    u["confidence"] = np.where(u.pred == 1, u.prob_fatigued, 1 - u.prob_fatigued)
    return u


def summarise(pred, level, cfg, rng):
    m = metrics(pred.label.to_numpy(), pred.score.to_numpy())
    ci = bootstrap_ci(pred, rng)
    return [{**cfg, "level": level, "metric": k, "value": m[k],
             "ci_low": ci[k][0], "ci_high": ci[k][1]} for k in METRICS]


def confusion(pred, cfg, level):
    p = (pred.score > 0).astype(int)
    return {**cfg, "level": level,
            "true_rested_pred_rested": int(((pred.label == 0) & (p == 0)).sum()),
            "true_rested_pred_fatigued": int(((pred.label == 0) & (p == 1)).sum()),
            "true_fatigued_pred_rested": int(((pred.label == 1) & (p == 0)).sum()),
            "true_fatigued_pred_fatigued": int(((pred.label == 1) & (p == 1)).sum())}


def per_participant(seg_pred, utt_pred, cfg):
    s = seg_pred.assign(ok=((seg_pred.score > 0).astype(int) == seg_pred.label))
    u = utt_pred.assign(ok=(utt_pred.pred == utt_pred.label))
    out = pd.DataFrame({"segment_accuracy": s.groupby("speaker_id").ok.mean(),
                        "utterance_accuracy": u.groupby("speaker_id").ok.mean()}).reset_index()
    for k, v in cfg.items():
        out.insert(0, k, v)
    return out


# -------------------------------------------------------------------------- main
def main():
    rng = np.random.default_rng(C.SEED)
    segs = pd.read_csv(C.RESULTS / "segments.csv")
    utts = pd.read_csv(C.RESULTS / "utterances.csv")
    folds = pd.read_csv(C.LOSO_FOLDS_CSV)
    assert set(folds.test_speaker) == set(segs.speaker_id), "loso_folds.csv does not match the data"

    summary, confs, parts, seg_preds, fold_logs = [], [], [], [], {}
    configs = [(f, n, p) for f in C.FEATURE_SETS for n in C.NORMALISATIONS
               for p in ["loso", "random_split"]]
    for feat, norm, protocol in configs:
        cfg = {"feature": feat, "normalisation": norm, "protocol": protocol}
        X = normalise(load_features(feat, segs), segs, norm)
        score, prob, log = (run_loso(X, segs, folds) if protocol == "loso"
                            else run_random_split(X, segs))
        sp = segs[["seg_id", "utt_id", "speaker_id", "label"]].assign(score=score, prob_fatigued=prob)
        up = to_utterances(sp)
        summary += summarise(sp, "segment", cfg, rng) + summarise(up, "utterance", cfg, rng)
        confs += [confusion(sp, cfg, "segment"), confusion(up, cfg, "utterance")]
        parts.append(per_participant(sp, up, cfg))
        seg_preds.append(sp.assign(**cfg))
        fold_logs["/".join(cfg.values())] = log
        uar = metrics(up.label.to_numpy(), up.score.to_numpy())["uar"]
        print(f"{feat:8s} {norm:16s} {protocol:13s} utterance UAR = {uar:.3f}", flush=True)

    # Supplementary sanity baseline (not in the paper): speech duration alone, utterance level, LOSO.
    Xd = utts[["speech_s"]].to_numpy(float)
    dseg = utts.rename(columns={"utt_id": "seg_id"}).assign(utt_id=utts.utt_id)
    score, prob, log = run_loso(Xd, dseg, folds, use_pca=False)
    cfg = {"feature": "speech_duration (supplementary)", "normalisation": "none", "protocol": "loso"}
    dp = dseg[["utt_id", "speaker_id", "label"]].assign(score=score, prob_fatigued=prob)
    summary += summarise(dp, "utterance", cfg, rng)
    confs.append(confusion(dp, cfg, "utterance"))
    fold_logs["/".join(cfg.values())] = log

    summary = pd.DataFrame(summary)
    summary.to_csv(C.RESULTS / "metrics_summary.csv", index=False)
    pd.DataFrame(confs).to_csv(C.RESULTS / "confusion_matrices.csv", index=False)
    pd.concat(parts).to_csv(C.RESULTS / "per_participant_accuracy.csv", index=False)
    pd.concat(seg_preds).to_csv(C.RESULTS / "predictions_segments.csv", index=False)
    (C.RESULTS / "fold_hyperparameters.json").write_text(json.dumps(fold_logs, indent=1, default=str))

    # Primary outcome (Sec. V-B): inflation of the speaker-dependent split over LOSO.
    wide = summary[summary.protocol.isin(["loso", "random_split"])].pivot_table(
        index=["feature", "normalisation", "level", "metric"], columns="protocol", values="value")
    wide["inflation"] = wide["random_split"] - wide["loso"]
    wide.reset_index().to_csv(C.RESULTS / "speaker_leakage_inflation.csv", index=False)
    print(wide.xs("utterance", level="level").xs("uar", level="metric").round(3).to_string())


if __name__ == "__main__":
    main()
