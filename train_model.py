"""
train_model.py
==============
Stages 3-4 of the pipeline (Sections 3.4.4 - 3.4.5).

  * TF-IDF (unigrams + bigrams) fitted on the TRAINING split only
  * Three classifiers: Multinomial Naive Bayes, Logistic Regression, Random Forest
  * Hyperparameters tuned with stratified k-fold cross-validation on the
    training split; SMOTE is applied INSIDE each fold and to the training
    split only, never to validation or test (Section 3.4.2.4)
  * Metrics per split: accuracy, precision, recall, F1 (Fake class),
    macro-F1, AUC-ROC, Brier score, confusion matrix, false-positive rate
    (genuine claims wrongly flagged as Fake). Train accuracy is reported so
    the train-test gap can flag overfitting (3.4.5)
  * Disaggregated (bias-audit) results by language, category, source,
    region, party reference, religion reference and ethnic reference
  * Figures for Chapter Four saved to models/figures/

Outputs (models/):
  vectorizer.pkl, naive_bayes.pkl, logistic_regression.pkl, random_forest.pkl,
  best_model_name.txt, metrics_report.json, metrics_summary.csv,
  disaggregated_report.csv, training_meta.json, figures/*.png

Usage:
  python train_model.py --datadir data --outdir models
"""

import argparse
import json
import os
import platform
import warnings
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB

warnings.filterwarnings("ignore", category=UserWarning)
RANDOM_STATE = 42
LABELS = ["Real", "Fake"]
POS = "Fake"
GROUP_COLUMNS = ["language_tag", "category", "source", "region", "party_ref",
                 "religion_ref", "ethnic_ref"]
DISPLAY = {"naive_bayes": "Naive Bayes", "logistic_regression": "Logistic Regression",
           "random_forest": "Random Forest"}


def load_split(datadir, name):
    path = os.path.join(datadir, f"{name}.csv")
    df = pd.read_csv(path, keep_default_na=False)
    missing = {"clean_text", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing {missing}. Run preprocess.py first.")
    for c in GROUP_COLUMNS:
        if c not in df.columns:
            df[c] = "Unknown"
    df["clean_text"] = df["clean_text"].fillna("").astype(str)
    return df.reset_index(drop=True)


def p_fake(model, X):
    classes = list(model.classes_)
    return model.predict_proba(X)[:, classes.index(POS)]


def compute_metrics(y_true, y_pred, pf):
    y_true = np.asarray(y_true)
    y_bin = (y_true == POS).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    tn, fp = cm[0, 0], cm[0, 1]  # row 0 = actual Real
    out = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_fake": float(precision_score(y_true, y_pred, pos_label=POS, zero_division=0)),
        "recall_fake": float(recall_score(y_true, y_pred, pos_label=POS, zero_division=0)),
        "f1_fake": float(f1_score(y_true, y_pred, pos_label=POS, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", labels=LABELS,
                                   zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else None,
        "brier_score": float(brier_score_loss(y_bin, pf)),
        "roc_auc": None,
        "confusion_matrix": cm.tolist(),
    }
    if len(set(y_bin)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_bin, pf))
    return out


def build_candidates(minority_train):
    # SMOTE needs k_neighbors < minority samples inside each CV training fold
    k = max(1, min(5, int(minority_train * 0.6) - 1))
    use_smote = minority_train >= 3

    def pipe(clf):
        steps = [("smote", SMOTE(k_neighbors=k, random_state=RANDOM_STATE))] if use_smote else []
        return ImbPipeline(steps + [("clf", clf)])

    return {
        "naive_bayes": (pipe(MultinomialNB()), {"clf__alpha": [0.1, 0.5, 1.0]}),
        "logistic_regression": (
            pipe(LogisticRegression(solver="liblinear", max_iter=2000,
                                    random_state=RANDOM_STATE)),
            {"clf__C": [0.1, 1, 10]}),
        "random_forest": (
            pipe(RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
            {"clf__n_estimators": [100, 200, 300], "clf__max_depth": [None, 30]}),
    }, use_smote, k


def disaggregate(model_key, split, df, y_pred, pf):
    rows = []
    for col in GROUP_COLUMNS:
        for grp, idx in df.groupby(col).groups.items():
            idx = np.asarray(list(idx))
            m = compute_metrics(df.loc[idx, "label"].values, y_pred[idx], pf[idx])
            rows.append({
                "model": DISPLAY[model_key], "split": split, "group_column": col,
                "group": grp, "n": m["n"], "accuracy": m["accuracy"],
                "f1_macro": m["f1_macro"], "precision_fake": m["precision_fake"],
                "recall_fake": m["recall_fake"], "false_positive_rate": m["false_positive_rate"],
                "small_sample_warning": m["n"] < 10,
            })
    return rows


def make_figures(outdir, results, test_df, test_scores):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    figdir = os.path.join(outdir, "figures")
    os.makedirs(figdir, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    colors = ["#2a6f97", "#c9732b", "#5a8f3d"]
    keys = list(results)

    # 1. Confusion matrices (test set)
    fig, axes = plt.subplots(1, len(keys), figsize=(4.4 * len(keys), 4))
    for ax, k in zip(np.atleast_1d(axes), keys):
        cm = np.array(results[k]["test"]["confusion_matrix"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=LABELS, yticklabels=LABELS)
        ax.set_title(DISPLAY[k])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.suptitle("Confusion matrices - test set")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "confusion_matrices_test.png"), dpi=150)
    plt.close(fig)

    # 2. ROC curves (test set)
    y_bin = (test_df["label"].values == POS).astype(int)
    if len(set(y_bin)) == 2:
        fig, ax = plt.subplots(figsize=(5.5, 5))
        for k, c in zip(keys, colors):
            fpr, tpr, _ = roc_curve(y_bin, test_scores[k])
            auc = results[k]["test"]["roc_auc"]
            ax.plot(fpr, tpr, color=c, lw=2, label=f"{DISPLAY[k]} (AUC {auc:.3f})")
        ax.plot([0, 1], [0, 1], ls="--", color="#999999", lw=1)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate (Fake recall)")
        ax.set_title("ROC curves - test set")
        ax.legend(loc="lower right", frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, "roc_curves_test.png"), dpi=150)
        plt.close(fig)

    # 3. Metric comparison (test set)
    metrics = ["accuracy", "precision_fake", "recall_fake", "f1_fake", "f1_macro", "roc_auc"]
    names = ["Accuracy", "Precision (Fake)", "Recall (Fake)", "F1 (Fake)", "Macro F1", "AUC-ROC"]
    x = np.arange(len(metrics))
    w = 0.8 / len(keys)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, (k, c) in enumerate(zip(keys, colors)):
        vals = [results[k]["test"][m] or 0 for m in metrics]
        ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=DISPLAY[k], color=c)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05)
    ax.set_title("Classifier comparison - test set")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "metric_comparison_test.png"), dpi=150)
    plt.close(fig)

    # 4. Train vs test accuracy (overfitting check)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = np.arange(len(keys))
    ax.bar(x - 0.18, [results[k]["train"]["accuracy"] for k in keys], 0.36,
           label="Train", color="#9db4c0")
    ax.bar(x + 0.18, [results[k]["test"]["accuracy"] for k in keys], 0.36,
           label="Test", color="#2a6f97")
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[k] for k in keys])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Train vs test accuracy (overfitting check)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "train_vs_test_accuracy.png"), dpi=150)
    plt.close(fig)

    # 5. Macro-F1 by language (test set)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    langs = sorted(test_df["language_tag"].unique())
    x = np.arange(len(langs))
    for i, (k, c) in enumerate(zip(keys, colors)):
        vals = [results[k]["test_by_language"].get(l, {}).get("f1_macro", 0) for l in langs]
        ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=DISPLAY[k], color=c)
    ax.set_xticks(x)
    ax.set_xticklabels({"en": "English", "pcm": "Pidgin / code-switched"}.get(l, l)
                       for l in langs)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Macro F1")
    ax.set_title("Performance by language - test set")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "f1_by_language_test.png"), dpi=150)
    plt.close(fig)
    return figdir


def run(datadir="data", outdir="models"):
    os.makedirs(outdir, exist_ok=True)
    train, val, test = (load_split(datadir, s) for s in ("train", "val", "test"))
    print(f"train={len(train)} val={len(val)} test={len(test)}")

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000,
                                 min_df=2 if len(train) >= 200 else 1, sublinear_tf=True)
    X_train = vectorizer.fit_transform(train["clean_text"])   # fit on train ONLY
    X_val = vectorizer.transform(val["clean_text"])
    X_test = vectorizer.transform(test["clean_text"])
    joblib.dump(vectorizer, os.path.join(outdir, "vectorizer.pkl"))
    print(f"TF-IDF vocabulary: {len(vectorizer.vocabulary_)} features")

    counts = train["label"].value_counts()
    minority = int(counts.min())
    n_splits = max(2, min(5, minority))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    candidates, use_smote, k = build_candidates(minority)

    smote_info = {"applied": use_smote, "k_neighbors": k,
                  "train_counts_before": {k_: int(v) for k_, v in counts.items()}}
    if use_smote:
        _, y_res = SMOTE(k_neighbors=k, random_state=RANDOM_STATE).fit_resample(
            X_train, train["label"])
        smote_info["train_counts_after"] = {k_: int(v) for k_, v in
                                            pd.Series(y_res).value_counts().items()}
        print(f"SMOTE on training split only: {smote_info['train_counts_before']} -> "
              f"{smote_info['train_counts_after']}")
    else:
        print("SMOTE skipped: minority class too small (<3).")

    results, disagg, test_scores = {}, [], {}
    for key, (pipe, grid) in candidates.items():
        print(f"\n=== {DISPLAY[key]}: {n_splits}-fold CV grid search ===")
        search = GridSearchCV(pipe, grid, scoring="f1_macro", cv=cv, n_jobs=-1)
        search.fit(X_train, train["label"])
        model = search.best_estimator_.named_steps["clf"]  # trained on SMOTE-balanced train
        print(f"Best params: {search.best_params_}  CV macro-F1={search.best_score_:.4f}")

        res = {"best_params": {k_.replace("clf__", ""): v for k_, v in search.best_params_.items()},
               "cv_f1_macro": float(search.best_score_)}
        for split, df, X in (("train", train, X_train), ("val", val, X_val), ("test", test, X_test)):
            pred = model.predict(X)
            pf = p_fake(model, X)
            res[split] = compute_metrics(df["label"].values, pred, pf)
            if split in ("val", "test"):
                disagg.extend(disaggregate(key, split, df, pred, pf))
            if split == "test":
                test_scores[key] = pf
                res["test_by_language"] = {
                    lang: compute_metrics(df.loc[idx, "label"].values, pred[np.asarray(list(idx))],
                                          pf[np.asarray(list(idx))])
                    for lang, idx in df.groupby("language_tag").groups.items()}
        t = res["test"]
        auc = f"{t['roc_auc']:.3f}" if t["roc_auc"] is not None else "n/a"
        print(f"  train acc={res['train']['accuracy']:.3f} | test acc={t['accuracy']:.3f} "
              f"P={t['precision_fake']:.3f} R={t['recall_fake']:.3f} F1={t['f1_fake']:.3f} "
              f"macroF1={t['f1_macro']:.3f} AUC={auc} Brier={t['brier_score']:.3f}")
        for lang, m in res["test_by_language"].items():
            print(f"    [{lang}] n={m['n']} macroF1={m['f1_macro']:.3f} acc={m['accuracy']:.3f}")
        results[key] = res
        joblib.dump(model, os.path.join(outdir, f"{key}.pkl"), compress=3)  # keeps files under GitHub upload limit

    best = max(results, key=lambda k_: results[k_]["val"]["f1_macro"])
    print(f"\nBest model by validation macro-F1: {DISPLAY[best]} "
          f"(test macro-F1 {results[best]['test']['f1_macro']:.4f}, unbiased estimate)")
    with open(os.path.join(outdir, "best_model_name.txt"), "w") as f:
        f.write(best)

    report = {"best_model_by_val_f1_macro": best, "models": results}
    with open(os.path.join(outdir, "metrics_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)

    summary = []
    for k_, r in results.items():
        for split in ("train", "val", "test"):
            m = r[split]
            summary.append({"model": DISPLAY[k_], "split": split, "n": m["n"],
                            "accuracy": m["accuracy"], "precision_fake": m["precision_fake"],
                            "recall_fake": m["recall_fake"], "f1_fake": m["f1_fake"],
                            "f1_macro": m["f1_macro"], "roc_auc": m["roc_auc"],
                            "brier_score": m["brier_score"],
                            "false_positive_rate": m["false_positive_rate"]})
    pd.DataFrame(summary).round(4).to_csv(os.path.join(outdir, "metrics_summary.csv"), index=False)
    pd.DataFrame(disagg).round(4).to_csv(os.path.join(outdir, "disaggregated_report.csv"),
                                         index=False)

    prep_log = {}
    p = os.path.join(datadir, "preprocess_log.json")
    if os.path.exists(p):
        with open(p) as f:
            prep_log = json.load(f)
    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "cv_folds": n_splits, "smote": smote_info,
        "is_sample_only": bool(prep_log.get("is_sample_only", False)),
        "input_files": prep_log.get("input_files", []),
        "python": platform.python_version(), "scikit_learn": sklearn.__version__,
        "decision_threshold_p_real": 0.5,
    }
    with open(os.path.join(outdir, "training_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    figdir = make_figures(outdir, results, test, test_scores)
    print(f"\nSaved models, reports and figures to {outdir}/ (figures in {figdir}/)")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", default="data")
    ap.add_argument("--outdir", default="models")
    a = ap.parse_args()
    run(a.datadir, a.outdir)
