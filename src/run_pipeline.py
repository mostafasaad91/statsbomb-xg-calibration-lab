"""
StatsBomb 360 xG Calibration — Custom Event + Freeze-Frame xG ML Benchmark
Author: Mostafa Saad
Data: Official StatsBomb Open Data (FIFA World Cup 2022, UEFA Euro 2024, Bundesliga 2023/24)
"""

import json
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "statsbomb_shots_real.csv"
DB_PATH = ROOT / "data" / "statsbomb_xg.db"
SQL_PATH = ROOT / "sql" / "01_xg_views.sql"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

BG = "#000000"
CARD = "#0A0C12"
CYAN = "#00E5FF"
VIOLET = "#A78BFA"
EMERALD = "#10B981"
AMBER = "#F59E0B"
ROSE = "#F43F5E"
TEXT = "#FFFFFF"
MUTED = "#94A3B8"
GRID = "#1E2433"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": CARD,
    "axes.edgecolor": "#262D3F",
    "axes.labelcolor": MUTED,
    "text.color": TEXT,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": GRID,
    "font.family": "DejaVu Sans",
})


def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if np.sum(mask) > 0:
            ece += (np.sum(mask) / n) * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def bootstrap_ci(y_true, y_prob, fn, n_boot=400, seed=42):
    rng = np.random.default_rng(seed)
    yt, yp = np.asarray(y_true), np.asarray(y_prob)
    vals = []
    n = len(yt)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(yt[idx])) < 2:
            continue
        vals.append(fn(yt[idx], yp[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    print("[1/5] Loading real StatsBomb shots into SQLite & building non-penalty xG views...")
    df_raw = pd.read_csv(RAW_CSV)
    conn = sqlite3.connect(DB_PATH)
    df_raw.to_sql("raw_statsbomb_shots", conn, if_exists="replace", index=False)
    conn.executescript(SQL_PATH.read_text(encoding="utf-8"))
    qa = pd.read_sql_query("SELECT * FROM vw_shots_qa", conn).iloc[0].to_dict()
    df = pd.read_sql_query("SELECT * FROM vw_shots_modeling", conn)
    conn.close()
    print("QA Summary:", qa)

    # Match-Grouped Split (60% Train / 20% Val / 20% Test) -> ZERO match leakage
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    trainval_idx, test_idx = next(gss1.split(df, df["is_goal"], groups=df["match_id"]))
    df_tv = df.iloc[trainval_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, val_idx = next(gss2.split(df_tv, df_tv["is_goal"], groups=df_tv["match_id"]))
    df_train = df_tv.iloc[train_idx].reset_index(drop=True)
    df_val = df_tv.iloc[val_idx].reset_index(drop=True)

    print(f"Match-Grouped Splits -> Train: {len(df_train):,} ({df_train['match_id'].nunique()} matches) | "
          f"Val: {len(df_val):,} ({df_val['match_id'].nunique()} matches) | "
          f"Test: {len(df_test):,} ({df_test['match_id'].nunique()} matches)")

    geom_cols = ["distance", "distance_sq", "goal_angle_rad", "is_inside_box", "is_six_yard_box", "is_central_lane"]
    ff_num_cols = geom_cols + [
        "is_header", "first_time", "one_on_one", "under_pressure",
        "defenders_in_cone", "defenders_within_2_5yd",
        "gk_dist_to_goal", "gk_dist_to_shot", "gk_in_cone", "gk_off_line",
    ]
    cat_cols = ["shot_type", "body_part", "technique", "play_pattern"]

    prep_geom = ColumnTransformer([("num", StandardScaler(), geom_cols)])
    prep_full = ColumnTransformer([
        ("num", StandardScaler(), ff_num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])

    y_train = df_train["is_goal"].values
    y_val = df_val["is_goal"].values
    y_test = df_test["is_goal"].values

    model_specs = {
        "Naive Conversion Rate": (prep_geom, DummyClassifier(strategy="prior")),
        "Geometry-Only Logistic": (prep_geom, LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
        "Full + Freeze-Frame Logistic": (prep_full, LogisticRegression(C=0.4, max_iter=1000, random_state=42)),
        "LightGBM xG (+Freeze-Frame)": (prep_full, LGBMClassifier(
            n_estimators=180, learning_rate=0.035, num_leaves=12, max_depth=4,
            min_child_samples=35, subsample=0.85, colsample_bytree=0.8,
            reg_alpha=1.5, reg_lambda=3.0, random_state=42, verbose=-1
        )),
        "XGBoost xG (+Freeze-Frame)": (prep_full, XGBClassifier(
            n_estimators=180, learning_rate=0.035, max_depth=3, min_child_weight=12,
            subsample=0.85, colsample_bytree=0.8, reg_alpha=1.5, reg_lambda=3.0,
            eval_metric="logloss", random_state=42, n_jobs=-1
        )),
    }

    print("[2/5] Training & benchmarking custom xG models vs official StatsBomb xG...")
    results = {}
    val_probs, test_probs, pipes = {}, {}, {}

    for name, (prep, clf) in model_specs.items():
        pipe = Pipeline([("prep", prep), ("clf", clf)])
        cols = geom_cols if prep is prep_geom else (ff_num_cols + cat_cols)
        pipe.fit(df_train[cols], y_train)
        pv = np.clip(pipe.predict_proba(df_val[cols])[:, 1], 1e-4, 1 - 1e-4)
        pt = np.clip(pipe.predict_proba(df_test[cols])[:, 1], 1e-4, 1 - 1e-4)
        val_probs[name], test_probs[name], pipes[name] = pv, pt, (pipe, cols)

        ll_v = log_loss(y_val, pv)
        ll_t = log_loss(y_test, pt)
        br_t = brier_score_loss(y_test, pt)
        auc_t = roc_auc_score(y_test, pt)
        ece_t = expected_calibration_error(y_test, pt)
        ci_l, ci_h = bootstrap_ci(y_test, pt, roc_auc_score) if name != "Naive Conversion Rate" else (0.5, 0.5)

        results[name] = {
            "val_log_loss": round(float(ll_v), 4),
            "test_log_loss": round(float(ll_t), 4),
            "test_brier": round(float(br_t), 4),
            "test_roc_auc": round(float(auc_t), 4),
            "test_roc_auc_ci95": [round(ci_l, 4), round(ci_h, 4)],
            "test_ece": round(float(ece_t), 4),
            "sum_pred_xg": round(float(pt.sum()), 2),
            "actual_goals": int(y_test.sum()),
        }
        print(f"  {name:30s} | Val LL={ll_v:.4f} | Test LL={ll_t:.4f} | Brier={br_t:.4f} | AUC={auc_t:.4f}")

    sb_pt = np.clip(df_test["statsbomb_xg"].values, 1e-4, 1 - 1e-4)
    sb_ci_l, sb_ci_h = bootstrap_ci(y_test, sb_pt, roc_auc_score)
    results["StatsBomb Official xG (Ref)"] = {
        "val_log_loss": round(float(log_loss(y_val, np.clip(df_val["statsbomb_xg"].values, 1e-4, 1 - 1e-4))), 4),
        "test_log_loss": round(float(log_loss(y_test, sb_pt)), 4),
        "test_brier": round(float(brier_score_loss(y_test, sb_pt)), 4),
        "test_roc_auc": round(float(roc_auc_score(y_test, sb_pt)), 4),
        "test_roc_auc_ci95": [round(sb_ci_l, 4), round(sb_ci_h, 4)],
        "test_ece": round(float(expected_calibration_error(y_test, sb_pt)), 4),
        "sum_pred_xg": round(float(sb_pt.sum()), 2),
        "actual_goals": int(y_test.sum()),
    }
    print(f"  {'StatsBomb Official xG (Ref)':30s} | Test LL={results['StatsBomb Official xG (Ref)']['test_log_loss']:.4f} | "
          f"Brier={results['StatsBomb Official xG (Ref)']['test_brier']:.4f} | AUC={results['StatsBomb Official xG (Ref)']['test_roc_auc']:.4f}")

    custom_models = [k for k in model_specs if k != "Naive Conversion Rate"]
    best_name = min(custom_models, key=lambda k: results[k]["val_log_loss"])
    print(f"Selected Best Custom xG Model (by Val Log-Loss): {best_name}")

    df_test["custom_xg"] = test_probs[best_name]
    slice_audit = []
    for col in ["competition", "is_header", "is_inside_box"]:
        for val, grp in df_test.groupby(col):
            if len(grp) < 20 or grp["is_goal"].nunique() < 2:
                continue
            slice_audit.append({
                "slice": f"{col}={val}",
                "shots": int(len(grp)),
                "actual_goals": int(grp["is_goal"].sum()),
                "sum_custom_xg": round(float(grp["custom_xg"].sum()), 2),
                "brier": round(float(brier_score_loss(grp["is_goal"], grp["custom_xg"])), 4),
                "roc_auc": round(float(roc_auc_score(grp["is_goal"], grp["custom_xg"])), 4),
            })

    best_pipe, best_cols = pipes[best_name]
    df["custom_xg"] = np.clip(best_pipe.predict_proba(df[best_cols])[:, 1], 1e-4, 1 - 1e-4)
    leaderboard = (
        df.groupby(["player", "team"], as_index=False)
        .agg(shots=("is_goal", "count"), goals=("is_goal", "sum"), custom_xg=("custom_xg", "sum"), statsbomb_xg=("statsbomb_xg", "sum"))
    )
    leaderboard = leaderboard[leaderboard["shots"] >= 10].copy()
    leaderboard["goals_minus_xg"] = (leaderboard["goals"] - leaderboard["custom_xg"]).round(2)
    leaderboard["custom_xg"] = leaderboard["custom_xg"].round(2)
    leaderboard["statsbomb_xg"] = leaderboard["statsbomb_xg"].round(2)
    leaderboard = leaderboard.sort_values("goals_minus_xg", ascending=False)
    leaderboard.to_csv(OUT_DIR / "finishing_leaderboard.csv", index=False)

    print("[3/5] Computing SHAP feature importance...")
    prep_fit = best_pipe.named_steps["prep"]
    clf_fit = best_pipe.named_steps["clf"]
    clean_names = [f.replace("num__", "").replace("cat__", "") for f in prep_fit.get_feature_names_out()]
    X_test_t = prep_fit.transform(df_test[best_cols])
    X_train_t = prep_fit.transform(df_train[best_cols])
    if isinstance(clf_fit, LogisticRegression):
        expl = shap.LinearExplainer(clf_fit, X_train_t)
        sv = expl.shap_values(X_test_t)
    else:
        expl = shap.TreeExplainer(clf_fit)
        raw_sv = expl.shap_values(X_test_t)
        sv = raw_sv[1] if isinstance(raw_sv, list) else (raw_sv[:, :, 1] if raw_sv.ndim == 3 else raw_sv)
    mean_abs = np.abs(sv).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:12]

    metrics_payload = {
        "dataset": "StatsBomb Open Data (FIFA World Cup 2022 + UEFA Euro 2024 + Bundesliga 2023/24)",
        "qa_summary": qa,
        "splits": {
            "train_shots": len(df_train), "val_shots": len(df_val), "test_shots": len(df_test),
            "train_matches": int(df_train["match_id"].nunique()),
            "val_matches": int(df_val["match_id"].nunique()),
            "test_matches": int(df_test["match_id"].nunique()),
        },
        "models": results,
        "selected_custom_model": best_name,
        "freeze_frame_logloss_gain": round(
            results["Geometry-Only Logistic"]["test_log_loss"] - results[best_name]["test_log_loss"], 4
        ),
        "slice_audit": slice_audit,
        "top_shap_features": [{"feature": clean_names[i], "mean_abs_shap": round(float(mean_abs[i]), 4)} for i in top_idx],
        "top5_overperformers": leaderboard.head(5).to_dict(orient="records"),
        "top5_underperformers": leaderboard.tail(5).sort_values("goals_minus_xg").to_dict(orient="records"),
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    print("[4/5] Rendering 4 AMOLED StatsBomb xG visualizations...")

    # Fig 1: Pitch Shot Locations + Distance Decay
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.4), dpi=160)
    fig.patch.set_facecolor(BG)
    fig.suptitle(f"StatsBomb Real Non-Penalty Shots ({len(df):,} Shots · {int(qa['total_matches'])} Matches)", color=TEXT, fontsize=14.5, fontweight="bold", y=0.97)

    ax1.set_facecolor(BG)
    ax1.plot([0, 80, 80, 0, 0], [78, 78, 120, 120, 78], color="#2D3548", lw=1.3)
    ax1.plot([18, 18, 62, 62], [120, 102, 102, 120], color="#2D3548", lw=1.2)
    ax1.plot([30, 30, 50, 50], [120, 114, 114, 120], color="#2D3548", lw=1.2)
    ax1.plot([36, 36, 44, 44], [120, 121.5, 121.5, 120], color=CYAN, lw=2.0)

    non_goals = df[df["is_goal"] == 0]
    goals = df[df["is_goal"] == 1]
    ax1.scatter(non_goals["y"], non_goals["x"], s=10, color=CYAN, alpha=0.14, edgecolors="none", label=f"No Goal ({len(non_goals):,})")
    ax1.scatter(goals["y"], goals["x"], s=24, color=EMERALD, alpha=0.72, edgecolors="#FFFFFF", lw=0.35, label=f"Goal ({len(goals):,})")
    ax1.set_xlim(-2, 82); ax1.set_ylim(76, 123); ax1.axis("off")
    ax1.set_title("Shot & Goal Locations (Attacking Third)", color=EMERALD, fontweight="bold")
    ax1.legend(loc="lower left", frameon=False, fontsize=8.5)

    df["dist_bin"] = pd.cut(df["distance"], bins=[0, 6, 10, 14, 18, 22, 28, 40], labels=["0-6", "6-10", "10-14", "14-18", "18-22", "22-28", "28+"])
    conv_d = df.groupby("dist_bin", observed=False)["is_goal"].mean() * 100
    pred_d = df.groupby("dist_bin", observed=False)["custom_xg"].mean() * 100
    ax2.plot(conv_d.index.astype(str), conv_d.values, marker="o", color=EMERALD, lw=2.5, label="Observed Conversion %")
    ax2.plot(pred_d.index.astype(str), pred_d.values, marker="s", color=CYAN, lw=2.1, ls="--", label=f"Predicted {best_name} %")
    ax2.set_xlabel("Distance to Goal (Yards)"); ax2.set_ylabel("Goal Probability (%)")
    ax2.set_title("Distance Decay: Observed vs Custom xG", color=CYAN, fontweight="bold")
    ax2.grid(ls=":", alpha=0.5); ax2.legend(frameon=False, fontsize=8.6)
    plt.tight_layout(rect=[0, 0.02, 1, 0.91])
    fig.savefig(FIG_DIR / "01_xg_pitch_heatmap.png", facecolor=BG)
    plt.close(fig)

    # Fig 2: Calibration & Ablation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.2), dpi=160)
    fig.patch.set_facecolor(BG)
    fig.suptitle("xG Calibration Benchmark — Custom Freeze-Frame ML vs Official StatsBomb xG", color=TEXT, fontsize=14, fontweight="bold", y=0.97)

    comp_keys = ["Geometry-Only Logistic", "Full + Freeze-Frame Logistic", "LightGBM xG (+Freeze-Frame)", "XGBoost xG (+Freeze-Frame)", "StatsBomb Official xG (Ref)"]
    ll_vals = [results[k]["test_log_loss"] for k in comp_keys]
    short_lbl = ["Geom-Only\nLogistic", "Full + FF\nLogistic", "LightGBM\n(+FF)", "XGBoost\n(+FF)", "StatsBomb\nOfficial"]
    bars = ax1.bar(short_lbl, ll_vals, color=[MUTED, CYAN, VIOLET, EMERALD, AMBER], width=0.58)
    for b, v in zip(bars, ll_vals):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.4f}", ha="center", color=TEXT, fontsize=8.8, family="monospace", fontweight="bold")
    ax1.set_ylim(min(ll_vals) * 0.85, max(ll_vals) * 1.12)
    ax1.set_ylabel("Test Log-Loss (Lower = Better)")
    ax1.set_title("Ablation: Value of Freeze-Frame Defenders & GK", color=CYAN, fontweight="bold")
    ax1.grid(axis="y", ls=":", alpha=0.5)

    for lbl_k, col in [("Geometry-Only Logistic", MUTED), (best_name, EMERALD), ("StatsBomb Official xG (Ref)", AMBER)]:
        probs = sb_pt if lbl_k.startswith("StatsBomb") else test_probs[lbl_k]
        pt_c, pp_c = calibration_curve(y_test, probs, n_bins=8, strategy="quantile")
        ax2.plot(pp_c, pt_c, marker="o", color=col, lw=2.1, markersize=5, label=f"{lbl_k} (Brier {results[lbl_k]['test_brier']:.4f})")
    ax2.plot([0, 0.55], [0, 0.55], color="#FFFFFF", ls="--", lw=1, alpha=0.6)
    ax2.set_xlabel("Mean Predicted xG"); ax2.set_ylabel("Observed Goal Rate")
    ax2.set_title("Probability Calibration Curve (Held-Out Matches)", color=EMERALD, fontweight="bold")
    ax2.grid(ls=":", alpha=0.5); ax2.legend(frameon=False, fontsize=8)
    plt.tight_layout(rect=[0, 0.02, 1, 0.91])
    fig.savefig(FIG_DIR / "02_xg_model_calibration.png", facecolor=BG)
    plt.close(fig)

    # Fig 3: SHAP Drivers
    fig, ax = plt.subplots(figsize=(10.5, 5.6), dpi=160)
    fig.patch.set_facecolor(BG)
    tf = [clean_names[i] for i in top_idx][::-1]
    tv = [mean_abs[i] for i in top_idx][::-1]
    cols_s = []
    for i in top_idx[::-1]:
        corr = np.corrcoef(X_test_t[:, i], sv[:, i])[0, 1]
        cols_s.append(EMERALD if (not np.isnan(corr) and corr > 0) else ROSE)
    ax.barh(tf, tv, color=cols_s, height=0.62, alpha=0.9)
    for idx_b, v in enumerate(tv):
        ax.text(v + 0.008, idx_b, f"{v:.3f}", va="center", color=TEXT, fontsize=8.6, family="monospace")
    ax.set_xlabel("Mean |SHAP Value| (Impact on Goal Log-Odds)")
    ax.set_title(f"Top 12 xG Probability Drivers ({best_name} · SHAP Explainability)", color=TEXT, fontsize=13.5, fontweight="bold", pad=12)
    ax.text(0.98, 0.06, "■ Green = Increases Goal Prob   ■ Red = Reduces Goal Prob",
            transform=ax.transAxes, ha="right", color=MUTED, fontsize=8.8, family="monospace")
    ax.grid(axis="x", ls=":", alpha=0.5)
    ax.set_xlim(0, max(tv) * 1.18)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "03_xg_shap_drivers.png", facecolor=BG)
    plt.close(fig)

    # Fig 4: Over & Under-Performing Finishers
    fig, ax = plt.subplots(figsize=(11.2, 5.6), dpi=160)
    fig.patch.set_facecolor(BG)
    comb = pd.concat([leaderboard.head(6), leaderboard.tail(6)]).sort_values("goals_minus_xg")
    lbls = [f"{r['player']} ({r['team']})" for _, r in comb.iterrows()]
    vals = comb["goals_minus_xg"].values
    ax.barh(lbls, vals, color=[EMERALD if v >= 0 else ROSE for v in vals], height=0.62)
    ax.axvline(0, color="#FFFFFF", lw=1.0, alpha=0.6)
    for idx_b, (_, r) in enumerate(comb.iterrows()):
        v = r["goals_minus_xg"]
        offset = 0.12 if v >= 0 else -0.12
        ha = "left" if v >= 0 else "right"
        ax.text(v + offset, idx_b, f"{v:+.2f} ({int(r['goals'])}G / {r['custom_xg']:.1f}xG)", va="center", ha=ha, color=TEXT, fontsize=8.2, family="monospace")
    ax.set_xlabel("Non-Penalty Goals Minus Custom xG (Min. 10 Shots)")
    ax.set_title("Clinical Finishers vs Under-Performers (WC 2022 · Euro 2024 · BL 23/24)", color=TEXT, fontsize=11.6, fontweight="bold", pad=12)
    ax.set_xlim(min(vals) - 2.3, max(vals) + 2.6)
    ax.grid(axis="x", ls=":", alpha=0.5)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "04_finishing_leaderboard.png", facecolor=BG)
    plt.close(fig)
    print("[5/5] StatsBomb 360 xG Calibration complete!")


if __name__ == "__main__":
    main()
