import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_xg_lab():
    m = json.loads((ROOT / "outputs" / "metrics.json").read_text(encoding="utf-8"))
    assert m["qa_summary"]["total_matches"] >= 100
    assert m["qa_summary"]["non_penalty_shots"] >= 2500

    conn = sqlite3.connect(ROOT / "data" / "statsbomb_xg.db")
    cnt = conn.execute("SELECT COUNT(*) FROM vw_shots_modeling").fetchone()[0]
    conn.close()
    assert cnt == m["qa_summary"]["non_penalty_shots"]

    best = m["selected_custom_model"]
    assert m["models"][best]["test_roc_auc"] >= 0.74
    assert m["models"][best]["test_brier"] <= 0.095
    assert m["freeze_frame_logloss_gain"] > 0.0

    for fig in [
        "01_xg_pitch_heatmap.png",
        "02_xg_model_calibration.png",
        "03_xg_shap_drivers.png",
        "04_finishing_leaderboard.png",
    ]:
        p = ROOT / "outputs" / "figures" / fig
        assert p.exists() and p.stat().st_size > 20_000, f"Missing figure {fig}"

    print("ALL STATSBOMB xG LAB TESTS PASSED!")


if __name__ == "__main__":
    test_xg_lab()
