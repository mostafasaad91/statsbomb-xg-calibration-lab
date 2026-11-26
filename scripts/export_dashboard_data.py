"""Compile all 3,642 real non-penalty StatsBomb shots, finishing leaderboard & metrics.json into compact docs/xg_dashboard_data.json."""

import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "statsbomb_shots_real.csv"
OUT_DIR = ROOT / "outputs"
DOCS_DIR = ROOT / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    metrics = json.loads((OUT_DIR / "metrics.json").read_text(encoding="utf-8"))
    lb = pd.read_csv(OUT_DIR / "finishing_leaderboard.csv")
    df = pd.read_csv(RAW_CSV)
    # Match modeling pool (non-penalty shots = 3,642)
    df = df[df["shot_type"] != "Penalty"].copy().reset_index(drop=True)

    # Build lookup tables to keep JSON super compact & fast
    comps = sorted(df["competition"].dropna().unique().tolist())
    teams = sorted(df["team"].dropna().unique().tolist())
    players = sorted(df["player"].dropna().unique().tolist())
    bodies = sorted(df["body_part"].dropna().unique().tolist())
    patterns = sorted(df["play_pattern"].dropna().unique().tolist())
    techs = sorted(df["technique"].dropna().unique().tolist())

    c_map = {v: i for i, v in enumerate(comps)}
    t_map = {v: i for i, v in enumerate(teams)}
    p_map = {v: i for i, v in enumerate(players)}
    b_map = {v: i for i, v in enumerate(bodies)}
    pat_map = {v: i for i, v in enumerate(patterns)}
    tech_map = {v: i for i, v in enumerate(techs)}

    # Compact shot tuples:
    # [comp_i, team_i, player_i, minute, x, y, dist, angle_deg, body_i, pat_i, tech_i,
    #  def_cone, def_25, gk_dist_shot, gk_in_cone, one_on_one, under_press, sb_xg, is_goal]
    shots = []
    for r in df.itertuples(index=False):
        shots.append([
            c_map[r.competition],
            t_map[r.team],
            p_map[r.player],
            int(r.minute),
            round(float(r.x), 1),
            round(float(r.y), 1),
            round(float(r.distance), 1),
            round(float(r.goal_angle_deg), 1),
            b_map[r.body_part],
            pat_map[r.play_pattern],
            tech_map[r.technique],
            int(r.defenders_in_cone),
            int(r.defenders_within_2_5yd),
            round(float(r.gk_dist_to_shot), 1) if pd.notna(r.gk_dist_to_shot) else -1,
            int(r.gk_in_cone) if pd.notna(r.gk_in_cone) else 0,
            int(r.one_on_one),
            int(r.under_pressure),
            round(float(r.statsbomb_xg), 3),
            int(r.is_goal),
        ])

    payload = {
        "metrics": metrics,
        "lookups": {
            "competitions": comps,
            "teams": teams,
            "players": players,
            "body_parts": bodies,
            "play_patterns": patterns,
            "techniques": techs,
        },
        "shots": shots,
        "leaderboard": lb.to_dict(orient="records"),
    }

    out_path = DOCS_DIR / "xg_dashboard_data.json"
    out_path.write_text(json.dumps(payload, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    print(f"Exported StatsBomb xG dashboard JSON ({len(shots)} real shots) -> {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
