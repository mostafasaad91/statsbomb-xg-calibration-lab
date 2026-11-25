-- ============================================================================
-- StatsBomb xG Calibration Lab — SQLite Shot QA & Feature Views
-- Data: Official StatsBomb Open Data (WC 2022, Euro 2024, Bundesliga 2023/24)
-- Author: Mostafa Saad
-- ============================================================================

DROP VIEW IF EXISTS vw_shots_qa;
CREATE VIEW vw_shots_qa AS
SELECT
    COUNT(*) AS total_shots,
    COUNT(DISTINCT match_id) AS total_matches,
    SUM(CASE WHEN shot_type = 'Penalty' THEN 1 ELSE 0 END) AS penalty_shots,
    SUM(CASE WHEN shot_type != 'Penalty' THEN 1 ELSE 0 END) AS non_penalty_shots,
    SUM(CASE WHEN shot_type != 'Penalty' AND is_goal = 1 THEN 1 ELSE 0 END) AS non_penalty_goals,
    ROUND(100.0 * SUM(CASE WHEN shot_type != 'Penalty' AND is_goal = 1 THEN 1 ELSE 0 END) /
          NULLIF(SUM(CASE WHEN shot_type != 'Penalty' THEN 1 ELSE 0 END), 0), 2) AS np_conversion_pct,
    ROUND(100.0 * AVG(has_freeze_frame), 1) AS freeze_frame_coverage_pct
FROM raw_statsbomb_shots;

DROP VIEW IF EXISTS vw_shots_modeling;
CREATE VIEW vw_shots_modeling AS
SELECT
    match_id,
    competition,
    match_date,
    team,
    player,
    minute,
    period,
    x,
    y,
    distance,
    ROUND(distance * distance, 3) AS distance_sq,
    goal_angle_rad,
    goal_angle_deg,
    shot_type,
    body_part,
    technique,
    play_pattern,
    first_time,
    one_on_one,
    under_pressure,
    defenders_in_cone,
    defenders_within_2_5yd,
    gk_dist_to_goal,
    gk_dist_to_shot,
    gk_in_cone,
    has_freeze_frame,
    CASE WHEN body_part = 'Head' THEN 1 ELSE 0 END AS is_header,
    CASE WHEN x >= 102.0 AND y BETWEEN 18.0 AND 62.0 THEN 1 ELSE 0 END AS is_inside_box,
    CASE WHEN x >= 114.0 AND y BETWEEN 30.0 AND 50.0 THEN 1 ELSE 0 END AS is_six_yard_box,
    CASE WHEN ABS(y - 40.0) <= 10.0 THEN 1 ELSE 0 END AS is_central_lane,
    CASE WHEN gk_dist_to_goal > 6.0 THEN 1 ELSE 0 END AS gk_off_line,
    statsbomb_xg,
    is_goal
FROM raw_statsbomb_shots
WHERE shot_type != 'Penalty';
