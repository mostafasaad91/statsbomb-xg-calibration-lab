# Project Story — StatsBomb Expected Goals (xG) Calibration & Freeze-Frame Lab
## قصة بناء المشروع: ليه عملناه، إزاي اتبنى، وإيه الأسئلة والتحديات اللي واجهتنا؟

---

## 1. Why Did We Build This Project? (ليه عملنا المشروع ده؟)

Most football analytics tutorials build a naive xG model using only `(x, y)` shot coordinates, report an inflated AUC on a random shot split, and stop there. Real club analytics departments care about three deeper questions:
1. **How much real probability accuracy do Freeze-Frame defender & goalkeeper coordinates actually add** over pure shot geometry?
2. **Is the model properly calibrated (Log-Loss, Brier Score, ECE)** so that summing xG over 30 matches actually matches real goals scored?
3. **How close can an open, reproducible ML model get to StatsBomb's proprietary commercial xG benchmark** on unseen matches?

**بالعربي:**
أغلب مشاريع الـ xG على الإنترنت بتاخد مكان التسديدة `(x, y)` بس وتعمل Logistic Regression بسيط وتقول "خلصنا". بس محلل البيانات المحترف في نادي كرة قدم بيسأل 3 أسئلة أعمق بكتير:
1. **هل بيانات الـ Freeze-Frame (أماكن المدافعين وحارس المرمى لحظة التسديد) بتفرق فعلاً بالأرقام؟**
2. **هل احتمالات الموديل متعايرة صح (Calibrated)؟** يعني لو مجموع الـ xG في 30 ماتش طلع 71، هل اتسجل فعلاً حوالي 73 هدف؟
3. **إلى أي مدى موديلنا المفتوح يقدر يقرب من موديل StatsBomb التجاري الرسمي؟**

---

## 2. How Was It Built? (إزاي اتبنى خطوة بخطوة؟)

### Stage 1 — Harvesting Real Official StatsBomb Open Data (149 Matches)
- Pulled live JSON event feeds directly from the official `statsbomb/open-data` repository across three elite competitions:
  - **FIFA World Cup 2022** (64 matches)
  - **UEFA Euro 2024** (51 matches)
  - **Bundesliga 2023/24 — Bayer Leverkusen's Invincible Season** (34 matches)
- Extracted **3,685 real shots** with exact `(x, y)` coordinates, body part, shot technique, play pattern, `under_pressure`, `first_time`, `one_on_one`, and full **Freeze-Frame geometry** (`98.9%` coverage):
  - `defenders_in_cone`: number of opposition outfield players inside the triangle formed by the shooter and both goalposts.
  - `defenders_within_2_5yd`: immediate shot-blocking pressure within 2.5 yards.
  - `gk_dist_to_shot` & `gk_dist_to_goal`: whether the goalkeeper rushed out to cut the angle or stayed deep on the line.

### Stage 2 — SQLite Spatial Shot Mart (`sql/01_xg_views.sql`)
- Loaded raw shots into SQLite and built analytical views separating **43 penalties** (which obey a constant ~0.76–0.78 conversion law, not open-play spatial physics) from **3,642 non-penalty shots** (**352 goals**, **9.67%** conversion).
- Engineered Euclidean distance to goal center `(120, 40)`, visible goal opening angle (`atan2` between posts at `y = 36` and `y = 44`), `is_inside_box`, `is_central_lane`, and `is_header` directly in SQL.

### Stage 3 — Match-Grouped Splitting & Freeze-Frame Ablation (`src/run_pipeline.py`)
- Used **`GroupShuffleSplit` by `match_id`**: **89 train matches (2,183 shots) / 30 val matches (701 shots) / 30 test matches (758 shots)** — guaranteeing zero matches leak across splits.
- Ran a controlled **ablation experiment**:
  - `Geometry-Only Logistic`: Test Log-Loss **0.2767**, Brier **0.0784**, AUC **0.7540**
  - `Full + Freeze-Frame Logistic`: Test Log-Loss **0.2531**, Brier **0.0724**, AUC **0.8163**
  - **Proven Freeze-Frame Gain:** **-0.0236 Log-Loss** and **+0.0623 ROC-AUC**!

### Stage 4 — SHAP Explainability, Slice Audit & Finishing Leaderboard
- SHAP confirmed the football physics: **`goal_angle_rad` (0.416)**, **`is_inside_box` (0.344)**, **`defenders_within_2_5yd` (0.316)**, **`distance` (0.246)**, and **`defenders_in_cone` (0.194)** dominate goal probability.
- Audited calibration across competitions, headers vs feet (`AUC 0.8048` vs `0.8200`), and inside vs outside box.

---

## 3. Key Questions & Challenges We Faced (إيه الأسئلة والتحديات اللي واجهتنا؟)

### Challenge 1 — Why Exclude Penalties From the Open-Play xG Model?
- **The Question:** Should penalties stay inside the training set?
- **Our Decision:** **No.** Penalties happen from a fixed point (`12 yards`, 0 defenders in cone, ~78% conversion) and inflate overall AUC artificially while distorting distance/defender coefficients. Excluding all **43 penalties** inside SQL keeps the open-play spatial physics pure.

### Challenge 2 — Why Match-Grouped Splits Instead of Random Shot Splits?
- **The Question:** Why not use a standard `train_test_split` on the 3,642 shots?
- **The Football Reality:** Shots from the same match share game state, goalkeeper form, weather, and tactical block. Random splitting leaks match context between train and test. Grouping strictly by `match_id` simulates true production deployment: predicting xG on **brand-new unseen matches**.

### Challenge 3 — The Surprising Winner: Why Did Logistic Regression Beat XGBoost & LightGBM?!
- **The Question:** Usually gradient boosted trees dominate tabular datasets. Why did **Full + Freeze-Frame Logistic** (`0.2531` Test Log-Loss, `0.8163` AUC) outperform both **XGBoost** (`0.2588` / `0.8074`) and **LightGBM** (`0.2599` / `0.8028`)?
- **The Scientific Answer:** With only **2,183 training shots** and **~210 goals**, goal outcomes are inherently high-variance Bernoulli events. Tree ensembles slightly overfit noisy splits in small-sample low-prevalence data, whereas regularized Logistic Regression on well-engineered domain physics (`goal_angle_rad`, `distance_sq`, `defenders_in_cone`) produces smoother, better-calibrated log-odds — coming within **0.0109 Log-Loss** of StatsBomb's commercial model (`0.2422`)!

### Challenge 4 — Visual QA Caught a Clipped Chart Title
- **The Issue:** During visual inspection of `04_finishing_leaderboard.png`, the right parenthesis of `Bundesliga 23/24)` touched the right border of the PNG.
- **The Fix:** Tightened the title string & font size (`11.6pt`) and re-ran the full pipeline + test suite to guarantee publication-grade alignment.

---

## 4. Real Football Takeaways From the Data (استنتاجات كروية حقيقية)
- **Bayer Leverkusen's Invincible 23/24 title run** wasn't just tactical dominance — **Alejandro Grimaldo (+4.29)**, **Florian Wirtz (+3.86)**, and **Josip Stanišić (+2.25)** delivered elite over-performance above xG, offsetting **Victor Boniface's (-2.36)** slight under-finishing (`10G` from `12.36 xG`).
- **Euro 2024 / WC 2022 Variance:** **Romelu Lukaku (-3.31, 0G from 3.31 xG)** and **Cristiano Ronaldo (-3.28, 0G from 3.28 xG)** highlight classic short-tournament finishing variance — exactly why recruitment teams evaluate **xG process**, not raw tournament goal tallies!
