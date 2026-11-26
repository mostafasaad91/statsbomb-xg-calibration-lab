/* StatsBomb 360 xG Calibration & 3,642-Shot Explorer Live Dashboard */
let DATA = null;
let CHARTS = {};
let lang = 'en';
let lbSort = { key: 'goals_minus_xg', dir: -1 };
let lastSelectedShot = null;

const I18N = {
  en: {
    t_exp: '01 · 3,642-Shot Pitch Map',
    t_abl: '02 · 360 Ablation & SHAP',
    t_lb: '03 · Finishing Leaderboard',
    k_shots: 'NON-PENALTY SHOTS',
    k_ff: 'FREEZE-FRAME GAIN',
    k_auc: 'HELD-OUT TEST AUC',
    k_brier: 'TEST BRIER / ECE',
    k_cal: 'HELD-OUT GOALS CAL.',
    k_fin: 'TOP FINISHER (G−xG)',
  },
  ar: {
    t_exp: '01 · خريطة 3,642 تسديدة حقيقية',
    t_abl: '02 · أثر 360 Freeze-Frame و SHAP',
    t_lb: '03 · قائمة أفضل الهدافين (G−xG)',
    k_shots: 'التسديدات (بدون ركلات جزاء)',
    k_ff: 'مكسب إحداثيات 360',
    k_auc: 'دقة الاختبار ROC-AUC',
    k_brier: 'معايرة Brier / ECE',
    k_cal: 'معايرة الأهداف (30 مباراة)',
    k_fin: 'أفضل إنهاء (G−xG)',
  },
};

// Shot tuple indices:
// 0:comp, 1:team, 2:player, 3:min, 4:x, 5:y, 6:dist, 7:angle, 8:body, 9:pat, 10:tech,
// 11:def_cone, 12:def_25, 13:gk_dist_shot, 14:gk_in_cone, 15:one_on_one, 16:under_press, 17:sb_xg, 18:is_goal

function chartColors() {
  const light = document.documentElement.getAttribute('data-theme') === 'light';
  return {
    text: light ? '#334155' : '#94a3b8',
    grid: light ? 'rgba(15,23,42,0.08)' : 'rgba(255,255,255,0.07)',
    emerald: light ? '#059669' : '#10b981',
    cyan: light ? '#0284c7' : '#00e5ff',
    violet: light ? '#7c3aed' : '#a78bfa',
    amber: light ? '#d97706' : '#f59e0b',
    rose: light ? '#e11d48' : '#f43f5e',
  };
}

function makeOrReplace(id, cfg) {
  if (CHARTS[id]) CHARTS[id].destroy();
  const ctx = document.getElementById(id);
  if (!ctx) return;
  CHARTS[id] = new Chart(ctx, cfg);
}

function getFilteredShots() {
  const compVal = document.getElementById('fComp').value;
  const teamVal = document.getElementById('fTeam').value;
  const playerQ = document.getElementById('fPlayer').value.trim().toLowerCase();
  const bodyVal = document.getElementById('fBody').value;
  const outVal = document.getElementById('fOutcome').value;
  const minXg = Number(document.getElementById('fMinXg').value) || 0;
  document.getElementById('fMinXgVal').textContent = minXg.toFixed(2);

  const L = DATA.lookups;
  return DATA.shots.filter((s) => {
    if (compVal !== '' && s[0] !== Number(compVal)) return false;
    if (teamVal !== '' && s[1] !== Number(teamVal)) return false;
    if (bodyVal !== '' && s[8] !== Number(bodyVal)) return false;
    if (outVal === 'goal' && s[18] !== 1) return false;
    if (outVal === 'miss' && s[18] !== 0) return false;
    if (s[17] < minXg) return false;
    if (playerQ && !L.players[s[2]].toLowerCase().includes(playerQ)) return false;
    return true;
  });
}

function inspectShot(s) {
  lastSelectedShot = s;
  const L = DATA.lookups;
  const isGoal = s[18] === 1;
  document.getElementById('inspTitle').textContent = `${L.players[s[2]]} (${L.teams[s[1]]} · ${s[3]}')`;
  const badge = document.getElementById('inspBadge');
  badge.textContent = isGoal ? `GOAL · xG ${s[17].toFixed(3)}` : `NO GOAL · xG ${s[17].toFixed(3)}`;
  badge.className = `badge-pill mono ${isGoal ? 'goal' : 'miss'}`;

  const cells = [
    ['Competition', L.competitions[s[0]].replace('FIFA World Cup ', 'WC ').replace('UEFA Euro ', 'Euro ').replace('Bundesliga ', 'BL ')],
    ['StatsBomb xG', `${(s[17] * 100).toFixed(1)}% (${s[17].toFixed(3)})`],
    ['Distance / Angle', `${s[6]} yd · ${s[7]}°`],
    ['Body / Technique', `${L.body_parts[s[8]]} · ${L.techniques[s[10]]}`],
    ['Play Pattern', L.play_patterns[s[9]]],
    ['Defenders in Cone', `${s[11]} defenders`],
    ['Defenders ≤2.5 yd', `${s[12]} tight pressure`],
    ['GK Dist to Shot', s[13] >= 0 ? `${s[13]} yd (${s[14] ? 'In Cone' : 'Off-Line'})` : 'N/A'],
    ['Context Flags', `${s[15] ? '1v1 · ' : ''}${s[16] ? 'Under Pressure' : 'Unpressured'}`],
  ];
  document.getElementById('inspGrid').innerHTML = cells
    .map(([k, v]) => `<div class="insp-cell"><span>${k}</span><strong>${v}</strong></div>`)
    .join('');
}

function renderExplorer() {
  const shots = getFilteredShots();
  const svg = document.getElementById('shotPitch');
  const light = document.documentElement.getAttribute('data-theme') === 'light';
  const lineCol = light ? 'rgba(15,23,42,0.28)' : 'rgba(255,255,255,0.26)';
  const goalCol = light ? '#059669' : '#10b981';

  // StatsBomb attacking half-pitch: x in [60..120], y in [0..80]
  // Map y [0..80] -> svg X [30..570] (540px wide)
  // Map x [120..60] -> svg Y [26..406] (380px tall; goal at top y=26)
  const mapX = (sbY) => 30 + (sbY / 80) * 540;
  const mapY = (sbX) => 26 + ((120 - Math.max(60, Math.min(120, sbX))) / 60) * 380;

  let parts = [];
  // Pitch markings
  parts.push(`<rect x="30" y="26" width="540" height="380" rx="6" fill="none" stroke="${lineCol}" stroke-width="1.5" />`);
  // Goal mouth: y in [36..44] at x=120
  parts.push(`<rect x="${mapX(36).toFixed(1)}" y="14" width="${(mapX(44) - mapX(36)).toFixed(1)}" height="12" fill="rgba(16,185,129,0.22)" stroke="${goalCol}" stroke-width="2" />`);
  // Penalty Box: x in [102..120], y in [18..62]
  parts.push(`<rect x="${mapX(18).toFixed(1)}" y="26" width="${(mapX(62) - mapX(18)).toFixed(1)}" height="${(mapY(102) - 26).toFixed(1)}" fill="none" stroke="${lineCol}" stroke-width="1.4" />`);
  // 6-Yard Box: x in [114..120], y in [30..50]
  parts.push(`<rect x="${mapX(30).toFixed(1)}" y="26" width="${(mapX(50) - mapX(30)).toFixed(1)}" height="${(mapY(114) - 26).toFixed(1)}" fill="none" stroke="${lineCol}" stroke-width="1.3" />`);
  // Penalty Spot (x=108, y=40)
  parts.push(`<circle cx="${mapX(40).toFixed(1)}" cy="${mapY(108).toFixed(1)}" r="2.6" fill="${lineCol}" />`);
  // Halfway arc
  parts.push(`<path d="M ${mapX(30).toFixed(1)} 406 A 64 64 0 0 1 ${mapX(50).toFixed(1)} 406" fill="none" stroke="${lineCol}" stroke-width="1.3" />`);

  // Draw non-goals first, then goals on top
  const nonGoals = shots.filter((s) => s[18] === 0);
  const goals = shots.filter((s) => s[18] === 1);
  // Cap background non-goals if >1800 only for ultra-smooth hover (keep all goals!)
  const step = nonGoals.length > 1800 ? 2 : 1;

  for (let i = 0; i < nonGoals.length; i += step) {
    const s = nonGoals[i];
    const r = Math.max(2.6, Math.min(10.5, 2.2 + Math.sqrt(s[17]) * 9.5));
    parts.push(
      `<circle class="shot-dot" data-idx="${DATA.shots.indexOf(s)}" cx="${mapX(s[5]).toFixed(1)}" cy="${mapY(s[4]).toFixed(1)}" r="${r.toFixed(1)}" fill="rgba(0,229,255,0.34)" stroke="rgba(0,229,255,0.72)" stroke-width="0.7" />`
    );
  }
  for (const s of goals) {
    const r = Math.max(4.0, Math.min(12.5, 3.0 + Math.sqrt(s[17]) * 10.5));
    parts.push(
      `<circle class="shot-dot" data-idx="${DATA.shots.indexOf(s)}" cx="${mapX(s[5]).toFixed(1)}" cy="${mapY(s[4]).toFixed(1)}" r="${r.toFixed(1)}" fill="rgba(16,185,129,0.88)" stroke="#ffffff" stroke-width="1.1" />`
    );
  }

  svg.innerHTML = parts.join('');
  svg.querySelectorAll('.shot-dot').forEach((el) => {
    const handler = () => inspectShot(DATA.shots[Number(el.dataset.idx)]);
    el.addEventListener('mouseenter', handler);
    el.addEventListener('click', handler);
  });

  // Filtered KPIs
  const totalShots = shots.length;
  const totalGoals = goals.length;
  const sumXg = shots.reduce((a, s) => a + s[17], 0);
  const diff = totalGoals - sumXg;
  const avgDist = totalShots ? shots.reduce((a, s) => a + s[6], 0) / totalShots : 0;
  const avgDef = totalShots ? shots.reduce((a, s) => a + s[11], 0) / totalShots : 0;

  document.getElementById('pitchTitle').textContent = `${totalShots.toLocaleString()} Shots · ${totalGoals} Goals · ${sumXg.toFixed(2)} xG`;
  document.getElementById('filteredKpis').innerHTML = [
    ['Filtered Shots', totalShots.toLocaleString(), 'cyan'],
    ['Actual Goals', totalGoals.toLocaleString(), 'emerald'],
    ['Sum StatsBomb xG', sumXg.toFixed(2), 'violet'],
    ['G − xG Delta', `${diff >= 0 ? '+' : ''}${diff.toFixed(2)}`, diff >= 0 ? 'emerald' : 'rose'],
    ['Avg Dist / Def', `${avgDist.toFixed(1)}yd · ${avgDef.toFixed(1)}d`, 'amber'],
  ]
    .map(([l, v, c]) => `<div class="fk-box"><span>${l}</span><strong class="${c}">${v}</strong></div>`)
    .join('');

  // Auto-inspect highest-xG goal or first shot in slice
  if (goals.length) inspectShot(goals[0]);
  else if (shots.length) inspectShot(shots[0]);

  // Defender Cone Pressure breakdown chart
  const buckets = ['0 Def', '1 Def', '2 Def', '3+ Def'];
  const bCounts = [0, 0, 0, 0];
  const bGoals = [0, 0, 0, 0];
  for (const s of shots) {
    const b = Math.min(3, s[11]);
    bCounts[b]++;
    if (s[18] === 1) bGoals[b]++;
  }
  const bConv = bCounts.map((n, i) => (n ? Number(((bGoals[i] / n) * 100).toFixed(1)) : 0));
  const col = chartColors();
  makeOrReplace('chartDefPressure', {
    data: {
      labels: buckets,
      datasets: [
        { type: 'bar', label: 'Shots', data: bCounts, backgroundColor: 'rgba(0,229,255,0.55)', borderRadius: 5, yAxisID: 'y' },
        { type: 'line', label: 'Goal Conversion %', data: bConv, borderColor: col.emerald, borderWidth: 2.5, pointRadius: 4, yAxisID: 'y1' },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { labels: { color: col.text } } },
      scales: {
        x: { ticks: { color: col.text }, grid: { color: col.grid } },
        y: { ticks: { color: col.text }, grid: { color: col.grid } },
        y1: { position: 'right', ticks: { color: col.emerald, callback: (v) => `${v}%` }, grid: { drawOnChartArea: false } },
      },
    },
  });
}

function renderAblation() {
  const c = chartColors();
  const models = DATA.metrics.models;
  const sel = DATA.metrics.selected_custom_model;
  const entries = Object.entries(models);

  makeOrReplace('chartAblation', {
    data: {
      labels: entries.map(([k]) => k.replace(' (Ref)', '').replace(' (+Freeze-Frame)', ' +FF')),
      datasets: [
        {
          type: 'bar',
          label: 'Test Log-Loss ↓ (Lower Better)',
          data: entries.map(([, m]) => m.test_log_loss),
          backgroundColor: entries.map(([k]) => (k === sel ? c.emerald : k.includes('StatsBomb') ? c.amber : 'rgba(0,229,255,0.52)')),
          borderRadius: 6,
          yAxisID: 'y',
        },
        {
          type: 'line',
          label: 'Test ROC-AUC ↑',
          data: entries.map(([, m]) => m.test_roc_auc),
          borderColor: c.violet,
          borderWidth: 2.6,
          pointRadius: 4,
          yAxisID: 'y1',
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { labels: { color: c.text } } },
      scales: {
        x: { ticks: { color: c.text, maxRotation: 25, font: { size: 10 } }, grid: { color: c.grid } },
        y: { min: 0.22, max: 0.34, ticks: { color: c.text }, grid: { color: c.grid } },
        y1: { position: 'right', min: 0.45, max: 0.90, ticks: { color: c.violet }, grid: { drawOnChartArea: false } },
      },
    },
  });

  // SHAP Horizontal Bar
  const shap = [...DATA.metrics.top_shap_features].reverse();
  makeOrReplace('chartXgShap', {
    type: 'bar',
    data: {
      labels: shap.map((s) => s.feature),
      datasets: [
        {
          label: 'Mean |SHAP|',
          data: shap.map((s) => s.mean_abs_shap),
          backgroundColor: shap.map((s) => (s.feature.includes('defenders') || s.feature.includes('gk_') ? c.emerald : c.cyan)),
          borderRadius: 5,
        },
      ],
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: c.text }, grid: { color: c.grid } },
        y: { ticks: { color: c.text, font: { size: 11 } }, grid: { display: false } },
      },
    },
  });

  // Tables
  document.querySelector('#modelBenchTable tbody').innerHTML = entries
    .map(([name, m]) => {
      const isWin = name === sel;
      return `
        <tr class="${isWin ? 'row-hi' : ''}">
          <td><strong>${name}${isWin ? ' ★' : ''}</strong></td>
          <td>${m.val_log_loss.toFixed(4)}</td>
          <td class="emerald">${m.test_log_loss.toFixed(4)}</td>
          <td>${m.test_brier.toFixed(4)}</td>
          <td class="cyan">${m.test_roc_auc.toFixed(4)} [${m.test_roc_auc_ci95[0].toFixed(3)}, ${m.test_roc_auc_ci95[1].toFixed(3)}]</td>
          <td>${m.sum_pred_xg.toFixed(2)}</td>
        </tr>`;
    })
    .join('');

  document.querySelector('#sliceTable tbody').innerHTML = DATA.metrics.slice_audit
    .map(
      (r) => `
      <tr>
        <td><strong>${r.slice}</strong></td>
        <td>${r.shots}</td>
        <td>${r.actual_goals}</td>
        <td class="cyan">${r.sum_custom_xg.toFixed(2)}</td>
        <td>${r.brier.toFixed(4)}</td>
        <td class="emerald">${r.roc_auc.toFixed(4)}</td>
      </tr>`
    )
    .join('');
}

function renderLeaderboard() {
  const c = chartColors();
  const lb = DATA.leaderboard;
  const maxVal = Math.max(...lb.map((r) => Math.max(r.custom_xg, r.goals))) + 1;

  makeOrReplace('chartFinScatter', {
    data: {
      datasets: [
        {
          type: 'line',
          label: 'Perfect Calibration (G = xG)',
          data: [{ x: 0, y: 0 }, { x: maxVal, y: maxVal }],
          borderColor: 'rgba(148,163,184,0.45)',
          borderDash: [5, 5],
          borderWidth: 1.5,
          pointRadius: 0,
        },
        {
          type: 'scatter',
          label: 'Overperformers (G > xG)',
          data: lb.filter((r) => r.goals_minus_xg >= 0).map((r) => ({ x: r.custom_xg, y: r.goals, meta: r })),
          backgroundColor: c.emerald,
          pointRadius: 5.5,
        },
        {
          type: 'scatter',
          label: 'Underperformers (G < xG)',
          data: lb.filter((r) => r.goals_minus_xg < 0).map((r) => ({ x: r.custom_xg, y: r.goals, meta: r })),
          backgroundColor: c.rose,
          pointRadius: 5.5,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: c.text } },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const m = ctx.raw.meta;
              if (!m) return 'G = xG Diagonal';
              return `${m.player} (${m.team}): ${m.goals}G vs ${m.custom_xg.toFixed(2)} xG (${m.goals_minus_xg >= 0 ? '+' : ''}${m.goals_minus_xg.toFixed(2)})`;
            },
          },
        },
      },
      scales: {
        x: { title: { display: true, text: 'Custom Freeze-Frame xG', color: c.text }, ticks: { color: c.text }, grid: { color: c.grid } },
        y: { title: { display: true, text: 'Actual Non-Penalty Goals', color: c.text }, ticks: { color: c.text }, grid: { color: c.grid } },
      },
    },
  });

  renderLbTable();
}

function renderLbTable() {
  const q = (document.getElementById('lbSearch')?.value || '').trim().toLowerCase();
  const rows = DATA.leaderboard
    .filter((r) => !q || r.player.toLowerCase().includes(q) || r.team.toLowerCase().includes(q))
    .sort((a, b) => {
      const av = a[lbSort.key];
      const bv = b[lbSort.key];
      return typeof av === 'string' ? av.localeCompare(bv) * lbSort.dir : (av - bv) * lbSort.dir;
    });

  const tbody = document.querySelector('#lbTable tbody');
  tbody.innerHTML = rows
    .map(
      (r) => `
      <tr class="clickable" data-jump-player="${r.player}">
        <td><strong>${r.player}</strong></td>
        <td>${r.team}</td>
        <td>${r.shots}</td>
        <td class="cyan">${r.goals}</td>
        <td>${r.custom_xg.toFixed(2)}</td>
        <td>${r.statsbomb_xg.toFixed(2)}</td>
        <td class="${r.goals_minus_xg >= 0 ? 'emerald' : 'rose'}"><strong>${r.goals_minus_xg >= 0 ? '+' : ''}${r.goals_minus_xg.toFixed(2)}</strong></td>
      </tr>`
    )
    .join('');

  tbody.querySelectorAll('tr[data-jump-player]').forEach((tr) => {
    tr.addEventListener('click', () => {
      document.getElementById('fComp').value = '';
      document.getElementById('fTeam').value = '';
      document.getElementById('fBody').value = '';
      document.getElementById('fOutcome').value = 'all';
      document.getElementById('fMinXg').value = '0';
      document.getElementById('fPlayer').value = tr.dataset.jumpPlayer;
      document.querySelector('.tab-btn[data-tab="explorer"]').click();
    });
  });
}

function applyLang(l) {
  lang = l;
  document.documentElement.lang = l;
  document.documentElement.dir = l === 'ar' ? 'rtl' : 'ltr';
  document.querySelectorAll('[data-i18n]').forEach((el) => {
    const k = el.getAttribute('data-i18n');
    if (I18N[l][k]) el.textContent = I18N[l][k];
  });
  document.getElementById('langToggle').textContent = l === 'en' ? 'عربي' : 'EN';
}

function renderActiveTab() {
  const active = document.querySelector('.tab-btn.active')?.dataset.tab || 'explorer';
  if (active === 'explorer') renderExplorer();
  if (active === 'ablation') renderAblation();
  if (active === 'leaderboard') renderLeaderboard();
}

document.addEventListener('DOMContentLoaded', () => {
  if (new URLSearchParams(location.search).get('embed') === '1') {
    document.body.classList.add('embed-mode');
  }

  document.querySelectorAll('.tab-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach((b) => { b.classList.remove('active'); b.setAttribute('aria-selected', 'false'); });
      document.querySelectorAll('.tab-panel').forEach((p) => p.classList.remove('active'));
      btn.classList.add('active');
      btn.setAttribute('aria-selected', 'true');
      document.getElementById(`panel-${btn.dataset.tab}`).classList.add('active');
      if (DATA) renderActiveTab();
    });
  });

  document.getElementById('themeToggle').addEventListener('click', () => {
    const next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    document.getElementById('themeToggle').textContent = next === 'dark' ? '☀' : '☾';
    if (DATA) renderActiveTab();
  });
  document.getElementById('langToggle').addEventListener('click', () => applyLang(lang === 'en' ? 'ar' : 'en'));

  ['fComp', 'fTeam', 'fPlayer', 'fBody', 'fOutcome', 'fMinXg'].forEach((id) =>
    document.getElementById(id).addEventListener('input', () => DATA && renderExplorer())
  );

  document.querySelectorAll('.star-btn[data-star]').forEach((b) => {
    b.addEventListener('click', () => {
      document.querySelectorAll('.star-btn[data-star]').forEach((x) => x.classList.remove('active'));
      b.classList.add('active');
      document.getElementById('fComp').value = '';
      document.getElementById('fTeam').value = '';
      document.getElementById('fBody').value = '';
      document.getElementById('fOutcome').value = 'all';
      document.getElementById('fMinXg').value = '0';
      document.getElementById('fPlayer').value = b.dataset.star;
      if (DATA) renderExplorer();
    });
  });

  document.getElementById('resetFiltersBtn').addEventListener('click', () => {
    document.querySelector('.star-btn[data-star=""]').click();
  });

  document.getElementById('lbSearch').addEventListener('input', () => DATA && renderLbTable());
  document.querySelectorAll('#lbTable th[data-sort]').forEach((th) => {
    th.addEventListener('click', () => {
      const k = th.dataset.sort;
      lbSort.dir = lbSort.key === k ? lbSort.dir * -1 : -1;
      lbSort.key = k;
      if (DATA) renderLbTable();
    });
  });

  fetch('xg_dashboard_data.json')
    .then((r) => r.json())
    .then((json) => {
      DATA = json;
      const L = DATA.lookups;
      const compSel = document.getElementById('fComp');
      L.competitions.forEach((v, i) => compSel.insertAdjacentHTML('beforeend', `<option value="${i}">${v}</option>`));
      const teamSel = document.getElementById('fTeam');
      L.teams.forEach((v, i) => teamSel.insertAdjacentHTML('beforeend', `<option value="${i}">${v}</option>`));
      const bodySel = document.getElementById('fBody');
      L.body_parts.forEach((v, i) => bodySel.insertAdjacentHTML('beforeend', `<option value="${i}">${v}</option>`));
      renderExplorer();
    });
});
