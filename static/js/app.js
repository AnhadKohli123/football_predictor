/* ============================================================
   Football Predictor — frontend
   ============================================================ */

(() => {
  "use strict";

  const $  = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  const state = {
    teams: [],
    leagues: [],
    selectedLeagues: new Set(),
    status: null,
  };

  /** Escape anything that came from the model, the API, or the user. */
  const esc = (value) =>
    String(value ?? "").replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
    );

  const pct = (n) => `${(Number(n) * 100).toFixed(1)}%`;

  /** "Home Win" -> "home", so it can pick a colour class. */
  const outcomeClass = (prediction) => {
    const p = String(prediction || "").toLowerCase();
    if (p.includes("home")) return "home";
    if (p.includes("away")) return "away";
    return "draw";
  };

  async function api(url, options) {
    const response = await fetch(url, options);
    let payload;
    try {
      payload = await response.json();
    } catch {
      throw new Error(`Server returned ${response.status} (${response.statusText}).`);
    }
    if (!response.ok || payload.ok === false) {
      throw new Error(payload.error || `Request failed (${response.status}).`);
    }
    return payload;
  }

  const errorBox = (message) => `<div class="msg error">${esc(message)}</div>`;
  const infoBox  = (message) => `<div class="msg info">${esc(message)}</div>`;

  // ============================================================
  // Tabs
  // ============================================================

  $$(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      $$(".tab").forEach((t) => t.setAttribute("aria-selected", String(t === tab)));
      $$(".panel").forEach((p) =>
        p.classList.toggle("active", p.id === `panel-${tab.dataset.panel}`)
      );
    });
  });

  // ============================================================
  // Searchable team picker
  // ============================================================

  function initCombo(root) {
    const input = $("input", root);
    const list  = $("[data-options]", root);
    let activeIndex = -1;

    const render = (matches) => {
      if (!matches.length) {
        list.innerHTML = `<div class="option"><span class="empty">No matching team</span></div>`;
      } else {
        list.innerHTML = matches
          .map((team, i) => `<div class="option${i === activeIndex ? " active" : ""}" data-value="${esc(team)}">${esc(team)}</div>`)
          .join("");
      }
      list.classList.add("open");
    };

    const matching = () => {
      const query = input.value.trim().toLowerCase();
      if (!query) return state.teams.slice(0, 80);
      // Rank teams that START with the query above ones that merely contain
      // it, so typing "en" offers England before Argentina.
      const starts = [], contains = [];
      state.teams.forEach((team) => {
        const lower = team.toLowerCase();
        if (lower.startsWith(query)) starts.push(team);
        else if (lower.includes(query)) contains.push(team);
      });
      return starts.concat(contains).slice(0, 80);
    };

    const close = () => { list.classList.remove("open"); activeIndex = -1; };

    input.addEventListener("focus", () => { activeIndex = -1; render(matching()); });
    input.addEventListener("input", () => { activeIndex = -1; render(matching()); });

    input.addEventListener("keydown", (event) => {
      const options = $$(".option[data-value]", list);
      if (event.key === "ArrowDown" || event.key === "ArrowUp") {
        event.preventDefault();
        if (!list.classList.contains("open")) { render(matching()); return; }
        activeIndex += event.key === "ArrowDown" ? 1 : -1;
        if (activeIndex < 0) activeIndex = options.length - 1;
        if (activeIndex >= options.length) activeIndex = 0;
        options.forEach((o, i) => o.classList.toggle("active", i === activeIndex));
        options[activeIndex]?.scrollIntoView({ block: "nearest" });
      } else if (event.key === "Enter") {
        if (activeIndex >= 0 && options[activeIndex]) {
          event.preventDefault();
          input.value = options[activeIndex].dataset.value;
          close();
        }
      } else if (event.key === "Escape") {
        close();
      }
    });

    list.addEventListener("mousedown", (event) => {
      const option = event.target.closest(".option[data-value]");
      if (!option) return;
      event.preventDefault();
      input.value = option.dataset.value;
      close();
    });

    input.addEventListener("blur", () => setTimeout(close, 120));
  }

  // ============================================================
  // Header status
  // ============================================================

  function renderStatus(status) {
    const pills = [];

    pills.push(status.model_ready
      ? `<span class="pill"><span class="dot on"></span> Model <b>ready</b></span>`
      : `<span class="pill"><span class="dot off"></span> <b>No model</b> — run train_model.py</span>`);

    if (status.matches) {
      pills.push(`<span class="pill"><span class="dot on"></span> <b>${status.matches.toLocaleString()}</b> matches</span>`);
      pills.push(`<span class="pill"><span class="dot on"></span> <b>${status.teams}</b> teams</span>`);
    }

    if (status.date_range) {
      pills.push(`<span class="pill">📅 ${esc(status.date_range[0])} → ${esc(status.date_range[1])}</span>`);
    }

    pills.push(status.api_key_set
      ? `<span class="pill"><span class="dot on"></span> API <b>${esc(status.provider || "?")}</b></span>`
      : `<span class="pill"><span class="dot warn"></span> <b>No API key</b> — fixtures unavailable</span>`);

    $("#pills").innerHTML = pills.join("");
  }

  // ============================================================
  // Match predictor
  // ============================================================

  function renderPrediction(p) {
    const cls = outcomeClass(p.prediction);
    const winner =
      cls === "home" ? p.home_team : cls === "away" ? p.away_team : "Draw";

    const warning = p.reliable
      ? ""
      : `<div class="note warn" style="margin-bottom:18px">
           <b>Limited history for ${esc((p.unknown_teams || []).join(", "))}.</b>
           This prediction falls back to league-average priors, so it carries much
           less information than the percentages suggest.
         </div>`;

    return `
      <div class="card">
        ${warning}
        <div class="verdict">
          <div class="matchup">
            ${esc(p.home_team)}<span class="sep">vs</span>${esc(p.away_team)}
            <span class="meta">${esc(p.date)}${p.league ? " · " + esc(p.league) : ""}</span>
          </div>
          <div class="call ${cls}">
            ${esc(cls === "draw" ? "Draw" : winner + " win")}
            <span class="conf">${esc(p.confidence)}</span>
          </div>
        </div>

        <div class="bars">
          ${bar("home", `${esc(p.home_team)} win`, p.prob_home_win)}
          ${bar("draw", "Draw", p.prob_draw)}
          ${bar("away", `${esc(p.away_team)} win`, p.prob_away_win)}
        </div>
      </div>`;
  }

  const bar = (cls, label, value) => `
    <div class="bar-row">
      <div class="bar-label">${label}</div>
      <div class="bar-track"><div class="bar-fill ${cls}" data-w="${(Number(value) * 100).toFixed(1)}"></div></div>
      <div class="bar-pct">${pct(value)}</div>
    </div>`;

  /** Bars animate from zero, so widths are applied after paint. */
  function animateBars(container) {
    requestAnimationFrame(() => {
      $$(".bar-fill", container).forEach((el) => { el.style.width = `${el.dataset.w}%`; });
    });
  }

  async function predictMatch() {
    const button = $("#predict-btn");
    const target = $("#predict-result");
    const home = $("#home-team").value.trim();
    const away = $("#away-team").value.trim();

    if (!home || !away) {
      target.innerHTML = errorBox("Pick both a home and an away team.");
      return;
    }
    if (home === away) {
      target.innerHTML = errorBox("A team cannot play itself.");
      return;
    }

    button.disabled = true;
    button.innerHTML = `<span class="spinner"></span> Predicting…`;
    target.innerHTML = "";

    try {
      const data = await api("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          home, away,
          league: $("#league").value,
          date: $("#match-date").value,
        }),
      });
      target.innerHTML = renderPrediction(data.prediction);
      animateBars(target);
    } catch (err) {
      target.innerHTML = errorBox(err.message);
    } finally {
      button.disabled = false;
      button.textContent = "Predict outcome";
    }
  }

  // ============================================================
  // Fixtures
  // ============================================================

  function renderFixtures(rows, meta) {
    if (!rows.length) {
      return `<div class="card"><div class="empty-state">
        <span class="ico">📅</span>
        ${esc(meta.message || "No fixtures found in that window.")}
      </div></div>`;
    }

    const byLeague = new Map();
    rows.forEach((row) => {
      if (!byLeague.has(row.league)) byLeague.set(row.league, []);
      byLeague.get(row.league).push(row);
    });

    let html = `<div class="card">
      <div class="card-title">${rows.length} fixture${rows.length === 1 ? "" : "s"}
        · ${meta.requests_made ? esc(meta.requests_made) + " API request(s)" : "served from cache"}</div>`;

    byLeague.forEach((fixtures, league) => {
      html += `<div class="league-head">${esc(league)}</div><div class="fixtures">`;
      fixtures.forEach((f) => {
        const cls = outcomeClass(f.prediction);
        const day = new Date(f.date + "T00:00:00");
        const weekday = isNaN(day) ? "" : day.toLocaleDateString(undefined, { weekday: "short" });
        const label = isNaN(day) ? f.date : day.toLocaleDateString(undefined, { day: "numeric", month: "short" });

        html += `
          <div class="fixture">
            <div class="fx-date"><b>${esc(label)}</b>${esc(weekday)}</div>
            <div class="fx-teams">
              ${esc(f.home_team)}<span class="sep">vs</span>${esc(f.away_team)}
              ${f.reliable ? "" : ` <span class="conf" title="Limited history — prediction uses priors">⚠</span>`}
            </div>
            <div class="fx-right">
              <div class="fx-mini" title="${pct(f.prob_home_win)} / ${pct(f.prob_draw)} / ${pct(f.prob_away_win)}">
                <span class="h" style="width:${(f.prob_home_win * 100).toFixed(1)}%"></span>
                <span class="d" style="width:${(f.prob_draw * 100).toFixed(1)}%"></span>
                <span class="a" style="width:${(f.prob_away_win * 100).toFixed(1)}%"></span>
              </div>
              <div class="fx-call ${cls}">
                ${esc(f.prediction)}
                <small>${esc(f.confidence)}</small>
              </div>
            </div>
          </div>`;
      });
      html += `</div>`;
    });

    return html + `</div>`;
  }

  async function fetchFixtures() {
    const button = $("#fixtures-btn");
    const target = $("#fixtures-result");

    button.disabled = true;
    button.innerHTML = `<span class="spinner"></span> Fetching…`;
    target.innerHTML = infoBox("Contacting the API. A cold fetch can take a few seconds per competition.");

    const params = new URLSearchParams({ days: $("#days").value });
    state.selectedLeagues.forEach((l) => params.append("league", l));

    try {
      const data = await api(`/api/fixtures?${params}`);
      target.innerHTML = renderFixtures(data.fixtures, data);
    } catch (err) {
      target.innerHTML = errorBox(err.message);
    } finally {
      button.disabled = false;
      button.innerHTML = "Fetch &amp; predict";
    }
  }

  // ============================================================
  // Model panel
  // ============================================================

  function renderMetrics(status) {
    const target = $("#model-metrics");
    const m = status.metrics;

    if (!m) {
      target.innerHTML = infoBox(
        "No metrics yet. Run  python train_model.py  and reload this page."
      );
      return;
    }

    const beatsBaseline = m.accuracy > m.baseline_always_home;
    const beatsPriors = m.log_loss < m.baseline_prior_log_loss;

    target.innerHTML = `
      <div class="metrics">
        <div class="metric">
          <div class="k">Accuracy</div>
          <div class="v">${(m.accuracy * 100).toFixed(1)}%</div>
          <div class="sub">${beatsBaseline ? "+" : ""}${((m.accuracy - m.baseline_always_home) * 100).toFixed(1)} pts vs baseline</div>
        </div>
        <div class="metric">
          <div class="k">Log loss</div>
          <div class="v">${m.log_loss.toFixed(3)}</div>
          <div class="sub">${beatsPriors ? "beats" : "WORSE than"} priors (${m.baseline_prior_log_loss.toFixed(3)})</div>
        </div>
        <div class="metric">
          <div class="k">Brier score</div>
          <div class="v plain">${m.brier.toFixed(3)}</div>
          <div class="sub">lower is better</div>
        </div>
        <div class="metric">
          <div class="k">Algorithm</div>
          <div class="v plain" style="font-size:1.15rem">${esc(m.model)}</div>
          <div class="sub">${esc(m.n_features)} features</div>
        </div>
        <div class="metric">
          <div class="k">Trained on</div>
          <div class="v plain" style="font-size:1.3rem">${Number(m.trained_on).toLocaleString()}</div>
          <div class="sub">matches</div>
        </div>
        <div class="metric">
          <div class="k">Tested on</div>
          <div class="v plain" style="font-size:1.3rem">${Number(m.tested_on).toLocaleString()}</div>
          <div class="sub">most recent matches</div>
        </div>
      </div>`;
  }

  // ============================================================
  // Boot
  // ============================================================

  async function boot() {
    $$("[data-combo]").forEach(initCombo);

    const dateInput = $("#match-date");
    dateInput.value = new Date().toISOString().slice(0, 10);

    const days = $("#days");
    const daysVal = $("#days-val");
    const syncDays = () => {
      daysVal.textContent = `${days.value} day${days.value === "1" ? "" : "s"}`;
      days.style.setProperty("--pct", `${((days.value - 1) / 59) * 100}%`);
    };
    days.addEventListener("input", syncDays);
    syncDays();

    $("#predict-btn").addEventListener("click", predictMatch);
    $("#fixtures-btn").addEventListener("click", fetchFixtures);

    // Enter in either team box runs the prediction.
    ["#home-team", "#away-team"].forEach((sel) => {
      $(sel).addEventListener("keydown", (event) => {
        if (event.key === "Enter" && !$(".options.open")) predictMatch();
      });
    });

    try {
      const status = await api("/api/status");
      state.status = status;
      state.leagues = status.leagues || [];
      renderStatus(status);
      renderMetrics(status);

      $("#league").innerHTML =
        state.leagues.map((l) => `<option value="${esc(l.name)}">${esc(l.name)}</option>`).join("") +
        `<option value="">— other / friendly —</option>`;

      $("#league-chips").innerHTML = state.leagues
        .map((l) => `<button type="button" class="chip" aria-pressed="false" data-key="${esc(l.key)}">${esc(l.name)}</button>`)
        .join("");

      $$("#league-chips .chip").forEach((chip) => {
        chip.addEventListener("click", () => {
          const on = chip.getAttribute("aria-pressed") === "true";
          chip.setAttribute("aria-pressed", String(!on));
          if (on) state.selectedLeagues.delete(chip.dataset.key);
          else state.selectedLeagues.add(chip.dataset.key);
        });
      });

      if (!status.api_key_set) {
        $("#fixtures-result").innerHTML = infoBox(
          "No API token configured, so upcoming fixtures cannot be fetched.\n" +
          "Add FOOTBALL_API_KEY to a .env file next to app.py and restart.\n\n" +
          "The Match Predictor tab works without a token."
        );
      }
    } catch (err) {
      $("#pills").innerHTML = `<span class="pill"><span class="dot off"></span> ${esc(err.message)}</span>`;
    }

    try {
      const data = await api("/api/teams");
      state.teams = data.teams || [];
      if (state.teams.length >= 2) {
        $("#home-team").placeholder = `e.g. ${state.teams[0]}`;
        $("#away-team").placeholder = `e.g. ${state.teams[1]}`;
      }
    } catch {
      $("#home-team").placeholder = "No history loaded";
      $("#away-team").placeholder = "No history loaded";
    }
  }

  document.addEventListener("DOMContentLoaded", boot);
})();
