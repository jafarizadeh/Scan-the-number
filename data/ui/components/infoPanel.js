import { state } from "../state.js";

export function infoPanelHTML(){
  return `
    <section class="card info-panel">
      <div class="panel-title">Current Info</div>

      <div class="info-card">
        <div class="info-label">Scan</div>
        <div id="scanState" class="info-val">STOP</div>
      </div>

      <div class="info-card">
        <div class="info-label">Last Number</div>
        <div id="lastNumber" class="last-number">--</div>
      </div>

      <div class="info-card">
        <div class="info-label">Last Match</div>
        <div id="lastMatch" class="info-val" style="font-size:22px">--</div>
      </div>
    </section>
  `;
}

export function renderInfoPanel(){
  const st = state.liveStatus || {};
  const scanning = !!st.monitoring;

  document.getElementById("scanState").textContent = scanning ? "SCANNING" : "STOP";
  document.getElementById("scanState").style.color = scanning ? "#1a9f54" : "#db3242";
  document.getElementById("lastNumber").textContent = st.lastOutputNumber ?? "--";
  document.getElementById("lastMatch").textContent = st.latestMatch ? st.latestMatch.number : "--";
}
