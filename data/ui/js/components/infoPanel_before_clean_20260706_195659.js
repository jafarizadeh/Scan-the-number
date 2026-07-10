import { state } from "../state.js";

export function infoPanelHTML(){
  return `
    <section class="card infoPanel">
      <div class="cardTitle">Current Info</div>

      <div class="infoBox">
        <div class="infoLabel">Scan</div>
        <div id="scanState" class="infoVal">STOP</div>
      </div>

      <div class="infoBox">
        <div class="infoLabel">Last Number</div>
        <div id="lastNumber" class="lastNumber">--</div>
      </div>

      <div class="infoBox">
        <div class="infoLabel">Last Match</div>
        <div id="lastMatch" class="infoVal" style="font-size:22px">--</div>
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
