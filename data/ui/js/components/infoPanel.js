import { state } from "../state.js";
import { numberClass } from "../utils.js";

export function infoPanelHTML(){
  return `
    <section class="card infoPanel">
      <div class="infoHeader">
        <div class="cardTitle infoTitle">Current Info</div>
      </div>

      <div class="infoStack">
        <div class="infoMetric">
          <div class="infoLabel">Scan</div>
          <div id="scanState" class="statusBadge stopped">STOP</div>
        </div>

        <div class="infoMetric">
          <div class="infoLabel">Last Number</div>
          <div id="lastNumber" class="numberBadge empty">--</div>
        </div>

        <div class="infoMetric">
          <div class="infoLabel">Last Match</div>
          <div id="lastMatch" class="matchBadge empty">--</div>
        </div>
      </div>
    </section>
  `;
}

export function renderInfoPanel(){
  const st = state.liveStatus || {};
  const scanning = !!st.monitoring;

  const scanEl = document.getElementById("scanState");
  scanEl.textContent = scanning ? "SCAN" : "STOP";
  scanEl.className = scanning ? "statusBadge running" : "statusBadge stopped";

  const lastNumberEl = document.getElementById("lastNumber");
  const lastNumber = st.lastOutputNumber;

  if(lastNumber === null || lastNumber === undefined){
    lastNumberEl.textContent = "--";
    lastNumberEl.className = "numberBadge empty";
  }else{
    lastNumberEl.textContent = lastNumber;
    lastNumberEl.className = `numberBadge ${numberClass(lastNumber)}`;
  }

  const lastMatchEl = document.getElementById("lastMatch");
  const match = st.latestMatch;

  if(match && match.number !== undefined && match.number !== null){
    lastMatchEl.textContent = match.number;
    lastMatchEl.className = `matchBadge ${numberClass(match.number)}`;
  }else{
    lastMatchEl.textContent = "--";
    lastMatchEl.className = "matchBadge empty";
  }
}
