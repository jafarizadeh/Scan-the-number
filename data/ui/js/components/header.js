import { icons } from "../icons.js";
import { state } from "../state.js";

export function headerHTML(){
  return `
    <header class="topbar">
      <div class="brand">
        <div class="logoBox">
          <img
            class="appLogo"
            src="/assets/logo.png"
            alt="Roulette Vision logo"
            onload="document.getElementById('logoFallback').style.display='none'"
            onerror="this.style.display='none';document.getElementById('logoFallback').style.display='flex'"
          >
          <div id="logoFallback" class="logoFallback">logo<br>path</div>
        </div>

        <div class="brandTitle">Roulette Vision</div>
      </div>

      <div class="topActions">
        <div class="scanPill" id="scanPill">
          <span class="statusDot"></span>
          <span>Stopped</span>
        </div>

        <button class="settingsBtn topBtn" id="settingsBtn">
          ${icons.settings}
          Settings
        </button>
        <button class="settingsBtn topBtn calibrationTopBtn" id="calibrationBtn">
          ${icons.settings}
          Calibration
        </button>

        <button class="shutdownBtn topBtn" id="shutdownBtn">
          <svg class="icon" viewBox="0 0 24 24">
            <path d="M12 2v10"></path>
            <path d="M6.7 6.7a7.5 7.5 0 1 0 10.6 0"></path>
          </svg>
          Power Off
        </button>
      </div>
    </header>
  `;
}

export function bindHeader(actions){
  document.getElementById("settingsBtn").onclick = actions.openSettings;

  const calibrationBtn = document.getElementById("calibrationBtn");
  if(calibrationBtn){
    calibrationBtn.onclick = () => {
      window.location.href = "http://" + window.location.hostname + ":8081/";
    };
  }
  document.getElementById("shutdownBtn").onclick = actions.openShutdownModal;
}

export function renderHeader(){
  const scanning = !!state.liveStatus?.monitoring;

  document.getElementById("scanPill").innerHTML = scanning
    ? `<span class="statusDot"></span><span>Scanning</span>`
    : `<span class="statusDot" style="background:#9aacbf"></span><span>Stopped</span>`;
}
