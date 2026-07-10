import { icons } from "../icons.js";
import { state } from "../state.js";

export function headerHTML(){
  return `
    <header class="topbar">
      <div class="brand">
        <div class="logoBox">
          <img src="/assets/logo.png" onerror="this.style.display='none';document.getElementById('logoFallback').style.display='block'">
          <div id="logoFallback" class="logoFallback">logo<br>path</div>
        </div>
        <div class="brandTitle">Roulette Vision</div>
      </div>

      <div class="topActions">
        <button class="topBtn" id="topSoundBtn">${icons.sound} Sound</button>
        <button class="topBtn" id="topArmBtn">${icons.arm} Arm</button>
        <div class="scanPill" id="scanPill"><span class="statusDot"></span><span>Stopped</span></div>
        <button class="settingsBtn topBtn" id="settingsBtn">${icons.settings} Settings</button>
      </div>
    </header>
  `;
}

export function bindHeader(actions){
  document.getElementById("topSoundBtn").onclick = actions.toggleSound;
  document.getElementById("topArmBtn").onclick = actions.toggleArm;
  document.getElementById("settingsBtn").onclick = actions.openSettings;
}

export function renderHeader(){
  const scanning = !!state.liveStatus?.monitoring;
  document.getElementById("scanPill").innerHTML = scanning
    ? `<span class="statusDot"></span><span>Scanning</span>`
    : `<span class="statusDot" style="background:#9aacbf"></span><span>Stopped</span>`;
}
