import { icons } from "../icons.js";
import { state } from "../state.js";
import { setSwitch } from "../utils.js";

export function controlsHTML(){
  return `
    <section class="card controlsCol">
      <div class="panel-title">Controls</div>

      <div class="toggle-card">
        <div class="toggle-left">${icons.camera} Camera</div>
        <div id="cameraSwitch" class="switch on"></div>
      </div>

      <div class="toggle-card">
        <div class="toggle-left">${icons.sound} Sound</div>
        <div id="soundSwitch" class="switch"></div>
      </div>

      <div class="toggle-card">
        <div class="toggle-left">${icons.arm} Arm</div>
        <div id="armSwitch" class="switch"></div>
      </div>

      <div class="status-card">
        <div class="status-line"><span class="dot" id="statusDot"></span>Status</div>
        <div id="statusValue" class="status-value">STOPPED</div>
        <div id="statusSub" class="status-text">System is stopped</div>
      </div>
    </section>
  `;
}

export function bindControls(actions){
  document.getElementById("cameraSwitch").onclick = actions.toggleCamera;
  document.getElementById("soundSwitch").onclick = actions.toggleSound;
  document.getElementById("armSwitch").onclick = actions.toggleArm;
}

export function renderControls(){
  const st = state.liveStatus || {};
  const scanning = !!st.monitoring;

  setSwitch("cameraSwitch", state.cameraOn);

  document.getElementById("statusValue").textContent = scanning ? "SCANNING" : "STOPPED";
  document.getElementById("statusValue").style.color = scanning ? "#1a9f54" : "#db3242";
  document.getElementById("statusSub").textContent = scanning ? "System is active" : "System is stopped";
  document.getElementById("statusDot").style.background = scanning ? "#25bf6b" : "#9aacbf";
}
