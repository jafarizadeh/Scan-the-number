import { icons } from "../icons.js";
import { state } from "../state.js";
import { setSwitch } from "../utils.js";

export function controlsHTML(){
  return `
    <section class="card controlsCol">
      <div class="cardTitle">Controls</div>

      <div class="toggleCard">
        <div class="toggleLeft">${icons.camera} Camera</div>
        <div id="cameraSwitch" class="switch on"></div>
      </div>

      <div class="toggleCard">
        <div class="toggleLeft">${icons.sound} Sound</div>
        <div id="soundSwitch" class="switch"></div>
      </div>

      <div class="toggleCard">
        <div class="toggleLeft">${icons.arm} Tapper</div>
        <div id="armSwitch" class="switch"></div>
      </div>

      <div class="statusCard">
        <div class="statusLine"><span class="statusDot" id="statusDot"></span>Status</div>
        <div id="statusValue" class="statusValue">STOPPED</div>
        <div id="statusSub" class="statusSub">System is stopped</div>
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
  const scanning = !!state.liveStatus?.monitoring;
  const hw = state.appSettings?.hardware || {};

  setSwitch("cameraSwitch", state.cameraOn);
  setSwitch("soundSwitch", !!hw.buzzerEnabled);
  setSwitch("armSwitch", !!hw.actuatorEnabled);

  document.getElementById("statusValue").textContent = scanning ? "SCANNING" : "STOPPED";
  document.getElementById("statusValue").style.color = scanning ? "#1a9f54" : "#db3242";
  document.getElementById("statusSub").textContent = scanning ? "System is active" : "System is stopped";
  document.getElementById("statusDot").style.background = scanning ? "#25bf6b" : "#9aacbf";
}
