import { icons } from "../icons.js";
import { state } from "../state.js";
import { setSwitch } from "../utils.js";

export function controlsHTML(){
  return `
<section class="controlsCol">
  <div class="cardTitle">Controls</div>

  <div id="cameraToggle" class="toggleCard">
    <div class="toggleLeft">${icons.camera} Camera</div>
    <div id="cameraSwitch" class="switch on"></div>
  </div>

  <div id="soundToggle" class="toggleCard">
    <div class="toggleLeft">${icons.sound} Sound</div>
    <div id="soundSwitch" class="switch"></div>
  </div>

  <div id="armToggle" class="toggleCard">
    <div class="toggleLeft">${icons.arm} Tapper</div>
    <div id="armSwitch" class="switch"></div>
  </div>

  <div id="doubleMatchToggle" class="toggleCard">
    <div class="toggleLeft">Double Match</div>
    <div id="doubleMatchSwitch" class="switch"></div>
  </div>

  <div id="spinAutoToggle" class="toggleCard">
    <div class="toggleLeft">Spin Auto</div>
    <div id="spinAutoSwitch" class="switch"></div>
  </div>
</section>
  `;
}

export function bindControls(actions){
  const bindToggle = (id, action) => {
    const element = document.getElementById(id);
    if(!element) return;

    element.addEventListener("pointerup", event => {
      event.preventDefault();
      event.stopPropagation();
      action();
    });
  };

  bindToggle("cameraToggle", actions.toggleCamera);
  bindToggle("soundToggle", actions.toggleSound);
  bindToggle("armToggle", actions.toggleArm);
  bindToggle("doubleMatchToggle", actions.toggleDoubleMatch);
  bindToggle("spinAutoToggle", actions.toggleSpinAuto);
}

export function renderControls(){
  const hw = state.appSettings?.hardware || {};

  setSwitch("cameraSwitch", state.cameraOn);
  setSwitch("soundSwitch", !!hw.buzzerEnabled);
  setSwitch("armSwitch", !!hw.actuatorEnabled);
  setSwitch(
    "doubleMatchSwitch",
    !!state.appSettings?.doubleMatchEnabled
  );

  setSwitch(
    "spinAutoSwitch",
    state.appSettings?.spinAutoEnabled !== false
  );
}
